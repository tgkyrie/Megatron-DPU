#!/usr/bin/env python3
"""本地 GDR MR 探针。

用途：
1. 直接用 ``cudaMalloc`` 分配一块 GPU 内存，然后对指定 offset 的子窗口做
   ``ibv_reg_mr``。
2. 用 PyTorch 分配一块 CUDA tensor，或者直接对现有 tensor 做同样测试。

推荐运行方式：

```bash
PYTHONPATH=/home/shenzu/Megatron-DPU/byteps \
python3 -m byteps.tools.gdr_probe \
  --source cudamalloc \
  --size-mib 392 \
  --offsets-mib 218,219,220 \
  --window-mib 1 \
  --ib-dev mlx5_0 \
  --gpu 0
```

如果要在已有 Python 进程里直接测真实 tensor：

```python
from byteps.tools.gdr_probe import run_probe_from_tensor

results = run_probe_from_tensor(
    tensor=my_tensor,
    offsets_mib=[218, 219, 220],
    window_mib=1,
    ib_dev="mlx5_0",
)
```
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Callable, Iterable


IBV_ACCESS_LOCAL_WRITE = 1
IBV_ACCESS_REMOTE_WRITE = 2
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12
MIB = 1024 * 1024
KIB = 1024


class ProbeError(RuntimeError):
    """Probe runtime error."""


def _load_library(names: Iterable[str], use_errno: bool = False) -> ctypes.CDLL:
    tried: list[str] = []
    for name in names:
        if not name:
            continue
        tried.append(name)
        try:
            return ctypes.CDLL(name, use_errno=use_errno)
        except OSError:
            continue
    raise ProbeError(f"无法加载动态库，已尝试: {', '.join(tried) or '<none>'}")


def _fmt_ptr(value: int | None) -> str:
    if value is None:
        return "None"
    return f"0x{value:x}"


def _fmt_size(value: int) -> str:
    mib = value / MIB
    if mib.is_integer():
        return f"{int(mib)}MiB"
    return f"{mib:.2f}MiB"


def _align_down_pow2(value: int, align: int) -> int:
    return value & ~(align - 1)


def _align_up_pow2(value: int, align: int) -> int:
    return (value + align - 1) & ~(align - 1)


def _round_up_pow2(value: int) -> int:
    rounded = 1
    while rounded < value:
        rounded <<= 1
    return rounded


def _byteps_exact_candidates(
    *,
    ptr: int,
    window_bytes: int,
    range_base: int,
    range_size: int,
    cache_window_bytes: int,
) -> list[tuple[str, int, int]]:
    """生成和 rdma_van.h 的 cuda-exact 路径一致的有界候选窗口。"""
    page = os.sysconf("SC_PAGESIZE") or 4096
    exact_local_cache = max(cache_window_bytes, page)
    if exact_local_cache & (exact_local_cache - 1):
        exact_local_cache = _round_up_pow2(exact_local_cache)

    req_end = ptr + window_bytes
    alloc_start = range_base
    alloc_end = range_base + range_size
    candidates: list[tuple[str, int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(mode: str, cand_start: int, cand_end: int) -> None:
        cand_start = max(cand_start, alloc_start)
        cand_end = min(cand_end, alloc_end)
        cand_start = _align_down_pow2(cand_start, page)
        cand_end = _align_up_pow2(cand_end, page)
        cand_start = max(cand_start, alloc_start)
        cand_end = min(cand_end, alloc_end)
        if cand_start > ptr or cand_end < req_end or cand_end <= cand_start:
            return
        key = (cand_start, cand_end)
        if key in seen:
            return
        seen.add(key)
        candidates.append((mode, cand_start, cand_end - cand_start))

    add(
        f"exact-cache-window:{exact_local_cache}",
        _align_down_pow2(ptr, exact_local_cache),
        _align_up_pow2(req_end, exact_local_cache),
    )
    add(f"exact-request:{window_bytes}", ptr, req_end)

    max_back = max(0, exact_local_cache - window_bytes)
    max_back = min(max_back, max(0, ptr - alloc_start))
    last_back = 0
    back = page
    while back <= max_back and back > last_back:
        add(f"exact-backoff-tail:{back}", ptr - back, req_end)
        last_back = back
        next_back = back << 1
        if next_back <= back or next_back > max_back:
            next_back = max_back
        back = next_back

    return candidates


def _parse_csv_ints(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("至少要给一个 offset")
    return values


def _parse_range_ints(text: str) -> list[int]:
    parts = [item.strip() for item in text.split(":")]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "range 格式必须是 start:end 或 start:end:step"
        )
    try:
        start = int(parts[0])
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
    except ValueError as exc:
        raise argparse.ArgumentTypeError("range 里只能有整数") from exc
    if step <= 0:
        raise argparse.ArgumentTypeError("step 必须大于 0")
    if end < start:
        raise argparse.ArgumentTypeError("end 必须大于等于 start")
    return list(range(start, end + 1, step))


def _access_flags_from_names(names: Iterable[str]) -> int:
    mapping = {
        "local_write": IBV_ACCESS_LOCAL_WRITE,
        "remote_write": IBV_ACCESS_REMOTE_WRITE,
    }
    flags = 0
    for name in names:
        key = name.strip().lower()
        if key not in mapping:
            raise ProbeError(f"不支持的 access flag: {name}")
        flags |= mapping[key]
    return flags


class IbvMr(ctypes.Structure):
    _fields_ = [
        ("context", ctypes.c_void_p),
        ("pd", ctypes.c_void_p),
        ("addr", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        ("handle", ctypes.c_uint32),
        ("lkey", ctypes.c_uint32),
        ("rkey", ctypes.c_uint32),
    ]


class CudaRuntime:
    def __init__(self) -> None:
        self.lib = _load_library(
            [
                ctypes.util.find_library("cudart"),
                "libcudart.so.12",
                "libcudart.so",
            ]
        )
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaSetDevice.restype = ctypes.c_int
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        self.lib.cudaMemset.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self._cuda_mem_get_address_range = getattr(
            self.lib, "cudaMemGetAddressRange", None
        )
        if self._cuda_mem_get_address_range is not None:
            self._cuda_mem_get_address_range.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_void_p,
            ]
            self._cuda_mem_get_address_range.restype = ctypes.c_int

        self.driver = None
        self._cu_init = None
        self._cu_pointer_get_attribute = None
        self._cu_mem_get_address_range = None
        self._cu_get_error_string = None
        try:
            self.driver = _load_library(
                [
                    ctypes.util.find_library("cuda"),
                    "libcuda.so.1",
                    "libcuda.so",
                ]
            )
            self._cu_init = getattr(self.driver, "cuInit", None)
            if self._cu_init is not None:
                self._cu_init.argtypes = [ctypes.c_uint]
                self._cu_init.restype = ctypes.c_int
                self._cu_init(0)
            self._cu_get_error_string = getattr(self.driver, "cuGetErrorString", None)
            if self._cu_get_error_string is not None:
                self._cu_get_error_string.argtypes = [
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_char_p),
                ]
                self._cu_get_error_string.restype = ctypes.c_int

            self._cu_pointer_get_attribute = getattr(
                self.driver, "cuPointerGetAttribute", None
            )
            if self._cu_pointer_get_attribute is not None:
                self._cu_pointer_get_attribute.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_uint64,
                ]
                self._cu_pointer_get_attribute.restype = ctypes.c_int

            self._cu_mem_get_address_range = getattr(
                self.driver, "cuMemGetAddressRange", None
            )
            if self._cu_mem_get_address_range is None:
                self._cu_mem_get_address_range = getattr(
                    self.driver, "cuMemGetAddressRange_v2", None
                )
            if self._cu_mem_get_address_range is not None:
                self._cu_mem_get_address_range.argtypes = [
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_size_t),
                    ctypes.c_uint64,
                ]
                self._cu_mem_get_address_range.restype = ctypes.c_int
        except ProbeError:
            self.driver = None

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise ProbeError(
                f"{what} 失败: rc={rc}, msg={self.error_string(rc)}"
            )

    def error_string(self, rc: int) -> str:
        try:
            raw = self.lib.cudaGetErrorString(rc)
        except Exception:
            raw = None
        return raw.decode("utf-8", errors="replace") if raw else "<unknown>"

    def driver_error_string(self, rc: int) -> str:
        if self._cu_get_error_string is None:
            return "<unknown>"
        raw = ctypes.c_char_p()
        ret = self._cu_get_error_string(rc, ctypes.byref(raw))
        if ret != 0 or not raw.value:
            return "<unknown>"
        return raw.value.decode("utf-8", errors="replace")

    def set_device(self, device: int) -> None:
        self._check(self.lib.cudaSetDevice(device), f"cudaSetDevice({device})")

    def malloc(self, size_bytes: int) -> int:
        ptr = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(ptr), size_bytes), "cudaMalloc")
        return int(ptr.value)

    def free(self, ptr: int) -> None:
        self._check(self.lib.cudaFree(ctypes.c_void_p(ptr)), "cudaFree")

    def memset(self, ptr: int, value: int, size_bytes: int) -> None:
        self._check(
            self.lib.cudaMemset(ctypes.c_void_p(ptr), value, size_bytes),
            "cudaMemset",
        )

    def get_address_range(self, ptr: int) -> tuple[int | None, int | None]:
        if self._cuda_mem_get_address_range is not None:
            base = ctypes.c_void_p()
            size = ctypes.c_size_t()
            rc = self._cuda_mem_get_address_range(
                ctypes.byref(base), ctypes.byref(size), ctypes.c_void_p(ptr)
            )
            if rc == 0 and base.value:
                return int(base.value), int(size.value)

        if self._cu_pointer_get_attribute is not None:
            base = ctypes.c_uint64()
            size = ctypes.c_size_t()
            rc1 = self._cu_pointer_get_attribute(
                ctypes.byref(base),
                CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                ctypes.c_uint64(ptr),
            )
            rc2 = self._cu_pointer_get_attribute(
                ctypes.byref(size),
                CU_POINTER_ATTRIBUTE_RANGE_SIZE,
                ctypes.c_uint64(ptr),
            )
            if rc1 == 0 and rc2 == 0 and base.value:
                return int(base.value), int(size.value)

        if self._cu_mem_get_address_range is not None:
            base = ctypes.c_uint64()
            size = ctypes.c_size_t()
            rc = self._cu_mem_get_address_range(
                ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(ptr)
            )
            if rc == 0 and base.value:
                return int(base.value), int(size.value)
        return None, None

    def prefault(self, ptr: int, size_bytes: int, step_bytes: int) -> None:
        if size_bytes == 0:
            return
        host = ctypes.create_string_buffer(1)
        for offset in range(0, size_bytes, step_bytes):
            self._check(
                self.lib.cudaMemcpy(
                    host,
                    ctypes.c_void_p(ptr + offset),
                    1,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                ),
                f"cudaMemcpy(prefault offset={offset})",
            )
        tail_offset = size_bytes - 1
        if tail_offset % step_bytes != 0:
            self._check(
                self.lib.cudaMemcpy(
                    host,
                    ctypes.c_void_p(ptr + tail_offset),
                    1,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                ),
                "cudaMemcpy(prefault tail)",
            )


class IbverbsRuntime:
    def __init__(self) -> None:
        self.lib = _load_library(
            [
                ctypes.util.find_library("ibverbs"),
                "libibverbs.so.1",
                "libibverbs.so",
            ],
            use_errno=True,
        )
        self.lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.ibv_free_device_list.restype = None
        self.lib.ibv_get_device_name.argtypes = [ctypes.c_void_p]
        self.lib.ibv_get_device_name.restype = ctypes.c_char_p
        self.lib.ibv_open_device.argtypes = [ctypes.c_void_p]
        self.lib.ibv_open_device.restype = ctypes.c_void_p
        self.lib.ibv_close_device.argtypes = [ctypes.c_void_p]
        self.lib.ibv_close_device.restype = ctypes.c_int
        self.lib.ibv_alloc_pd.argtypes = [ctypes.c_void_p]
        self.lib.ibv_alloc_pd.restype = ctypes.c_void_p
        self.lib.ibv_dealloc_pd.argtypes = [ctypes.c_void_p]
        self.lib.ibv_dealloc_pd.restype = ctypes.c_int
        self.lib.ibv_reg_mr.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.ibv_reg_mr.restype = ctypes.POINTER(IbvMr)
        self.lib.ibv_dereg_mr.argtypes = [ctypes.POINTER(IbvMr)]
        self.lib.ibv_dereg_mr.restype = ctypes.c_int

    def list_devices(self) -> list[str]:
        count = ctypes.c_int()
        dev_list = self.lib.ibv_get_device_list(ctypes.byref(count))
        if not dev_list:
            err = ctypes.get_errno()
            raise ProbeError(f"ibv_get_device_list 失败: errno={err} {os.strerror(err)}")
        try:
            names: list[str] = []
            for idx in range(count.value):
                dev = dev_list[idx]
                if not dev:
                    continue
                raw = self.lib.ibv_get_device_name(dev)
                names.append(raw.decode("utf-8", errors="replace"))
            return names
        finally:
            self.lib.ibv_free_device_list(dev_list)

    def open_pd(self, preferred_dev: str | None) -> tuple[str, ctypes.c_void_p, ctypes.c_void_p]:
        count = ctypes.c_int()
        dev_list = self.lib.ibv_get_device_list(ctypes.byref(count))
        if not dev_list:
            err = ctypes.get_errno()
            raise ProbeError(f"ibv_get_device_list 失败: errno={err} {os.strerror(err)}")
        chosen_name: str | None = None
        chosen_dev = None
        try:
            for idx in range(count.value):
                dev = dev_list[idx]
                if not dev:
                    continue
                raw = self.lib.ibv_get_device_name(dev)
                name = raw.decode("utf-8", errors="replace")
                if preferred_dev is None or preferred_dev == name:
                    chosen_name = name
                    chosen_dev = dev
                    break
            if chosen_dev is None or chosen_name is None:
                devices = ", ".join(self.list_devices())
                raise ProbeError(
                    f"找不到 IB 设备 {preferred_dev!r}，当前设备: {devices or '<none>'}"
                )
            ctx = self.lib.ibv_open_device(chosen_dev)
            if not ctx:
                err = ctypes.get_errno()
                raise ProbeError(
                    f"ibv_open_device({chosen_name}) 失败: errno={err} {os.strerror(err)}"
                )
            pd = self.lib.ibv_alloc_pd(ctx)
            if not pd:
                err = ctypes.get_errno()
                self.lib.ibv_close_device(ctx)
                raise ProbeError(
                    f"ibv_alloc_pd({chosen_name}) 失败: errno={err} {os.strerror(err)}"
                )
            return chosen_name, ctx, pd
        finally:
            self.lib.ibv_free_device_list(dev_list)

    def close_pd(self, ctx: ctypes.c_void_p, pd: ctypes.c_void_p) -> None:
        if pd:
            rc = self.lib.ibv_dealloc_pd(pd)
            if rc != 0:
                err = ctypes.get_errno()
                raise ProbeError(f"ibv_dealloc_pd 失败: errno={err} {os.strerror(err)}")
        if ctx:
            rc = self.lib.ibv_close_device(ctx)
            if rc != 0:
                err = ctypes.get_errno()
                raise ProbeError(
                    f"ibv_close_device 失败: errno={err} {os.strerror(err)}"
                )

    def reg_mr(
        self,
        pd: ctypes.c_void_p,
        ptr: int,
        size_bytes: int,
        access_flags: int,
    ) -> tuple[bool, int, str, ctypes.POINTER(IbvMr) | None]:
        ctypes.set_errno(0)
        mr = self.lib.ibv_reg_mr(pd, ctypes.c_void_p(ptr), size_bytes, access_flags)
        if mr:
            return True, 0, "Success", mr
        err = ctypes.get_errno()
        return False, err, os.strerror(err), None


@dataclass
class Allocation:
    source: str
    base_ptr: int
    size_bytes: int
    cleanup: Callable[[], None]
    keepalive: object | None = None


@dataclass
class ProbeResult:
    source: str
    offset_bytes: int
    window_ptr: int
    window_bytes: int
    range_base: int | None
    range_size: int | None
    prefault_ok: bool
    reg_ok: bool
    errno_num: int
    errno_msg: str
    lkey: int | None
    rkey: int | None
    reused: bool = False


def _alloc_cudamalloc(cuda: CudaRuntime, device: int, size_bytes: int) -> Allocation:
    cuda.set_device(device)
    ptr = cuda.malloc(size_bytes)
    cuda.memset(ptr, 0, size_bytes)

    def cleanup() -> None:
        cuda.free(ptr)

    return Allocation(
        source="cudamalloc",
        base_ptr=ptr,
        size_bytes=size_bytes,
        cleanup=cleanup,
    )


def _load_factory(factory_spec: str) -> Callable[..., object]:
    path, sep, func_name = factory_spec.partition(":")
    if not sep or not path or not func_name:
        raise ProbeError("--tensor-factory 必须是 '/abs/path/file.py:function'")
    module_name = f"_gdr_probe_factory_{abs(hash(factory_spec))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ProbeError(f"无法加载 factory 文件: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, func_name, None)
    if fn is None:
        raise ProbeError(f"{path} 里找不到函数 {func_name}")
    return fn


def _torch_dtype_from_name(torch_module: object, dtype_name: str) -> object:
    dtype = getattr(torch_module, dtype_name, None)
    if dtype is None:
        raise ProbeError(f"不支持的 torch dtype: {dtype_name}")
    return dtype


def _alloc_torch_tensor(
    device: int,
    size_bytes: int,
    dtype_name: str,
    factory_spec: str | None,
) -> Allocation:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ProbeError("PyTorch 模式需要先安装 torch") from exc

    torch.cuda.set_device(device)
    if factory_spec:
        factory = _load_factory(factory_spec)
        try:
            tensor = factory(size_bytes=size_bytes, device=device)
        except TypeError:
            tensor = factory(size_bytes, device)
    else:
        dtype = _torch_dtype_from_name(torch, dtype_name)
        tensor = torch.empty(size_bytes, dtype=dtype, device=f"cuda:{device}")

    if not getattr(tensor, "is_cuda", False):
        raise ProbeError("factory 返回的 tensor 不是 CUDA tensor")
    ptr = int(tensor.data_ptr())
    alloc_size = int(tensor.numel() * tensor.element_size())

    def cleanup() -> None:
        nonlocal tensor
        tensor = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return Allocation(
        source="pytorch",
        base_ptr=ptr,
        size_bytes=alloc_size,
        cleanup=cleanup,
        keepalive=tensor,
    )


def _probe_windows(
    *,
    source: str,
    base_ptr: int,
    offsets_bytes: list[int],
    window_bytes: int,
    ib_dev: str | None,
    access_flags: int,
    prefault: bool,
    prefault_step_bytes: int,
    keep_mr: bool,
    stop_on_fail: bool,
    cache_window_bytes: int | None,
    max_active_bytes: int | None,
    registration_policy: str,
) -> list[ProbeResult]:
    cuda = CudaRuntime()
    verbs = IbverbsRuntime()
    chosen_ib, ctx, pd = verbs.open_pd(ib_dev)
    print(
        f"[INFO] source={source} ib_dev={chosen_ib} window={window_bytes}B "
        f"({_fmt_size(window_bytes)}) access_flags={access_flags} "
        f"keep_mr={keep_mr} cache_window="
        f"{cache_window_bytes if cache_window_bytes is not None else 'None'} "
        f"policy={registration_policy}"
    )
    results: list[ProbeResult] = []
    registered_windows: list[tuple[int, int, ctypes.POINTER(IbvMr)]] = []
    total_new_mr = 0
    active_bytes = 0
    peak_active_bytes = 0
    pagesize = os.sysconf("SC_PAGESIZE") or 4096

    def evict_until_fits(incoming_bytes: int) -> None:
        nonlocal active_bytes
        if max_active_bytes is None:
            return
        while registered_windows and active_bytes + incoming_bytes > max_active_bytes:
            old_start, old_end, old_mr = registered_windows.pop(0)
            old_len = old_end - old_start
            rc = verbs.lib.ibv_dereg_mr(old_mr)
            if rc != 0:
                err = ctypes.get_errno()
                raise ProbeError(
                    f"ibv_dereg_mr(evict) 失败: errno={err} {os.strerror(err)}"
                )
            active_bytes -= old_len
            print(
                "[EVICT] "
                f"cache.ptr={_fmt_ptr(old_start)} cache.len={old_len} "
                f"active_bytes={active_bytes}"
            )

    try:
        for offset in offsets_bytes:
            ptr = base_ptr + offset
            range_base, range_size = cuda.get_address_range(ptr)
            prefault_ok = True

            for cached_start, cached_end, cached_mr in registered_windows:
                if ptr >= cached_start and ptr + window_bytes <= cached_end:
                    print(
                        "[REUSE] "
                        f"offset={offset}({_fmt_size(offset)}) "
                        f"ptr={_fmt_ptr(ptr)} len={window_bytes} "
                        f"cache.ptr={_fmt_ptr(cached_start)} "
                        f"cache.len={cached_end - cached_start}"
                    )
                    results.append(
                        ProbeResult(
                            source=source,
                            offset_bytes=offset,
                            window_ptr=ptr,
                            window_bytes=window_bytes,
                            range_base=range_base,
                            range_size=range_size,
                            prefault_ok=True,
                            reg_ok=True,
                            errno_num=0,
                            errno_msg="Success",
                            lkey=int(cached_mr.contents.lkey),
                            rkey=int(cached_mr.contents.rkey),
                            reused=True,
                        )
                    )
                    break
            if results and results[-1].offset_bytes == offset and results[-1].reused:
                continue

            candidates: list[tuple[str, int, int]]
            if registration_policy == "byteps-exact":
                exact_cache_window = cache_window_bytes or (4 * MIB)
                if range_base is None or range_size is None:
                    candidates = [("original", ptr, window_bytes)]
                else:
                    candidates = _byteps_exact_candidates(
                        ptr=ptr,
                        window_bytes=window_bytes,
                        range_base=range_base,
                        range_size=range_size,
                        cache_window_bytes=exact_cache_window,
                    )
            elif cache_window_bytes is not None:
                if cache_window_bytes <= 0:
                    raise ProbeError("cache_window_bytes 必须大于 0")
                reg_ptr = _align_down_pow2(ptr, cache_window_bytes)
                reg_size = (
                    _align_up_pow2(ptr + window_bytes, cache_window_bytes)
                    - reg_ptr
                )
                if range_base is not None and range_size is not None:
                    alloc_start = range_base
                    alloc_end = range_base + range_size
                    if reg_ptr < alloc_start:
                        reg_ptr = alloc_start
                    reg_end = min(reg_ptr + reg_size, alloc_end)
                    reg_size = reg_end - reg_ptr
                candidates = [("cache-window", reg_ptr, reg_size)]
            else:
                candidates = [("original", ptr, window_bytes)]

            if not candidates:
                raise ProbeError(
                    f"没有可用候选窗口 offset={offset} ptr={_fmt_ptr(ptr)} "
                    f"len={window_bytes} range.base={_fmt_ptr(range_base)} "
                    f"range.size={range_size}"
                )

            reg_ok = False
            errno_num = 0
            errno_msg = "Success"
            mr = None
            lkey = None
            rkey = None
            reg_ptr = candidates[-1][1]
            reg_size = candidates[-1][2]
            final_mode = candidates[-1][0]
            for mode, cand_ptr, cand_size in candidates:
                if cand_size <= 0:
                    continue
                req_end = ptr + window_bytes
                if cand_ptr > ptr or cand_ptr + cand_size < req_end:
                    raise ProbeError(
                        f"候选窗口未覆盖请求 mode={mode} ptr={_fmt_ptr(ptr)} "
                        f"len={window_bytes} cand.ptr={_fmt_ptr(cand_ptr)} "
                        f"cand.len={cand_size}"
                    )
                if range_base is not None and range_size is not None:
                    if cand_ptr < range_base or cand_ptr + cand_size > range_base + range_size:
                        raise ProbeError(
                            f"候选窗口越过 CUDA allocation mode={mode} "
                            f"cand.ptr={_fmt_ptr(cand_ptr)} cand.len={cand_size} "
                            f"range.base={_fmt_ptr(range_base)} range.size={range_size}"
                        )
                if registration_policy == "byteps-exact":
                    exact_limit = cache_window_bytes or (4 * MIB)
                    exact_limit = max(exact_limit, pagesize)
                    if exact_limit & (exact_limit - 1):
                        exact_limit = _round_up_pow2(exact_limit)
                    if cand_size > exact_limit:
                        raise ProbeError(
                            f"byteps-exact 候选超过本地窗口 mode={mode} "
                            f"cand.len={cand_size} limit={exact_limit}"
                        )

                reg_ptr = cand_ptr
                reg_size = cand_size
                final_mode = mode
                prefault_ok = True
                if prefault:
                    try:
                        cuda.prefault(reg_ptr, reg_size, prefault_step_bytes)
                    except Exception as exc:
                        prefault_ok = False
                        print(
                            f"[WARN] prefault 失败 offset={offset} "
                            f"mode={mode} ptr={_fmt_ptr(reg_ptr)}: {exc}"
                        )
                if keep_mr:
                    evict_until_fits(reg_size)
                reg_ok, errno_num, errno_msg, mr = verbs.reg_mr(
                    pd, reg_ptr, reg_size, access_flags
                )
                lkey = int(mr.contents.lkey) if mr else None
                rkey = int(mr.contents.rkey) if mr else None
                print(
                    "[TRY] "
                    f"offset={offset}({_fmt_size(offset)}) mode={mode} "
                    f"ptr={_fmt_ptr(ptr)} len={window_bytes} "
                    f"reg.ptr={_fmt_ptr(reg_ptr)} reg.len={reg_size} "
                    f"range.base={_fmt_ptr(range_base)} "
                    f"range.size={range_size if range_size is not None else 'None'} "
                    f"prefault={'ok' if prefault_ok else 'fail'} "
                    f"reg={'ok' if reg_ok else 'fail'} "
                    f"errno={errno_num}({errno_msg}) "
                    f"lkey={lkey} rkey={rkey}"
                )
                if reg_ok:
                    break

            print(
                "[RESULT] "
                f"offset={offset}({_fmt_size(offset)}) mode={final_mode} "
                f"ptr={_fmt_ptr(ptr)} len={window_bytes} "
                f"reg.ptr={_fmt_ptr(reg_ptr)} reg.len={reg_size} "
                f"range.base={_fmt_ptr(range_base)} "
                f"range.size={range_size if range_size is not None else 'None'} "
                f"prefault={'ok' if prefault_ok else 'fail'} "
                f"reg={'ok' if reg_ok else 'fail'} "
                f"errno={errno_num}({errno_msg}) "
                f"lkey={lkey} rkey={rkey}"
            )
            if mr and keep_mr:
                registered_windows.append((reg_ptr, reg_ptr + reg_size, mr))
                total_new_mr += 1
                active_bytes += reg_size
                if active_bytes > peak_active_bytes:
                    peak_active_bytes = active_bytes
            elif mr:
                rc = verbs.lib.ibv_dereg_mr(mr)
                if rc != 0:
                    err = ctypes.get_errno()
                    raise ProbeError(
                        f"ibv_dereg_mr 失败: errno={err} {os.strerror(err)}"
                    )
            results.append(
                ProbeResult(
                    source=source,
                    offset_bytes=offset,
                    window_ptr=ptr,
                    window_bytes=window_bytes,
                    range_base=range_base,
                    range_size=range_size,
                    prefault_ok=prefault_ok,
                    reg_ok=reg_ok,
                    errno_num=errno_num,
                    errno_msg=errno_msg,
                    lkey=lkey,
                    rkey=rkey,
                    reused=False,
                )
            )
            if stop_on_fail and not reg_ok:
                break
        print(
            f"[SUMMARY] total_offsets={len(results)} "
            f"active_mr={len(registered_windows)} "
            f"total_new_mr={total_new_mr} "
            f"peak_active_bytes={peak_active_bytes}"
        )
        return results
    finally:
        for _start, _end, mr in reversed(registered_windows):
            rc = verbs.lib.ibv_dereg_mr(mr)
            if rc != 0:
                err = ctypes.get_errno()
                raise ProbeError(
                    f"ibv_dereg_mr(held) 失败: errno={err} {os.strerror(err)}"
                )
        verbs.close_pd(ctx, pd)


def run_probe_from_cudamalloc(
    *,
    size_mib: int,
    offsets_mib: list[int],
    window_mib: int = 1,
    ib_dev: str | None = None,
    gpu: int = 0,
    prefault: bool = True,
    prefault_step_kib: int = 64,
    access_flags: int = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
    keep_mr: bool = False,
    stop_on_fail: bool = False,
    cache_window_mib: int | None = None,
    max_active_mib: int | None = None,
    registration_policy: str = "cache-window",
) -> list[ProbeResult]:
    cuda = CudaRuntime()
    alloc = _alloc_cudamalloc(cuda, gpu, size_mib * MIB)
    print(
        f"[INFO] alloc source={alloc.source} gpu={gpu} base={_fmt_ptr(alloc.base_ptr)} "
        f"size={alloc.size_bytes}({_fmt_size(alloc.size_bytes)})"
    )
    try:
        return _probe_windows(
            source=alloc.source,
            base_ptr=alloc.base_ptr,
            offsets_bytes=[x * MIB for x in offsets_mib],
            window_bytes=window_mib * MIB,
            ib_dev=ib_dev,
            access_flags=access_flags,
            prefault=prefault,
            prefault_step_bytes=prefault_step_kib * KIB,
            keep_mr=keep_mr,
            stop_on_fail=stop_on_fail,
            cache_window_bytes=(
                cache_window_mib * MIB if cache_window_mib is not None else None
            ),
            max_active_bytes=(
                max_active_mib * MIB if max_active_mib is not None else None
            ),
            registration_policy=registration_policy,
        )
    finally:
        alloc.cleanup()


def run_probe_from_tensor(
    *,
    tensor: object,
    offsets_mib: list[int],
    window_mib: int = 1,
    ib_dev: str | None = None,
    prefault: bool = True,
    prefault_step_kib: int = 64,
    access_flags: int = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
    keep_mr: bool = False,
    stop_on_fail: bool = False,
    cache_window_mib: int | None = None,
    max_active_mib: int | None = None,
    registration_policy: str = "cache-window",
) -> list[ProbeResult]:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ProbeError("run_probe_from_tensor 需要 torch") from exc

    if not isinstance(tensor, torch.Tensor):
        raise ProbeError("tensor 参数必须是 torch.Tensor")
    if not tensor.is_cuda:
        raise ProbeError("tensor 必须在 CUDA 上")

    device_index = tensor.device.index if tensor.device.index is not None else 0
    torch.cuda.set_device(device_index)
    base_ptr = int(tensor.data_ptr())
    size_bytes = int(tensor.numel() * tensor.element_size())
    print(
        f"[INFO] alloc source=existing-pytorch-tensor base={_fmt_ptr(base_ptr)} "
        f"size={size_bytes}({_fmt_size(size_bytes)}) "
        f"device={device_index} "
        f"dtype={tensor.dtype} shape={tuple(tensor.shape)} "
        f"contiguous={tensor.is_contiguous()}"
    )
    return _probe_windows(
        source="existing-pytorch-tensor",
        base_ptr=base_ptr,
        offsets_bytes=[x * MIB for x in offsets_mib],
        window_bytes=window_mib * MIB,
        ib_dev=ib_dev,
        access_flags=access_flags,
        prefault=prefault,
        prefault_step_bytes=prefault_step_kib * KIB,
        keep_mr=keep_mr,
        stop_on_fail=stop_on_fail,
        cache_window_bytes=(
            cache_window_mib * MIB if cache_window_mib is not None else None
        ),
        max_active_bytes=(
            max_active_mib * MIB if max_active_mib is not None else None
        ),
        registration_policy=registration_policy,
    )


def run_probe_from_new_torch_tensor(
    *,
    size_mib: int,
    offsets_mib: list[int],
    window_mib: int = 1,
    ib_dev: str | None = None,
    gpu: int = 0,
    torch_dtype: str = "uint8",
    tensor_factory: str | None = None,
    prefault: bool = True,
    prefault_step_kib: int = 64,
    access_flags: int = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
    keep_mr: bool = False,
    stop_on_fail: bool = False,
    cache_window_mib: int | None = None,
    max_active_mib: int | None = None,
    registration_policy: str = "cache-window",
) -> list[ProbeResult]:
    alloc = _alloc_torch_tensor(gpu, size_mib * MIB, torch_dtype, tensor_factory)
    print(
        f"[INFO] alloc source={alloc.source} gpu={gpu} base={_fmt_ptr(alloc.base_ptr)} "
        f"size={alloc.size_bytes}({_fmt_size(alloc.size_bytes)})"
    )
    try:
        return _probe_windows(
            source=alloc.source,
            base_ptr=alloc.base_ptr,
            offsets_bytes=[x * MIB for x in offsets_mib],
            window_bytes=window_mib * MIB,
            ib_dev=ib_dev,
            access_flags=access_flags,
            prefault=prefault,
            prefault_step_bytes=prefault_step_kib * KIB,
            keep_mr=keep_mr,
            stop_on_fail=stop_on_fail,
            cache_window_bytes=(
                cache_window_mib * MIB if cache_window_mib is not None else None
            ),
            max_active_bytes=(
                max_active_mib * MIB if max_active_mib is not None else None
            ),
            registration_policy=registration_policy,
        )
    finally:
        alloc.cleanup()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="本地 GPUDirect RDMA MR 注册探针",
    )
    parser.add_argument(
        "--source",
        choices=["cudamalloc", "pytorch"],
        default="cudamalloc",
        help="测试来源：纯 cudaMalloc，或者新分配的 PyTorch tensor",
    )
    parser.add_argument(
        "--size-mib",
        type=int,
        default=392,
        help="分配大小，单位 MiB，默认 392",
    )
    parser.add_argument(
        "--offsets-mib",
        type=_parse_csv_ints,
        default=[218, 219, 220],
        help="要测试的 offset 列表，单位 MiB，例如 218,219,220",
    )
    parser.add_argument(
        "--offset-range-mib",
        type=_parse_range_ints,
        default=None,
        help="用范围生成 offset，格式 start:end[:step]，闭区间，例如 0:220:1",
    )
    parser.add_argument(
        "--window-mib",
        type=int,
        default=1,
        help="每个子窗口大小，单位 MiB，默认 1",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA 设备号")
    parser.add_argument("--ib-dev", default=None, help="IB 设备名，例如 mlx5_0")
    parser.add_argument(
        "--prefault-step-kib",
        type=int,
        default=64,
        help="prefault 步长，单位 KiB，默认 64",
    )
    parser.add_argument(
        "--no-prefault",
        action="store_true",
        help="关闭 prefault，直接调 ibv_reg_mr",
    )
    parser.add_argument(
        "--torch-dtype",
        default="uint8",
        help="PyTorch 模式默认分配 dtype，默认 uint8",
    )
    parser.add_argument(
        "--tensor-factory",
        default=None,
        help="PyTorch 模式自定义工厂，格式 /abs/path/file.py:function",
    )
    parser.add_argument(
        "--access-flags",
        default="local_write,remote_write",
        help="MR access flags，逗号分隔，默认 local_write,remote_write",
    )
    parser.add_argument(
        "--list-ib-devs",
        action="store_true",
        help="只列出本机 IB 设备然后退出",
    )
    parser.add_argument(
        "--keep-mr",
        action="store_true",
        help="成功注册后不立即 dereg，直到本次 probe 结束才统一释放",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="遇到第一个注册失败就停止",
    )
    parser.add_argument(
        "--cache-window-mib",
        type=int,
        default=None,
        help="模拟按固定窗口缓存复用 MR，例如 2 表示按 2MiB 对齐窗口注册并复用",
    )
    parser.add_argument(
        "--max-active-mib",
        type=int,
        default=None,
        help="限制活跃已注册总字节，超出时按 LRU 先 dereg 老窗口",
    )
    parser.add_argument(
        "--registration-policy",
        choices=["cache-window", "byteps-exact"],
        default="cache-window",
        help=(
            "注册候选策略：cache-window 保持原探针行为；byteps-exact "
            "按 rdma_van.h 的 CUDA exact 路径依次尝试 4MiB cache-window、"
            "exact-request 和有界 backoff-tail"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    verbs = IbverbsRuntime()
    if args.list_ib_devs:
        for name in verbs.list_devices():
            print(name)
        return 0

    offsets_mib = (
        args.offset_range_mib if args.offset_range_mib is not None else args.offsets_mib
    )
    access_flags = _access_flags_from_names(args.access_flags.split(","))
    try:
        if args.source == "cudamalloc":
            results = run_probe_from_cudamalloc(
                size_mib=args.size_mib,
                offsets_mib=offsets_mib,
                window_mib=args.window_mib,
                ib_dev=args.ib_dev,
                gpu=args.gpu,
                prefault=not args.no_prefault,
                prefault_step_kib=args.prefault_step_kib,
                access_flags=access_flags,
                keep_mr=args.keep_mr,
                stop_on_fail=args.stop_on_fail,
                cache_window_mib=args.cache_window_mib,
                max_active_mib=args.max_active_mib,
                registration_policy=args.registration_policy,
            )
        else:
            results = run_probe_from_new_torch_tensor(
                size_mib=args.size_mib,
                offsets_mib=offsets_mib,
                window_mib=args.window_mib,
                ib_dev=args.ib_dev,
                gpu=args.gpu,
                torch_dtype=args.torch_dtype,
                tensor_factory=args.tensor_factory,
                prefault=not args.no_prefault,
                prefault_step_kib=args.prefault_step_kib,
                access_flags=access_flags,
                keep_mr=args.keep_mr,
                stop_on_fail=args.stop_on_fail,
                cache_window_mib=args.cache_window_mib,
                max_active_mib=args.max_active_mib,
                registration_policy=args.registration_policy,
            )
    except ProbeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if all(item.reg_ok for item in results):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
