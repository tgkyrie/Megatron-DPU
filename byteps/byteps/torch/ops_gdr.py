import torch
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

from byteps.torch.ops import _do_push_pull_async,synchronize

# ===== 全局通信 Buffer 池 =====
class _CommBufferPool:
    """
    为 (numel, dtype, device) 预分配固定 GPU buffer。
    支持双缓冲（用于 overlap copy + communication）
    """
    def __init__(self):
        # key: (numel, dtype_str, device_id) -> [buffer0, buffer1, ...]
        self._buffers: Dict[Tuple[int, str, int], List[torch.Tensor]] = defaultdict(list)
        self._max_buffers_per_key = 2  # 双缓冲足够

    def get_buffer(self, tensor: torch.Tensor) -> torch.Tensor:
        """获取一个与 tensor 同 shape/dtype/device 的固定 buffer"""
        numel = tensor.numel()
        dtype_str = str(tensor.dtype)
        device_id = tensor.device.index if tensor.device.type == 'cuda' else -1
        key = (numel, dtype_str, device_id)

        if not self._buffers[key]:
            # 预分配多个 buffer（双缓冲）
            for _ in range(self._max_buffers_per_key):
                buf = torch.empty_like(tensor, device=tensor.device)
                self._buffers[key].append(buf)

        # 简单轮询（实际可用更复杂策略）
        buf = self._buffers[key].pop(0)
        self._buffers[key].append(buf)
        return buf

# 全局单例
_COMM_BUFFER_POOL = _CommBufferPool()


# # ===== EventGroup（同前）=====
# class EventGroup:
#     def __init__(self, events: List[TensorEvent]):
#         self.events = events

#     def wait(self):
#         for ev in self.events:
#             if ev is not None:
#                 ev.Sync()
#                 ev.Release()

#     def query(self) -> bool:
#         return all((ev is None) or ev.query() for ev in self.events)

def allreduce(
    tensor: torch.Tensor,
    async_op: bool = False,
    average: int =False,
    name=None,
) :
    if name == None:
        raise AssertionError("To manually call push_pull, you must specify a name by name=...")
    # --- 1. 获取固定通信 buffer（与 tensor 同 shape/dtype）---
    comm_buf = _COMM_BUFFER_POOL.get_buffer(tensor)

    # --- 2. 拷贝 input tensor → comm_buf（HtoD）---
    # 使用当前 stream（通常由 PyTorch 上下文管理）
    comm_buf.copy_(tensor)
    handle=_do_push_pull_async(comm_buf,comm_buf,False,name,0,0)

    tensor.copy_(comm_buf)
    if async_op:
        return handle
    
    return synchronize(handle)
    
# ===== 核心函数：ps_allreduce with fixed buffer =====
# def ps_allreduce(
#     tensor: torch.Tensor,
#     async_op: bool = False,
#     partition_bytes: int = 2 * 1024 * 1024,  # 2MB
#     pipeline_depth: int = 2,
#     base_key: Optional[int] = None
# ) -> Optional[EventGroup]:
#     """
#     PS AllReduce using pre-allocated fixed buffers for RDMA.
#     - Input tensor address may change each iteration.
#     - Communication happens on fixed-address buffers (MR registered once).
#     """
#     if base_key is None:
#         raise ValueError("base_key must be provided")

#     if not tensor.is_cuda:
#         raise NotImplementedError("Only CUDA tensor supported")

#     # --- 1. 获取固定通信 buffer（与 tensor 同 shape/dtype）---
#     comm_buf = _COMM_BUFFER_POOL.get_buffer(tensor)

#     # --- 2. 拷贝 input tensor → comm_buf（HtoD）---
#     # 使用当前 stream（通常由 PyTorch 上下文管理）
#     comm_buf.copy_(tensor)

#     # --- 3. 分块参数（基于 comm_buf）---
#     element_size = comm_buf.element_size()
#     total_bytes = comm_buf.numel() * element_size

#     alignment = 64
#     chunk_size_bytes = min(partition_bytes, total_bytes)
#     chunk_size_bytes = (chunk_size_bytes // alignment) * alignment
#     if chunk_size_bytes == 0:
#         chunk_size_bytes = alignment

#     chunk_size_elems = chunk_size_bytes // element_size
#     num_chunks = (total_bytes + chunk_size_bytes - 1) // chunk_size_bytes
#     if num_chunks == 0:
#         num_chunks = 1

#     chunk_keys = [base_key + i for i in range(num_chunks)]

#     # --- 4. 流水线通信（操作 comm_buf，而非原 tensor）---
#     # push_events: List[Optional[TensorEvent]] = [None] * num_chunks
#     # pull_events: List[Optional[TensorEvent]] = [None] * num_chunks
#     pushpull_ev: List[Optional[TensorEvent]]=[None]*num_chunks
#     for i in range(num_chunks):
#         start = i * chunk_size_elems
#         end = min(start + chunk_size_elems, comm_buf.numel())
#         chunk = comm_buf[start:end]

#         # # Push current chunk
#         # push_ev = stepmesh.push(chunk, chunk_keys[i], need_event=True)
#         # push_events[i] = push_ev

#         # # Pull previous chunk
#         # if i > 0:
#         #     prev_start = (i - 1) * chunk_size_elems
#         #     prev_end = min(prev_start + chunk_size_elems, comm_buf.numel())
#         #     prev_chunk = comm_buf[prev_start:prev_end]
#         #     pull_ev = stepmesh.pull(prev_chunk, chunk_keys[i - 1], need_event=True)
#         #     pull_events[i - 1] = pull_ev
#         pushpull_ev[i]=bps.push_pull(chunk,chunk_keys)
#         # 控制流水线深度
#         # if i >= pipeline_depth:
#         #     idx_to_wait = i - pipeline_depth
#         #     if pull_events[idx_to_wait] is not None:
#         #         pull_events[idx_to_wait].Sync()
#         #         pull_events[idx_to_wait].Release()
#         #         pull_events[idx_to_wait] = None

#     # Pull last chunk
#     # last_start = (num_chunks - 1) * chunk_size_elems
#     # last_end = min(last_start + chunk_size_elems, comm_buf.numel())
#     # last_chunk = comm_buf[last_start:last_end]
#     # pull_ev = stepmesh.pull(last_chunk, chunk_keys[num_chunks - 1], need_event=True)
#     # pull_events[num_chunks - 1] = pull_ev

#     # --- 5. 拷贝结果 comm_buf → original tensor（DtoH）---
#     # 注意：必须等所有 pull 完成后再拷贝！
#     all_pull_events = [ev for ev in pull_events if ev is not None]
    
#     if async_op:
#         # 异步：返回事件组，调用方负责 wait + 拷贝
#         class _AsyncResult:
#             def __init__(self, event_group, src_buf, dst_tensor):
#                 self.event_group = event_group
#                 self.src_buf = src_buf
#                 self.dst_tensor = dst_tensor

#             def wait(self):
#                 self.event_group.wait()
#                 self.dst_tensor.copy_(self.src_buf)

#         return _AsyncResult(EventGroup(all_pull_events), comm_buf, tensor)
#     else:
#         # 同步：立即等待 + 拷贝
#         final_group = EventGroup(all_pull_events)
#         final_group.wait()
#         tensor.copy_(comm_buf)
#         return None