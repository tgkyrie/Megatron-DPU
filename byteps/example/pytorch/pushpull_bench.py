#!/usr/bin/env python3
"""
纯通信 BytePS push_pull 基准，不跑模型、不用 backward。

通过 --size-mb 控制张量大小，模拟大梯度（如 VGG16 / Transformer）。
输出指标与 NCCL allreduce bench 对齐。
"""

import argparse
import os
import time

import torch
import byteps.torch as bps
from byteps.torch.ops import push_pull_group_sync_inplace as byteps_push_pull_group
from byteps.torch.ops import get_pushpull_speed
from byteps.torch.ops_gdr import allreduce


def parse_args():
    p = argparse.ArgumentParser(description="BytePS push_pull-only benchmark")
    p.add_argument("--size-mb", type=float, default=520.0,
                   help="单次 push_pull 张量大小 (MB)")
    p.add_argument("--iters", type=int, default=50,
                   help="正式迭代次数")
    p.add_argument("--warmup", type=int, default=5,
                   help="热身次数")
    p.add_argument("--dtype", type=str, default="float16",
                   help="数据类型 (float32/float16)")
    p.add_argument("--cpu-tensor",default=False,action="store_true",
                   help="cpu tensor")
    p.add_argument("--check", action="store_true",
                   help="先做一次正确性校验，再进入 benchmark")
    p.add_argument("--check-only", action="store_true",
                   help="只做一次正确性校验，不跑 benchmark")
    p.add_argument("--check-size-mb", type=float, default=4.0,
                   help="正确性校验时使用的张量大小 (MB)")
    return p.parse_args()

def push_pull_grad_group_sync(tensor):
    name="test"

    handle, grad_count = byteps_push_pull_group(tensor, average=True,
            name="Gradient."+name)
    return handle, grad_count

def do_allreduce(tensor, name):
    return allreduce(tensor, False, True, name)


def validate_collective(args, rank, world_size, dtype):
    elem_size = torch.tensor([], dtype=dtype).element_size()
    elems = max(1, int(args.check_size_mb * 1024 * 1024 / elem_size))
    expected = float(sum(range(1, world_size + 1))) / float(world_size)

    if args.cpu_tensor:
        tensor = torch.full((elems,), float(rank + 1), dtype=dtype)
    else:
        tensor = torch.full((elems,), float(rank + 1), device="cuda", dtype=dtype)

    do_allreduce(tensor, "gdr_check")
    if not args.cpu_tensor:
        torch.cuda.synchronize()

    max_err = (tensor.float() - expected).abs().max().item()
    tol = 5e-3 if dtype == torch.float16 else 1e-6
    print(
        f"[byteps-pushpull-check] rank={rank} expected={expected:.6f} "
        f"max_err={max_err:.6e} tol={tol:.6e}"
    )
    if max_err > tol:
        raise RuntimeError(
            f"GDR push_pull check failed on rank={rank}: "
            f"max_err={max_err} > tol={tol}"
        )

def main():
    args = parse_args()

    # ---------------------------
    # BytePS init
    # ---------------------------
    bps.init()
    rank = bps.rank()
    world_size = bps.size()

    local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("BYTEPS_LOCAL_RANK", bps.local_rank()))
    )
    if not args.cpu_tensor:
        print("Using GPU Tensor")
        torch.cuda.set_device(local_rank)

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    elem_size = torch.tensor([], dtype=dtype).element_size()

    elems = int(args.size_mb * 1024 * 1024 / elem_size)
    if args.cpu_tensor:
        tensor = torch.ones(elems, dtype=dtype)
    else:
        tensor = torch.ones(elems, device="cuda", dtype=dtype)

    def do_push_pull():
        handle, gc = push_pull_grad_group_sync(tensor)
        return handle
    if rank == 0:
        print(
            f"[byteps-pushpull] "
            f"world_size={world_size} size_mb={args.size_mb} dtype={args.dtype}"
        )

    if args.check or args.check_only:
        validate_collective(args, rank, world_size, dtype)
        if rank == 0:
            print("[byteps-pushpull-check] validation passed")
        if args.check_only:
            return

    # ---------------------------
    # warmup
    # ---------------------------
    for _ in range(args.warmup):
        do_allreduce(tensor, "gdr_bench")

    # ---------------------------
    # benchmark
    # ---------------------------
    for i in range(args.iters):
        time.sleep(0.5)
        t0 = time.time()
        do_allreduce(tensor, "gdr_bench")
        dur_ms = (time.time() - t0) * 1000.0

        if True:
            bytes_len = args.size_mb * 1024 * 1024
            sec = dur_ms / 1000.0

            per_rank_GBps = bytes_len / sec / 1e9
            agg_GBps = (bytes_len * world_size) / sec / 1e9
            bus_GBps = (
                bytes_len * (2 * (world_size - 1) / world_size)
            ) / sec / 1e9

            print(
                f"[byteps-pushpull] iter={i} time_ms={dur_ms:.3f} "
                f"per_rank_GBps={per_rank_GBps:.2f} "
                f"agg_GB/s={agg_GBps:.2f} "
                f"bus_GB/s={bus_GBps:.2f}"
            )




if __name__ == "__main__":
    main()
