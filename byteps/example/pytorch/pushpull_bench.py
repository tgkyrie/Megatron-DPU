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


def parse_args():
    p = argparse.ArgumentParser(description="BytePS push_pull-only benchmark")
    p.add_argument("--size-mb", type=float, default=520.0,
                   help="单次 push_pull 张量大小 (MB)")
    p.add_argument("--iters", type=int, default=50,
                   help="正式迭代次数")
    p.add_argument("--warmup", type=int, default=5,
                   help="热身次数")
    p.add_argument("--dtype", type=str, default="float32",
                   help="数据类型 (float32/float16)")
    p.add_argument("--cpu-tensor",default=False,action="store_true",
                   help="cpu tensor")
    return p.parse_args()

def push_pull_grad_group_sync(tensor):
    name="test"

    handle, grad_count = byteps_push_pull_group(tensor, average=True,
            name="Gradient."+name)
    return handle, grad_count

def main():
    args = parse_args()

    # ---------------------------
    # BytePS init
    # ---------------------------
    bps.init()
    rank = bps.rank()
    world_size = bps.size()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
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
        # bps.push_pull(tensor, average=True, name="bench")
        handle,gc=push_pull_grad_group_sync(tensor)
        if not args.cpu_tensor:
            torch.cuda.synchronize()
        return handle
    if rank == 0:
        print(
            f"[byteps-pushpull] "
            f"world_size={world_size} size_mb={args.size_mb} dtype={args.dtype}"
        )

    # ---------------------------
    # warmup
    # ---------------------------
    for _ in range(args.warmup):
        handle=do_push_pull()
        bps.synchronize(handle=handle)

    # ---------------------------
    # benchmark
    # ---------------------------
    for i in range(args.iters):
        time.sleep(0.5)
        t0 = time.time()
        handle=do_push_pull()
        bps.synchronize(handle=handle)
        dur_ms = (time.time() - t0) * 1000.0

        if rank == 0:
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
