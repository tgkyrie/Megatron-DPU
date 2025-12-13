#!/usr/bin/env python3
"""
纯通信 allreduce 基准，不跑模型。
通过 size_mb 控制张量大小，可模拟 VGG16 级别的梯度量。

用法示例（两机各起一进程）:
  MASTER_ADDR=192.168.1.12 MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 \
    NCCL_SOCKET_IFNAME=ens39f1np1 NCCL_IB_HCA=mlx5_1 NCCL_IB_GID_INDEX=0 \
    python3 allreduce_bench.py --size-mb 520 --iters 10
  另一台把 RANK 设为 1，其余相同。
"""
import argparse
import os
import time

import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser(description="AllReduce-only benchmark")
    p.add_argument("--size-mb", type=float, default=520.0, help="单次 allreduce 张量大小 (MB)")
    p.add_argument("--iters", type=int, default=10, help="正式迭代次数")
    p.add_argument("--warmup", type=int, default=5, help="热身次数")
    p.add_argument("--dtype", type=str, default="float32", help="数据类型 (float32/float16)")
    return p.parse_args()


def main():
    args = parse_args()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    elems = int(args.size_mb * 1024 * 1024 / torch.tensor([], dtype=dtype).element_size())
    tensor = torch.ones(elems, device="cuda", dtype=dtype)

    def do_allreduce():
        dist.all_reduce(tensor, async_op=False)
        torch.cuda.synchronize()

    if rank == 0:
        print(f"[allreduce] world_size={world_size} size_mb={args.size_mb} dtype={args.dtype}")

    # warmup
    for _ in range(args.warmup):
        do_allreduce()

    dist.barrier()

    for i in range(args.iters):
        t0 = time.time()
        do_allreduce()
        dur_ms = (time.time() - t0) * 1000.0
        if rank == 0:
            bytes_len = args.size_mb * 1024 * 1024
            bw = bytes_len * world_size / dur_ms / 1e6  # MB/s 聚合带宽
            print(f"[allreduce] iter={i} time_ms={dur_ms:.3f} agg_bw_MBps={bw:.1f}")

    dist.barrier()


if __name__ == "__main__":
    main()
