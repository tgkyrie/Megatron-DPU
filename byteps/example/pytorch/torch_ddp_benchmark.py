#!/usr/bin/env python3
"""
PyTorch DDP synthetic benchmark，格式与 BytePS 示例保持一致。
可选 profiler（仅 rank0 记录 trace 与通信热点）。
"""
import argparse
import os
import sys
import timeit

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch DDP Synthetic Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="resnet50", help="model to benchmark")
    parser.add_argument("--batch-size", type=int, default=32, help="input batch size")
    parser.add_argument("--num-warmup-batches", type=int, default=10, help="warmup batches")
    parser.add_argument("--num-batches-per-iter", type=int, default=10, help="batches per iter")
    parser.add_argument("--num-iters", type=int, default=10, help="benchmark iterations")
    parser.add_argument("--num-classes", type=int, default=1000, help="classes")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disable CUDA")
    parser.add_argument(
        "--profiler",
        action="store_true",
        default=False,
        help="enable autograd profiler on rank 0，并导出 trace",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="./traces_ddp",
        help="profile 导出的目录（仅 rank0 会写入）",
    )
    return parser.parse_args()


def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def main():
    args = parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    rank, world_size, local_rank = setup_ddp()

    if args.cuda:
        torch.cuda.set_device(local_rank)
        cudnn.benchmark = True

    model = getattr(models, args.model)(num_classes=args.num_classes)
    if args.cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank] if args.cuda else None
    )

    # fake data
    datasets = []
    targets = []
    for _ in range(100):
        data = torch.rand(args.batch_size, 3, 224, 224)
        target = torch.randint(0, args.num_classes, (args.batch_size,))
        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        datasets.append(data)
        targets.append(target)
    data_index = 0

    def benchmark_step():
        nonlocal data_index
        data = datasets[data_index % len(datasets)]
        target = targets[data_index % len(targets)]
        data_index += 1
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    def log(msg):
        if rank == 0:
            print(msg)
            sys.stdout.flush()

    device = "GPU" if args.cuda else "CPU"
    log(f"Model: {args.model}")
    log(f"Batch size: {args.batch_size}")
    log(f"Number of {device}s: {world_size}")

    log("Running warmup...")
    timeit.timeit(benchmark_step, number=args.num_warmup_batches)

    log("Running benchmark...")
    img_secs = []
    enable_profiling = args.profiler and (rank == 0)

    if enable_profiling:
        from torch.profiler import profile, ProfilerActivity

        acts = [ProfilerActivity.CPU]
        if args.cuda:
            acts.append(ProfilerActivity.CUDA)
        os.makedirs(args.trace_dir, exist_ok=True)
        trace_path = os.path.join(args.trace_dir, "ddp_rank0_trace.json")

        with profile(
            activities=acts,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for x in range(args.num_iters):
                t = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
                img_sec = args.batch_size * args.num_batches_per_iter / t
                log(f"Iter #{x}: {img_sec:.1f} img/sec per {device}")
                img_secs.append(img_sec)
            prof.export_chrome_trace(trace_path)
            nccl_rows = [
                e
                for e in prof.key_averages()
                if ("nccl" in e.key.lower()) or ("all_reduce" in e.key.lower())
            ]
            log("---- Profiler (nccl/all_reduce related) ----")
            for e in nccl_rows[:10]:
                log(
                    f"{e.key}: cuda_time_total={e.cuda_time_total:.2f}us "
                    f"cpu_time_total={e.cpu_time_total:.2f}us"
                )
    else:
        for x in range(args.num_iters):
            t = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
            img_sec = args.batch_size * args.num_batches_per_iter / t
            log(f"Iter #{x}: {img_sec:.1f} img/sec per {device}")
            img_secs.append(img_sec)

    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log(f"Img/sec per {device}: {img_sec_mean:.1f} +- {img_sec_conf:.1f}")
    log(
        f"Total img/sec on {world_size} {device}(s): "
        f"{world_size * img_sec_mean:.1f} +- {world_size * img_sec_conf:.1f}"
    )


if __name__ == "__main__":
    main()
