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
    parser.add_argument(
        "--no-trace",
        action="store_true",
        default=False,
        help="开启 profiler 时不导出 json trace，只输出控制台统计",
    )
    parser.add_argument(
        "--no-comm-log",
        dest="comm_log",
        action="store_false",
        help="关闭逐次 allreduce 的通信日志（默认开启）",
    )
    parser.add_argument(
        "--sync-comm",
        action="store_true",
        default=False,
        help="通信改为同步等待，测纯通信时用；默认异步允许与计算重叠",
    )
    parser.add_argument(
        "--bucket-cap-mb",
        type=float,
        default=None,
        help="DDP bucket_cap_mb（不设则用 PyTorch 默认）；调大可减少 bucket 数、趋向单大包",
    )
    parser.set_defaults(comm_log=True)
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

    ddp_kwargs = {}
    if args.bucket_cap_mb is not None:
        ddp_kwargs["bucket_cap_mb"] = args.bucket_cap_mb

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank] if args.cuda else None, **ddp_kwargs
    )

    # 记录每个 allreduce bucket 的通信字节数和耗时（日志仅 rank0 输出，默认开启，可通过 --no-comm-log 关闭）
    comm_state = {
        "iter": 0,
        "bucket": 0,
        "rank": rank,
        "bytes_sum": 0,
        "time_sum": 0.0,
        "bucket_count": 0,
    }
    if args.comm_log:
        import time
        import threading

        _comm_lock = threading.Lock()

        def log_hook(state, bucket):
            start = time.time()
            tensor = bucket.buffer()
            bytes_size = tensor.numel() * tensor.element_size()

            with _comm_lock:
                bucket_id = state["bucket"]
                state["bucket"] += 1
                state["bytes_sum"] += bytes_size
                state["bucket_count"] += 1

            if args.sync_comm:
                # 同步通信：测纯通信时间，牺牲与计算重叠
                dist.all_reduce(tensor, async_op=False)
                if args.cuda:
                    torch.cuda.synchronize()
                dur_ms = (time.time() - start) * 1000.0
                with _comm_lock:
                    state["time_sum"] += dur_ms
                if state["rank"] == 0:
                    print(
                        f"[comm] rank=0 iter={state['iter']} bucket={bucket_id} "
                        f"bytes={bytes_size} time_ms={dur_ms:.3f}"
                    )
                    sys.stdout.flush()
                fut = torch.futures.Future()
                fut.set_result(tensor / dist.get_world_size())
                return fut

            # 异步通信：保持默认的计算-通信重叠
            fut = dist.all_reduce(tensor, async_op=True).get_future()

            def _cb(fut):
                dur_ms = (time.time() - start) * 1000.0
                with _comm_lock:
                    state["time_sum"] += dur_ms
                if state["rank"] == 0:
                    print(
                        f"[comm] rank=0 iter={state['iter']} bucket={bucket_id} "
                        f"bytes={bytes_size} time_ms={dur_ms:.3f}"
                    )
                    sys.stdout.flush()
                t = fut.value()[0] / dist.get_world_size()
                return t

            return fut.then(_cb)

        ddp_model.register_comm_hook(comm_state, log_hook)


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
        stats_path = args.nccl_stats_file or os.path.join(args.trace_dir, "nccl_stats.txt")

        with profile(
            activities=acts,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for x in range(args.num_iters):
                comm_state["iter"] = x
                comm_state["bucket"] = 0
                comm_state["bytes_sum"] = 0
                comm_state["time_sum"] = 0.0
                comm_state["bucket_count"] = 0
                t = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
                if args.cuda:
                    torch.cuda.synchronize()
                img_sec = args.batch_size * args.num_batches_per_iter / t
                log(f"Iter #{x}: {img_sec:.1f} img/sec per {device}")
                img_secs.append(img_sec)
                if rank == 0:
                    print(
                        f"[comm] iter={x} buckets={comm_state['bucket_count']} "
                        f"total_bytes={comm_state['bytes_sum']} total_time_ms={comm_state['time_sum']:.3f}"
                    )
        if rank == 0:
            if not args.no_trace:
                prof.export_chrome_trace(trace_path)
            nccl_rows = [
                e
                for e in prof.key_averages()
                if ("nccl" in e.key.lower()) or ("all_reduce" in e.key.lower())
            ]
            total_cuda = sum(e.cuda_time_total for e in nccl_rows)
            total_cpu = sum(e.cpu_time_total for e in nccl_rows)
            log("---- Profiler (nccl/all_reduce related) ----")
            for e in nccl_rows[:10]:
                log(
                    f"{e.key}: cuda_time_total={e.cuda_time_total:.2f}us "
                    f"cpu_time_total={e.cpu_time_total:.2f}us count={e.count}"
                )
            log(f"Total nccl/all_reduce cuda_time={total_cuda:.2f}us cpu_time={total_cpu:.2f}us")
    else:
        for x in range(args.num_iters):
            comm_state["iter"] = x
            comm_state["bucket"] = 0
            comm_state["bytes_sum"] = 0
            comm_state["time_sum"] = 0.0
            comm_state["bucket_count"] = 0
            t = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
            if args.cuda:
                torch.cuda.synchronize()
            img_sec = args.batch_size * args.num_batches_per_iter / t
            log(f"Iter #{x}: {img_sec:.1f} img/sec per {device}")
            img_secs.append(img_sec)
            if rank == 0:
                print(
                    f"[comm] iter={x} buckets={comm_state['bucket_count']} "
                    f"total_bytes={comm_state['bytes_sum']} total_time_ms={comm_state['time_sum']:.3f}"
                )

    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log(f"Img/sec per {device}: {img_sec_mean:.1f} +- {img_sec_conf:.1f}")
    log(
        f"Total img/sec on {world_size} {device}(s): "
        f"{world_size * img_sec_mean:.1f} +- {world_size * img_sec_conf:.1f}"
    )


if __name__ == "__main__":
    main()
