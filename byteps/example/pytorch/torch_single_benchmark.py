#!/usr/bin/env python3
"""
单机 PyTorch synthetic benchmark，参数与 BytePS 示例对齐：
  --model vgg16 --batch-size 32 --num-warmup-batches 10 --num-batches-per-iter 10 --num-iters 10
"""
import argparse
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Single-node PyTorch Synthetic Benchmark",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", type=str, default="vgg16", help="model name")
    p.add_argument("--batch-size", type=int, default=32, help="batch size")
    p.add_argument("--num-warmup-batches", type=int, default=10, help="warmup batches")
    p.add_argument("--num-batches-per-iter", type=int, default=10, help="batches per iter")
    p.add_argument("--num-iters", type=int, default=10, help="benchmark iterations")
    p.add_argument("--num-classes", type=int, default=1000, help="classes")
    p.add_argument("--no-cuda", action="store_true", default=False, help="disable CUDA")
    p.add_argument("--profiler", action="store_true", default=False, help="enable autograd profiler")
    return p.parse_args()


def main():
    args = parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.cuda:
        cudnn.benchmark = True

    model = getattr(models, args.model)(num_classes=args.num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    datasets = []
    for _ in range(100):
        data = torch.rand(args.batch_size, 3, 224, 224, device=device)
        target = torch.randint(0, args.num_classes, (args.batch_size,), device=device)
        datasets.append((data, target))
    data_index = 0

    def benchmark_step():
        nonlocal data_index
        data, target = datasets[data_index % len(datasets)]
        data_index += 1
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    def log(msg):
        print(msg)
        sys.stdout.flush()

    log(f"Model: {args.model}")
    log(f"Batch size: {args.batch_size}")
    log(f"Device: {device}")

    log("Running warmup...")
    for _ in range(args.num_warmup_batches):
        benchmark_step()

    log("Running benchmark...")
    img_secs = []
    enable_profiling = args.profiler
    for i in range(args.num_iters):
        start = time.perf_counter()
        for _ in range(args.num_batches_per_iter):
            benchmark_step()
        dur = time.perf_counter() - start
        img_sec = args.batch_size * args.num_batches_per_iter / dur
        img_secs.append(img_sec)
        log(f"Iter #{i}: {img_sec:.1f} img/sec")

    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log(f"Img/sec per device: {img_sec_mean:.1f} +- {img_sec_conf:.1f}")


if __name__ == "__main__":
    main()
