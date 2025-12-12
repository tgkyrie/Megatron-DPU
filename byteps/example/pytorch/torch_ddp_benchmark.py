#!/usr/bin/env python3
"""
PyTorch DDP synthetic benchmark，计时和输出形式对齐 BytePS synthetic benchmark。

用法示例：
  torchrun --nproc_per_node=8 ddp_synthetic_benchmark.py \
    --model resnet50 --batch-size 32 \
    --num-warmup-batches 10 --num-batches-per-iter 10 --num-iters 10 \
    --profiler
"""

import argparse
import os
import sys
import timeit

import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch DDP Synthetic Benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model', type=str, default='resnet50',
                        help='model to benchmark')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size')

    parser.add_argument('--num-warmup-batches', type=int, default=10,
                        help='number of warm-up batches that don\'t count towards benchmark')
    parser.add_argument('--num-batches-per-iter', type=int, default=10,
                        help='number of batches per benchmark iteration')
    parser.add_argument('--num-iters', type=int, default=10,
                        help='number of benchmark iterations')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='number of classes')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # 和 BytePS 脚本一样的 profiler 开关
    parser.add_argument('--profiler', action='store_true', default=False,
                        help='enable autograd profiler on rank 0')

    return parser.parse_args()


def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return rank, world_size, local_rank


def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    rank, world_size, local_rank = setup_ddp()

    if args.cuda:
        torch.cuda.set_device(local_rank)
        cudnn.benchmark = True

    # 标准模型
    model = getattr(models, args.model)(num_classes=args.num_classes)
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # DDP 包装
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank] if args.cuda else None
    )

    # ---- Fake data：结构仿照 BytePS synthetic benchmark ----
    global target
    datasets = []
    for _ in range(100):
        data = torch.rand(args.batch_size, 3, 224, 224)
        # BytePS 脚本里写死 1000，这里保持一致
        target = torch.LongTensor(args.batch_size).random_() % 1000
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        datasets.append(data)
    global data_index
    data_index = 0

    def benchmark_step():
        global data_index, target

        data = datasets[data_index % len(datasets)]
        data_index += 1
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    def log(s, nl=True):
        if rank != 0:
            return
        print(s, end='\n' if nl else '')
        sys.stdout.flush()

    device = 'GPU' if args.cuda else 'CPU'
    log('Model: %s' % args.model)
    log('Batch size: %d' % args.batch_size)
    log('Number of %ss: %d' % (device, world_size))

    # ---- Warm-up：完全照 BytePS 写法 ----
    log('Running warmup...')
    timeit.timeit(benchmark_step, number=args.num_warmup_batches)

    # ---- Benchmark：完全对齐 BytePS 逻辑 ----
    log('Running benchmark...')
    img_secs = []
    # 注意：沿用 BytePS 中的按位与写法（功能等价于 and）
    enable_profiling = args.profiler & (rank == 0)

    with torch.autograd.profiler.profile(enabled=enable_profiling, use_cuda=True) as prof:
        for x in range(args.num_iters):
            t = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
            img_sec = args.batch_size * args.num_batches_per_iter / t
            log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
            img_secs.append(img_sec)

    # ---- Results：同样照搬 BytePS 打印格式 ----
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
    log('Total img/sec on %d %s(s): %.1f +-%.1f' %
        (world_size, device, world_size * img_sec_mean, world_size * img_sec_conf))


if __name__ == '__main__':
    main()
