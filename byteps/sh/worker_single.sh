#!/bin/bash
set -e

# 单机 PyTorch benchmark（无分布式），参数与 BytePS 示例相近
# 若机器有多块 GPU，可在外部或此处固定一块，例如：
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "[single] running single-node benchmark..."
exec python3 /usr/local/byteps/example/pytorch/torch_single_benchmark.py \
     --model vgg16 --batch-size 32 --num-warmup-batches 10 --num-batches-per-iter 10 --num-iters 10 \
     "$@"
