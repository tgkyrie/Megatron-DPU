#!/bin/bash
set -e
# 用法：
# MASTER_ADDR=192.168.1.11 WORLD_SIZE=4 RANK=0 bash /usr/local/byteps/sh/worker_ddp.sh
# RANK=1 bash /usr/local/byteps/sh/worker_ddp.sh

#################### 基本配置（可按需覆盖） ####################
# 类似 BytePS 的 scheduler：选一台机器作为 master
MASTER_ADDR=${MASTER_ADDR:-192.168.1.10}
MASTER_PORT=${MASTER_PORT:-29500}

# 总机器数（每机一个进程，一个 GPU）
WORLD_SIZE=${WORLD_SIZE:-2}

# 本机 rank（0 ~ WORLD_SIZE-1）
RANK=${RANK:-0}


# 启用 NCCL 的 IB/RDMA
export NCCL_IB_DISABLE=0

# 让 NCCL 用 ens39f1np1 这个网卡建连接（和你训练通信的 IP 一致）
export NCCL_SOCKET_IFNAME=ens39f1np1

# 明确指定用 ACTIVE 的这块 HCA
export NCCL_IB_HCA=mlx5_1
export NCCL_IB_GID_INDEX=4
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET


# 每台只用一张卡，比如第 0 张；可在外部覆盖
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# NCCL 建议指定网卡，对齐 BytePS 的 DMLC_INTERFACE
export NCCL_SOCKET_IFNAME=ens39f1np1

#################### 不太需要改的部分 ####################
export MASTER_ADDR MASTER_PORT
export WORLD_SIZE RANK

# 单进程单卡，本机就是 local_rank 0
export LOCAL_RANK=${LOCAL_RANK:-0}

echo "[DDP] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE RANK=$RANK"
echo "[DDP] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES LOCAL_RANK=$LOCAL_RANK"

#################### 启动 DDP benchmark ####################
python3 /usr/local/byteps/example/pytorch/torch_ddp_benchmark.py \
  --model vgg16 --num-iters 10 "$@"
