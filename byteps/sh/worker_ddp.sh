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

# 每台只用一张卡，比如第 0 张；可在外部覆盖
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 启用 NCCL 的 IB/RDMA（请按实际网卡/HCA 调整 IFNAME、HCA、GID_INDEX）
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens39f1np1}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}

#################### 不太需要改的部分 ####################
export MASTER_ADDR MASTER_PORT
export WORLD_SIZE RANK

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,GRAPH,INIT,NET   # COLL 会打印每次 collective 的 count/bytes 和参与 rank
export NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log     # 避免刷屏，各进程各写一个文件


# 单进程单卡，本机就是 local_rank 0
export LOCAL_RANK=${LOCAL_RANK:-0}

echo "[DDP] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE RANK=$RANK"
echo "[DDP] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES LOCAL_RANK=$LOCAL_RANK"
echo "[DDP] NCCL_IB_DISABLE=$NCCL_IB_DISABLE NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
echo "[DDP] NCCL_DEBUG=$NCCL_DEBUG NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"

#################### 启动 DDP benchmark ####################
python3 /usr/local/byteps/example/pytorch/torch_ddp_benchmark.py \
  --model vgg16 --num-iters 10 "$@"
