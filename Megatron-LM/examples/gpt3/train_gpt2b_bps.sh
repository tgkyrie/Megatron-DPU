#!/bin/bash

############################################
# Environment variables

# use ***--use-dpu-reduce ***

############################################
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_DEBUG_FILE=nccl_${HOSTNAME}_rank${NODE_RANK}.log
export NCCL_DEBUG_LEVEL=TRACE

export CUDA_VISIBLE_DEVICES=1   # 多机单卡
export NODE_RANK=${NODE_RANK:-0}

export BYTEPS_RDMA_RX_DEPTH=${BYTEPS_RDMA_RX_DEPTH:-512}
export BYTEPS_RDMA_START_DEPTH=${BYTEPS_RDMA_START_DEPTH:-16}

export DMLC_ENABLE_RDMA=${DMLC_ENABLE_RDMA:-ibverbs}
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=${DMLC_NUM_SERVER:-2}
export DMLC_INTERFACE=${DMLC_INTERFACE:-ens39f1np1}
export DMLC_PS_ROOT_URI=${DMLC_PS_ROOT_URI:-192.168.1.10}   # scheduler：gpu01
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-9010}
# Set per host to its local IP (machine-specific).
export DMLC_NODE_HOST=${DMLC_NODE_HOST:-$(ip -4 -o addr show dev "$DMLC_INTERFACE" | awk '{print $4}' | cut -d/ -f1 | head -n1)}

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$DMLC_INTERFACE}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$DMLC_INTERFACE}

TOKENIZER_ARG=${3:-"MOCK"}   # HF tokenizer path or MOCK
DATA_ARG=${4:-"MOCK"}        # data prefix or MOCK

############################################
# Distributed setup
############################################
GPUS_PER_NODE=${GPUS_PER_NODE:-1}  # 每台机器上的 GPU 数
NUM_NODES=${NUM_NODES:-2}     # 机器总数
MASTER_ADDR=${MASTER_ADDR:-192.168.1.10}
MASTER_PORT=${MASTER_PORT:-19002}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# BytePS: number of worker nodes (per host), not processes.
export DMLC_NUM_WORKER=$NUM_NODES
# BytePS: local processes per host (nproc_per_node).
export BYTEPS_LOCAL_SIZE=$GPUS_PER_NODE

torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  pretrain_gpt.py \
  --num-layers 24 \
  --hidden-size 2048 \
  --num-attention-heads 16 \
  --micro-batch-size 4 \
  --use-flash-attn \
  --global-batch-size 16 \
  --max-position-embeddings 3072 \
  --seq-length 3072 \
  --vocab-size 50257 \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-file ../vocab/vocab.json \
  --merge-file ../vocab/merges.txt \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --transformer-impl local \
  --no-persist-layer-norm \
  --mock-data \
  --fp16 \
  --train-iters 30 \
  --lr 0.00015 \
  --min-lr 1.0e-5 \
  --lr-decay-style cosine \
  --log-interval 10 \
  --eval-iters 0 \
  --timing-log-level 2 \
  --recompute-activations