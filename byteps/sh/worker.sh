#!/bin/bash
set -e

# ==== 手动指定这个 worker 是第几号 ====
WORKER_ID=${WORKER_ID:-0}   # 可以在外面 export WORKER_ID 覆盖

# ===== RDMA & BytePS 基本配置 =====
export DMLC_ENABLE_RDMA=${DMLC_ENABLE_RDMA:-ibverbs}
export DMLC_NUM_WORKER=${DMLC_NUM_WORKER:-2}
export DMLC_NUM_SERVER=${DMLC_NUM_SERVER:-2}
export DMLC_ROLE=worker
export DMLC_WORKER_ID=${WORKER_ID}

export DMLC_INTERFACE=${DMLC_INTERFACE:-ens39f1np1}

detect_ip() {
  ip -4 addr show dev "${DMLC_INTERFACE}" 2>/dev/null \
    | awk '/inet / {print $2}' | cut -d/ -f1 | head -n1
}

# scheduler 地址：默认 192.168.1.10:9010
export DMLC_PS_ROOT_URI=${DMLC_PS_ROOT_URI:-192.168.1.10}
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-9010}

# 本机 IP 自动查
export DMLC_NODE_HOST=${DMLC_NODE_HOST:-$(detect_ip)}

export BYTEPS_RDMA_RX_DEPTH=${BYTEPS_RDMA_RX_DEPTH:-1024}
export BYTEPS_RDMA_START_DEPTH=${BYTEPS_RDMA_START_DEPTH:-32}

echo "[worker ${WORKER_ID}] ROOT=${DMLC_PS_ROOT_URI}:${DMLC_PS_ROOT_PORT} "\
     "HOST=${DMLC_NODE_HOST} NUM_WORKER=${DMLC_NUM_WORKER} NUM_SERVER=${DMLC_NUM_SERVER} "\
     "IF=${DMLC_INTERFACE}"

exec bpslaunch python3 /usr/local/byteps/example/pytorch/benchmark_byteps.py \
  --model vgg16 --num-iters 10
