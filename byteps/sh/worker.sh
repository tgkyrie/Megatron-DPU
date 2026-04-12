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

# 启用 IPC 和异步通信以提升性能
export BYTEPS_ENABLE_IPC=${BYTEPS_ENABLE_IPC:-0}
# export BYTEPS_ENABLE_ASYNC=${BYTEPS_ENABLE_ASYNC:-0}

# 分片大小：默认 4MB，可以外面 export BYTEPS_PARTITION_BYTES 覆盖
export BYTEPS_PARTITION_BYTES=${BYTEPS_PARTITION_BYTES:-1024000}

export BYTEPS_SCHEDULING_CREDIT=${BYTEPS_SCHEDULING_CREDIT:-0}

export DMLC_USE_GDR=${DMLC_USE_GDR:-1}  # GPU Direct RDMA，默认关闭

extract_primary_hca() {
  local hca="${NCCL_IB_HCA:-}"
  hca="${hca%%,*}"
  hca="${hca%%:*}"
  hca="${hca#^}"
  hca="${hca#=}"
  printf '%s' "${hca}"
}

detect_iface_from_hca() {
  local hca="$1"
  if [[ -z "${hca}" ]]; then
    return
  fi
  ls "/sys/class/infiniband/${hca}/device/net" 2>/dev/null | head -n1
}

detect_ip() {
  ip -4 addr show dev "${DMLC_INTERFACE}" 2>/dev/null \
    | awk '/inet / {print $2}' | cut -d/ -f1 | head -n1
}

export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1}
PRIMARY_HCA=$(extract_primary_hca)
AUTO_DMLC_INTERFACE=$(detect_iface_from_hca "${PRIMARY_HCA}")
export DMLC_INTERFACE=${DMLC_INTERFACE:-${AUTO_DMLC_INTERFACE:-ens39f1np1}}

# scheduler 地址：默认 192.168.1.10:9010
export DMLC_PS_ROOT_URI=${DMLC_PS_ROOT_URI:-192.168.1.10}
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-9010}

# 本机 IP 自动查
export DMLC_NODE_HOST=${DMLC_NODE_HOST:-$(detect_ip)}

# 本机 GPU 拓扑：单机多卡需显式告诉 BytePS
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export BYTEPS_LOCAL_SIZE=${BYTEPS_LOCAL_SIZE:-1}   # 本机 GPU 数
export BYTEPS_LOCAL_RANK=${BYTEPS_LOCAL_RANK:-0}   # 本进程使用的本地 GPU 编号

# RDMA队列深度配置 - 适度增加以修复ENOMEM错误，避免内存不足
export BYTEPS_RDMA_RX_DEPTH=${BYTEPS_RDMA_RX_DEPTH:-1024}
export BYTEPS_RDMA_START_DEPTH=${BYTEPS_RDMA_START_DEPTH:-32}

echo "[worker ${WORKER_ID}] ROOT=${DMLC_PS_ROOT_URI}:${DMLC_PS_ROOT_PORT} "\
     "HOST=${DMLC_NODE_HOST} NUM_WORKER=${DMLC_NUM_WORKER} NUM_SERVER=${DMLC_NUM_SERVER} "\
     "IF=${DMLC_INTERFACE}"

exec bpslaunch python3 /usr/local/byteps/example/pytorch/benchmark_byteps.py \
  --model vgg16 --num-iters 10 "$@"