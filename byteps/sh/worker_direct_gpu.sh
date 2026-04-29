#!/bin/bash
set -e

# Direct worker launcher: one shell process owns one GPU and registers as one
# ps-lite worker. By default this does not enable legacy GDR; it bypasses local
# multi-GPU aggregation by using BYTEPS_LOCAL_SIZE=1 and still uses the normal
# CPU-staging BytePS path.

GPU_ID=${GPU_ID:-0}
NODE_RANK=${NODE_RANK:-0}
GPU_WORKERS_PER_NODE=${GPU_WORKERS_PER_NODE:-${GPUS_PER_NODE:-1}}
NUM_NODES=${NUM_NODES:-1}
WORLD_SIZE=${WORLD_SIZE:-$((NUM_NODES * GPU_WORKERS_PER_NODE))}
WORKER_ID=${WORKER_ID:-$((NODE_RANK * GPU_WORKERS_PER_NODE + GPU_ID))}
NUM_SERVERS=${NUM_SERVERS:-${WORLD_SIZE}}
export WORLD_SIZE

if [ "${BYTEPS_USE_TP:-0}" = "1" ] || [ "${DMLC_PS_VAN_TYPE:-}" = "tp" ]; then
  unset DMLC_ENABLE_RDMA
  unset DMLC_ENABLE_UCX
  export DMLC_PS_VAN_TYPE=${DMLC_PS_VAN_TYPE:-tp}
else
  export DMLC_ENABLE_RDMA=${DMLC_ENABLE_RDMA:-ibverbs}
fi
export DMLC_NUM_WORKER=${DMLC_NUM_WORKER:-${WORLD_SIZE}}
export DMLC_NUM_SERVER=${DMLC_NUM_SERVER:-${NUM_SERVERS}}
export DMLC_ROLE=worker
export DMLC_WORKER_ID=${WORKER_ID}
export BYTEPS_GLOBAL_RANK=${BYTEPS_GLOBAL_RANK:-${WORKER_ID}}

export DMLC_USE_GDR=${DMLC_USE_GDR:-0}
if [ "${DMLC_USE_GDR}" != "0" ]; then
  echo "worker_direct_gpu.sh is using the legacy DMLC_USE_GDR path, not bccl-github GDR."
  export BYTEPS_GDR_DIRECT=${BYTEPS_GDR_DIRECT:-1}
else
  export BYTEPS_GDR_DIRECT=${BYTEPS_GDR_DIRECT:-0}
fi
unset BYTEPS_REDUCE_ROOTS

# This script masks one physical GPU per process, so the CUDA ordinal visible
# inside the process is always 0.
export CUDA_VISIBLE_DEVICES=${BYTEPS_CUDA_VISIBLE_DEVICES:-${GPU_ID}}
# bpslaunch uses NVIDIA_VISIBLE_DEVICES, not CUDA_VISIBLE_DEVICES, to decide
# how many local worker threads to start. Keep it single-device in direct mode.
export NVIDIA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export LOCAL_RANK=${LOCAL_RANK:-0}
export RANK=${RANK:-${WORKER_ID}}
export BYTEPS_VISIBLE_DEVICE=${BYTEPS_VISIBLE_DEVICE:-0}
export BYTEPS_LOCAL_SIZE=${BYTEPS_LOCAL_SIZE:-1}
export BYTEPS_LOCAL_RANK=${BYTEPS_LOCAL_RANK:-0}

# With BYTEPS_LOCAL_SIZE=1 every local process has local_rank=0, so its local
# socket/shared-memory names must be unique.
export BYTEPS_UUID=${BYTEPS_UUID:-byteps_direct_${DMLC_PS_ROOT_PORT:-9010}_w${WORKER_ID}}

export BYTEPS_ENABLE_IPC=${BYTEPS_ENABLE_IPC:-0}
export BYTEPS_PARTITION_BYTES=${BYTEPS_PARTITION_BYTES:-1024000}
export BYTEPS_SCHEDULING_CREDIT=${BYTEPS_SCHEDULING_CREDIT:-0}
export BYTEPS_RDMA_RX_DEPTH=${BYTEPS_RDMA_RX_DEPTH:-1024}
export BYTEPS_RDMA_START_DEPTH=${BYTEPS_RDMA_START_DEPTH:-32}

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
export DMLC_PS_ROOT_URI=${DMLC_PS_ROOT_URI:-192.168.1.10}
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-9010}
export DMLC_NODE_HOST=${DMLC_NODE_HOST:-$(detect_ip)}

echo "[direct-worker ${WORKER_ID}] GPU_ID=${GPU_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} "\
     "ROOT=${DMLC_PS_ROOT_URI}:${DMLC_PS_ROOT_PORT} HOST=${DMLC_NODE_HOST} "\
     "NUM_WORKER=${DMLC_NUM_WORKER} NUM_SERVER=${DMLC_NUM_SERVER} IF=${DMLC_INTERFACE} "\
     "UUID=${BYTEPS_UUID}"

exec bpslaunch python3 /usr/local/byteps/example/pytorch/benchmark_byteps.py \
  --model vgg16 --num-iters 10 "$@"
