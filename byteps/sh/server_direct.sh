#!/bin/bash
set -e

# Direct per-GPU process server launcher. Start one shell for each server
# process. This does not imply legacy GDR.

GPU_WORKERS_PER_NODE=${GPU_WORKERS_PER_NODE:-${GPUS_PER_NODE:-1}}
NUM_NODES=${NUM_NODES:-1}
WORLD_SIZE=${WORLD_SIZE:-$((NUM_NODES * GPU_WORKERS_PER_NODE))}
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
export DMLC_ROLE=server
if [ "${BYTEPS_USE_PREFERRED_RANK:-0}" != "0" ]; then
  SERVER_ID=${SERVER_ID:-0}
  export DMLC_SERVER_ID=${DMLC_SERVER_ID:-${SERVER_ID}}
else
  SERVER_ID=${SERVER_ID:-auto}
  unset DMLC_SERVER_ID
fi

export BYTEPS_ENABLE_IPC=${BYTEPS_ENABLE_IPC:-0}
export BYTEPS_SERVER_ENABLE_SCHEDULE=${BYTEPS_SERVER_ENABLE_SCHEDULE:-1}
export BYTEPS_SCHEDULING_CREDIT=${BYTEPS_SCHEDULING_CREDIT:-0}
export BYTEPS_OMP_THREAD_PER_GPU=${BYTEPS_OMP_THREAD_PER_GPU:-4}
export BYTEPS_RDMA_RX_DEPTH=${BYTEPS_RDMA_RX_DEPTH:-512}
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

detect_numa_from_iface() {
  cat "/sys/class/net/${DMLC_INTERFACE}/device/numa_node" 2>/dev/null || echo "-1"
}

detect_cpulist_from_iface() {
  cat "/sys/class/net/${DMLC_INTERFACE}/device/local_cpulist" 2>/dev/null || echo ""
}

build_numactl_prefix() {
  local node="$1"
  local cpulist="$2"

  if ! command -v numactl >/dev/null 2>&1; then
    echo ""
    return
  fi

  if [ -n "${cpulist}" ]; then
    if [ -n "${node}" ] && [ "${node}" != "-1" ]; then
      echo "numactl --physcpubind=${cpulist} --membind=${node}"
    else
      echo "numactl --physcpubind=${cpulist} --localalloc"
    fi
    return
  fi

  if [ -n "${node}" ] && [ "${node}" != "-1" ]; then
    echo "numactl --cpunodebind=${node} --membind=${node}"
  else
    echo ""
  fi
}

export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1}
PRIMARY_HCA=$(extract_primary_hca)
AUTO_DMLC_INTERFACE=$(detect_iface_from_hca "${PRIMARY_HCA}")
export DMLC_INTERFACE=${DMLC_INTERFACE:-${AUTO_DMLC_INTERFACE:-ens39f1np1}}
export DMLC_PS_ROOT_URI=${DMLC_PS_ROOT_URI:-192.168.1.10}
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-9010}
export DMLC_NODE_HOST=${DMLC_NODE_HOST:-$(detect_ip)}

NUMA_NODE=${NUMA_NODE:-$(detect_numa_from_iface)}
CPU_LIST=${CPU_LIST:-$(detect_cpulist_from_iface)}
NUMACTL_PREFIX=$(build_numactl_prefix "${NUMA_NODE}" "${CPU_LIST}")

echo "[direct-server ${SERVER_ID}] NUMA_NODE=${NUMA_NODE} CPU_LIST='${CPU_LIST}' "\
     "ROOT=${DMLC_PS_ROOT_URI}:${DMLC_PS_ROOT_PORT} HOST=${DMLC_NODE_HOST} "\
     "NUM_WORKER=${DMLC_NUM_WORKER} NUM_SERVER=${DMLC_NUM_SERVER} IF=${DMLC_INTERFACE}"

${NUMACTL_PREFIX} python3 - <<'PY' &
import byteps.server
PY
PID=$!
STOPPED=0
cleanup() {
  if [ $STOPPED -eq 1 ]; then return; fi
  STOPPED=1
  echo "[direct-server ${SERVER_ID}] stopping"
  kill -TERM $PID 2>/dev/null || true
  wait $PID 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM
wait $PID
