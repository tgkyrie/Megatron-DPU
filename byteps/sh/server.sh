#!/bin/bash
set -e

# ===== RDMA & BytePS 基本配置 =====
export BYTEPS_RDMA_RX_DEPTH=${BYTEPS_RDMA_RX_DEPTH:-1024}
export BYTEPS_RDMA_START_DEPTH=${BYTEPS_RDMA_START_DEPTH:-64}
export DMLC_ENABLE_RDMA=${DMLC_ENABLE_RDMA:-ibverbs}

export DMLC_NUM_WORKER=${DMLC_NUM_WORKER:-2}
export DMLC_NUM_SERVER=${DMLC_NUM_SERVER:-2}
export DMLC_ROLE=server

# 启用 IPC 和异步通信以提升性能
export BYTEPS_ENABLE_IPC=${BYTEPS_ENABLE_IPC:-0}
# export BYTEPS_ENABLE_ASYNC=${BYTEPS_ENABLE_ASYNC:-1}

# 开启为 1 后 server 端也按优先级调度分片，可能在高并发场景更均衡，但会增加一些调度开销。
export BYTEPS_SERVER_ENABLE_SCHEDULE=${BYTEPS_SERVER_ENABLE_SCHEDULE:-1}

# 分片大小：默认 4MB，可以外面 export BYTEPS_PARTITION_BYTES 覆盖
export BYTEPS_PARTITION_BYTES=${BYTEPS_PARTITION_BYTES:-4096000}
export BYTEPS_SCHEDULING_CREDIT=${BYTEPS_SCHEDULING_CREDIT:-0}

export DMLC_INTERFACE=${DMLC_INTERFACE:-ens39f1np1}

detect_ip() {
  ip -4 addr show dev "${DMLC_INTERFACE}" 2>/dev/null \
    | awk '/inet / {print $2}' | cut -d/ -f1 | head -n1
}

# scheduler 的地址：默认 192.168.1.10:9010，可被外部 export 覆盖
export DMLC_PS_ROOT_URI=${DMLC_PS_ROOT_URI:-192.168.1.10}
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-9010}

# 本机 IP：自动按网卡查
export DMLC_NODE_HOST=${DMLC_NODE_HOST:-$(detect_ip)}

# server engine 线程
export BYTEPS_SERVER_ENGINE_THREAD=${BYTEPS_SERVER_ENGINE_THREAD:-64}

# ===== NUMA 绑定：默认按网卡对应 NUMA 节点自动检测 =====
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

NUMA_NODE=${NUMA_NODE:-$(detect_numa_from_iface)}
CPU_LIST=${CPU_LIST:-$(detect_cpulist_from_iface)}
NUMACTL_PREFIX=$(build_numactl_prefix "${NUMA_NODE}" "${CPU_LIST}")

echo "[server] NUMA_NODE=${NUMA_NODE} CPU_LIST='${CPU_LIST}' ROOT=${DMLC_PS_ROOT_URI}:${DMLC_PS_ROOT_PORT} "\
     "NUM_WORKER=${DMLC_NUM_WORKER} NUM_SERVER=${DMLC_NUM_SERVER} "\
     "IF=${DMLC_INTERFACE} HOST=${DMLC_NODE_HOST} ENGINE_THREAD=${BYTEPS_SERVER_ENGINE_THREAD}"

# 启动 server（后台），并确保退出时释放端口/进程
${NUMACTL_PREFIX} python3 - <<'PY' &
import byteps.server  # import 即启动
PY
PID=$!
STOPPED=0
cleanup() {
  if [ $STOPPED -eq 1 ]; then return; fi
  STOPPED=1
  echo "[server] stopping"
  kill -TERM $PID 2>/dev/null || true
  wait $PID 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM
wait $PID
