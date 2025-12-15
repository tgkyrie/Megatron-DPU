#!/bin/bash
set -e

# ===== RDMA & BytePS 基本配置 =====
export BYTEPS_RDMA_RX_DEPTH=${BYTEPS_RDMA_RX_DEPTH:-1024}
export BYTEPS_RDMA_START_DEPTH=${BYTEPS_RDMA_START_DEPTH:-32}
export DMLC_ENABLE_RDMA=${DMLC_ENABLE_RDMA:-ibverbs}

export DMLC_NUM_WORKER=${DMLC_NUM_WORKER:-2}
export DMLC_NUM_SERVER=${DMLC_NUM_SERVER:-2}
export DMLC_ROLE=server

# 启用 IPC 和异步通信以提升性能
export BYTEPS_ENABLE_IPC=1
export BYTEPS_ENABLE_ASYNC=1

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

echo "[server] ROOT=${DMLC_PS_ROOT_URI}:${DMLC_PS_ROOT_PORT} "\
     "NUM_WORKER=${DMLC_NUM_WORKER} NUM_SERVER=${DMLC_NUM_SERVER} "\
     "IF=${DMLC_INTERFACE} HOST=${DMLC_NODE_HOST} "\
     "ENGINE_THREAD=${BYTEPS_SERVER_ENGINE_THREAD}"

# 捕捉 Ctrl+C/TERM，确保子进程被杀掉，端口释放
python3 - <<'PY' &
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
