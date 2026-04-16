#!/bin/bash

set -euo pipefail

############################################
# TP + DP BytePS validation script
#
# Notes:
# 1. This script enables both TP and DP with BytePS
# 2. DP gradient sync via --use-dpu-reduce
# 3. TP all-reduce via --use-dpu-tp-reduce
# 4. All communication goes through cross-node RDMA network
############################################

############################################
# Environment variables
############################################
export NODE_RANK=${NODE_RANK:-0}

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions_${USER:-root}}
mkdir -p "${TORCH_EXTENSIONS_DIR}"

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-COLL}
export NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-nccl_${HOSTNAME}_rank${NODE_RANK}.log}
export NCCL_DEBUG_LEVEL=${NCCL_DEBUG_LEVEL:-TRACE}

# Enable both DP and TP by default
export USE_DPU_DP=${USE_DPU_DP:-1}
export USE_DPU_TP=${USE_DPU_TP:-1}
export USE_OVERLAP=${USE_OVERLAP:-1}

# RDMA configuration
export DMLC_USE_GDR=${DMLC_USE_GDR:-0}
export BYTEPS_RDMA_RX_DEPTH=${BYTEPS_RDMA_RX_DEPTH:-512}
export BYTEPS_RDMA_START_DEPTH=${BYTEPS_RDMA_START_DEPTH:-16}
export DMLC_ENABLE_RDMA=${DMLC_ENABLE_RDMA:-ibverbs}
export BYTEPS_PARTITION_BYTES=${BYTEPS_PARTITION_BYTES:-4096000}

# Keep the current default behavior: multi-node single-GPU (GPU1)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

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

detect_ip_from_iface() {
    local iface="$1"
    if [[ -z "${iface}" ]]; then
        return
    fi
    ip -4 -o addr show dev "${iface}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n1
}

export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1}
PRIMARY_HCA=$(extract_primary_hca)
AUTO_DMLC_INTERFACE=$(detect_iface_from_hca "${PRIMARY_HCA}")
export DMLC_INTERFACE=${DMLC_INTERFACE:-${AUTO_DMLC_INTERFACE:-ens39f1np1}}
LOCAL_DETECTED_IP=$(detect_ip_from_iface "${DMLC_INTERFACE}")
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-9010}
export DMLC_NODE_HOST=${DMLC_NODE_HOST:-${LOCAL_DETECTED_IP}}

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$DMLC_INTERFACE}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$DMLC_INTERFACE}

MASTER_ADDR_VALUE=${MASTER_ADDR:-}
DMLC_PS_ROOT_URI_VALUE=${DMLC_PS_ROOT_URI:-}
if [[ -z "${MASTER_ADDR_VALUE}" && -n "${DMLC_PS_ROOT_URI_VALUE}" ]]; then
    MASTER_ADDR_VALUE="${DMLC_PS_ROOT_URI_VALUE}"
fi
if [[ -z "${DMLC_PS_ROOT_URI_VALUE}" && -n "${MASTER_ADDR_VALUE}" ]]; then
    DMLC_PS_ROOT_URI_VALUE="${MASTER_ADDR_VALUE}"
fi
if [[ -z "${MASTER_ADDR_VALUE}" ]]; then
    if [[ "${NODE_RANK}" == "0" && -n "${LOCAL_DETECTED_IP}" ]]; then
        MASTER_ADDR_VALUE="${LOCAL_DETECTED_IP}"
    else
        MASTER_ADDR_VALUE="192.168.1.10"
    fi
fi
if [[ -z "${DMLC_PS_ROOT_URI_VALUE}" ]]; then
    DMLC_PS_ROOT_URI_VALUE="${MASTER_ADDR_VALUE}"
fi
export MASTER_ADDR="${MASTER_ADDR_VALUE}"
export DMLC_PS_ROOT_URI="${DMLC_PS_ROOT_URI_VALUE}"

TOKENIZER_ARG=${3:-"MOCK"}
DATA_ARG=${4:-"MOCK"}

############################################
# NUMA binding
############################################
detect_numa_from_iface() {
    cat "/sys/class/net/${DMLC_INTERFACE}/device/numa_node" 2>/dev/null || echo "-1"
}

detect_cpulist_from_iface() {
    cat "/sys/class/net/${DMLC_INTERFACE}/device/local_cpulist" 2>/dev/null || echo ""
}

build_numactl_prefix() {
    local node="$1"
    local cpulist="$2"

    NUMACTL_PREFIX=()
    if ! command -v numactl >/dev/null 2>&1; then
        return
    fi

    if [[ -n "${cpulist}" ]]; then
        if [[ -n "${node}" && "${node}" != "-1" ]]; then
            NUMACTL_PREFIX=(numactl --physcpubind="${cpulist}" --membind="${node}")
        else
            NUMACTL_PREFIX=(numactl --physcpubind="${cpulist}" --localalloc)
        fi
        return
    fi

    if [[ -n "${node}" && "${node}" != "-1" ]]; then
        NUMACTL_PREFIX=(numactl --cpunodebind="${node}" --membind="${node}")
    fi
}

NUMA_NODE=${NUMA_NODE:-$(detect_numa_from_iface)}
CPU_LIST=${CPU_LIST:-$(detect_cpulist_from_iface)}
NUMACTL_PREFIX=()
build_numactl_prefix "${NUMA_NODE}" "${CPU_LIST}"

############################################
# Distributed setup
############################################
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NUM_NODES=${NUM_NODES:-4}
MASTER_PORT=${MASTER_PORT:-19002}
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

# BytePS: one worker per host (node)
export DMLC_NUM_WORKER=${DMLC_NUM_WORKER:-$NUM_NODES}
export DMLC_NUM_SERVER=${DMLC_NUM_SERVER:-$NUM_NODES}
export BYTEPS_LOCAL_SIZE=${BYTEPS_LOCAL_SIZE:-$GPUS_PER_NODE}

PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

############################################
# Model configuration: Qwen-3B
############################################
# TP_SIZE must be set, DP_SIZE is derived from WORLD_SIZE / TP_SIZE
TP_SIZE=${TP_SIZE:-2}
CP_SIZE=${CP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}

if [[ "${CP_SIZE}" -ne 1 || "${PP_SIZE}" -ne 1 ]]; then
    echo "[error] This script expects CP_SIZE=1 and PP_SIZE=1."
    echo "        CP_SIZE=${CP_SIZE}, PP_SIZE=${PP_SIZE}"
    exit 1
fi

if (( WORLD_SIZE % TP_SIZE != 0 )); then
    echo "[error] WORLD_SIZE must be divisible by TP_SIZE."
    echo "        WORLD_SIZE=${WORLD_SIZE}, TP_SIZE=${TP_SIZE}"
    exit 1
fi

DP_SIZE=$((WORLD_SIZE / TP_SIZE))
if [[ "${DP_SIZE}" -lt 1 ]]; then
    echo "[error] DP_SIZE must be at least 1."
    echo "        DP_SIZE=${DP_SIZE}, WORLD_SIZE=${WORLD_SIZE}, TP_SIZE=${TP_SIZE}"
    exit 1
fi

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * DP_SIZE))}

NUM_LAYERS=${NUM_LAYERS:-32}
HIDDEN_SIZE=${HIDDEN_SIZE:-2048}
NUM_HEADS=${NUM_HEADS:-16}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-11008}
KV_CHANNELS=${KV_CHANNELS:-128}

SEQ_LENGTH=${SEQ_LENGTH:-256}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-2048}

DTYPE=${DTYPE:-fp32}

DATA_CACHE_PATH="${PWD}/benchmark_cache_qwen_3b_tp_dp_byteps"
mkdir -p "$DATA_CACHE_PATH"

############################################
# torchrun args
############################################
DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
    --no_python
)

############################################
# Model args
############################################
MODEL_ARGS=(
    --transformer-impl local
    --use-mcore-models
    --no-persist-layer-norm
    --recompute-activations

    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_HEADS"
    --kv-channels "$KV_CHANNELS"

    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"

    --position-embedding-type rope
    --rotary-base 10000
    --rotary-percent 1.0

    --attention-dropout 0.0
    --hidden-dropout 0.0

    --swiglu
    --attention-backend fused

    --disable-bias-linear
    --apply-layernorm-1p
    --init-method-std 0.02

    --log-interval 1
    --train-iters 10
    --no-rope-fusion

    --tensor-model-parallel-size "$TP_SIZE"
    --context-parallel-size "$CP_SIZE"
)

# Enable DP reduce via BytePS
if [[ "${USE_DPU_DP}" == "1" ]]; then
    MODEL_ARGS+=(--use-dpu-reduce)
fi

# Enable TP reduce via BytePS
if [[ "${USE_DPU_TP}" == "1" ]]; then
    MODEL_ARGS+=(--use-dpu-tp-reduce)
fi

############################################
# Training args
############################################
TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"

    --lr 3.0e-4
    --min-lr 3.0e-5
    --lr-decay-style cosine
    --weight-decay 0.1
    --clip-grad 1.0

    --adam-beta1 0.9
    --adam-beta2 0.95

    --cross-entropy-loss-fusion
    --calculate-per-token-loss

    --manual-gc
    --empty-unused-memory-level 1
    --exit-duration-in-mins 235
)

if [[ "${USE_OVERLAP}" == "1" ]]; then
    TRAINING_ARGS+=(--overlap-grad-reduce)
fi

############################################
# FP8 args
############################################
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        --fp8-format hybrid
        --fp8-amax-history-len 1024
        --fp8-amax-compute-algo max
    )
elif [[ "$DTYPE" == "bf16" ]]; then
    DTYPE_ARGS+=(--bf16)
fi

############################################
# Data args
############################################
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]]; then
    DATA_ARGS_LIST+=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size 4096
        --split '99,1,0'
        --no-mmap-bin-files
        --num-workers 1
    )
else
    DATA_ARGS_LIST+=(
        --data-path "$DATA_ARG"
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "$TOKENIZER_ARG"
        --vocab-size 4096
        --data-cache-path "$DATA_CACHE_PATH"
        --split '99,1,0'
        --no-mmap-bin-files
        --num-workers 1
    )
fi

############################################
# Logging / eval
############################################
EVAL_AND_LOGGING_ARGS=(
    --eval-iters 10
    --eval-interval 100
    --save-interval 1000
    --log-throughput
    --ckpt-format torch_dist
    --distributed-timeout-minutes 60
    --timing-log-level 1
)

############################################
# Base command
############################################
CMD=(python "$PRETRAIN_SCRIPT_PATH"
    "${MODEL_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${DTYPE_ARGS[@]}"
    "${DATA_ARGS_LIST[@]}"
    "${EVAL_AND_LOGGING_ARGS[@]}"
)

############################################
# Launch
############################################
echo "[net] HCA=${PRIMARY_HCA:-unset} IF=${DMLC_INTERFACE} LOCAL_IP=${LOCAL_DETECTED_IP:-unset} MASTER_ADDR=${MASTER_ADDR} ROOT=${DMLC_PS_ROOT_URI}"
echo "[parallel] WORLD_SIZE=${WORLD_SIZE} TP_SIZE=${TP_SIZE} DP_SIZE=${DP_SIZE} PP_SIZE=${PP_SIZE} CP_SIZE=${CP_SIZE}"
echo "[byteps] USE_DPU_DP=${USE_DPU_DP} USE_DPU_TP=${USE_DPU_TP} DMLC_NUM_WORKER=${DMLC_NUM_WORKER} DMLC_NUM_SERVER=${DMLC_NUM_SERVER}"
echo "[overlap] USE_OVERLAP=${USE_OVERLAP}"
echo "[batch] MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "[numa] NUMA_NODE=${NUMA_NODE} CPU_LIST='${CPU_LIST}' IF=${DMLC_INTERFACE} HOST=${DMLC_NODE_HOST}"

"${NUMACTL_PREFIX[@]}" torchrun "${DISTRIBUTED_ARGS[@]}" bash -c '
export DMLC_ROLE=worker
export DMLC_NUM_WORKER='"$DMLC_NUM_WORKER"'
export DMLC_NUM_SERVER='"$DMLC_NUM_SERVER"'
export DMLC_PS_ROOT_URI='"$DMLC_PS_ROOT_URI"'
export DMLC_PS_ROOT_PORT='"$DMLC_PS_ROOT_PORT"'
export DMLC_NODE_HOST='"$DMLC_NODE_HOST"'
export BYTEPS_LOCAL_SIZE='"$BYTEPS_LOCAL_SIZE"'
export BYTEPS_LOCAL_RANK="$LOCAL_RANK"
export DMLC_WORKER_ID="$NODE_RANK"

echo "[tp-dp-byteps] NODE_RANK='"$NODE_RANK"' RANK=${RANK:-unset} LOCAL_RANK=$LOCAL_RANK CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[tp-dp-byteps] MASTER_ADDR='"$MASTER_ADDR"' MASTER_PORT='"$MASTER_PORT"'"
echo "[tp-dp-byteps] DMLC_NODE_HOST=$DMLC_NODE_HOST DMLC_PS_ROOT_URI=$DMLC_PS_ROOT_URI DMLC_PS_ROOT_PORT=$DMLC_PS_ROOT_PORT"
echo "[tp-dp-byteps] DMLC_NUM_WORKER=$DMLC_NUM_WORKER BYTEPS_LOCAL_SIZE=$BYTEPS_LOCAL_SIZE BYTEPS_LOCAL_RANK=$BYTEPS_LOCAL_RANK DMLC_WORKER_ID=$DMLC_WORKER_ID"
echo "[tp-dp-byteps] TP_SIZE='"$TP_SIZE"' DP_SIZE='"$DP_SIZE"' USE_DPU_DP='"$USE_DPU_DP"' USE_DPU_TP='"$USE_DPU_TP"'"

exec "$@"
' bash "${CMD[@]}"
