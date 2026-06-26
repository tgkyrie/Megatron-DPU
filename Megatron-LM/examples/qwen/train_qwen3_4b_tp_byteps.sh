#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export USE_DPU=${USE_DPU:-1}
export USE_DPU_DP=${USE_DPU_DP:-${USE_DPU}}
export USE_DPU_TP=${USE_DPU_TP:-${USE_DPU}}
export USE_OVERLAP=${USE_OVERLAP:-0}
export NUM_NODES=${NUM_NODES:-8}
export GPUS_PER_NODE=${GPUS_PER_NODE:-1}
export TP_SIZE=${TP_SIZE:-$((NUM_NODES * GPUS_PER_NODE))}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1}
export SEQ_LENGTH=${SEQ_LENGTH:-6144}
export MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-${SEQ_LENGTH}}
export VOCAB_SIZE=${VOCAB_SIZE:-4096}
export MODEL_NAME=${MODEL_NAME:-qwen3_4b_tp}

exec bash "${SCRIPT_DIR}/train_qwen3_4b_tp_dp_byteps.sh" "$@"
