#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Approximation: true DeciLM-6B uses per-layer KV heads
# [4, 2, ..., 1, ..., 4], which the current Megatron TP=2 script
# cannot express. NUM_QUERY_GROUPS=2 is the closest TP-compatible
# fixed-GQA approximation.
export MODEL_NAME=${MODEL_NAME:-decilm_6b_approx_gqa2}
export NUM_LAYERS=${NUM_LAYERS:-32}
export HIDDEN_SIZE=${HIDDEN_SIZE:-4096}
export NUM_HEADS=${NUM_HEADS:-32}
export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-11008}
export KV_CHANNELS=${KV_CHANNELS:-128}
export NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-2}
export SEQ_LENGTH=${SEQ_LENGTH:-256}
export MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-4096}
export VOCAB_SIZE=${VOCAB_SIZE:-32000}
export MAKE_VOCAB_SIZE_DIVISIBLE_BY=${MAKE_VOCAB_SIZE_DIVISIBLE_BY:-1}
export NORMALIZATION=${NORMALIZATION:-RMSNorm}
export NORM_EPSILON=${NORM_EPSILON:-1e-5}
export ROTARY_BASE=${ROTARY_BASE:-10000}
export ROTARY_PERCENT=${ROTARY_PERCENT:-1.0}
export ACTIVATION=${ACTIVATION:-swiglu}
export UNTIE_EMBEDDINGS=${UNTIE_EMBEDDINGS:-1}
export DISABLE_BIAS_LINEAR=${DISABLE_BIAS_LINEAR:-1}

exec bash "${SCRIPT_DIR}/train_qwen3_4b_tp_dp_byteps.sh" "$@"
