#!/bin/bash

############################################
# Environment variables
############################################
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_DEBUG_FILE=nccl_${HOSTNAME}_rank${NODE_RANK}.log
export NCCL_DEBUG_LEVEL=TRACE

export CUDA_VISIBLE_DEVICES=0,1
export NODE_RANK=${NODE_RANK:-0}

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=0
export NCCL_SOCKET_IFNAME=ens39f1np1
export NCCL_IB_HCA=mlx5_1
export GLOO_SOCKET_IFNAME=ens39f1np1

export DMLC_ENABLE_RDMA=ibverbs
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=1
export DMLC_INTERFACE=ens39f1np1
export DMLC_PS_ROOT_URI=192.168.1.10
export DMLC_PS_ROOT_PORT=9010
# Set per host to its local IP (machine-specific).
export DMLC_NODE_HOST=${DMLC_NODE_HOST:-192.168.1.10}

TOKENIZER_ARG=${3:-"MOCK"}   # HF tokenizer path or MOCK
DATA_ARG=${4:-"MOCK"}        # data prefix or MOCK

############################################
# Distributed setup
############################################
GPUS_PER_NODE=2
NUM_NODES=2
MASTER_ADDR=${MASTER_ADDR:-192.168.1.10}
MASTER_PORT=${MASTER_PORT:-19002}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# BytePS: number of worker nodes (per host), not processes.
export DMLC_NUM_WORKER=$NUM_NODES
# BytePS: local processes per host (nproc_per_node).
export BYTEPS_LOCAL_SIZE=$GPUS_PER_NODE

PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

############################################
# Model configuration: Qwen-3B
############################################
TP_SIZE=1
CP_SIZE=1
PP_SIZE=1

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=256

NUM_LAYERS=8
HIDDEN_SIZE=2048
NUM_HEADS=16
FFN_HIDDEN_SIZE=5504
KV_CHANNELS=128

SEQ_LENGTH=2048
MAX_POSITION_EMBEDDINGS=2048

DTYPE="fp8"   # fp8 | bf16

DATA_CACHE_PATH="${PWD}/benchmark_cache_qwen_3b"
mkdir -p "$DATA_CACHE_PATH"

############################################
# torchrun args
############################################
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --no_python
)
############################################
# Model args (Qwen style)
############################################
MODEL_ARGS=(
    --transformer-impl local
    --use-mcore-models
    --no-persist-layer-norm
    --recompute-activations

    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_HEADS
    --kv-channels $KV_CHANNELS

    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS

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
    --train-iters 5
    --no-rope-fusion
    --use-dpu-reduce
)

############################################
# Training args
############################################
TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE

    --lr 3.0e-4
    --min-lr 3.0e-5
    --lr-decay-style cosine
    --weight-decay 0.1
    --clip-grad 1.0

    --adam-beta1 0.9
    --adam-beta2 0.95

    # --bf16
    # --grad-reduce-in-bf16

    --cross-entropy-loss-fusion
    --calculate-per-token-loss

    # --use-distributed-optimizer
    # --overlap-grad-reduce
    # --overlap-param-gather

    --manual-gc
    --empty-unused-memory-level 1
    --exit-duration-in-mins 235
)

############################################
# FP8 args (optional)
############################################
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        --fp8-format hybrid
        --fp8-amax-history-len 1024
        --fp8-amax-compute-algo max
        # --fp8-param-gather
    )
fi

############################################
# Parallelism
############################################
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    # --sequence-parallel
)

############################################
# Data args
############################################
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]]; then
    DATA_ARGS_LIST+=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size 151936
        --split '99,1,0'
        --no-mmap-bin-files
        --num-workers 1
    )
else
    DATA_ARGS_LIST+=(
        --data-path $DATA_ARG
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model $TOKENIZER_ARG
        --vocab-size 151936
        --data-cache-path $DATA_CACHE_PATH
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
)

############################################
# Run
############################################
CMD=(python "$PRETRAIN_SCRIPT_PATH"
    "${MODEL_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${DTYPE_ARGS[@]}"
    "${MODEL_PARALLEL_ARGS[@]}"
    "${DATA_ARGS_LIST[@]}"
    "${EVAL_AND_LOGGING_ARGS[@]}"
)

torchrun "${DISTRIBUTED_ARGS[@]}" bash -c '
export BYTEPS_LOCAL_RANK="$LOCAL_RANK"
export DMLC_WORKER_ID="$NODE_RANK"
echo "[byteps] NODE_RANK=${NODE_RANK} RANK=${RANK} LOCAL_RANK=${LOCAL_RANK} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[byteps] DMLC_NODE_HOST=${DMLC_NODE_HOST} DMLC_PS_ROOT_URI=${DMLC_PS_ROOT_URI} DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT}"
echo "[byteps] DMLC_NUM_WORKER=${DMLC_NUM_WORKER} BYTEPS_LOCAL_SIZE=${BYTEPS_LOCAL_SIZE} BYTEPS_LOCAL_RANK=${BYTEPS_LOCAL_RANK} DMLC_WORKER_ID=${DMLC_WORKER_ID}"
exec "$@"
' bash "${CMD[@]}"
