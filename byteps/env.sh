export DMLC_NUM_WORKER=2
export DMLC_NUM_SERVER=2
export BYTEPS_VISIBLE_CPU_CORES=53-63
export BYTEPS_PARTITION_BYTES=4000000
export BYTEPS_RDMA_RX_DEPTH=512
export BYTEPS_RDMA_START_DEPTH=32

export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0         
export NCCL_IB_GID_INDEX=0        
export NCCL_SOCKET_IFNAME=ens39f1np1
export NCCL_IB_HCA=mlx5_1         
export GLOO_SOCKET_IFNAME=ens39f1np1
export MASTER_ADDR=192.168.1.10
export MASTER_PORT=19011
export NNODES=2
export NODE_RANK=1
export CUDA_VISIBLE_DEVICES=1

torchrun --nproc_per_node=1 --nnodes=$NNODES --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  pretrain_gpt.py \
  --num-layers 24 \
  --hidden-size 2048 \
  --num-attention-heads 16 \
  --micro-batch-size 4 \
  --global-batch-size 64 \
  --max-position-embeddings 1024 \
  --seq-length 1024 \
  --vocab-size 50257 \
  --legacy-tokenizer \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-file ../vocab/vocab.json \
  --merge-file ../vocab/merges.txt \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --transformer-impl local \
  --no-persist-layer-norm \
  --mock-data \
  --fp16 \
  --recompute-activations \
  --train-iters 50 \
  --lr 0.00015 \
  --min-lr 1.0e-5 \
  --lr-decay-style cosine \
  --use-distributed-optimizer \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --log-interval 10