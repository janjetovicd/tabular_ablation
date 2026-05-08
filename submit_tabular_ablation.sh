#!/bin/bash
# OBJECTIVE: Train a 1.54B proxy model on one serialization format.
#
# Specifically:
#   1. Launch a Megatron training job on GPU nodes using the tokenized .bin/.idx
#      files produced in Phase 2b (submit_tokenize.sh).
#   2. Train a 1.54B model from random initialization and log validation loss to WandB.
#
# Usage:
#   sbatch submit_tabular_ablation.sh csv
#   sbatch submit_tabular_ablation.sh sql_schema
#   sbatch submit_tabular_ablation.sh keyvalue
#   sbatch submit_tabular_ablation.sh markdown
#   sbatch submit_tabular_ablation.sh json

# SLURM job configuration

#SBATCH --account=a139
#SBATCH --time=12:00:00
#SBATCH --job-name=tabular-ablation
#SBATCH --output=/iopsstor/scratch/cscs/djanjetovic/apertus/Megatron-LM/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/djanjetovic/apertus/Megatron-LM/logs/slurm/training/%x-%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue

echo "START TIME: $(date)"

# Experiment config

FORMAT_NAME=${1:-csv}

if [[ ! "$FORMAT_NAME" =~ ^(csv|sql_schema|keyvalue|markdown|json)$ ]]; then
    echo "ERROR: Unknown format '$FORMAT_NAME'."
    echo "Valid options: csv, sql_schema, keyvalue, markdown, json"
    exit 1
fi

echo "  Format: $FORMAT_NAME"

# Container

CONTAINER_ENV=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tabular_container.toml

# Paths

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/djanjetovic/Megatron-LM
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/djanjetovic/datasets/cache

# Build data path list from all non-empty shards (00-23)

BASE_DATA_DIR=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tokenized/$FORMAT_NAME
DATA_PATH_LIST=()
for i in $(seq -f "%05g" 0 23); do
    shard="${BASE_DATA_DIR}/train_${i}_tokens"
    if [ -s "${shard}.bin" ]; then
        DATA_PATH_LIST+=(1.0 "$shard")
    else
        echo "WARNING: Skipping missing or empty shard: ${shard}.bin" # Shards 24-27 are 0-byte placeholders from empty ranks — skip them.
    fi
done

if [ ${#DATA_PATH_LIST[@]} -eq 0 ]; then
    echo "ERROR: No valid shards found for format '$FORMAT_NAME' in $BASE_DATA_DIR"
    exit 1
fi

echo "  Found $((${#DATA_PATH_LIST[@]} / 2)) valid shards for $FORMAT_NAME"

# Training hyperparameters

MBS=2
GBS=256
SEQ_LEN=4096
CHECKPOINT_STEPS=500

# NOTE: With 12h wall time you will hit ~10-15B tokens before the job ends.
# The job will checkpoint every 500 steps so you can resume with RESUME_TRAINING=true.
# To check your actual wall time limit with: sacctmgr show qos
TARGET_TOKENS=30000000000

TRAINING_STEPS=$((TARGET_TOKENS / (GBS * SEQ_LEN)))

RESUME_TRAINING=false

# Logging setup

PROJECT_NAME=tabular-ablation
EXP_NAME=tabular-$FORMAT_NAME-1p5b
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME
EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

# Environment variables

export WANDB_API_KEY=$(grep -A2 "api.wandb.ai" ~/.netrc 2>/dev/null | grep password | awk '{print $2}')
export WANDB__FILE_STREAM_RETRY_MAX=10
export HF_HUB_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS

# Model architecture — 1.54B proxy model

NETWORK_SIZE_ARGS=(
    --num-layers 32
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 4
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --rotary-base 500000
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --make-vocab-size-divisible-by 128
)

TRANSFORMER_ENGINE_ARGS=(
    --main-grads-dtype fp32
    --log-params-norm
)

LOGGING_ARGS=(
    --log-throughput
    --tensorboard-dir $TENSORBOARD_DIR
    --no-log-loss-scale-to-tensorboard
    --log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters $TRAINING_STEPS
    --log-interval 10
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --eval-interval 500
    --eval-iters 50
)

INITIALIZATION_ARGS=(
    --seed 42 #Same seed for all 5 format runs so any difference in val loss is due to the format, not random initialization
    --init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style cosine
    --lr-warmup-iters 100
)

# Checkpoint loading

if [ "$RESUME_TRAINING" = true ]; then
    echo "RESUME MODE: Loading from $CKPT_DIR"
    LOAD_DIR=$CKPT_DIR
else
    echo "FRESH START: Training from random initialization (seed 42)"
    LOAD_DIR=""
fi

CHECKPOINTING_ARGS=(
    --save $CKPT_DIR
    --save-interval $CHECKPOINT_STEPS
    --ckpt-format torch_dist
)

if [ -n "$LOAD_DIR" ]; then
    CHECKPOINTING_ARGS+=(--load $LOAD_DIR)
fi

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model /iopsstor/scratch/cscs/djanjetovic/tokenizer_cache/apertus
)

# Data arguments

DATA_ARGS=(
    --split 98.3,1.7,0
    --seq-length $SEQ_LEN
    --reset-position-ids
    --no-create-attention-mask-in-dataloader
    --eod-mask-loss
    --num-workers 32
)

# Create directories

mkdir -p $CKPT_DIR $PROJECT_DIR $LOGGING_DIR $TENSORBOARD_DIR
export PYTHONPATH=$MEGATRON_LM_DIR

DATA_ARGS="${DATA_ARGS[@]} --data-path ${DATA_PATH_LIST[@]} --data-cache-path $DATASET_CACHE_DIR"

# Build training command 

TRAINING_CMD="python3 $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    $DATA_ARGS"

# WandB logging

if [ -n "$WANDB_API_KEY" ]; then
    echo "WandB logging enabled. Project: $PROJECT_NAME, Experiment: $EXP_NAME"
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOGGING_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME"
    # NOTE: removed $SLURM_JOB_ID from exp name so resumed runs continue
    # the same WandB entry instead of creating a new one each time.
else
    export WANDB_MODE=disabled
    echo "No WandB API key found. Logging disabled."
fi

# Launch training

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --mpi=pmix \
    --environment=$CONTAINER_ENV \
    --network=disable_rdzv_get \
    -lu \
    bash -c "
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $TRAINING_CMD
    "

echo "END TIME: $(date)"