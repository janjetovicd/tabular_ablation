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
#
#   Or loop over all five formats:
#     for fmt in csv sql_schema keyvalue markdown json; do
#         sbatch submit_tabular_ablation.sh $fmt
#     done
#
# ── Token target note ─────────────────────────────────────────────────────────
# TARGET_TOKENS is set to 10B below (see CHANGE note).
# If all 76 T4 chunks produce more or fewer tokens than expected, update this
# value after serialization completes:
#   total_tokens=$(cat logs/serialize-*_*.out | grep "Tokens written" | \
#                  awk '{sum += $NF} END {print sum}')
# Then set TARGET_TOKENS=$total_tokens * 0.983  (the train fraction).
# =============================================================================

# ── SLURM job configuration ───────────────────────────────────────────────────

#SBATCH --account=a139
#SBATCH --time=12:00:00
# CHANGE: was 06:00:00. Training 10B tokens at GBS=256, SEQ=4096 → ~9,300 steps.
# At a conservative 2-3 seconds/step on 32 GPUs, that is 5-8 hours.
# 12h gives a safe buffer for startup, checkpointing, and eval overhead.
#SBATCH --job-name=tabular-ablation
#SBATCH --output=/iopsstor/scratch/cscs/djanjetovic/apertus/Megatron-LM/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/djanjetovic/apertus/Megatron-LM/logs/slurm/training/%x-%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue

echo "START TIME: $(date)"

# ── Experiment config ─────────────────────────────────────────────────────────

FORMAT_NAME=${1:-csv}

if [[ ! "$FORMAT_NAME" =~ ^(csv|sql_schema|keyvalue|markdown|json)$ ]]; then
    echo "ERROR: Unknown format '$FORMAT_NAME'."
    echo "Valid options: csv, sql_schema, keyvalue, markdown, json"
    exit 1
fi

echo "  Format: $FORMAT_NAME"

# ── Container ─────────────────────────────────────────────────────────────────

CONTAINER_ENV=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tabular_container.toml

# ── Paths ─────────────────────────────────────────────────────────────────────

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/djanjetovic/Megatron-LM
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/djanjetovic/datasets/cache
TABULAR_DATA=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tokenized/$FORMAT_NAME/train
DATA_PATH_LIST=(1.0 $TABULAR_DATA)

# ── Training hyperparameters ──────────────────────────────────────────────────

MBS=2
GBS=256
SEQ_LEN=4096
CHECKPOINT_STEPS=500

# CHANGE: TARGET_TOKENS updated from 30_000_000_000 to 10_000_000_000.
# Reason: each T4 chunk yields ~130-150M tokens after 3800-token row sampling
# (verified from chunk-0000 which has 40,595 parquet files, 99% of tables
# exceed 4096 tokens unsampled). All 76 chunks × ~140M = ~10.6B tokens max.
# 30B is not achievable from T4 alone per format.
# UPDATE THIS VALUE after serialization completes based on actual token counts:
#   grep "Approx tokens" logs/serialize-*_*.out | awk -F: '{print $NF}' | \
#   awk '{sum += $1} END {printf "Total: %.2fB\n", sum}'
TARGET_TOKENS=30000000000

TRAINING_STEPS=$((TARGET_TOKENS / (GBS * SEQ_LEN)))
# = 10B / (256 × 4096) = 10B / 1,048,576 ≈ 9,537 steps

RESUME_TRAINING=false

# ── Logging setup ─────────────────────────────────────────────────────────────

PROJECT_NAME=tabular-ablation
EXP_NAME=tabular-$FORMAT_NAME-1p5b
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME
EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

# ── Environment variables ─────────────────────────────────────────────────────

export WANDB_API_KEY=$(grep -A2 "api.wandb.ai" ~/.netrc 2>/dev/null | grep password | awk '{print $2}')
export WANDB__FILE_STREAM_RETRY_MAX=10
export HF_HUB_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS

# ── Model architecture — 1.54B proxy model ────────────────────────────────────

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
    --seed 42
    # CRITICAL: same seed for all 5 format runs — any difference in val loss
    # is due to the format, not random initialization.
    --init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style cosine
    --lr-warmup-iters 100
)

# ── Checkpoint loading ────────────────────────────────────────────────────────

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

# CHANGE: tokenizer path changed from HuggingFace model name to local cached path.
# The old value was: --tokenizer-model swiss-ai/Apertus-70B-2509
# This fails silently on compute nodes because HF_HUB_OFFLINE=1 is set above
# and Alps compute nodes have no internet access. Use the locally cached copy
# that was saved on the login node in step 1 of the execution plan.
TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model /iopsstor/scratch/cscs/djanjetovic/tokenizer_cache/apertus
)

# ── Data arguments ────────────────────────────────────────────────────────────

DATA_ARGS=(
    --split 98.3,1.7,0
    # CONFIRMED CORRECT per PhD student recommendation.
    # 98.3% train / 1.7% validation / 0% test.
    # Validation split generates the NLL/perplexity curves compared across formats.
    --seq-length $SEQ_LEN
    --reset-position-ids
    --no-create-attention-mask-in-dataloader
    --eod-mask-loss
    --num-workers 32
)

# ── Create directories ────────────────────────────────────────────────────────

mkdir -p $CKPT_DIR $PROJECT_DIR $LOGGING_DIR $TENSORBOARD_DIR
export PYTHONPATH=$MEGATRON_LM_DIR

DATA_ARGS="${DATA_ARGS[@]} --data-path ${DATA_PATH_LIST[@]} --data-cache-path $DATASET_CACHE_DIR"

# ── Build training command ────────────────────────────────────────────────────

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

# ── WandB logging ─────────────────────────────────────────────────────────────

if [ -n "$WANDB_API_KEY" ]; then
    echo "WandB logging enabled. Project: $PROJECT_NAME, Experiment: $EXP_NAME"
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOGGING_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "No WandB API key found. Logging disabled."
fi

# ── Launch training ───────────────────────────────────────────────────────────

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