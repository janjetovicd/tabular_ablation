#!/bin/bash
# OBJECTIVE: Train a 1.54B proxy model on one serialization format.
#
# Specifically:
#   1. Launch a Megatron training job on GPU nodes using the tokenized .bin/.idx
#   files produced in Phase 2b. 
#.  2. Train a 1.54B model from random initialization and logs validation loss to WandB.
#
# Usage:
#   FIX: FORMAT_NAME is now accepted as a command-line argument so you don't
#   need to manually edit the script 5 times. Pass it at submission time:
#     sbatch submit_tabular_ablation.sh csv
#     sbatch submit_tabular_ablation.sh sql_schema
#     sbatch submit_tabular_ablation.sh keyvalue
#     sbatch submit_tabular_ablation.sh markdown
#     sbatch submit_tabular_ablation.sh json
#
#   The old loop approach still works too (no manual editing needed):
#     for fmt in csv sql_schema keyvalue markdown json; do
#         sbatch submit_tabular_ablation.sh $fmt
#     done
#
# Output:
#   Checkpoints: $EXP_DIR/checkpoints/
#   Logs:        $EXP_DIR/logging/
#   Loss curves: WandB project "tabular-ablation"
# =============================================================================

# ── SLURM job configuration ───────────────────────────────────────────────────

#SBATCH --account=a139
#SBATCH --time=06:00:00             # 6 hours — enough for 1.5B token proxy run
#SBATCH --job-name=tabular-ablation # FIX: generic name; WandB exp name encodes format
#SBATCH --output=/iopsstor/scratch/cscs/djanjetovic/apertus/Megatron-LM/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/djanjetovic/apertus/Megatron-LM/logs/slurm/training/%x-%j.err
#SBATCH --nodes=8                   # 8 nodes × 4 GPUs = 32 GPUs total
                                    # Much smaller than audio script (128 nodes for 8B)
                                    # because our proxy model is 1.54B not 8B
#SBATCH --ntasks-per-node=4         # 4 GPUs per node — one process per GPU
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue

echo "START TIME: $(date)"

# ── Experiment config ─────────────────────────────────────────────────────────

# FIX: accept FORMAT_NAME as a command-line argument instead of hardcoding it.
# $1 = first argument passed to sbatch, e.g.: sbatch submit_tabular_ablation.sh csv
# ${1:-csv} means: use $1 if provided, otherwise default to "csv"
FORMAT_NAME=${1:-csv}

# Validate that FORMAT_NAME is one of the expected values
if [[ ! "$FORMAT_NAME" =~ ^(csv|sql_schema|keyvalue|markdown|json)$ ]]; then
    echo "ERROR: Unknown format '$FORMAT_NAME'."
    echo "Valid options: csv, sql_schema, keyvalue, markdown, json"
    exit 1
fi

echo "  Format: $FORMAT_NAME"

# ── Container ─────────────────────────────────────────────────────────────────

# Container provided by Ayush — pre-built NGC environment with PyTorch and CUDA
# The container is the same regardless of which format we're training on —
# it just provides the software environment, not data or model specifics
CONTAINER_ENV=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tabular_container.toml

# ── Paths ─────────────────────────────────────────────────────────────────────

# Megatron source — Swiss AI fork with multimodal support
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/djanjetovic/Megatron-LM

# Dataset cache directory for Megatron's internal bookkeeping
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/djanjetovic/datasets/cache

# Binary training data for this format — produced by submit_tokenize.sh (Phase 2b)
TABULAR_DATA=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tokenized/$FORMAT_NAME/train

# Data path list — Megatron format: alternating "weight path" pairs
# Weight 1.0 means 100% of tokens come from this dataset (only one format per job)
DATA_PATH_LIST=(1.0 $TABULAR_DATA)

# ── Training hyperparameters ──────────────────────────────────────────────────

MBS=2                               # Micro batch size per GPU
GBS=256                             # Global batch size across all GPUs
SEQ_LEN=4096                        # Context window (matches token_budget + overhead)
CHECKPOINT_STEPS=500                # Save checkpoint every 500 steps
TARGET_TOKENS=1500000000            # 1.5B tokens total for this ablation run
                                    # Enough to see stable loss curves
TRAINING_STEPS=$((TARGET_TOKENS / (GBS * SEQ_LEN)))
# Formula: total tokens / (sequences per step × tokens per sequence)

RESUME_TRAINING=false               # Set to true to resume from a checkpoint

# ── Logging setup ─────────────────────────────────────────────────────────────

PROJECT_NAME=tabular-ablation
EXP_NAME=tabular-$FORMAT_NAME-1p5b  # Experiment name encodes format and model size
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME
EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

# ── Environment variables ─────────────────────────────────────────────────────

# WandB API key — loaded from secure file set up during CSCS configuration
export WANDB_API_KEY=$(grep -A2 "api.wandb.ai" ~/.netrc 2>/dev/null | grep password | awk '{print $2}')
export WANDB__FILE_STREAM_RETRY_MAX=10

# Disable HuggingFace Hub network access — cluster nodes have no internet
export HF_HUB_OFFLINE=1

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Distributed training setup — SLURM provides these automatically
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS     # Total number of GPU processes across all nodes

# ── Model architecture — 1.54B proxy model ────────────────────────────────────
# These parameters define the model Megatron builds from scratch.
# Chosen to be small enough for fast ablation while large enough to show
# meaningful format differences in validation loss.
# Source: Ayush's recommendation in project notes.

NETWORK_SIZE_ARGS=(
    --num-layers 32                 # Transformer depth
    --hidden-size 2048              # Embedding dimension (2048 vs 4096 for 8B Apertus)
    --ffn-hidden-size 6144          # Feed-forward size = 3 × hidden_size
    --num-attention-heads 16        # Attention heads
    --group-query-attention         # Enable GQA (grouped query attention) for efficiency
    --num-query-groups 4            # KV heads = 4 (fewer than query heads = GQA)
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope  # Rotary position embeddings (same as Apertus)
    --rotary-base 500000
    --normalization RMSNorm         # RMS normalization (same as Apertus)
    --untie-embeddings-and-output-weights
    --make-vocab-size-divisible-by 128
)

TRANSFORMER_ENGINE_ARGS=(
    --main-grads-dtype fp32         # Keep gradients in fp32 for stability
    --log-params-norm               # Log parameter norms for debugging
)

LOGGING_ARGS=(
    --log-throughput                # Log tokens/second
    --tensorboard-dir $TENSORBOARD_DIR
    --no-log-loss-scale-to-tensorboard
    --log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0                 # Gradient clipping for stability
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
    --optimizer adam                # Standard Adam (audio uses AdEMAMix, simpler here)
    --dataloader-type single
    --eval-interval 500             # Evaluate every 500 steps — produces loss curve points
    --eval-iters 50                 # Run 50 eval batches per evaluation
)

INITIALIZATION_ARGS=(
    --seed 42                       # CRITICAL: same seed for all 5 format runs so they
                                    # start from identical random weights. Any difference
                                    # in val loss is due to format, not initialization.
    --init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style cosine         # Cosine decay schedule
    --lr-warmup-iters 100           # Short warmup since training from scratch
)

# ── Checkpoint loading ────────────────────────────────────────────────────────

if [ "$RESUME_TRAINING" = true ]; then
    # Resume from previous checkpoint (same experiment, continuing training)
    echo "RESUME MODE: Loading from $CKPT_DIR"
    LOAD_DIR=$CKPT_DIR
else
    # Fresh start — no checkpoint to load, initialize from random weights
    # --no-load-optim and --no-load-rng not needed since there's no --load
    echo "FRESH START: Training from random initialization (seed 42)"
    LOAD_DIR=""
fi

CHECKPOINTING_ARGS=(
    --save $CKPT_DIR
    --save-interval $CHECKPOINT_STEPS
    --ckpt-format torch_dist
)

# Only add --load if we're resuming
if [ -n "$LOAD_DIR" ]; then
    CHECKPOINTING_ARGS+=(--load $LOAD_DIR)
fi

MIXED_PRECISION_ARGS=(
    --bf16                          # bfloat16 training — standard for LLM training
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 2  # Split model across 2 GPUs (needed for 1.54B)
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

# ── Tokenizer ─────────────────────────────────────────────────────────────────

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model swiss-ai/Apertus-70B-2509
    # Must match the tokenizer used in Phase 2b (preprocess_data.py).
    # Using local CSCS path since HF_HUB_OFFLINE=1.
)

# ── Data arguments ────────────────────────────────────────────────────────────

DATA_ARGS=(
    --split 98,2,0                  # 98% train, 2% validation, 0% test
                                    # Validation split produces the loss curves
                                    # we compare across formats
    --seq-length $SEQ_LEN
    --reset-position-ids            # Reset position IDs at document boundaries
    --no-create-attention-mask-in-dataloader
    --eod-mask-loss                 # Mask loss on end-of-document tokens
    --num-workers 32
)

# ── Create directories ────────────────────────────────────────────────────────

# FIX: added TENSORBOARD_DIR to mkdir — it wasn't being created before, which
# caused Megatron to error when it tried to write tensorboard logs.
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

# WandB records loss at every step — you compare 5 curves on wandb.ai
# to determine which format produces the lowest validation loss
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

# srun launches one process per GPU across all nodes, inside the container.
# RANK = global GPU index (0 to WORLD_SIZE-1)
# LOCAL_RANK = GPU index on this specific node (0 to 3)
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