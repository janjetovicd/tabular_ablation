#!/bin/bash
# Train a 1.54B proxy model on one serialization format, then auto-submit eval.
#
# Usage:
#   sbatch submit_tabular_ablation.sh csv
#   sbatch submit_tabular_ablation.sh json
#   sbatch submit_tabular_ablation.sh keyvalue
#   sbatch submit_tabular_ablation.sh markdown
#   sbatch submit_tabular_ablation.sh sql_schema
#
# Submit all 5 at once — they run in parallel on separate node allocations.

#SBATCH --account=a139
#SBATCH --time=12:00:00
#SBATCH --job-name=tabular-ablation
#SBATCH --output=/iopsstor/scratch/cscs/djanjetovic/swiss-megatron/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/djanjetovic/swiss-megatron/logs/slurm/training/%x-%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue

echo "START TIME: $(date)"

# ── Format ─────────────────────────────────────────────────────────────────────

FORMAT_NAME=${1:-csv}

if [[ ! "$FORMAT_NAME" =~ ^(csv|sql_schema|keyvalue|markdown|json)$ ]]; then
    echo "ERROR: Unknown format '$FORMAT_NAME'. Valid: csv sql_schema keyvalue markdown json"
    exit 1
fi

echo "Format: $FORMAT_NAME"

# ── Paths ──────────────────────────────────────────────────────────────────────

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/djanjetovic/swiss-megatron-new
ABLATION_DIR=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/djanjetovic/datasets/cache
CONTAINER_ENV=$ABLATION_DIR/tabular_container.toml

# STORE: permanent, backed-up project storage (no auto-cleanup). Tokenized data
# is read from here, and final checkpoints + logs are staged out to here after
# training (see the stage-out block at the end of this script). Scratch
# (/iopsstor) purges anything untouched for 14 days.
STORE_DIR=/capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation

# ── Data shards ────────────────────────────────────────────────────────────────

# Read tokenized shards from permanent store (written there by submit_tokenize.sh).
BASE_DATA_DIR=$STORE_DIR/tokenized/$FORMAT_NAME
DATA_PATH_LIST=()
for i in $(seq -f "%05g" 0 23); do
    shard="${BASE_DATA_DIR}/train_${i}_tokens"
    if [ -s "${shard}.bin" ]; then
        DATA_PATH_LIST+=(1.0 "$shard")
    else
        echo "WARNING: Skipping missing/empty shard: ${shard}.bin"
    fi
done

if [ ${#DATA_PATH_LIST[@]} -eq 0 ]; then
    echo "ERROR: No valid shards found in $BASE_DATA_DIR — did tokenization run?"
    exit 1
fi
echo "Found $((${#DATA_PATH_LIST[@]} / 2)) valid shards."

# ── Hyperparameters ────────────────────────────────────────────────────────────

MBS=2
GBS=256
SEQ_LEN=4096
CHECKPOINT_STEPS=500        # checkpoint every 500 steps in case of preemption
TARGET_TOKENS=30000000000
TRAINING_STEPS=$((TARGET_TOKENS / (GBS * SEQ_LEN)))
RESUME_TRAINING=false

# ── Experiment directories ─────────────────────────────────────────────────────

PROJECT_NAME=tabular-ablation
# Use SLURM_JOB_ID in the WandB name so this creates a new run,
# not appending to the deleted previous run.
EXP_NAME=tabular-$FORMAT_NAME-1p5b-$SLURM_JOB_ID
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME
EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

mkdir -p $CKPT_DIR $PROJECT_DIR $LOGGING_DIR $TENSORBOARD_DIR
mkdir -p $MEGATRON_LM_DIR/logs/slurm/training

# ── Environment ────────────────────────────────────────────────────────────────

export WANDB_API_KEY=$(grep -A2 "api.wandb.ai" ~/.netrc 2>/dev/null | grep password | awk '{print $2}')
export WANDB__FILE_STREAM_RETRY_MAX=10
export HF_HUB_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS
export PYTHONPATH=$MEGATRON_LM_DIR

# ── Model args ─────────────────────────────────────────────────────────────────

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
    --init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style cosine
    --lr-warmup-iters 100
)

if [ "$RESUME_TRAINING" = true ]; then
    echo "RESUME MODE: loading from $CKPT_DIR"
    LOAD_DIR=$CKPT_DIR
else
    echo "FRESH START: training from scratch (seed 42)"
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

MIXED_PRECISION_ARGS=(--bf16)

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
    --trust-remote-code
)

DATA_ARGS=(
    --split 98.3,1.7,0
    --seq-length $SEQ_LEN
    --reset-position-ids
    --no-create-attention-mask-in-dataloader
    --eod-mask-loss
    --num-workers 32
)

DATA_ARGS="${DATA_ARGS[@]} --data-path ${DATA_PATH_LIST[@]} --data-cache-path $DATASET_CACHE_DIR"

# ── Training command ───────────────────────────────────────────────────────────

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

if [ -n "$WANDB_API_KEY" ]; then
    echo "WandB: project=$PROJECT_NAME exp=$EXP_NAME"
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOGGING_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME"
else
    export WANDB_MODE=disabled
    echo "No WandB API key found — logging disabled."
fi

# ── Run training ───────────────────────────────────────────────────────────────

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --mpi=pmix \
    --environment=$CONTAINER_ENV \
    --network=disable_rdzv_get \
    -lu \
    bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $TRAINING_CMD"

TRAIN_EXIT=$?
echo "Training finished with exit code $TRAIN_EXIT"

# ── Auto-submit eval ───────────────────────────────────────────────────────────
# Submit the eval job immediately after training, passing the checkpoint path
# so it knows exactly where the weights are. No need for permanent storage —
# eval will start within minutes of training finishing.

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "Submitting eval job for $FORMAT_NAME ..."
    sbatch $ABLATION_DIR/submit_eval_inference.sh $FORMAT_NAME $CKPT_DIR
    echo "Eval job submitted."

    # ── Stage-out checkpoints + logs to permanent store ─────────────────────────
    # Training wrote checkpoints to scratch (fast, frequent writes). Now copy the
    # whole experiment dir (checkpoints + logging + tensorboard) to backed-up
    # store so it survives the 14-day scratch cleanup and stays available for
    # later analysis and re-evaluation. This runs on the dedicated xfer partition.
    STORE_CKPT_DEST=$STORE_DIR/checkpoints/$EXP_NAME
    echo "Submitting stage-out: $EXP_DIR -> $STORE_CKPT_DEST"
    sbatch --job-name=stage_out_$FORMAT_NAME \
        $ABLATION_DIR/stage.sbatch "$EXP_DIR" "$STORE_CKPT_DEST"
    echo "Stage-out job submitted."
else
    echo "Training failed — not submitting eval or stage-out."
fi

echo "END TIME: $(date)"
