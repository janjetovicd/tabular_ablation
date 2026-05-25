#!/bin/bash
# submit_eval_inference.sh
#
# OBJECTIVE: Run run_eval_inference.py for one serialization format.
#
# Usage:
#   sbatch submit_eval_inference.sh <format>              # normal (own-format checkpoint)
#   sbatch submit_eval_inference.sh <format> <model_fmt>  # cross-format: model_fmt checkpoint
#                                                          # run on <format> prompts
#
# Examples:
#   sbatch submit_eval_inference.sh json           # json model, json prompts
#   sbatch submit_eval_inference.sh csv json       # json model, csv prompts (cross-format)
#
# Submit all 5 formats at once:
#   for fmt in csv json keyvalue markdown sql_schema; do
#       sbatch submit_eval_inference.sh $fmt
#   done

# It fails immediately if any command exits with a non-zero status.
#     Without this, bash ignores errors and "Done" prints even if inference failed.
set -e
# ▲▲▲

# ── SLURM config ──────────────────────────────────────────────────────────────

#SBATCH --account=a139
#SBATCH --job-name=eval-inference
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=72
#SBATCH --output=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/logs/eval_inference_%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/logs/eval_inference_%x_%j.err

# ── Arguments ─────────────────────────────────────────────────────────────────

PROMPT_FORMAT=$1      # format of the prompt file to read
MODEL_FORMAT=${2:-$1} # checkpoint format (defaults to same as prompt format)

VALID_FORMATS="csv json keyvalue markdown sql_schema"

if [ -z "$PROMPT_FORMAT" ]; then
    echo "ERROR: No format specified."
    echo "Usage: sbatch submit_eval_inference.sh <prompt_format> [model_format]"
    echo "Formats: $VALID_FORMATS"
    exit 1
fi

for F in "$PROMPT_FORMAT" "$MODEL_FORMAT"; do
    if [[ ! "$F" =~ ^(csv|json|keyvalue|markdown|sql_schema)$ ]]; then
        echo "ERROR: Unknown format '$F'. Valid: $VALID_FORMATS"
        exit 1
    fi
done

# ── Paths ─────────────────────────────────────────────────────────────────────

MEGATRON_ROOT=/iopsstor/scratch/cscs/djanjetovic/swiss-megatron
TABULAR_ABLATION=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation
EVAL_DIR=$TABULAR_ABLATION/eval
TOKENIZER_PATH=/iopsstor/scratch/cscs/djanjetovic/tokenizer_cache/apertus
CONTAINER_ENV=$TABULAR_ABLATION/tabular_container.toml

# checkpoint for the selected model format
CHECKPOINT_DIR=$MEGATRON_ROOT/logs/Meg-Runs/tabular-ablation/tabular-${MODEL_FORMAT}-1p5b/checkpoints

# prompt file always comes from the prompt format
PROMPT_FILE=$EVAL_DIR/prompts_${PROMPT_FORMAT}.jsonl

# output file name encodes both model and prompt format
if [ "$MODEL_FORMAT" = "$PROMPT_FORMAT" ]; then
    OUTPUT_FILE=$EVAL_DIR/predictions_${PROMPT_FORMAT}.jsonl
else
    OUTPUT_FILE=$EVAL_DIR/cross_format_predictions_${MODEL_FORMAT}_model_${PROMPT_FORMAT}_prompts.jsonl
fi

SCRIPT=$TABULAR_ABLATION/run_eval_inference.py

mkdir -p $EVAL_DIR
mkdir -p $TABULAR_ABLATION/logs

echo "=== eval inference ==="
echo "  Prompt format:  $PROMPT_FORMAT"
echo "  Model format:   $MODEL_FORMAT"
echo "  Checkpoint:     $CHECKPOINT_DIR"
echo "  Prompt file:    $PROMPT_FILE"
echo "  Output file:    $OUTPUT_FILE"
echo "  Container:      $CONTAINER_ENV"

# validate inputs exist
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: prompt file not found: $PROMPT_FILE"
    echo "Run prepare_eval_data.py first."
    exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# ── Launch ────────────────────────────────────────────────────────────────────

# Previously only wiped /dev/shm/$USER/enrootrun (the subfolder).
#     Enroot also leaves broken state in the parent /dev/shm/$USER/ directory
#     itself, so the subfolder deletion wasn't enough — the container kept
#     failing with Permission denied when trying to recreate dirs inside the
#     still-broken parent.
#
#     Fix: wipe the entire /dev/shm/$USER/ tree on the allocated node.
#     The || true at the end means: if rm fails for any reason (e.g. nothing
#     to delete), don't let set -e abort the whole job — just continue.
#     The echo lines confirm in the .out log whether cleanup actually ran.
srun --ntasks=1 bash -c "
    echo 'Cleaning /dev/shm/$USER on node \$(hostname)...'
    rm -rf /dev/shm/$USER && echo 'Cleanup OK' || echo 'Cleanup had nothing to remove or failed (non-fatal)'
"

# Explicit failure check on the container srun.
#     Previously if this step failed, bash kept going and printed "Done"
#     even though no inference ran. Now it prints the real exit code and
#     exits with failure so SLURM marks the job as failed, not completed.
srun --mpi=pmix --environment=$CONTAINER_ENV bash -c "
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    export WORLD_SIZE=\$SLURM_NTASKS
    export MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n1)
    export MASTER_PORT=6000

    cd $MEGATRON_ROOT

    python $SCRIPT \
        --prompt-file  $PROMPT_FILE \
        --output-file  $OUTPUT_FILE \
        --load         $CHECKPOINT_DIR \
        --num-layers   32 \
        --hidden-size  2048 \
        --ffn-hidden-size 6144 \
        --num-attention-heads 16 \
        --group-query-attention \
        --num-query-groups 4 \
        --max-position-embeddings 4096 \
        --position-embedding-type rope \
        --rotary-base 500000 \
        --normalization RMSNorm \
        --untie-embeddings-and-output-weights \
        --make-vocab-size-divisible-by 128 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --bf16 \
        --disable-bias-linear \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model $TOKENIZER_PATH \
        --trust-remote-code \
        --micro-batch-size 1 \
        --no-load-optim \
        --no-load-rng \
        --distributed-backend nccl
" || { echo "ERROR: container srun failed — inference did not run. Check .err log."; exit 1; }

echo "Done: eval inference for prompt_format=$PROMPT_FORMAT model_format=$MODEL_FORMAT"