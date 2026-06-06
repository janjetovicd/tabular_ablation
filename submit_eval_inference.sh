#!/bin/bash
# Run BLiMP-style inference eval for one serialization format.
#
# Auto-submitted by submit_tabular_ablation.sh — you don't normally call this directly.
# If you need to run it manually:
#   sbatch submit_eval_inference.sh csv /path/to/checkpoints/dir
#
# Args:
#   $1 = format (csv | json | keyvalue | markdown | sql_schema)
#   $2 = checkpoint directory (the EXP_DIR/checkpoints path from training)

#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=eval-inference
#SBATCH --output=/iopsstor/scratch/cscs/djanjetovic/swiss-megatron/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/djanjetovic/swiss-megatron/logs/slurm/training/%x-%j.err

echo "START TIME: $(date)"
echo "Running on: $(hostname)"

# ── Args ───────────────────────────────────────────────────────────────────────

FORMAT=${1:-csv}
CKPT_DIR=${2:-}

if [[ ! "$FORMAT" =~ ^(csv|json|keyvalue|markdown|sql_schema)$ ]]; then
    echo "ERROR: Unknown format '$FORMAT'. Valid: csv json keyvalue markdown sql_schema"
    exit 1
fi

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint directory given. Pass it as the second argument."
    exit 1
fi

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint directory does not exist: $CKPT_DIR"
    exit 1
fi

echo "Format:     $FORMAT"
echo "Checkpoint: $CKPT_DIR"

# ── Paths ──────────────────────────────────────────────────────────────────────

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/djanjetovic/swiss-megatron
ABLATION_DIR=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation
CONTAINER_ENV=$ABLATION_DIR/tabular_container.toml
PAIRS_DIR=$ABLATION_DIR/eval
OUTPUT_DIR=$ABLATION_DIR/eval_results

mkdir -p $OUTPUT_DIR
mkdir -p $MEGATRON_LM_DIR/logs/slurm/training

# ── Checks ─────────────────────────────────────────────────────────────────────

PAIRS_FILE=$PAIRS_DIR/pairs_${FORMAT}.jsonl
if [ ! -f "$PAIRS_FILE" ]; then
    echo "ERROR: Pairs file not found: $PAIRS_FILE"
    exit 1
fi

echo "Pairs file: $PAIRS_FILE ($(wc -l < $PAIRS_FILE) lines)"
echo "Output dir: $OUTPUT_DIR"

# ── Environment ────────────────────────────────────────────────────────────────

export MEGATRON_LM_DIR=$MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8889
export WORLD_SIZE=$SLURM_NPROCS

# ── Run eval ───────────────────────────────────────────────────────────────────

EVAL_CMD="python3 $ABLATION_DIR/run_eval_inference.py \
    --format     $FORMAT \
    --ckpt-dir   $CKPT_DIR \
    --pairs-dir  $PAIRS_DIR \
    --output-dir $OUTPUT_DIR \
    --max-seq-len 4096"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --mpi=pmix \
    --environment=$CONTAINER_ENV \
    --network=disable_rdzv_get \
    -lu \
    bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $EVAL_CMD"

echo "END TIME: $(date)"
