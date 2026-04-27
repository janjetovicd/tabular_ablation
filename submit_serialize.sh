#!/bin/bash
# OBJECTIVE: Serialize T4 parquet files into .jsonl format.
#
# Specifically:
#   Runs serialize_t4.py on T4 chunks in parallel via a SLURM array job.
#   Each task handles one chunk zip file and writes one .jsonl output file.
#
#
# ── Two-phase usage ───────────────────────────────────────────────────────────
# PHASE A — Test run (chunk 0 only, ~3-4h):
#   sbatch --array=0 submit_serialize.sh csv
#   Then inspect logs: grep "Tokens written" logs/serialize-*-0.out
#   This tells you the real per-chunk token yield.
#
# PHASE B — Full run (all 76 chunks, after Phase A confirms output):
#   for fmt in csv sql_schema keyvalue markdown json; do
#       sbatch --array=0-75 submit_serialize.sh $fmt
#   done
#
# Output per format:
#   /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/jsonl/{format}/chunk-0000.jsonl
#   ...
#   /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/jsonl/{format}/chunk-0075.jsonl
# =============================================================================

# ── SLURM job configuration ───────────────────────────────────────────────────

#SBATCH --account=a139
#SBATCH --time=12:00:00
#SBATCH --job-name=serialize-tabular
#SBATCH --output=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/logs/serialize-%A-%a.out
#SBATCH --error=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/logs/serialize-%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# CHANGE: --array is now a comment placeholder only — DO NOT set it here.
# Pass it explicitly at submission time so you can switch between test (0)
# and full (0-75) without editing this file:
#   sbatch --array=0       submit_serialize.sh csv    # test: chunk 0 only
#   sbatch --array=0-75    submit_serialize.sh csv    # full: all 76 chunks
# The array line below is left commented out as documentation of the range:
# #SBATCH --array=0-75

# ── Arguments ─────────────────────────────────────────────────────────────────

FORMAT=$1

if [ -z "$FORMAT" ]; then
    echo "ERROR: No format specified."
    echo "Usage: sbatch --array=0 submit_serialize.sh <format>"
    echo "       sbatch --array=0-75 submit_serialize.sh <format>"
    echo "Formats: csv, sql_schema, keyvalue, markdown, json"
    exit 1
fi

if [[ ! "$FORMAT" =~ ^(csv|sql_schema|keyvalue|markdown|json)$ ]]; then
    echo "ERROR: Unknown format '$FORMAT'."
    echo "Valid options: csv, sql_schema, keyvalue, markdown, json"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────

T4_BASE=/capstor/store/cscs/swissai/a139/datasets/mlfoundations_t4_full
CHUNK=$(printf "chunk-%04d" $SLURM_ARRAY_TASK_ID)
CHUNK_ZIP=$T4_BASE/${CHUNK}.zip
OUTPUT_DIR=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/jsonl/$FORMAT

# CHANGE: Pre-cached local tokenizer path.
# Compute nodes have no internet — HuggingFace Hub downloads fail silently.
# Cache the tokenizer on the login node BEFORE submitting any jobs (see step 1
# in the execution plan). Path must match where you saved it.
TOKENIZER_PATH=/iopsstor/scratch/cscs/djanjetovic/tokenizer_cache/apertus

mkdir -p /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/logs
mkdir -p $OUTPUT_DIR

echo "Starting serialization"
echo "  Format:    $FORMAT"
echo "  Chunk:     $CHUNK  (task $SLURM_ARRAY_TASK_ID of array)"
echo "  Input:     $CHUNK_ZIP"
echo "  Output:    $OUTPUT_DIR/$CHUNK.jsonl"
echo "  Tokenizer: $TOKENIZER_PATH"

# ── Environment ───────────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# ── Run serialization ─────────────────────────────────────────────────────────

# CHANGE: removed --token_target argument entirely.
# The old value (300000000) was causing early stops that appeared correct
# but were actually just stopping near the natural end of a chunk anyway.
# Without --token_target, serialize_t4.py processes ALL parquet files in
# the chunk and reports the real token yield in the log.
# To inspect yield after chunk 0 completes:
#   grep "Tokens written\|Approx tokens" logs/serialize-*_0.out
python /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/serialize_t4.py \
    --chunk_zip $CHUNK_ZIP \
    --format $FORMAT \
    --output $OUTPUT_DIR/$CHUNK.jsonl \
    --tokenizer $TOKENIZER_PATH

echo "Done: $CHUNK for format $FORMAT"
echo "Output file size: $(du -sh $OUTPUT_DIR/$CHUNK.jsonl 2>/dev/null || echo 'not found')"