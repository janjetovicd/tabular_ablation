#!/bin/bash
# OBJECTIVE: Serialize T4 parquet files into .jsonl format.
#
# Specifically:
#   1. Runs serialize_t4.py on 10 T4 chunks in parallel using a SLURM array job.
#
# Why SLURM array job:
#   Processing 10 chunks one by one would take 10x longer. With --array=0-9,
#   SLURM launches 10 jobs simultaneously, each handling one chunk, finishing
#   in roughly the same time as processing one chunk alone.
#
# Usage (run once per format from CSCS login node):
#   sbatch submit_serialize.sh csv
#   sbatch submit_serialize.sh sql_schema
#   sbatch submit_serialize.sh keyvalue
#   sbatch submit_serialize.sh markdown
#   sbatch submit_serialize.sh json
#
# Output per format:
#   /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/jsonl/{format}/chunk-0000.jsonl
#   /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/jsonl/{format}/chunk-0001.jsonl
#   ... (10 files total per format)
# =============================================================================

# ── SLURM job configuration ───────────────────────────────────────────────────

#SBATCH --account=a139           # CSCS project account to bill compute hours to
#SBATCH --time=02:00:00             # Max time per job — 2 hours per chunk is generous
#SBATCH --job-name=serialize-tabular
#SBATCH --output=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/logs/serialize-%A-%a.out
# %A = parent job ID, %a = array task index (0-9)
# Each job writes its own log file so you can inspect individual chunk progress
#SBATCH --error=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/logs/serialize-%A-%a.err
#SBATCH --nodes=1                   # 1 node per chunk — serialization is CPU only
#SBATCH --ntasks=1                  # 1 process per job
#SBATCH --cpus-per-task=16          # 16 CPU cores for parallel parquet reading
#SBATCH --array=0-9                 # Launch 10 jobs: task IDs 0,1,2,...,9
                                    # Each task ID maps to one chunk directory

# ── Arguments ─────────────────────────────────────────────────────────────────

# FORMAT is passed when you submit: sbatch submit_serialize.sh csv
# $1 means "first argument after the script name"
FORMAT=$1

# Validate that FORMAT was provided
if [ -z "$FORMAT" ]; then
    echo "ERROR: No format specified."
    echo "Usage: sbatch submit_serialize.sh <format>"
    echo "Formats: csv, sql_schema, keyvalue, markdown, json"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────

# Root directory of T4 data on CSCS (read-only, managed by a139 group)
T4_BASE=/capstor/store/cscs/swissai/a139/datasets/mlfoundations_t4_full

# Convert SLURM array task ID (0-9) to chunk directory name (chunk-0000 to chunk-0009)
# printf with %04d pads with zeros: 0 → 0000, 3 → 0003, 9 → 0009
CHUNK=$(printf "chunk-%04d" $SLURM_ARRAY_TASK_ID)
CHUNK_DIR=$T4_BASE/$CHUNK

# Output directory for this format's .jsonl files
OUTPUT_DIR=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/jsonl/$FORMAT

# FIX: create logs and output dirs before the job tries to write to them.
# Without this, the --output log path above fails and the job silently dies.
mkdir -p /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/logs
mkdir -p $OUTPUT_DIR

echo "Starting serialization"
echo "  Format:    $FORMAT"
echo "  Chunk:     $CHUNK"
echo "  Input:     $CHUNK_DIR"
echo "  Output:    $OUTPUT_DIR/$CHUNK.jsonl"

# ── Environment ───────────────────────────────────────────────────────────────

# Activate conda — needed because serialize_t4.py uses pyarrow, pandas,
# transformers which are installed in the base conda environment.
# This does NOT use a container — serialization is plain Python, no GPU needed.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# ── Run serialization ─────────────────────────────────────────────────────────

# FIX: removed inline comments that were placed after backslash continuations.
# In bash, a line ending with \ continues to the next line, so a # comment
# on the continuation line is NOT treated as a comment — it gets passed as
# a literal argument to Python, causing a confusing syntax error.
# All explanatory comments are now placed above the command block instead.

# 300M tokens per chunk × 10 chunks = 3B tokens total per format.
# Enough for proxy training without processing all 76 chunks.
# Tokenizer is used only for token counting during row sampling,
# not for producing the final binary training data (that's Phase 2b).
python /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/serialize_t4.py \
    --chunk_dir $CHUNK_DIR \
    --format $FORMAT \
    --output $OUTPUT_DIR/$CHUNK.jsonl \
    --token_target 300000000 \
    --tokenizer swiss-ai/Apertus-70B-2509

echo "Done: $CHUNK for format $FORMAT"