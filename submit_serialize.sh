#!/bin/bash
# OBJECTIVE: Serialize T4 parquet files into .jsonl format.
#
# Specifically:
#   1. Runs serialize_t4.py on 10 T4 chunks in parallel using a SLURM array job.

# Arguments

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

# Paths

# Root directory of T4 data on CSCS (read-only, managed by a139 group)
T4_BASE=/capstor/store/cscs/swissai/a139/datasets/mlfoundations_t4_full

# Convert SLURM array task ID (0-9) to chunk directory name (chunk-0000 to chunk-0009)
# printf with %04d pads with zeros: 0 → 0000, 3 → 0003, 9 → 0009
CHUNK=$(printf "chunk-%04d" $SLURM_ARRAY_TASK_ID)
CHUNK_DIR=$T4_BASE/$CHUNK

# Output directory for this format's .jsonl files
OUTPUT_DIR=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/jsonl/$FORMAT

echo "Starting serialization"
echo "  Format:    $FORMAT"
echo "  Chunk:     $CHUNK"
echo "  Input:     $CHUNK_DIR"
echo "  Output:    $OUTPUT_DIR/$CHUNK.jsonl"

# Environment

# Activate conda — needed because serialize_t4.py uses pyarrow, pandas,
# transformers which are installed in the base conda environment.
# This does NOT use a container — serialization is plain Python, no GPU needed.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Run serialization

python /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/serialize_t4.py \
    --chunk_dir $CHUNK_DIR \
    --format $FORMAT \
    --output $OUTPUT_DIR/$CHUNK.jsonl \
    --token_target 300000000 \
    # 300M tokens per chunk × 10 chunks = 3B tokens total per format
    # Enough for proxy training without processing all 76 chunks
    --tokenizer swiss-ai/Apertus-70B-2509
    # Apertus tokenizer — used only for token counting during sampling,
    # not for producing the final binary training data (that's Phase 2b)

echo "Done: $CHUNK for format $FORMAT"
