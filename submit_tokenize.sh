#!/bin/bash
# OBJECTIVE: Tokenize .jsonl files into Megatron binary format (.bin/.idx).
#
# Specifivally:
#   1. Merges all 10 chunk .jsonl files for a format into one merged.jsonl
#   2. Runs Megatron's preprocess_data.py to convert text → integer token IDs
#      and write binary .bin and .idx files the training job reads directly
#
# Why this step is needed:
#   Megatron cannot read .jsonl files directly. It needs binary files where
#   every token is already converted to its integer ID and packed sequentially.
#   preprocess_data.py does this conversion using the Apertus tokenizer.
#   The .idx file is an index so Megatron can jump to any document instantly.
#
# This is where the REAL tokenization happens. In Phase 2a, the tokenizer
# was only used to COUNT tokens. Here it actually converts text to IDs that
# get saved permanently to disk as training data.

# Arguments

FORMAT=$1

if [ -z "$FORMAT" ]; then
    echo "ERROR: No format specified."
    echo "Usage: sbatch submit_tokenize.sh <format>"
    exit 1
fi

# Paths

# Directory containing the 10 chunk .jsonl files from T4 serilization step
JSONL_DIR=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/jsonl/$FORMAT

# Output prefix for .bin and .idx files
# Megatron appends _text_document.bin and _text_document.idx automatically
OUTPUT_PREFIX=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tokenized/$FORMAT/train

# Swiss AI Megatron fork — preprocess_data.py lives in its tools/ directory
MEGATRON=/iopsstor/scratch/cscs/djanjetovic/Megatron-LM

echo "Starting tokenization"
echo "  Format:  $FORMAT"
echo "  Input:   $JSONL_DIR"
echo "  Output:  $OUTPUT_PREFIX"

# Environment

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Create output directory if it doesn't exist
mkdir -p /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tokenized/$FORMAT

# Merge all chunk files

# Concatenate all 10 chunk .jsonl files into one merged file
# preprocess_data.py reads a single input file, not a directory
echo "Merging chunk files..."
cat $JSONL_DIR/chunk-*.jsonl > $JSONL_DIR/merged.jsonl
echo "Merged. Total lines: $(wc -l < $JSONL_DIR/merged.jsonl)"

# Run tokenization

python $MEGATRON/tools/preprocess_data.py \
    --input $JSONL_DIR/merged.jsonl \
    # The merged .jsonl file — one {"text": "..."} line per table
    --output-prefix $OUTPUT_PREFIX \
    # Where to write .bin and .idx files
    --tokenizer-type HuggingFaceTokenizer \
    # Use HuggingFace tokenizer format (required for Apertus BPE tokenizer)
    --tokenizer-model swiss-ai/Apertus-70B-2509 \
    # Apertus tokenizer with 128k vocab — must match what training uses
    --workers 32 \
    # Parallel workers — matches --cpus-per-task above for full utilization
    --append-eod
    # Append End-Of-Document token between documents so Megatron knows
    # where one table ends and the next begins when packing sequences

echo "Done tokenizing $FORMAT"
echo "Output files:"
ls -lh /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tokenized/$FORMAT/
