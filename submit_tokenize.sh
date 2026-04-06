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
#
# Usage (run once per format, after Phase 2a finishes):
#   sbatch submit_tokenize.sh csv
#   sbatch submit_tokenize.sh sql_schema
#   sbatch submit_tokenize.sh keyvalue
#   sbatch submit_tokenize.sh markdown
#   sbatch submit_tokenize.sh json
#
# Output per format:
#   /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tokenized/{format}/train.bin
#   /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tokenized/{format}/train.idx
# =============================================================================

# ── SLURM job configuration ───────────────────────────────────────────────────

#SBATCH --account=a139
#SBATCH --time=04:00:00             # Tokenization takes longer than serialization
#SBATCH --job-name=tokenize-tabular
#SBATCH --output=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/logs/tokenize-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/logs/tokenize-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # More cores = faster tokenization (--workers 32 below)

# ── Arguments ─────────────────────────────────────────────────────────────────

FORMAT=$1

if [ -z "$FORMAT" ]; then
    echo "ERROR: No format specified."
    echo "Usage: sbatch submit_tokenize.sh <format>"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────

# Directory containing the 10 chunk .jsonl files from Phase 2a
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

# ── Environment ───────────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Create output directory if it doesn't exist
mkdir -p /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tokenized/$FORMAT

# ── Merge all chunk files ─────────────────────────────────────────────────────

# Concatenate all 10 chunk .jsonl files into one merged file.
# preprocess_data.py reads a single input file, not a directory.
echo "Merging chunk files..."
cat $JSONL_DIR/chunk-*.jsonl > $JSONL_DIR/merged.jsonl
echo "Merged. Total lines: $(wc -l < $JSONL_DIR/merged.jsonl)"

# ── Run tokenization ──────────────────────────────────────────────────────────

# FIX: removed all inline comments from inside the python \ command block.
# In bash, after a line continuation backslash \, a # on the next line is NOT
# a comment — it is passed as a literal argument to Python, breaking the call.
# All explanatory notes are placed above the command block instead.
#
# What each argument does:
#   --input:          the merged .jsonl file, one {"text": "..."} per line
#   --output-prefix:  where to write .bin and .idx (Megatron appends suffixes)
#   --tokenizer-type: HuggingFaceTokenizer = required for Apertus BPE tokenizer
#   --tokenizer-model: Apertus 128k vocab tokenizer — must match training
#   --workers:        32 parallel workers, matches --cpus-per-task above
#   --append-eod:     inserts end-of-document token so Megatron knows where
#                     one table ends and the next begins when packing sequences
python $MEGATRON/tools/preprocess_data.py \
    --input $JSONL_DIR/merged.jsonl \
    --output-prefix $OUTPUT_PREFIX \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model swiss-ai/Apertus-70B-2509 \
    --workers 32 \
    --append-eod

# FIX: delete the merged.jsonl after tokenization to free disk space.
# At 3B tokens the merged file is ~12-15 GB. With 5 formats running,
# leaving them all on disk wastes 60-75 GB of scratch space.
echo "Cleaning up merged file..."
#rm $JSONL_DIR/merged.jsonl
echo "Removed $JSONL_DIR/merged.jsonl"

echo "Done tokenizing $FORMAT"
echo "Output files:"
ls -lh /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tokenized/$FORMAT/
