#!/bin/bash
#SBATCH --account=a139
#SBATCH --job-name=tokenize
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --partition=normal
#SBATCH --output=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/logs/tokenize_%x_%j.out

# OBJECTIVE: Tokenize .jsonl files into Megatron binary format (.bin/.idx).
# Uses datatrove MegatronDocumentTokenizer with a pre-cached local tokenizer.
#
# CHANGES from previous version:
#   1. Removed the merge step entirely. datatrove reads chunk files directly
#      from the directory, which is faster and avoids the 63GB merge overhead.
#   2. Replaced python3 - <<EOF heredoc with a real .py file written to disk.
#      The heredoc caused a forkserver crash because worker processes tried to
#      re-import <stdin> which is not a real file on disk.
#
# Usage (run once per format, after all serialize jobs for that format complete):
#   sbatch submit_tokenize.sh csv
#   sbatch submit_tokenize.sh sql_schema
#   sbatch submit_tokenize.sh keyvalue
#   sbatch submit_tokenize.sh markdown
#   sbatch submit_tokenize.sh json

FORMAT=$1

if [ -z "$FORMAT" ]; then
    echo "ERROR: No format specified."
    echo "Usage: sbatch submit_tokenize.sh <format>"
    echo "Formats: csv, sql_schema, keyvalue, markdown, json"
    exit 1
fi

if [[ ! "$FORMAT" =~ ^(csv|sql_schema|keyvalue|markdown|json)$ ]]; then
    echo "ERROR: Unknown format '$FORMAT'."
    echo "Valid options: csv, sql_schema, keyvalue, markdown, json"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────

JSONL_DIR=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/jsonl/$FORMAT
OUTPUT_DIR=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tokenized/$FORMAT
TOKENIZER_PATH=/iopsstor/scratch/cscs/djanjetovic/tokenizer_cache/apertus

echo "Starting tokenization with datatrove"
echo "  Format:    $FORMAT"
echo "  Input:     $JSONL_DIR"
echo "  Output:    $OUTPUT_DIR"
echo "  Tokenizer: $TOKENIZER_PATH"

# Validate that the tokenizer cache exists
if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "ERROR: Tokenizer cache not found at $TOKENIZER_PATH"
    echo "Run the following on the LOGIN NODE first:"
    echo "  python -c \""
    echo "  from transformers import AutoTokenizer"
    echo "  tok = AutoTokenizer.from_pretrained('swiss-ai/Apertus-70B-2509')"
    echo "  tok.save_pretrained('$TOKENIZER_PATH')"
    echo "  print('Tokenizer cached.')\""
    exit 1
fi

# Validate that chunk files exist
CHUNK_COUNT=$(ls $JSONL_DIR/chunk-*.jsonl 2>/dev/null | wc -l)
echo "  Chunk files found: $CHUNK_COUNT"
if [ "$CHUNK_COUNT" -eq 0 ]; then
    echo "ERROR: No chunk-*.jsonl files found in $JSONL_DIR"
    echo "Run submit_serialize.sh first."
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

mkdir -p $OUTPUT_DIR

# Write the tokenization script to a real .py file on disk.
# This is required because datatrove uses forkserver multiprocessing — worker
# processes need to re-import the main script by file path. The old heredoc
# approach (python3 - <<EOF) made the script appear as <stdin> which is not
# a real path, causing all 28 workers to crash with FileNotFoundError.
TOKENIZE_SCRIPT=$OUTPUT_DIR/run_tokenize.py

cat > $TOKENIZE_SCRIPT << EOF
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import MegatronDocumentTokenizer

executor = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(
            "$JSONL_DIR",
            text_key="text",
            glob_pattern="chunk-*.jsonl",
        ),
        MegatronDocumentTokenizer(
            output_folder="$OUTPUT_DIR",
            tokenizer_name_or_path="$TOKENIZER_PATH",
            save_filename="train",
        ),
    ],
    tasks=28,
    workers=28,
    logging_dir="$OUTPUT_DIR/logs",
)
executor.run()
EOF

python3 $TOKENIZE_SCRIPT

echo "Done tokenizing $FORMAT"
echo "Output files:"
ls -lh $OUTPUT_DIR/*.bin $OUTPUT_DIR/*.idx 2>/dev/null || echo "No .bin/.idx files found — check logs above."