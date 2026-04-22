#!/bin/bash
#SBATCH --account=a139
# CHANGE: added --account=a139 to match the other scripts and ensure correct
# project billing on CSCS. Without this the job may be rejected or billed wrong.
#SBATCH --job-name=tokenize
#SBATCH --time=06:00:00
# CHANGE: was 04:00:00. With ~10B tokens across 76 chunks merged into one file
# and 28 parallel workers, tokenization takes 3-5h. 6h gives a safe buffer.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --partition=normal
#SBATCH --output=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/logs/tokenize_%x_%j.out

# OBJECTIVE: Tokenize .jsonl files into Megatron binary format (.bin/.idx).
# Uses datatrove MegatronDocumentTokenizer with a pre-cached local tokenizer
# instead of preprocess_data.py, which fails silently on compute nodes
# because they have no internet access and cannot load from HuggingFace Hub.
#
# Usage (run once per format, after all serialize jobs for that format complete):
#   sbatch submit_tokenize.sh csv
#   sbatch submit_tokenize.sh sql_schema
#   sbatch submit_tokenize.sh keyvalue
#   sbatch submit_tokenize.sh markdown
#   sbatch submit_tokenize.sh json
#
# Prerequisites:
#   1. Tokenizer must be cached locally (step 1 in execution plan, login node).
#   2. All chunk-*.jsonl files for this format must exist in JSONL_DIR.
#      Check: ls /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/jsonl/<format>/
#      Should show 76 files (chunk-0000.jsonl to chunk-0075.jsonl).

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

# CHANGE: tokenizer path must match exactly where you cached it in step 1.
# The path in serialize_t4.py and submit_serialize.sh uses the same location.
# All three scripts must agree on this path.
TOKENIZER_PATH=/iopsstor/scratch/cscs/djanjetovic/tokenizer_cache/apertus

echo "Starting tokenization with datatrove"
echo "  Format:    $FORMAT"
echo "  Input:     $JSONL_DIR"
echo "  Output:    $OUTPUT_DIR"
echo "  Tokenizer: $TOKENIZER_PATH"

# Validate that the tokenizer cache exists before submitting
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

# Merge all chunk files into one for datatrove
# CHANGE: added a check so we don't re-merge if merged.jsonl already exists
# and appears complete (non-zero size). This saves time on retries.
MERGED=$JSONL_DIR/merged.jsonl
if [ -f "$MERGED" ] && [ -s "$MERGED" ]; then
    echo "merged.jsonl already exists ($(du -sh $MERGED | cut -f1)). Skipping merge."
else
    echo "Merging $CHUNK_COUNT chunk files..."
    cat $JSONL_DIR/chunk-*.jsonl > $MERGED
    echo "Merged. Lines: $(wc -l < $MERGED)"
    echo "Merged file size: $(du -sh $MERGED | cut -f1)"
fi

# Run tokenization using datatrove
python3 - <<EOF
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import MegatronDocumentTokenizer

executor = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(
            "$MERGED",
            text_key="text",
        ),
        MegatronDocumentTokenizer(
            output_folder="$OUTPUT_DIR",
            tokenizer_name_or_path="$TOKENIZER_PATH",
            max_tokens_per_file=1e9,   # ~1B tokens per .bin shard
            save_filename="train",
        ),
    ],
    tasks=28,      # matches --cpus-per-task
    workers=28,
    logging_dir="$OUTPUT_DIR/logs",
)
executor.run()
EOF

echo "Done tokenizing $FORMAT"
echo "Output files:"
ls -lh $OUTPUT_DIR/*.bin $OUTPUT_DIR/*.idx 2>/dev/null || echo "No .bin/.idx files found — check logs above."