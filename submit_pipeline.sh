#!/bin/bash
# Submits the full pipeline for all 5 formats in parallel with SLURM dependencies.
# Run this once from the login node — it will chain serialize → tokenize → train → eval
# automatically for each format. Everything runs in parallel across formats.
#
# Usage:
#   bash submit_pipeline.sh
#
# What it submits (per format, all in parallel):
#   1. serialize  (array job: 76 tasks, one per T4 chunk) — reads from capstor
#   2. tokenize   (starts after all 76 serialize tasks finish)
#   3. train      (starts after tokenize finishes) — auto-submits eval when done

set -e

ABLATION_DIR=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation
FORMATS=(csv json keyvalue markdown sql_schema)

# ── Sanity check: tokenizer cache ─────────────────────────────────────────────
# The tokenizer is used by both serialize and training. Check it still exists
# since it's also on iopsstor/scratch and may have been purged.
TOKENIZER_CHECK=/iopsstor/scratch/cscs/djanjetovic/tokenizer_cache/apertus/tokenizer.json
if [ ! -f "$TOKENIZER_CHECK" ]; then
    echo "ERROR: Tokenizer cache missing: $TOKENIZER_CHECK"
    echo ""
    echo "Re-cache it on the LOGIN NODE first (takes ~2 min, needs internet):"
    echo "  python3 -c \""
    echo "  from transformers import AutoTokenizer"
    echo "  tok = AutoTokenizer.from_pretrained('swiss-ai/Apertus-70B-2509')"
    echo "  tok.save_pretrained('/iopsstor/scratch/cscs/djanjetovic/tokenizer_cache/apertus')"
    echo "  print('Done.')\""
    echo ""
    echo "Then re-run this script."
    exit 1
fi
echo "Tokenizer cache OK."

# ── Submit pipeline for each format ───────────────────────────────────────────

for FORMAT in "${FORMATS[@]}"; do
    echo ""
    echo "=== Submitting pipeline for: $FORMAT ==="

    # Step 1: Serialize (array job, 76 tasks = chunks 0-75)
    SERIALIZE_JOB=$(sbatch --parsable --array=0-75 \
        $ABLATION_DIR/submit_serialize.sh $FORMAT)
    echo "  Serialize job:  $SERIALIZE_JOB (76 array tasks)"

    # Step 2: Tokenize — waits for ALL 76 serialize tasks to finish successfully
    TOKENIZE_JOB=$(sbatch --parsable \
        --dependency=afterok:$SERIALIZE_JOB \
        $ABLATION_DIR/submit_tokenize.sh $FORMAT)
    echo "  Tokenize job:   $TOKENIZE_JOB (starts after serialize)"

    # Step 3: Train — waits for tokenize to finish
    # Training script auto-submits the eval job when it finishes.
    TRAIN_JOB=$(sbatch --parsable \
        --dependency=afterok:$TOKENIZE_JOB \
        $ABLATION_DIR/submit_tabular_ablation.sh $FORMAT)
    echo "  Training job:   $TRAIN_JOB (starts after tokenize)"
    echo "  Eval:           auto-submitted by training when done"
done

echo ""
echo "All pipelines submitted. Check status with:"
echo "  squeue -u djanjetovic"
echo ""
echo "Timeline estimate:"
echo "  Serialize: ~2-4h  (76 chunks in parallel)"
echo "  Tokenize:  ~1-2h"
echo "  Train:     ~12h"
echo "  Eval:      ~2-4h"
echo "  Total:     ~17-22h  → done well before Tuesday"
