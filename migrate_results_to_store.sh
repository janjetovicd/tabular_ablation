#!/bin/bash -l
#
# ONE-SHOT migration: copy ALL existing tabular-ablation results from scratch
# (/iopsstor, auto-purged after 14 days) to permanent, backed-up project store
# (/capstor/store/cscs/swissai/a139). Runs as a single job on the dedicated
# `xfer` partition.
#
#   Submit with:  sbatch migrate_results_to_store.sh
#   Watch it:     squeue --me   /   tail -f migrate-<jobid>.out
#
# This COPIES (rsync), it does not delete the scratch originals. Once you've
# confirmed everything landed (see the verification at the end of the log),
# scratch will clean itself up, or you can delete the originals manually.
#
# rsync is restartable: if the job times out, just submit it again and it skips
# whatever already copied.

#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --partition=xfer
#SBATCH --job-name=migrate_to_store
#SBATCH --output=migrate-%j.out
#SBATCH --error=migrate-%j.err

set -u

# ── Source (scratch) and destination (store) roots ──────────────────────────────
SCRATCH_ABLATION=/iopsstor/scratch/cscs/djanjetovic/tabular_ablation
SCRATCH_MEGATRON=/iopsstor/scratch/cscs/djanjetovic/swiss-megatron-new
SCRATCH_TOKCACHE=/iopsstor/scratch/cscs/djanjetovic/tokenizer_cache

STORE_DIR=/capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation

RSYNC="rsync -av --partial"

mkdir -p "$STORE_DIR" "$STORE_DIR/checkpoints"

echo "=================================================================="
echo " Migration started: $(date)"
echo " Source (scratch):  $SCRATCH_ABLATION  +  Megatron checkpoints"
echo " Destination:       $STORE_DIR"
echo "=================================================================="

copy() {
    # copy <source-dir> <dest-dir>  — only if the source exists
    local src="$1" dst="$2"
    if [ -e "$src" ]; then
        echo -e "\n>>> Copying: $src  ->  $dst"
        mkdir -p "$(dirname "$dst")"
        srun -n 1 $RSYNC "$src" "$dst"
        echo ">>> Done ($(date)), rsync exit=$?"
    else
        echo -e "\n--- Skipping (not found): $src"
    fi
}

# 1. Serialized JSONL datasets  (serialization job output)
copy "$SCRATCH_ABLATION/jsonl"          "$STORE_DIR/"

# 2. Tokenized .bin/.idx shards (tokenization job output)
copy "$SCRATCH_ABLATION/tokenized"      "$STORE_DIR/"

# 3. Eval pairs + candidates cache (eval-prep output, consumed by inference)
copy "$SCRATCH_ABLATION/eval"           "$STORE_DIR/"
copy "$SCRATCH_ABLATION/eval_cpu"       "$STORE_DIR/"   # if you used the CPU prep variant

# 4. Eval results (logs + results_*.jsonl)
copy "$SCRATCH_ABLATION/eval_results"   "$STORE_DIR/"

# 5. Training checkpoints + logs + tensorboard (all experiments)
copy "$SCRATCH_MEGATRON/logs/Meg-Runs/tabular-ablation" "$STORE_DIR/checkpoints_megatron"

# 6. Tokenizer cache (small but annoying to re-create on a purged scratch)
copy "$SCRATCH_TOKCACHE"                "$STORE_DIR/tokenizer_cache"

# ── Verification ────────────────────────────────────────────────────────────────
echo -e "\n=================================================================="
echo " Migration finished: $(date)"
echo " Destination tree (top level):"
ls -la "$STORE_DIR"
echo
echo " Disk usage on store (this may take a moment):"
du -sh "$STORE_DIR"/* 2>/dev/null
echo "=================================================================="
echo " NOTE: originals on scratch were NOT deleted. Confirm the sizes above"
echo "       look right, then let scratch auto-purge or remove them yourself."
echo "       Check your store quota with:  quota   (or ask your PI if low)."
