#!/bin/bash
# 05_run_fixed_table_eval.sh — submit BLiMP eval at the rows-equalized
# ("fixed table size") checkpoints chosen by 03_fixed_table_checkpoints.py.
#
# It reads eval_plan.json and, for every planned (format, seed) experiment,
# submits submit_eval_inference.sh pointed at that experiment's *view* dir
# (the symlinked matched checkpoint). Results are tagged "<fmt>_seed<seed>_matched"
# so they never collide with the fixed-token (final-checkpoint) eval.
#
# Usage (login node):
#   ./05_run_fixed_table_eval.sh                       # mode=matched (default)
#   ./05_run_fixed_table_eval.sh --mode final          # eval at final checkpoints
#   ./05_run_fixed_table_eval.sh --plan eval_plan.json --submit-script ../submit_eval_inference.sh
#
# Prerequisite: eval pairs must already exist at $STORE_DIR/eval/pairs_<fmt>.jsonl
# (produced by prepare_eval_data_v2.py — see RUNBOOK.md).

set -euo pipefail

PLAN="eval_plan.json"
SUBMIT_SCRIPT="../submit_eval_inference.sh"
MODE="matched"          # matched | final

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan)          PLAN="$2"; shift 2 ;;
        --submit-script) SUBMIT_SCRIPT="$2"; shift 2 ;;
        --mode)          MODE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ ! -f "$PLAN" ]; then
    echo "ERROR: plan file not found: $PLAN  (run 03_fixed_table_checkpoints.py first)"
    exit 1
fi
if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "ERROR: submit script not found: $SUBMIT_SCRIPT"
    exit 1
fi

echo "Plan:   $PLAN"
echo "Mode:   $MODE   (matched = rows-equalized view dirs, final = real final ckpt)"
echo "Submit: $SUBMIT_SCRIPT"
echo

# Emit "format<TAB>seed<TAB>ckpt_dir<TAB>matched_iter" lines from the plan.
python3 - "$PLAN" "$MODE" <<'PY' | while IFS=$'\t' read -r FMT SEED CKPT ITER; do
import json, sys
plan = json.load(open(sys.argv[1]))
mode = sys.argv[2]
for e in plan["plan"]:
    seed = e["seed"] if e["seed"] is not None else "NA"
    if mode == "final":
        ckpt = e["real_ckpt_dir"]            # Megatron loads its latest iter
        tag  = f"{seed}_final"
    else:
        ckpt = e["view_ckpt_dir"]            # symlinked matched checkpoint
        tag  = f"{seed}_matched"
    print(f"{e['format']}\t{tag}\t{ckpt}\t{e['matched_iter']}")
PY
    echo ">> $FMT  seed-tag=$SEED  iter=$ITER"
    echo "   ckpt: $CKPT"
    EVAL_ENABLED=1 sbatch "$SUBMIT_SCRIPT" "$FMT" "$CKPT" "$SEED"
done

echo
echo "All eval jobs submitted. Watch with: squeue --me"
echo "When done, aggregate with:"
echo "  python3 ../aggregate_results.py --results-dir \\"
echo "    /capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation/eval_results"
