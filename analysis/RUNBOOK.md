# Fixed-table-size assessment + evaluation — runbook

Scripts to **assess** the finished tabular-ablation training runs, **compare** the
five formats at a fair *fixed table size* (rows seen) rather than fixed tokens, and
**launch evaluation** at the matched checkpoints. Everything runs on a CSCS login
node except the eval jobs themselves (Slurm/GPU).

## Why fixed table size, not tokens

All 5 models trained to the same **30 B-token** budget (28,610 iters). But the
formats are not equally token-efficient: CSV packs ~2× more rows into 30 B tokens
than keyvalue. So comparing at the final checkpoint is a *fixed-token* comparison —
it confounds "which serialization is better" with "which model simply saw more
table data."

Clara's fix: compare each format at the checkpoint where it has consumed the same
number of **rows** (≡ cells, since the row stream is identical across formats).
Because every format serializes the same tables with the same cleaning and the same
deterministic shuffle, the total row count is shared and cancels out — leaving a
clean result:

```
matched_iter(format) = 28,610 × dataset_tokens(format) / max_k dataset_tokens(k)
```

The most verbose format (most dataset tokens — expected to be keyvalue) is the
binding constraint and stays at its final checkpoint; every other format is rolled
back to the proportional earlier checkpoint. We only need each format's tokenized
**dataset token count** to compute this — that is what `02_token_stats.py` measures.

> **Rows vs cells vs segments — why rows.** Segments are a 3,800-token packing
> artifact (a CSV segment holds ~2× the rows of a keyvalue segment), so equalizing
> segments does *not* equalize information. Tables vary wildly in size, so "tables
> seen" is coarse and ambiguous for partially-consumed tables. Rows (≡ cells here)
> is the faithful unit of "table information seen," and it falls straight out of the
> token counts — no extra bookkeeping.

## Paths (confirmed against the cluster, all on permanent store `a139`)

| What | Path |
|---|---|
| Checkpoints (store) | `…/a139/djanjetovic/tabular_ablation/checkpoints/<exp>/<exp>/checkpoints/iter_*` |
| Tokenized shards | `…/a139/djanjetovic/tabular_ablation/tokenized/<format>/train_*_tokens.{bin,idx}` |
| Eval pairs | `…/a139/djanjetovic/tabular_ablation/eval/pairs_<format>.jsonl` ✅ already prepared |
| Checkpoint save interval | every 500 iters (all intermediate checkpoints kept) |

Scripts default to store only (scratch is never relied on). The double-nested
`<exp>/<exp>/checkpoints` layout from rsync stage-out is detected automatically.

### Choosing which run to evaluate

There are **two batches** per (format, seed):
- the **first attempt** (job IDs ~`2595xxx`) — reached ~28,500 steps but used the
  older container that had failures on some runs;
- the **fixed-container rerun** (job IDs ~`2625xxx`) — scientifically clean, but
  stopped earlier (~21,000–21,500 steps).

The fixed-table-size method does **not** require any specific final step — it
equalizes every run at the largest row count they all reached. So:

- **Recommended:** use the rerun batch for container-consistency:
  `03_fixed_table_checkpoints.py --min-jobid 2625000`
- To use the (more-complete) first batch instead, drop the flag (the script auto-
  picks the most-complete copy per format+seed).

Run `01_assess_training.py` first to see exactly how far each run got, then decide.

## Steps

```bash
cd analysis/

# 1. Assess: did all 5 formats (× seeds) finish? what checkpoints exist?
python3 01_assess_training.py --sacct --out training_assessment.json

# 2. Token stats: tokens + segments per format (drives the matching)
python3 02_token_stats.py --out token_stats.json
#    override root if needed:
#    --tokenized-root /capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation/tokenized

# 3. Compute matched checkpoints + build non-destructive view dirs
#    --min-jobid 2625000 pins the fixed-container rerun batch (recommended).
#    Add --dry-run first to preview without creating symlinks.
python3 03_fixed_table_checkpoints.py \
    --token-stats token_stats.json \
    --assessment  training_assessment.json \
    --min-jobid 2625000 --dry-run        # then re-run without --dry-run

# 4. (optional but recommended) Re-plot loss vs rows seen
#    Option A — from tensorboard logs:
python3 04_replot_loss_vs_rows.py --token-stats token_stats.json \
    --tb-root /capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation/checkpoints \
    --out loss_vs_rows.png
#    Option B — from a WandB CSV export (columns: format,step,loss):
#    python3 04_replot_loss_vs_rows.py --token-stats token_stats.json \
#        --loss-csv wandb_loss.csv --out loss_vs_rows.png

# 5. Make sure eval pairs exist (held-out chunks 24–25), then submit eval
#    at the matched checkpoints:
python3 ../prepare_eval_data_v2.py        # only if pairs_<fmt>.jsonl not already on store
./05_run_fixed_table_eval.sh              # mode=matched (fixed table size)
./05_run_fixed_table_eval.sh --mode final # optional: fixed-token baseline for contrast

# 6. Aggregate accuracy (mean ± std across seeds) once eval jobs finish
python3 ../aggregate_results.py \
    --results-dir /capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation/eval_results
```

## Reading the result

- **`aggregate_results.py`** gives accuracy per format. The `_matched` run-ids are
  the fair, fixed-table-size comparison; `_final` are the fixed-token baseline.
  Compare the two to see how much the earlier ranking was a data-quantity artifact.
- **Accuracy is the primary metric.** LM loss is *not* directly comparable across
  formats (different token distributions), so treat `loss_vs_rows.png` as a sanity
  check on training health and relative data exposure, not as the verdict.
- Random chance on the statement-vs-distractor task is **0.50**.

## What each script writes

| Script | Output |
|---|---|
| `01_assess_training.py` | `training_assessment.json` + console table |
| `02_token_stats.py` | `token_stats.json` + console table |
| `03_fixed_table_checkpoints.py` | `eval_plan.json` + checkpoint view dirs |
| `04_replot_loss_vs_rows.py` | `loss_vs_rows.png` |
| `05_run_fixed_table_eval.sh` | submits Slurm eval jobs |
