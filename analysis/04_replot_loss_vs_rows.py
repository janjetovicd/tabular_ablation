#!/usr/bin/env python3
"""
04_replot_loss_vs_rows.py — re-draw the training-loss curves with the x-axis
converted from training step (== fixed tokens) to ROWS SEEN (== fixed table
size). This is the visual version of Clara's fair comparison: read off each
format's loss at the SAME amount of table data instead of the same step.

X-AXIS CONVERSION
  rows_seen(f, step) ∝ step / tokens_per_row(f) ∝ step / dataset_tokens(f)
  We rescale every curve into "reference-format token-equivalents" so that equal
  x means equal rows:

      x(f, step) = step * tokens_per_iter * (max_tokens / dataset_tokens[f])

  The reference (most verbose) format is unchanged; efficient formats get
  stretched right (same step = more rows). A vertical line at 30B marks the common
  rows budget — each curve's height there is its loss at equal data exposure.

LOSS INPUT — two options (pick whichever is easier on the cluster):
  A) --tb-root <dir>   Glob each experiment's tensorboard logs and pull the
                       'lm loss' scalar.  (needs `tensorboard` importable)
  B) --loss-csv <csv>  A tidy CSV you export from WandB with columns:
                       format,step,loss   (one row per logged point)

Usage:
  python3 04_replot_loss_vs_rows.py --token-stats token_stats.json \
      --tb-root /capstor/.../tabular_ablation/checkpoints --out loss_vs_rows.png
  python3 04_replot_loss_vs_rows.py --token-stats token_stats.json \
      --loss-csv wandb_loss.csv --out loss_vs_rows.png
"""

import os
import re
import glob
import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GBS, SEQ_LEN = 256, 4096
TOKENS_PER_ITER = GBS * SEQ_LEN
TARGET_TOKENS = 30_000_000_000
TARGET_ITERS = TARGET_TOKENS // TOKENS_PER_ITER          # 28,610
# Rows-budget x-position = reference format's final rows-equivalent (exactly the
# tokens actually consumed at the last iter, so the reference curve reaches it).
BUDGET_X_B = TARGET_ITERS * TOKENS_PER_ITER / 1e9
FORMATS = ["csv", "json", "keyvalue", "markdown", "sql_schema"]
COLORS = {"csv": "#1f77b4", "json": "#2ca02c", "keyvalue": "#ff7f0e",
          "markdown": "#9467bd", "sql_schema": "#d62728"}
EXP_RE = re.compile(r"tabular-(csv|json|keyvalue|markdown|sql_schema)-1p5b")


def load_from_tensorboard(tb_root):
    """Return {format: (steps[], losses[])} by reading tensorboard scalars."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out = defaultdict(lambda: ([], []))
    event_files = glob.glob(os.path.join(tb_root, "**", "events.out.tfevents.*"),
                            recursive=True)
    by_fmt_dir = defaultdict(list)
    for ev in event_files:
        m = EXP_RE.search(ev)
        if m:
            by_fmt_dir[m.group(1)].append(ev)
    for fmt, files in by_fmt_dir.items():
        steps, losses = [], []
        for ev in sorted(files):
            ea = EventAccumulator(ev, size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            # prefer an exact 'lm loss', else any tag containing 'loss'
            tag = next((t for t in tags if t.lower() == "lm loss"), None)
            if tag is None:
                tag = next((t for t in tags if "loss" in t.lower()
                            and "scale" not in t.lower()), None)
            if tag is None:
                continue
            for e in ea.Scalars(tag):
                steps.append(e.step)
                losses.append(e.value)
        if steps:
            order = np.argsort(steps)
            out[fmt] = (np.array(steps)[order], np.array(losses)[order])
    return out


def load_from_csv(path):
    """CSV columns: format,step,loss."""
    import csv
    tmp = defaultdict(lambda: ([], []))
    with open(path) as fh:
        for row in csv.DictReader(fh):
            fmt = row["format"].strip()
            tmp[fmt][0].append(float(row["step"]))
            tmp[fmt][1].append(float(row["loss"]))
    out = {}
    for fmt, (s, l) in tmp.items():
        s, l = np.array(s), np.array(l)
        order = np.argsort(s)
        out[fmt] = (s[order], l[order])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-stats", default="token_stats.json")
    ap.add_argument("--tb-root", default=None)
    ap.add_argument("--loss-csv", default=None)
    ap.add_argument("--out", default="loss_vs_rows.png")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="Optional EMA smoothing factor in [0,1), e.g. 0.9.")
    args = ap.parse_args()

    if not args.tb_root and not args.loss_csv:
        raise SystemExit("Provide --tb-root or --loss-csv.")

    token_stats = json.load(open(args.token_stats))
    dataset_tokens = {f: token_stats[f]["total_tokens"]
                      for f in FORMATS if token_stats.get(f, {}).get("total_tokens")}
    max_tokens = max(dataset_tokens.values())
    ref_fmt = max(dataset_tokens, key=dataset_tokens.get)

    curves = (load_from_csv(args.loss_csv) if args.loss_csv
              else load_from_tensorboard(args.tb_root))
    if not curves:
        raise SystemExit("No loss curves found in the given source.")

    def ema(y, a):
        if a <= 0:
            return y
        out = np.empty_like(y, dtype=float)
        acc = y[0]
        for i, v in enumerate(y):
            acc = a * acc + (1 - a) * v
            out[i] = acc
        return out

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: loss vs step (fixed tokens — the original, unfair-for-formats view)
    for fmt in FORMATS:
        if fmt not in curves:
            continue
        s, l = curves[fmt]
        ax1.plot(s, ema(l, args.ema), color=COLORS[fmt], label=fmt, lw=1.2)
    ax1.set_title("Fixed tokens: loss vs training step")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("LM loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: loss vs rows-equivalents (fixed table size — the fair view)
    for fmt in FORMATS:
        if fmt not in curves or fmt not in dataset_tokens:
            continue
        s, l = curves[fmt]
        x_rows = s * TOKENS_PER_ITER * (max_tokens / dataset_tokens[fmt]) / 1e9
        ax2.plot(x_rows, ema(l, args.ema), color=COLORS[fmt], label=fmt, lw=1.2)
    ax2.axvline(BUDGET_X_B, color="k", ls="--", lw=1,
                label="common rows budget")
    ax2.set_title("Fixed table size: loss vs rows seen")
    ax2.set_xlabel(f"rows seen  ({ref_fmt} token-equivalents, B)")
    ax2.set_ylabel("LM loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"Wrote {args.out}")

    # Print loss at the common rows budget (interpolated) per format
    print("\nLoss at the common rows budget (fixed table size):")
    budget_x = BUDGET_X_B
    rows = []
    for fmt in FORMATS:
        if fmt not in curves or fmt not in dataset_tokens:
            continue
        s, l = curves[fmt]
        x_rows = s * TOKENS_PER_ITER * (max_tokens / dataset_tokens[fmt]) / 1e9
        if x_rows[-1] >= budget_x:
            loss_at = float(np.interp(budget_x, x_rows, ema(l, args.ema)))
            rows.append((fmt, loss_at))
            print(f"  {fmt:<12} {loss_at:.4f}")
        else:
            print(f"  {fmt:<12} (curve ends at {x_rows[-1]:.1f}B < budget)")
    if rows:
        best = min(rows, key=lambda r: r[1])
        print(f"\nLowest loss at equal data exposure: {best[0]} ({best[1]:.4f})")
        print("NOTE: loss is not directly comparable across formats (different "
              "token distributions). Use this as a sanity check; the eval "
              "accuracy at matched checkpoints is the primary metric.")


if __name__ == "__main__":
    main()
