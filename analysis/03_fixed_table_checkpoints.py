#!/usr/bin/env python3
"""
03_fixed_table_checkpoints.py — the "fixed table size, not tokens" comparison.

THE PROBLEM (Clara's point)
  The models trained on a token budget, but token-efficient formats (csv) pack
  more rows into the same tokens than verbose ones (keyvalue). Comparing at the
  final checkpoint is a fixed-TOKEN comparison and confounds "which format is
  better" with "which model saw more table data."

THE FIX
  Compare each format at the checkpoint where it has consumed the SAME NUMBER OF
  ROWS (≡ cells, since every format serializes the same tables in the same row
  order). Rows seen ∝ iter / dataset_tokens(format), because
      tokens_per_row(f) = dataset_tokens(f) / total_rows   (total_rows shared).

  We pick the largest rows budget that EVERY run can reach:
      rows_at_final(run)  = final_iter(run) / dataset_tokens(format)
      budget              = min over runs of rows_at_final
      matched_iter(run)   = budget * dataset_tokens(format)        (snap to disk)

  The binding run (smallest rows-at-final — the most verbose and/or least-trained)
  stays at its final checkpoint; every other run rolls back to the matched earlier
  checkpoint. This works even when runs stopped at different steps — we do NOT
  assume everyone hit 28,610.

  NOTE rows vs cells vs segments: rows ≡ cells here (the i-th row has the same
  width in every format), and segments are a token-packing artifact (a csv segment
  holds ~2x the rows of a keyvalue one), so rows is the only fair unit.

WHAT IT DOES
  1. Reads token_stats.json (02_…) and training_assessment.json (01_…).
  2. Picks one run per (format, seed) — the most-complete copy — optionally
     restricted to a batch via --min-jobid / --jobid-substr.
  3. Computes each run's matched checkpoint and snaps to the nearest one on disk.
  4. Builds a NON-DESTRUCTIVE symlink "view" dir per run so eval can load the
     matched step without touching the real checkpoints.
  5. Writes eval_plan.json (+ prints a table).
"""

import os
import json
import argparse
from collections import defaultdict

GBS, SEQ_LEN = 256, 4096
TOKENS_PER_ITER = GBS * SEQ_LEN
FORMATS = ["csv", "json", "keyvalue", "markdown", "sql_schema"]


def snap(target_iter, available):
    return min(available, key=lambda it: abs(it - target_iter)) if available else None


def real_ckpt_dir(info):
    return info.get("store_ckpt_dir") or info.get("ckpt_dir")


def available_iters(info):
    return info.get("store_iters") or info.get("iters") or []


def make_view(views_dir, exp_name, ckpt_dir, matched_iter, dry_run):
    view = os.path.join(views_dir, f"{exp_name}__iter{matched_iter}")
    src_iter = os.path.join(ckpt_dir, f"iter_{matched_iter:07d}")
    link = os.path.join(view, f"iter_{matched_iter:07d}")
    marker = os.path.join(view, "latest_checkpointed_iteration.txt")
    if dry_run:
        return view
    os.makedirs(view, exist_ok=True)
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(src_iter, link)
    with open(marker, "w") as fh:
        fh.write(str(matched_iter))
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-stats", default="token_stats.json")
    ap.add_argument("--assessment", default="training_assessment.json")
    ap.add_argument(
        "--views-dir",
        default="/capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation/ckpt_views",
    )
    ap.add_argument("--out", default="eval_plan.json")
    ap.add_argument("--min-jobid", type=int, default=0,
                    help="Only consider experiments whose jobid >= this "
                         "(e.g. 2625000 to pin the fixed-container rerun batch).")
    ap.add_argument("--jobid-substr", default=None,
                    help="Only consider experiments whose name contains this string.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan but do not create view dirs.")
    args = ap.parse_args()

    token_stats = json.load(open(args.token_stats))
    assessment = json.load(open(args.assessment))

    dataset_tokens = {f: token_stats[f]["total_tokens"]
                      for f in FORMATS if token_stats.get(f, {}).get("total_tokens")}
    if not dataset_tokens:
        raise SystemExit("No token counts in token_stats.json — run 02_token_stats.py first.")

    # ── Filter + select one run per (format, seed) ───────────────────────────
    def keep(exp_name, info):
        if info["format"] not in dataset_tokens:
            return False
        if not available_iters(info):
            return False
        try:
            jid = int(info.get("jobid") or 0)
        except ValueError:
            jid = 0
        if jid < args.min_jobid:
            return False
        if args.jobid_substr and args.jobid_substr not in exp_name:
            return False
        return True

    best = {}
    for exp_name, info in assessment.items():
        if not keep(exp_name, info):
            continue
        key = (info["format"], info["seed"])
        avail = available_iters(info)
        score = (max(avail), len(avail))                 # most-complete copy wins
        if key not in best or score > best[key][0]:
            best[key] = (score, exp_name, info)
    selected = {e: i for _, e, i in best.values()}
    if not selected:
        raise SystemExit("No runs match the filters. Loosen --min-jobid / --jobid-substr.")

    # ── Common rows budget across ALL selected runs ──────────────────────────
    # rows_at_final ∝ final_iter / dataset_tokens(format)
    rows_at_final = {
        e: max(available_iters(i)) / dataset_tokens[i["format"]]
        for e, i in selected.items()
    }
    budget = min(rows_at_final.values())
    binding = min(rows_at_final, key=rows_at_final.get)

    print("=" * 100)
    print("FIXED TABLE SIZE — matched checkpoints (rows-equalized across all runs)")
    print(f"Binding run (stays at final): {binding}  "
          f"(final iter {max(available_iters(selected[binding]))})")
    print("=" * 100)
    print(f"{'experiment':<46}{'fmt':<11}{'seed':>5}{'final':>8}"
          f"{'target':>9}{'snapped':>9}{'rows Δ%':>8}")
    print("-" * 100)

    plan = []
    for exp_name in sorted(selected):
        info = selected[exp_name]
        fmt = info["format"]
        avail = available_iters(info)
        final_iter = max(avail)
        target = round(budget * dataset_tokens[fmt])
        snapped = snap(target, avail)
        rows_delta = 100.0 * (snapped - target) / target if target else 0.0
        ckpt_dir = real_ckpt_dir(info)
        view = make_view(args.views_dir, exp_name, ckpt_dir, snapped, args.dry_run)
        plan.append({
            "experiment": exp_name, "format": fmt, "seed": info["seed"],
            "real_ckpt_dir": ckpt_dir, "final_iter": final_iter,
            "target_iter": target, "matched_iter": snapped,
            "rows_delta_pct": round(rows_delta, 2), "view_ckpt_dir": view,
        })
        print(f"{exp_name:<46}{fmt:<11}{str(info['seed']):>5}{final_iter:>8}"
              f"{target:>9}{snapped:>9}{rows_delta:>7.1f}%")
    print("-" * 100)

    dropped = sorted(e for e in assessment
                     if e not in selected
                     and assessment[e]["format"] in dataset_tokens
                     and available_iters(assessment[e]))
    if dropped:
        print(f"\nNot used ({len(dropped)} superseded/filtered copy(ies)): "
              + ", ".join(dropped))

    with open(args.out, "w") as fh:
        json.dump({"binding_run": binding, "budget_rows_proxy": budget,
                   "dataset_tokens": dataset_tokens, "plan": plan}, fh, indent=2)
    print(f"\nWrote {args.out}  ({len(plan)} run(s) planned).")
    print("View dirs %s under: %s"
          % ("WOULD be created" if args.dry_run else "created", args.views_dir))
    print("\nNext: ./05_run_fixed_table_eval.sh   (reads eval_plan.json)")


if __name__ == "__main__":
    main()
