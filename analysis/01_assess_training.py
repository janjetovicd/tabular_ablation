#!/usr/bin/env python3
"""
01_assess_training.py — assess the finished tabular-ablation training jobs.

Run on a CSCS login node (no GPU, no container needed).

For every (format, seed) experiment it finds, it reports:
  - the latest checkpoint iteration written
  - whether the 30B-token target (28,610 iters) was reached
  - the full list of available checkpoint iterations on disk
  - whether a permanent copy exists on store

It scans BOTH locations because training writes to scratch first and stages
out to store afterwards:
  scratch: $MEGATRON_LM_DIR/logs/Meg-Runs/tabular-ablation/<exp>/checkpoints
  store:   $STORE_DIR/checkpoints/<exp>

Experiment dir names look like:
  tabular-<format>-1p5b-seed<seed>-<jobid>      (current multi-seed runs)
  tabular-<format>-1p5b-<jobid>                 (older single-seed runs)

Usage:
  python3 01_assess_training.py
  python3 01_assess_training.py --out training_assessment.json
  python3 01_assess_training.py --sacct      # also pull sacct state per jobid
"""

import os
import re
import json
import glob
import argparse
import subprocess
from collections import defaultdict

# ── Defaults (override on the CLI if your paths differ) ───────────────────────
# Primary source is the PERMANENT store (a139). Scratch is purged every 14 days,
# so we don't rely on it; pass --scratch-runs only if you want to also scan it.
STORE_CKPTS  = "/capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation/checkpoints"
SCRATCH_RUNS = ""   # empty = skip scratch

FORMATS      = ["csv", "json", "keyvalue", "markdown", "sql_schema"]
GBS          = 256
SEQ_LEN      = 4096
TARGET_TOKENS = 30_000_000_000
TOKENS_PER_ITER = GBS * SEQ_LEN                      # 1,048,576
TARGET_ITERS = TARGET_TOKENS // TOKENS_PER_ITER      # 28,610

EXP_RE = re.compile(
    r"^tabular-(?P<format>csv|json|keyvalue|markdown|sql_schema)-1p5b"
    r"(?:-seed(?P<seed>\d+))?-(?P<jobid>\d+)$"
)
ITER_RE = re.compile(r"^iter_(\d+)$")


def list_iters(ckpt_dir):
    """Return sorted list of checkpoint iterations present in a checkpoints dir."""
    iters = []
    if not os.path.isdir(ckpt_dir):
        return iters
    for name in os.listdir(ckpt_dir):
        m = ITER_RE.match(name)
        if m and os.path.isdir(os.path.join(ckpt_dir, name)):
            iters.append(int(m.group(1)))
    return sorted(iters)


def latest_marker(ckpt_dir):
    """Read latest_checkpointed_iteration.txt if present."""
    p = os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt")
    if os.path.isfile(p):
        try:
            return int(open(p).read().strip())
        except ValueError:
            return None
    return None


def find_ckpt_dir(exp_dir):
    """Locate the dir that actually contains iter_* folders.

    Handles the double-nesting created by `rsync <exp_dir> <dest>` during
    stage-out, which yields  <dest>/<exp>/<exp>/checkpoints/iter_*  as well as
    the simpler <exp>/checkpoints/iter_*  layout. Falls back to a full walk.
    """
    direct = [os.path.join(exp_dir, "checkpoints"), exp_dir]
    for cand in direct:
        if os.path.isdir(cand) and any(
            ITER_RE.match(n) for n in os.listdir(cand)
        ):
            return cand
    for dirpath, dirnames, _ in os.walk(exp_dir):
        if any(ITER_RE.match(d) for d in dirnames):
            return dirpath
    return os.path.join(exp_dir, "checkpoints")


def scan_location(root, is_store):
    """Yield (exp_name, info_dict) for every experiment dir under `root`."""
    if not os.path.isdir(root):
        return
    for exp_name in sorted(os.listdir(root)):
        m = EXP_RE.match(exp_name)
        if not m:
            continue
        exp_dir = os.path.join(root, exp_name)
        ckpt_dir = find_ckpt_dir(exp_dir)
        iters = list_iters(ckpt_dir)
        yield exp_name, {
            "format":  m.group("format"),
            "seed":    int(m.group("seed")) if m.group("seed") else None,
            "jobid":   m.group("jobid"),
            "ckpt_dir": ckpt_dir,
            "iters":   iters,
            "latest_iter": iters[-1] if iters else None,
            "latest_marker": latest_marker(ckpt_dir),
            "location": "store" if is_store else "scratch",
        }


def sacct_state(jobid):
    try:
        out = subprocess.run(
            ["sacct", "-j", jobid, "-n", "-X",
             "--format=State,Elapsed,ExitCode"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return out.splitlines()[0].split() if out else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-runs", default=SCRATCH_RUNS)
    ap.add_argument("--store-ckpts",  default=STORE_CKPTS)
    ap.add_argument("--out", default=None, help="Write the assessment as JSON.")
    ap.add_argument("--sacct", action="store_true",
                    help="Also query sacct for each jobid (login node only).")
    args = ap.parse_args()

    # Collect everything, keyed by exp_name; store entries supplement scratch.
    found = {}
    for exp_name, info in scan_location(args.scratch_runs, is_store=False):
        found[exp_name] = info
    for exp_name, info in scan_location(args.store_ckpts, is_store=True):
        if exp_name in found:
            found[exp_name]["store_ckpt_dir"] = info["ckpt_dir"]
            found[exp_name]["store_iters"] = info["iters"]
        else:
            found[exp_name] = info

    if not found:
        print("No experiment directories found. Check --scratch-runs / --store-ckpts.")
        print(f"  scratch: {args.scratch_runs}")
        print(f"  store:   {args.store_ckpts}")
        raise SystemExit(1)

    if args.sacct:
        for info in found.values():
            info["sacct"] = sacct_state(info["jobid"])

    # ── Print report ──────────────────────────────────────────────────────────
    print("=" * 100)
    print(f"TARGET: {TARGET_ITERS:,} iters  ({TARGET_TOKENS/1e9:.0f}B tokens, "
          f"{TOKENS_PER_ITER:,} tokens/iter)")
    print("=" * 100)
    hdr = f"{'experiment':<46}{'fmt':<11}{'seed':>5}{'latest':>9}{'reached?':>10}{'#ckpts':>8}  loc"
    print(hdr)
    print("-" * 100)

    by_format = defaultdict(list)
    for exp_name in sorted(found):
        info = found[exp_name]
        latest = info["latest_iter"]
        reached = (latest is not None and latest >= TARGET_ITERS)
        flag = "YES" if reached else (f"no({latest})" if latest else "NONE")
        loc = info["location"]
        if "store_ckpt_dir" in info:
            loc = "scratch+store"
        print(f"{exp_name:<46}{info['format']:<11}"
              f"{str(info['seed']):>5}{str(latest):>9}{flag:>10}"
              f"{len(info['iters']):>8}  {loc}")
        by_format[info["format"]].append((exp_name, info, reached))

    print("-" * 100)
    print("\nPer-format completeness:")
    for fmt in FORMATS:
        runs = by_format.get(fmt, [])
        n_done = sum(1 for _, _, r in runs if r)
        seeds = sorted({i["seed"] for _, i, _ in runs if i["seed"] is not None})
        print(f"  {fmt:<12} {len(runs)} run(s), {n_done} reached target"
              f"{'  seeds=' + str(seeds) if seeds else ''}")

    # Missing-format warning
    missing = [f for f in FORMATS if f not in by_format]
    if missing:
        print(f"\n  !! No runs found for: {', '.join(missing)}")

    if args.sacct:
        print("\nsacct state:")
        for exp_name in sorted(found):
            print(f"  {exp_name:<46} {found[exp_name].get('sacct')}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(found, fh, indent=2)
        print(f"\nWrote {args.out}")

    print("\nNext: run 02_token_stats.py to get per-format token counts, then "
          "03_fixed_table_checkpoints.py for the rows-equalized comparison.")


if __name__ == "__main__":
    main()
