#!/usr/bin/env python3
"""
02_token_stats.py — count tokens and segments per format from the tokenized
Megatron shards. These counts drive the rows-equalized ("fixed table size")
checkpoint matching in 03_fixed_table_checkpoints.py.

WHY THIS IS THE RIGHT INVARIANT
  All five formats serialize the SAME underlying tables, with the SAME cleaning
  and the SAME deterministic row shuffle. The only thing that differs is how many
  tokens each format spends to encode a given row. So:

      tokens_per_row(format) = dataset_tokens(format) / total_rows

  and total_rows is identical across formats. When we later equalize on rows seen,
  total_rows cancels and ONLY the per-format dataset token count matters — which is
  exactly what this script measures. (See 03_… for the algebra.)

WHAT IT READS
  Megatron .idx index files (datatrove MegatronDocumentTokenizer / Megatron-core
  IndexedDataset format). The .idx stores the length of every sequence (segment),
  so total tokens = sum of sequence lengths and #segments = sequence count. If an
  .idx can't be parsed, it falls back to bin_bytes / dtype_itemsize.

Usage (login node, no GPU):
  python3 02_token_stats.py
  python3 02_token_stats.py --tokenized-root /capstor/.../tabular_ablation/tokenized
  python3 02_token_stats.py --out token_stats.json
"""

import os
import glob
import json
import struct
import argparse

import numpy as np

FORMATS = ["csv", "json", "keyvalue", "markdown", "sql_schema"]

# Megatron-core IndexedDataset header
_INDEX_MAGIC = b"MMIDIDX\x00\x00"
_DTYPES = {  # code -> numpy dtype (megatron.core.datasets.indexed_dataset.DType)
    1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
    5: np.int64, 6: np.float64, 7: np.float32, 8: np.uint16,
}


def read_idx(idx_path):
    """Return (n_segments, total_tokens, dtype) from a Megatron .idx file."""
    with open(idx_path, "rb") as f:
        magic = f.read(len(_INDEX_MAGIC))
        if magic != _INDEX_MAGIC:
            raise ValueError(f"bad magic in {idx_path!r}: {magic!r}")
        (version,) = struct.unpack("<Q", f.read(8))
        (dtype_code,) = struct.unpack("<B", f.read(1))
        (seq_count,) = struct.unpack("<Q", f.read(8))
        (doc_count,) = struct.unpack("<Q", f.read(8))
        dtype = _DTYPES[dtype_code]
        # sequence_lengths: int32[seq_count] starts right after the header
        lengths = np.frombuffer(f.read(seq_count * 4), dtype=np.int32)
    return seq_count, int(lengths.sum()), np.dtype(dtype)


def bin_fallback(bin_path, dtype):
    """Token count from raw .bin size when .idx parse fails."""
    return os.path.getsize(bin_path) // np.dtype(dtype).itemsize


def stats_for_format(root, fmt):
    fmt_dir = os.path.join(root, fmt)
    idx_files = sorted(glob.glob(os.path.join(fmt_dir, "*.idx")))
    if not idx_files:
        # some layouts nest one level deeper
        idx_files = sorted(glob.glob(os.path.join(fmt_dir, "**", "*.idx"),
                                     recursive=True))
    total_tokens = 0
    total_segments = 0
    shards = []
    for idx in idx_files:
        try:
            n_seg, n_tok, dtype = read_idx(idx)
        except Exception as e:
            bin_path = idx[:-4] + ".bin"
            n_tok = bin_fallback(bin_path, np.int32) if os.path.exists(bin_path) else 0
            n_seg = 0
            print(f"    (idx parse failed for {os.path.basename(idx)}: {e}; "
                  f"used bin fallback -> {n_tok:,} tokens)")
        total_tokens += n_tok
        total_segments += n_seg
        shards.append({"shard": os.path.basename(idx),
                       "segments": n_seg, "tokens": n_tok})
    return {
        "format": fmt,
        "dir": fmt_dir,
        "n_shards": len(idx_files),
        "total_segments": total_segments,
        "total_tokens": total_tokens,
        "shards": shards,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokenized-root",
        default="/capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation/tokenized",
        help="Dir containing one subfolder per format with .bin/.idx shards.",
    )
    ap.add_argument("--out", default="token_stats.json")
    args = ap.parse_args()

    results = {}
    print("=" * 78)
    print(f"{'format':<12}{'shards':>8}{'segments':>14}{'tokens':>18}{'tok/seg':>10}")
    print("-" * 78)
    for fmt in FORMATS:
        s = stats_for_format(args.tokenized_root, fmt)
        results[fmt] = s
        tps = (s["total_tokens"] / s["total_segments"]) if s["total_segments"] else 0
        print(f"{fmt:<12}{s['n_shards']:>8}{s['total_segments']:>14,}"
              f"{s['total_tokens']:>18,}{tps:>10.1f}")
    print("=" * 78)

    toks = {f: results[f]["total_tokens"] for f in FORMATS if results[f]["total_tokens"]}
    if toks:
        ref = max(toks, key=toks.get)
        print(f"\nMost verbose (most tokens, the binding constraint): {ref}")
        print("Relative dataset size (= matched-checkpoint fraction of training):")
        for f in FORMATS:
            if f in toks:
                print(f"  {f:<12} {toks[f] / toks[ref]:.4f}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
