"""
Reads T4 chunks 24–25, samples 300 tables, prepares prompts in all 5 formats
simultaneously, and saves ground truth + cleaned dataframes.

Output (written to /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/eval/):
  prompts_csv.jsonl
  prompts_json.jsonl
  prompts_keyvalue.jsonl
  prompts_markdown.jsonl
  prompts_sql_schema.jsonl
  ground_truth.jsonl        — {table_id, ground_truth_dict, col_dtypes}
  dataframes/{table_id}.pkl — cleaned df for cross-format robustness test

Usage (login node, CPU only, ~10 min):
  python prepare_eval_data.py
"""

import os
import io
import json
import random
import logging
import pickle
import zipfile

import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# Import serializers from the training script
import sys
sys.path.insert(0, "/iopsstor/scratch/cscs/djanjetovic/tabular_ablation")
from serialize_t4 import (
    clean_df,
    serialize_csv,
    serialize_keyvalue,
    serialize_markdown,
    serialize_json_records,
    serialize_sql_schema,
    get_row_token_counts,
)

# Config

T4_BASE       = "/capstor/store/cscs/swissai/a139/datasets/mlfoundations_t4_full"
TOKENIZER_PATH = "/iopsstor/scratch/cscs/djanjetovic/tokenizer_cache/apertus"
EVAL_DIR      = "/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/eval"
CHUNK_IDS     = [24, 25]
N_TABLES      = 300
RANDOM_SEED   = 42
TOKEN_BUDGET  = 3800
FORMATS       = ["csv", "json", "keyvalue", "markdown", "sql_schema"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_prompt(df, format_name, arrow_schema, tokenizer):
    """
    Build a prompt for next-row prediction.

    Returns (prompt_str, last_row_dict) or raises ValueError with reason.
    The prompt contains rows 0..k-1 (context) serialized in format_name.
    The last row (row k) is held out as ground truth.
    """
    serializers = {
        "csv":        lambda d: serialize_csv(d),
        "keyvalue":   lambda d: serialize_keyvalue(d),
        "markdown":   lambda d: serialize_markdown(d),
        "json":       lambda d: serialize_json_records(d),
        "sql_schema": lambda d: serialize_sql_schema(d, arrow_schema),
    }
    serialize_fn = serializers[format_name]

    # ground truth: last row
    last_row  = df.iloc[[-1]]
    context_df = df.iloc[:-1]

    # how many tokens does the last row cost (in this format)
    try:
        last_row_costs, _ = get_row_token_counts(last_row, format_name, tokenizer, arrow_schema)
        last_row_tokens = last_row_costs[0]
    except Exception as e:
        raise ValueError(f"last-row token count failed: {e}")

    actual_budget = TOKEN_BUDGET - last_row_tokens

    # per-row costs for context rows
    try:
        row_costs, header_cost = get_row_token_counts(context_df, format_name, tokenizer, arrow_schema)
    except Exception as e:
        raise ValueError(f"context token count failed: {e}")

    if header_cost >= actual_budget:
        raise ValueError("header alone exceeds actual_budget")

    # greedy accumulation of context rows
    accumulated = header_cost
    k = 0
    for cost in row_costs:
        if accumulated + cost > actual_budget:
            break
        accumulated += cost
        k += 1

    if k == 0:
        raise ValueError("header + first row exceeds actual_budget")

    prompt_df = context_df.iloc[:k]
    try:
        prompt_text = serialize_fn(prompt_df)
    except Exception as e:
        raise ValueError(f"serialization failed: {e}")

    return prompt_text, last_row


def iter_tables(chunk_ids, t4_base):
    """Yield (table_id, df, arrow_schema) for every parquet in the given chunks."""
    for cid in chunk_ids:
        zip_path = os.path.join(t4_base, f"chunk-{cid:04d}.zip")
        log.info(f"Opening {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                parquet_names = sorted(n for n in zf.namelist() if n.endswith(".parquet"))
                for name in parquet_names:
                    table_id = f"chunk{cid:04d}__{name.replace('/', '__')}"
                    try:
                        with zf.open(name) as f:
                            arrow_table = pq.read_table(io.BytesIO(f.read()))
                        df = clean_df(arrow_table.to_pandas())
                        schema = arrow_table.schema
                        yield table_id, df, schema
                    except Exception as e:
                        log.warning(f"Skipping {name} in chunk {cid}: {e}")
        except Exception as e:
            log.error(f"Cannot open {zip_path}: {e}")


def is_valid_table(df, tokenizer, formats, arrow_schema):
    """Quick pre-filter: skip empty, single-row, or header-too-large tables."""
    if df.empty or len(df) < 2:
        return False
    # check header cost for every format
    for fmt in formats:
        try:
            _, header_cost = get_row_token_counts(df.head(1), fmt, tokenizer, arrow_schema)
            if header_cost >= TOKEN_BUDGET:
                return False
        except Exception:
            return False
    return True


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    df_dir = os.path.join(EVAL_DIR, "dataframes")
    os.makedirs(df_dir, exist_ok=True)

    log.info(f"Loading tokenizer from {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Phase 1: collect all candidate tables
    log.info("Phase 1: scanning chunks 24–25 for candidate tables …")
    candidates = []
    for table_id, df, schema in iter_tables(CHUNK_IDS, T4_BASE):
        if is_valid_table(df, tokenizer, FORMATS, schema):
            candidates.append((table_id, df, schema))

    log.info(f"Found {len(candidates)} valid candidate tables")

    if len(candidates) < N_TABLES:
        log.warning(
            f"Only {len(candidates)} candidates — using all of them (< {N_TABLES} requested)"
        )
        selected = candidates
    else:
        rng = random.Random(RANDOM_SEED)
        selected = rng.sample(candidates, N_TABLES)

    log.info(f"Sampled {len(selected)} tables (seed={RANDOM_SEED})")

    # Phase 2: build prompts & ground truth
    log.info("Phase 2: building prompts in all 5 formats …")

    prompt_files = {fmt: open(os.path.join(EVAL_DIR, f"prompts_{fmt}.jsonl"), "w") for fmt in FORMATS}
    gt_file      = open(os.path.join(EVAL_DIR, "ground_truth.jsonl"), "w")

    written = 0
    skipped = 0

    for table_id, df, schema in selected:
        results = {}
        ok = True
        for fmt in FORMATS:
            try:
                prompt_text, last_row_df = build_prompt(df, fmt, schema, tokenizer)
                results[fmt] = prompt_text
            except ValueError as e:
                log.warning(f"Skipping {table_id} [{fmt}]: {e}")
                ok = False
                break

        if not ok:
            skipped += 1
            continue

        # ground truth: last row as dict (format-agnostic)
        last_row_df = df.iloc[[-1]]
        gt_dict = last_row_df.iloc[0].to_dict()
        # convert numpy scalars to python natives for JSON serialisation
        gt_dict = {k: (v.item() if hasattr(v, "item") else v) for k, v in gt_dict.items()}

        # col dtypes for type-aware scoring
        col_dtypes = {col: str(df[col].dtype) for col in df.columns}

        # write prompts
        for fmt in FORMATS:
            prompt_files[fmt].write(
                json.dumps({"table_id": table_id, "prompt": results[fmt]}, ensure_ascii=False) + "\n"
            )

        # write ground truth
        gt_file.write(
            json.dumps({"table_id": table_id, "ground_truth_dict": gt_dict, "col_dtypes": col_dtypes},
                       ensure_ascii=False, default=str) + "\n"
        )

        # save cleaned df as pickle (for cross-format test)
        with open(os.path.join(df_dir, f"{table_id}.pkl"), "wb") as pf:
            pickle.dump(df, pf)

        written += 1
        if written % 50 == 0:
            log.info(f"  {written} tables written so far …")

    for f in prompt_files.values():
        f.close()
    gt_file.close()

    log.info(f"Done. Written: {written} | Skipped: {skipped}")
    log.info(f"Output dir: {EVAL_DIR}")


if __name__ == "__main__":
    main()
