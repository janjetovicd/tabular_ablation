
"""
OBJECTIVE: Read raw parquet files from one chunk, filter if necessary and sample as many rows as they fit within the token budget (3.8k tokens)
and serializes them in the chosen format, and writes one JSON line per table to an output .jsonl file

"""

import os
import json
import random
import hashlib
import argparse
import logging
import pyarrow.parquet as pq
import pandas as pd
from transformers import AutoTokenizer
import io
import zipfile

# Helper functions

def get_seed_from_filename(filename):
    """
    Derive a deterministic random seed from the parquet filename to shuffle rows before sampling and ensure model sees diverse row input.
    It is deterministic to ensure it is reproducible when running different ablation trials (same rows, different format) for consistency. 

    """

    hash_part = filename.split("=")[0]
    return int(hashlib.md5(hash_part.encode()).hexdigest()[:8], 16)


def clean_df(df):
    """
    Remove columns that carry no semantic signal for pretraining, since it is required for our 
    training objective (next-token prediction on multiple rows) differs from T4's original objective 
    (single-row classification). Specifically:

    1. Remove artifact columns ('Unnamed: X', 'index') since pandas adds these when
       reading CSVs without explicit headers; they hold no meaning.

    2. Remove long-value columns where any cell exceeds 500 characters
       (free-text, HTML, base64 blobs, long URLs). With multiple rows per
       sample, one such column can push a single row over the 3800 token
       budget, making the entire table unusable.
    """

    # Remove pandas artifact columns
    drop_cols = [c for c in df.columns
                 if str(c).startswith("Unnamed") or str(c) in ["index"]]
    df = df.drop(columns=drop_cols)

    # Remove columns that are entirely null — no signal at all
    df = df.dropna(axis=1, how="all")

    # Remove columns with excessively long cell values
    max_cell_chars = 500
    cols_to_drop = []
    for col in df.columns:
            if df[col].dtype == object:
                try:
                    if df[col].astype(str).str.len().max() > max_cell_chars:
                        cols_to_drop.append(col)
                except (UnicodeDecodeError, Exception):
                    cols_to_drop.append(col)  # drop columns that can't be decoded
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def sample_rows_to_budget(df, serialize_fn, tokenizer, token_budget=3800, seed=42):
    """
    Find the maximum number of rows that fit within token_budget using
    binary search, then return that subset of the dataframe.

    Returns None if even a single row exceeds the budget, signals to the
    caller that this table should be skipped entirely.
    """

    # Shuffle rows with deterministic seed for reproducibility
    rng = random.Random(seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    df_shuffled = df.iloc[indices].reset_index(drop=True)

    low, high = 1, min(len(df_shuffled), 200)
    while low < high:
        mid = (low + high + 1) // 2
        # +1 prevents infinite loop when low and high are adjacent
        tokens = len(tokenizer.encode(serialize_fn(df_shuffled.head(mid))))
        if tokens <= token_budget:
            low = mid       # mid rows fit, try more
        else:
            high = mid - 1  # mid rows exceed budget, try fewer

    # After loop: low == high == maximum rows that fit
    # Final check: if even 1 row exceeds the budget, skip this table
    final_tokens = len(tokenizer.encode(serialize_fn(df_shuffled.head(low))))
    if final_tokens > token_budget:
        return None

    return df_shuffled.head(low)


# Serialization functions: each function takes a cleaned, row-sampled dataframe and returns 
# a string that the model trains on

def serialize_keyvalue(df):
    """
    Format A — key:value per row.
    Example output:
        The city is Zurich. The population is 0.42. The area is 88.
        The city is Geneva. The population is 0.20. The area is 16.

    Pros: explicit column-value association in every row, strong signal.
    Cons: most verbose — column names repeat for every cell, uses ~2x more
          tokens than CSV, fitting fewer rows per sample.
    This is the baseline from TabLLM, UniPredict, and T4's own serialization.
    """

    lines = []
    for _, row in df.iterrows():
        pairs = [f"The {col} is {val}"
                 for col, val in row.items()
                 if pd.notna(val) and str(val).strip() != ""]
        if pairs:
            lines.append(". ".join(pairs) + ".")
    return "\n".join(lines)


def serialize_csv(df):
    """
    Format B — header row + CSV data rows.
    Example output:
        city,population,area
        Zurich,0.42,88
        Geneva,0.20,16

    Pros: most token-efficient — column names appear once in the header,
          fits ~2x more rows than key:value per sample.
    Cons: model must learn positional associations (3rd value = 3rd column).
    Well-motivated: CSV is extremely common in the base model's training data.
    """

    return df.to_csv(index=False).rstrip('\n')


def serialize_markdown(df):
    """
    Format C — markdown table.
    Example output:
        | city | population | area |
        |------|------------|------|
        | Zurich | 0.42   | 88   |

    Pros: matches format the base model saw in GitHub READMEs and Wikipedia.
    Cons: separator row and alignment padding waste tokens — less efficient
          than CSV but more efficient than key:value.
    """

    return df.to_markdown(index=False)


def serialize_json_records(df):
    """
    Format D — JSON array of records.
    Example output:
        [{"city": "Zurich", "population": 0.42, "area": 88},
         {"city": "Geneva", "population": 0.20, "area": 16}]

    Pros: extremely common on the web and GitHub, base model has seen lots.
    Cons: JSON syntax overhead (braces, quotes, commas) makes it similar in
          verbosity to key:value — fits similar rows per sample.
    """

    return json.dumps(df.to_dict(orient="records"), default=str)


def serialize_sql_schema(df, parquet_schema):
    """
    Format F — SQL schema-aware serialization. Novel application for pretraining.

    Example output:
        CREATE TABLE data (
            "city" VARCHAR,
            "population" FLOAT  -- range: 0.13 to 0.42,
            "area" INT  -- range: 16 to 88
        );
        city,population,area
        Zurich,0.42,88
        Geneva,0.20,16

    Objective is to mirror how a data scientist approaches a new dataset (schema first, then data).
    The type and range annotation replace the semantically weak column names. 
    
    CREATE TABLE is used in NL-to-SQL finetuning literature but has
    never been used as a pretraining serialization format for general tabular
    understanding. This is the unexplored direction being tested.
    """

    ## Read column type metadata directly from Parquet file
    # Parquet stores column statistics as metadata — no extra compute needed
    col_defs = []
    for field in parquet_schema:
        col_name = field.name
        if str(col_name).startswith("Unnamed") or col_name == "index":
            continue
        if col_name not in df.columns:
            continue
        col = df[col_name]
        field_type = str(field.type)
        if field_type in ["double", "float", "float32", "float64"]:
            sql_type = "FLOAT"
            range_str = (f" -- range: {col.dropna().min():.3g} to {col.dropna().max():.3g}"
                        if col.notna().any() else "")
        elif field_type.startswith("int"):
            sql_type = "INT"
            range_str = (f" -- range: {int(col.dropna().min())} to {int(col.dropna().max())}"
                         if col.notna().any() else "")
        else:
            sql_type = "VARCHAR"
            range_str = ""
        col_defs.append(f'    "{col_name}" {sql_type}{range_str}')
    header = "CREATE TABLE data (\n" + ",\n".join(col_defs) + "\n);\n"
    return header + df.to_csv(index=False)

def build_serializer(format_name, arrow_schema=None):
    """
    Return a serializer function that takes only a dataframe as argument.
    For sql_schema, arrow_schema must be passed (read once per file, not per binary search call).
    """
    if format_name == "csv":
        return serialize_csv
    elif format_name == "keyvalue":
        return serialize_keyvalue
    elif format_name == "markdown":
        return serialize_markdown
    elif format_name == "json":
        return serialize_json_records
    elif format_name == "sql_schema":
        return lambda df: serialize_sql_schema(df, arrow_schema)
    else:
        raise ValueError(f"Unknown format: {format_name}. "
                         f"Choose from: csv, keyvalue, markdown, json, sql_schema")


# Core processing function

def process_chunk(chunk_zip_path, format_name, output_path, tokenizer,
                  token_target, token_budget=3800):
    """
    Process all parquet files inside one T4 chunk zip file.

    T4 chunks are stored as .zip files, each containing many .parquet files.
    We read each parquet directly from the zip into memory — no extraction needed.

    For each parquet file:
        1. Load and clean the table
        2. Sample max rows that fit in token_budget
        3. Serialize sampled rows in chosen format
        4. Write {"text": "..."} as one line in output .jsonl

    Stops once token_target tokens have been written.
    """

    stats = {
        "tables_processed": 0,
        "tables_skipped_empty": 0,
        "tables_skipped_too_long": 0,
        "tokens_written": 0,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with zipfile.ZipFile(chunk_zip_path, 'r') as zf:
        parquet_names = sorted(n for n in zf.namelist() if n.endswith('.parquet'))

        with open(output_path, "w", encoding="utf-8") as out_f:
            for name in parquet_names:

                if stats["tokens_written"] >= token_target:
                    break

                # Load parquet directly from zip into memory — no disk extraction
                try:
                    with zf.open(name) as f:
                        arrow_table = pq.read_table(io.BytesIO(f.read()))
                        df_raw = arrow_table.to_pandas()
                        arrow_schema = arrow_table.schema
                except Exception as e:
                    logging.warning(f"Failed to read {name}: {e}")
                    stats["tables_skipped_empty"] += 1
                    continue

                df = clean_df(df_raw)
                if df.empty or len(df.columns) == 0:
                    stats["tables_skipped_empty"] += 1
                    continue

                # Build serializer — schema passed here once, not re-read per binary search call
                serialize_fn = build_serializer(format_name, arrow_schema=arrow_schema)

                seed = get_seed_from_filename(name)
                df_sampled = sample_rows_to_budget(
                    df, serialize_fn, tokenizer,
                    token_budget=token_budget, seed=seed
                )
                if df_sampled is None:
                    stats["tables_skipped_too_long"] += 1
                    continue

                text = serialize_fn(df_sampled)
                n_tokens = len(tokenizer.encode(text))
                out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                stats["tables_processed"] += 1
                stats["tokens_written"] += n_tokens

    return stats


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Serialize T4 parquet tables to .jsonl for Megatron pretraining."
    )
    parser.add_argument("--chunk_zip", type=str, required=True,
                        help="Path to T4 chunk .zip file (e.g. chunk-0000.zip).")
    parser.add_argument("--format", type=str, required=True,
                        choices=["csv", "keyvalue", "markdown", "json", "sql_schema"],
                        help="Serialization format to use for this run.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write the output .jsonl file.")
    parser.add_argument("--token_target", type=int, default=300_000_000,
                        help="Stop after this many tokens. 10 chunks x 300M = 3B per format.")
    parser.add_argument("--token_budget", type=int, default=3800,
                        help="Max tokens per sample (4096 window - 300 overhead).")
    parser.add_argument("--tokenizer", type=str, default="swiss-ai/Apertus-70B-2509",
                        help="HuggingFace tokenizer name or local CSCS path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    logging.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logging.info(f"Processing chunk: {args.chunk_zip}")
    logging.info(f"Format:           {args.format}")
    logging.info(f"Output:           {args.output}")
    logging.info(f"Token target:     {args.token_target:,}")

    stats = process_chunk(
        chunk_zip_path=args.chunk_zip,
        format_name=args.format,
        output_path=args.output,
        tokenizer=tokenizer,
        token_target=args.token_target,
        token_budget=args.token_budget,
    )

    logging.info("─── Done ───────────────────────────────────────────────")
    logging.info(f"  Tables processed:          {stats['tables_processed']:>8,}")
    logging.info(f"  Skipped (empty):           {stats['tables_skipped_empty']:>8,}")
    logging.info(f"  Skipped (row too long):    {stats['tables_skipped_too_long']:>8,}")
    logging.info(f"  Tokens written:            {stats['tokens_written']:>8,}")


if __name__ == "__main__":
    main()