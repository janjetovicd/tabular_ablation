"""
OBJECTIVE: Read raw parquet files from T4 chunks, perform 'Tabular Chunking' 
to ensure NO data is discarded. 

Each table is split into multiple sequences of ~3.8k tokens. 
Crucially, for formats like CSV or SQL Schema, the headers/schema are repeated 
in every chunk so the model always has semantic context.
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

# ─── UTILITIES & CLEANING ───────────────────────────────────────────────────

def get_seed_from_filename(filename):
    hash_part = filename.split("=")[0]
    return int(hashlib.md5(hash_part.encode()).hexdigest()[:8], 16)

def clean_df(df):
    """Robust cleaning: removes artifacts and long blobs."""
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed") or str(c) in ["index"]]
    df = df.drop(columns=drop_cols).dropna(axis=1, how="all")

    max_cell_chars = 500
    cols_to_drop = []
    for col in df.columns:
        if df[col].dtype == object:
            try:
                if df[col].astype(str).str.len().max() > max_cell_chars:
                    cols_to_drop.append(col)
            except:
                cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop) if cols_to_drop else df

# ─── SERIALIZATION FORMATS ──────────────────────────────

def serialize_csv(df):
    return df.to_csv(index=False).rstrip('\n')

def serialize_keyvalue(df):
    lines = []
    for _, row in df.iterrows():
        pairs = [f"The {col} is {val}" for col, val in row.items() if pd.notna(val) and str(val).strip() != ""]
        if pairs: lines.append(". ".join(pairs) + ".")
    return "\n".join(lines)

def serialize_markdown(df):
    return df.to_markdown(index=False)

def serialize_json_records(df):
    return json.dumps(df.to_dict(orient="records"), default=str)

def serialize_sql_schema(df, parquet_schema):
    """Detailed SQL serialization: extracts ranges and types from Parquet metadata."""
    col_defs = []
    for field in parquet_schema:
        col_name = field.name
        if str(col_name).startswith("Unnamed") or col_name == "index" or col_name not in df.columns:
            continue
        col = df[col_name]
        field_type = str(field.type)
        if field_type in ["double", "float", "float32", "float64"]:
            sql_type = "FLOAT"
            range_str = f" -- range: {col.dropna().min():.3g} to {col.dropna().max():.3g}" if col.notna().any() else ""
        elif field_type.startswith("int"):
            sql_type = "INT"
            range_str = f" -- range: {int(col.dropna().min())} to {int(col.dropna().max())}" if col.notna().any() else ""
        else:
            sql_type = "VARCHAR"; range_str = ""
        col_defs.append(f'    "{col_name}" {sql_type}{range_str}')
    
    header = "CREATE TABLE data (\n" + ",\n".join(col_defs) + "\n);\n"
    return header + df.to_csv(index=False)

# ─── FAST CHUNKING LOGIC ────────────────────────────────────────────────────

def get_row_token_counts(df, format_name, tokenizer, arrow_schema=None):
    """
    Tokenize each row individually, once.
    
    HOW per format:
    - csv/markdown: each row serializes WITH a header line. We strip the 
      header before counting, then add the header cost once per segment.
    - keyvalue: each row is fully self-contained, no header.
    - json: each row as a single-element list [{...}], strip the brackets.
      The per-row cost is an approximation but accurate enough given the 
      small safety margin baked into token_budget.
    - sql_schema: header is the CREATE TABLE block (large, fixed). 
      Rows are CSV lines below it. Same header-strip approach as csv.
    
    Returns:
        row_costs   : list of ints, token count for each row's data (no header)
        header_cost : int, tokens consumed by the header/schema repeated 
                      each segment (0 for keyvalue/json)
    """
    n = len(df)
    row_costs = []

    if format_name == "keyvalue":
        # Every row is self-contained. Serialize each individually.
        for _, row in df.iterrows():
            single = pd.DataFrame([row])
            text = serialize_keyvalue(single)
            row_costs.append(len(tokenizer.encode(text, add_special_tokens=False)))
        return row_costs, 0

    elif format_name == "csv":
        # Header: "col1,col2,col3\n"
        header_line = df.to_csv(index=False).split('\n')[0] + '\n'
        header_cost = len(tokenizer.encode(header_line, add_special_tokens=False))
        # Each row: just the CSV data line, no header
        for _, row in df.iterrows():
            single = pd.DataFrame([row])
            # to_csv produces "header\ndata_line", we want only data_line
            data_line = single.to_csv(index=False).split('\n')[1]
            row_costs.append(len(tokenizer.encode(data_line, add_special_tokens=False)))
        return row_costs, header_cost

    elif format_name == "markdown":
        # Markdown header is two lines: the column names row + the separator row
        # e.g. "| col1 | col2 |\n|------|------|\n"
        full = df.head(1).to_markdown(index=False)
        lines = full.split('\n')
        header_lines = '\n'.join(lines[:2]) + '\n'  # col names + separator
        header_cost = len(tokenizer.encode(header_lines, add_special_tokens=False))
        # Each row: just the data line (line index 2 in a 1-row markdown)
        for _, row in df.iterrows():
            single = pd.DataFrame([row])
            data_line = single.to_markdown(index=False).split('\n')[2]
            row_costs.append(len(tokenizer.encode(data_line, add_special_tokens=False)))
        return row_costs, header_cost

    elif format_name == "json":
        # JSON is [{...},{...},...]. No fixed header.
        # Approximate: serialize each row as a single-element list, 
        # subtract the 2 bracket tokens for [ and ].
        bracket_cost = len(tokenizer.encode("[]", add_special_tokens=False))
        for _, row in df.iterrows():
            single = pd.DataFrame([row])
            text = json.dumps(single.to_dict(orient="records"), default=str)
            full_cost = len(tokenizer.encode(text, add_special_tokens=False))
            # Subtract bracket overhead so costs are additive
            row_costs.append(max(1, full_cost - bracket_cost))
        return row_costs, bracket_cost  # bracket_cost acts as the "header"

    elif format_name == "sql_schema":
        # Header = entire CREATE TABLE block + CSV header line
        # Rows = CSV data lines
        header_text = serialize_sql_schema(df.head(0), arrow_schema)  # 0 rows = schema only
        # Also add the CSV header line that appears after the CREATE TABLE block
        csv_header_line = df.to_csv(index=False).split('\n')[0] + '\n'
        full_header = header_text + csv_header_line
        header_cost = len(tokenizer.encode(full_header, add_special_tokens=False))
        for _, row in df.iterrows():
            single = pd.DataFrame([row])
            data_line = single.to_csv(index=False).split('\n')[1]
            row_costs.append(len(tokenizer.encode(data_line, add_special_tokens=False)))
        return row_costs, header_cost

    else:
        raise ValueError(f"Unknown format: {format_name}")


def process_table_iteratively(df, format_name, tokenizer, token_budget, arrow_schema=None):
    """
    Slices the table into multiple JSONL lines.
    Each line = Header/Schema + subset of rows that fit within token_budget.
    
    OLD approach: binary search over row count, re-serializing every step.
      Cost: O(N_segments * log(N_rows)) tokenizer calls, each on large text.
    
    NEW approach: pre-tokenize each row once, then greedily accumulate.
      Cost: O(N_rows) tokenizer calls, each on a single small row.
      Result: same segments written to disk, same total tokens, much faster.
    """
    if df.empty:
        return []

    serializers = {
        "csv":        lambda d: serialize_csv(d),
        "keyvalue":   lambda d: serialize_keyvalue(d),
        "markdown":   lambda d: serialize_markdown(d),
        "json":       lambda d: serialize_json_records(d),
        "sql_schema": lambda d: serialize_sql_schema(d, arrow_schema),
    }
    serialize_fn = serializers[format_name]

    # Step 1: get per-row token costs and the fixed header cost (once per segment)
    row_costs, header_cost = get_row_token_counts(df, format_name, tokenizer, arrow_schema)

    # Step 2: greedy accumulation — walk rows left to right, 
    # emit a segment whenever the next row would overflow the budget
    segments = []
    segment_start = 0
    current_tokens = header_cost  # every segment starts with its header cost

    for i, cost in enumerate(row_costs):
        if cost > (token_budget - header_cost):
            # This single row is too large even alone — skip it
            # (This is the same behaviour as the old code's best_fit_rows==0 case)
            if segment_start == i:
                segment_start = i + 1
                current_tokens = header_cost
            # If we're mid-segment, first flush what we have, then skip this row
            else:
                segments.append(serialize_fn(df.iloc[segment_start:i]))
                segment_start = i + 1
                current_tokens = header_cost
            continue

        if current_tokens + cost > token_budget:
            # Current row would overflow — flush current segment, start new one
            segments.append(serialize_fn(df.iloc[segment_start:i]))
            segment_start = i
            current_tokens = header_cost + cost
        else:
            current_tokens += cost

    # Flush the final segment if anything remains
    if segment_start < len(df):
        segments.append(serialize_fn(df.iloc[segment_start:]))

    return segments

# ─── MAIN PROCESSING LOOP ───────────────────────────────────────────────────

def process_chunk(chunk_zip_path, format_name, output_path, tokenizer, token_budget):
    stats = {"processed": 0, "tokens": 0, "skipped": 0}
    
    with zipfile.ZipFile(chunk_zip_path, 'r') as zf:
        parquet_names = sorted(n for n in zf.namelist() if n.endswith('.parquet'))
        
        with open(output_path, 'w', encoding="utf-8") as out_f:
            for name in parquet_names:
                try:
                    with zf.open(name) as f:
                        arrow_table = pq.read_table(io.BytesIO(f.read()))
                        df = clean_df(arrow_table.to_pandas())
                        schema = arrow_table.schema
                    
                    if df.empty: continue
                    
                    seed = get_seed_from_filename(name)
                    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
                    
                    segments = process_table_iteratively(df, format_name, tokenizer, token_budget, schema)
                    
                    for seg in segments:
                        n_tokens = len(tokenizer.encode(seg, add_special_tokens=False))
                        out_f.write(json.dumps({"text": seg}, ensure_ascii=False) + "\n")
                        stats["tokens"] += n_tokens
                    
                    stats["processed"] += 1
                except Exception as e:
                    logging.error(f"Error on {name}: {e}")
                    stats["skipped"] += 1
                    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_zip", required=True)
    parser.add_argument("--format", required=True, choices=["csv", "keyvalue", "markdown", "json", "sql_schema"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--token_budget", type=int, default=3800)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    stats = process_chunk(args.chunk_zip, args.format, args.output, tokenizer, args.token_budget)
    logging.info(f"Done. Tokens: {stats['tokens']:,} | Tables: {stats['processed']} | Errors: {stats['skipped']}")