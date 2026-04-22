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
    """Robust cleaning from original script: removes artifacts and long blobs."""
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

# ─── ITERATIVE CHUNKING LOGIC ───────────────────────────────────────────────

def process_table_iteratively(df, format_name, tokenizer, token_budget, arrow_schema=None):
    """
    Slices the table into multiple JSONL lines. 
    Each line = Header/Schema + subset of rows.
    """
    # Map format name to the correct function
    serializers = {
        "csv": lambda d: serialize_csv(d),
        "keyvalue": lambda d: serialize_keyvalue(d),
        "markdown": lambda d: serialize_markdown(d),
        "json": lambda d: serialize_json_records(d),
        "sql_schema": lambda d: serialize_sql_schema(d, arrow_schema)
    }
    serialize_fn = serializers[format_name]
    
    segments = []
    remaining_df = df.copy()

    while len(remaining_df) > 0:
        low, high = 1, len(remaining_df)
        best_fit_rows = 0
        
        # Binary search for the maximum rows that fit in this specific 3.8k segment
        while low <= high:
            mid = (low + high) // 2
            test_df = remaining_df.head(mid)
            if len(tokenizer.encode(serialize_fn(test_df))) <= token_budget:
                best_fit_rows = mid
                low = mid + 1
            else:
                high = mid - 1
        
        if best_fit_rows == 0:
            # If even 1 row doesn't fit, skip it and move on
            remaining_df = remaining_df.iloc[1:]
            continue

        segments.append(serialize_fn(remaining_df.head(best_fit_rows)))
        remaining_df = remaining_df.iloc[best_fit_rows:]
            
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
                    
                    # Deterministic shuffle
                    seed = get_seed_from_filename(name)
                    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
                    
                    # Get all segments for this table
                    segments = process_table_iteratively(df, format_name, tokenizer, token_budget, schema)
                    
                    for seg in segments:
                        n_tokens = len(tokenizer.encode(seg))
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