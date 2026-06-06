# Generates BLiMP-style evaluation data from held-out T4 tables (chunks 24-25).
#   - Instead of a question+answer pair, we generate a STATEMENT + DISTRACTOR pair.
#   - At inference time, we compute log P(statement | table_context) and
#     log P(distractor | table_context) and check whether the model assigns
#     higher probability to the correct one.
#
# Output files (written to EVAL_DIR):
#   pairs_{fmt}.jsonl        — one record per (table, pair) with the table serialized
#                              in `fmt`, plus the correct statement and distractor.
#   blimp_pairs.jsonl        — format-agnostic ground truth (statement + distractor only).
#   dataframes/<table_id>.pkl — pickled cleaned DataFrames for later inspection.
#   completed_table_ids.txt  — checkpoint file; one table_id per line.
#   candidates_cache.pkl     — cached Phase 1 results so re-runs skip the 13-min scan.
#
# Resumable: re-running the script skips tables already listed in
# completed_table_ids.txt and appends to existing output files.
# Phase 1 is also cached to disk so re-runs skip the chunk scanning entirely.
#
# Flags:
#   --smoke-test   Run end-to-end with 2 synthetic tables, a mock tokenizer, and a
#                  mock LLaMA pipe. No GPU, no cluster paths needed. Use this to
#                  verify file I/O, serialization, checkpoint logic, and JSON parsing.
#   --skip-model   Load the real tokenizer and cluster data, but mock the LLaMA
#                  generation. Useful for testing serialization on real tables without
#                  burning GPU time.
#   --eval-dir DIR Override EVAL_DIR (default from constant below).

import os, io, json, random, logging, pickle, zipfile, re, signal, sys, time, argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ── Paths & constants ──────────────────────────────────────────────────────────

T4_BASE    = "/capstor/store/cscs/swissai/a139/datasets/mlfoundations_t4_full"
EVAL_DIR   = "/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/eval"
LLAMA_PATH = (
    "/capstor/store/cscs/swissai/infra01/apertus_probes/mera-runs/"
    "processed_datasets/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
CHUNK_IDS        = [24, 25]
N_TABLES         = 100
RANDOM_SEED      = 42
TOKEN_BUDGET     = 3800
FORMATS          = ["csv", "json", "keyvalue", "markdown", "sql_schema"]

# Suppress OpenBLAS thread warnings
os.environ.setdefault("OPENBLAS_NUM_THREADS", "128")

# ── Argument parsing (done early so EVAL_DIR override is in effect for logging) ─

parser = argparse.ArgumentParser(description="Generate BLiMP-style eval data from T4 tables.")
parser.add_argument("--smoke-test",  action="store_true",
                    help="Use synthetic tables + mock tokenizer/pipe. No GPU or cluster paths needed.")
parser.add_argument("--skip-model",  action="store_true",
                    help="Load real tokenizer + data but mock LLaMA generation.")
parser.add_argument("--eval-dir",        default=None,
                    help="Override output directory (checkpoints, pairs, etc.).")
parser.add_argument("--candidates-cache", default=None,
                    help="Path to candidates_cache.pkl. Defaults to <eval-dir>/candidates_cache.pkl. "
                         "Pass the real eval dir's cache explicitly when using --eval-dir for testing.")
args = parser.parse_args()

if args.eval_dir:
    EVAL_DIR = args.eval_dir
elif args.smoke_test:
    EVAL_DIR = f"/tmp/eval_smoke_{int(time.time())}"

CHECKPOINT_FILE  = os.path.join(EVAL_DIR, "completed_table_ids.txt")
# Cache defaults to EVAL_DIR, but can be overridden so a test run reuses the
# real cache without re-scanning the chunks.
CANDIDATES_CACHE = args.candidates_cache if args.candidates_cache else os.path.join(EVAL_DIR, "candidates_cache.pkl")

# ── Logging ────────────────────────────────────────────────────────────────────

os.makedirs(EVAL_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(EVAL_DIR, "eval_run.log")),
    ],
)
log = logging.getLogger(__name__)

# ── serialize_t4 import — try cluster path first, fall back to script directory ─

_SERIALIZE_PATHS = [
    "/iopsstor/scratch/cscs/djanjetovic/tabular_ablation",
    os.path.dirname(os.path.abspath(__file__)),
]
for _p in _SERIALIZE_PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from serialize_t4 import (
        clean_df, serialize_csv, serialize_keyvalue, serialize_markdown,
        serialize_json_records, serialize_sql_schema, get_row_token_counts,
    )
    log.info(f"Imported serialize_t4 successfully.")
except ImportError as e:
    if args.smoke_test:
        log.warning(f"Could not import serialize_t4 ({e}) — defining local stubs for smoke test.")
        # Minimal stubs so the smoke test runs without the cluster copy
        def clean_df(df): return df
        def serialize_csv(df): return df.to_csv(index=False).rstrip('\n')
        def serialize_keyvalue(df):
            lines = []
            for _, row in df.iterrows():
                parts = [f"The {c} is {v}" for c, v in row.items() if pd.notna(v)]
                if parts: lines.append(". ".join(parts) + ".")
            return "\n".join(lines)
        def serialize_markdown(df): return df.to_markdown(index=False)
        def serialize_json_records(df): return json.dumps(df.to_dict(orient="records"), default=str)
        def serialize_sql_schema(df, schema):
            col_defs = "\n".join(f'    "{c}" VARCHAR' for c in df.columns)
            return f"CREATE TABLE data (\n{col_defs}\n);\n" + df.to_csv(index=False)
        def get_row_token_counts(df, fmt, tokenizer, schema=None):
            costs = [len(tokenizer.encode(str(row.to_dict()))) for _, row in df.iterrows()]
            return costs, 10
    else:
        raise

# ── LLaMA prompt ──────────────────────────────────────────────────────────────

STATEMENT_SYSTEM = """You are an evaluation data generator for tabular AI models.
Given a table in markdown format, produce exactly 5 statement-distractor pairs that test whether a model can read and reason over table contents.

Each pair must have:
- "statement": a true declarative sentence whose truth is verifiable from the table (e.g. "The revenue for Apple in 2022 is 394B")
- "distractor": the same sentence with a WRONG but plausible value substituted — ideally another value that actually appears in the table (e.g. "The revenue for Apple in 2022 is 210B")
- "type": one of lookup | aggregation | comparison | inference | multihop

Rules:
- Include exactly one pair of each type
- Both statement and distractor must be grammatically complete sentences
- The distractor value must be WRONG but plausible (not obviously absurd)
- Prefer distractors that use real values from elsewhere in the table — this makes the contrast harder
- Statements must be answerable only from the table, not from general knowledge
- Keep statements concise (max 20 words)
- Do NOT ask about column names, row counts, or formatting

Output valid JSON only. No preamble. No markdown fences. No explanation. Exactly this structure:
{"pairs": [
  {"statement": "...", "distractor": "...", "type": "lookup"},
  {"statement": "...", "distractor": "...", "type": "aggregation"},
  {"statement": "...", "distractor": "...", "type": "comparison"},
  {"statement": "...", "distractor": "...", "type": "inference"},
  {"statement": "...", "distractor": "...", "type": "multihop"}
]}"""

# ── Mock objects for smoke test / skip-model ───────────────────────────────────

class MockTokenizer:
    """Word-split approximation — no model files needed."""
    def encode(self, text, add_special_tokens=True, **kwargs):
        return text.split()
    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}

_CANNED_PAIRS = json.dumps({"pairs": [
    {"statement": "The revenue for Apple is 394B.",   "distractor": "The revenue for Apple is 280B.",   "type": "lookup"},
    {"statement": "The total revenue across all companies is 790B.", "distractor": "The total revenue is 500B.", "type": "aggregation"},
    {"statement": "Apple has higher revenue than Google.", "distractor": "Google has higher revenue than Apple.", "type": "comparison"},
    {"statement": "The company with the most employees also has the highest revenue.", "distractor": "The company with the fewest employees has the highest revenue.", "type": "inference"},
    {"statement": "The company founded earliest has the highest revenue per employee.", "distractor": "The company founded latest has the highest revenue per employee.", "type": "multihop"},
]})

def mock_pipe(messages, **kwargs):
    return [{"generated_text": _CANNED_PAIRS}]

def make_smoke_test_candidates():
    """Two small synthetic tables with a mock pyarrow schema each."""
    df1 = pd.DataFrame({
        "company":      ["Apple", "Google", "Meta"],
        "revenue_B":    [394,     280,      116],
        "employees_K":  [165,     190,       87],
        "founded":      [1976,    1998,     2004],
    })
    schema1 = pa.schema([
        pa.field("company",     pa.string()),
        pa.field("revenue_B",   pa.int64()),
        pa.field("employees_K", pa.float64()),
        pa.field("founded",     pa.int64()),
    ])
    df2 = pd.DataFrame({
        "country":    ["USA",  "China", "Germany", "Brazil"],
        "pop_M":      [331,    1412,    83,         215],
        "gdp_B_USD":  [23000,  17700,   4200,       1800],
    })
    schema2 = pa.schema([
        pa.field("country",   pa.string()),
        pa.field("pop_M",     pa.int64()),
        pa.field("gdp_B_USD", pa.int64()),
    ])
    return [
        ("smoke_table_001", df1, schema1),
        ("smoke_table_002", df2, schema2),
    ]

# ── Graceful shutdown on SIGTERM ───────────────────────────────────────────────

_SHUTDOWN = False

def _handle_sigterm(signum, frame):
    global _SHUTDOWN
    log.warning("SIGTERM received — will exit cleanly after current table.")
    _SHUTDOWN = True

signal.signal(signal.SIGTERM, _handle_sigterm)

# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        log.info("No checkpoint file found — starting fresh.")
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        ids = {line.strip() for line in f if line.strip()}
    log.info(f"Checkpoint: {len(ids)} tables already completed — will skip them.")
    return ids

def save_checkpoint(table_id: str) -> None:
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(table_id + "\n")

# ── Serialization helpers ──────────────────────────────────────────────────────

def serialize_table_all_formats(
    df: pd.DataFrame,
    schema,
    tokenizer,
) -> dict[str, str | None]:
    fn_map = {
        "csv":      serialize_csv,
        "keyvalue": serialize_keyvalue,
        "markdown": serialize_markdown,
        "json":     serialize_json_records,
    }
    result: dict[str, str | None] = {}

    for fmt, fn in fn_map.items():
        result[fmt] = None
        for n_rows in range(len(df), 0, -1):
            sub = df.iloc[:n_rows]
            try:
                text = fn(sub)
                if len(tokenizer.encode(text)) <= TOKEN_BUDGET:
                    result[fmt] = text
                    break
            except Exception:
                continue

    result["sql_schema"] = None
    if schema is not None:
        try:
            for n_rows in range(len(df), 0, -1):
                sub = df.iloc[:n_rows]
                text = serialize_sql_schema(sub, schema)
                if len(tokenizer.encode(text)) <= TOKEN_BUDGET:
                    result["sql_schema"] = text
                    break
        except Exception as e:
            log.warning(f"sql_schema serialization failed: {e}")
    else:
        # No arrow schema available (e.g. smoke test with stub) — fall back to CSV
        try:
            result["sql_schema"] = serialize_csv(df)
        except Exception as e:
            log.warning(f"sql_schema fallback (csv) failed: {e}")

    return result


def truncate_to_budget(
    df: pd.DataFrame,
    fmt: str,
    schema,
    tokenizer,
    budget: int = TOKEN_BUDGET,
) -> pd.DataFrame:
    serializers = {
        "csv":      serialize_csv,
        "keyvalue": serialize_keyvalue,
        "markdown": serialize_markdown,
        "json":     serialize_json_records,
    }
    fn = serializers.get(fmt)
    if fn is None:
        return df
    for n_rows in range(len(df), 0, -1):
        sub = df.iloc[:n_rows]
        try:
            text = fn(sub)
            if len(tokenizer.encode(text)) <= budget:
                return sub
        except Exception:
            continue
    return df.iloc[:1]

# ── LLaMA generation ──────────────────────────────────────────────────────────

def generate_pairs_for_table(
    df: pd.DataFrame,
    table_id: str,
    tokenizer,
    pipe,
    budget: int = TOKEN_BUDGET,
    max_retries: int = 2,
) -> list[dict]:
    df_trunc = truncate_to_budget(df, "markdown", None, tokenizer, budget)
    table_md = serialize_markdown(df_trunc)

    messages = [
        {"role": "system", "content": STATEMENT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Table (id={table_id}):\n{table_md}\n\n"
                "Generate 5 statement-distractor pairs."
            ),
        },
    ]

    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"  Generation attempt {attempt}/{max_retries}...")
            t_gen = time.time()
            out = pipe(
                messages,
                max_new_tokens=900,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                return_full_text=False,
            )
            log.info(f"  Generation call returned in {time.time()-t_gen:.1f}s")

            raw = out[0]["generated_text"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()

            json_start = raw.find("{")
            if json_start > 0:
                log.warning(f"  Trimmed {json_start} leading chars before JSON for {table_id}")
                raw = raw[json_start:]

            data = json.loads(raw)
            pairs = data["pairs"]

            expected_types = {"lookup", "aggregation", "comparison", "inference", "multihop"}
            got_types = {p.get("type") for p in pairs}
            missing = expected_types - got_types
            if missing:
                log.warning(f"  Attempt {attempt}: missing pair types {missing} for {table_id}")
                if attempt < max_retries:
                    continue

            for p in pairs:
                p["table_id"] = table_id
            return pairs

        except Exception as e:
            log.warning(f"  Attempt {attempt}/{max_retries} failed for {table_id}: {e}")
            if attempt < max_retries:
                time.sleep(1)

    return []

# ── Table scanning ─────────────────────────────────────────────────────────────

def iter_tables(chunk_ids: list[int], t4_base: str):
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


def is_valid_table(
    df: pd.DataFrame,
    tokenizer,
    formats: list[str],
    arrow_schema,
) -> bool:
    if df.empty or len(df) < 2:
        return False
    for fmt in formats:
        try:
            _, header_cost = get_row_token_counts(df.head(1), fmt, tokenizer, arrow_schema)
            if header_cost >= TOKEN_BUDGET:
                return False
        except Exception:
            return False
    return True

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    df_dir = os.path.join(EVAL_DIR, "dataframes")
    os.makedirs(df_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("Starting eval data generation")
    log.info(f"Mode: {'SMOKE TEST' if args.smoke_test else 'skip-model' if args.skip_model else 'FULL'}")
    log.info(f"EVAL_DIR: {EVAL_DIR}")
    log.info(f"N_TABLES: {N_TABLES}  SEED: {RANDOM_SEED}  BUDGET: {TOKEN_BUDGET}")
    log.info("=" * 60)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    if args.smoke_test:
        log.info("SMOKE TEST: using MockTokenizer (word-split approximation).")
        tokenizer = MockTokenizer()
    else:
        from transformers import AutoTokenizer
        log.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
        log.info("Tokenizer loaded.")

    # ── Model / pipe ──────────────────────────────────────────────────────────
    if args.smoke_test or args.skip_model:
        log.info("Using mock pipe (no model weights loaded).")
        pipe = mock_pipe
    else:
        import torch
        from transformers import AutoModelForCausalLM
        from transformers import pipeline as hf_pipeline

        log.info(f"CUDA available: {torch.cuda.is_available()}")
        log.info("Loading LLaMA model weights...")
        t_load = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_PATH,
            torch_dtype=torch.bfloat16,   # FIX: was `dtype=`, should be `torch_dtype=`
            device_map="auto",
        )
        log.info(f"Model loaded in {time.time()-t_load:.1f}s")
        log.info(f"Model device: {next(model.parameters()).device}")
        if torch.cuda.is_available():
            log.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # FIX: clear the model's baked-in max_length=20 from generation_config
        # so it doesn't conflict with our max_new_tokens=900
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None
            log.info("Cleared model.generation_config.max_length (was 20).")

        pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer)
        log.info("Pipeline ready.")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    completed = load_checkpoint()

    # ── Phase 1: candidates ───────────────────────────────────────────────────
    if args.smoke_test:
        log.info("SMOKE TEST: using 2 synthetic tables, skipping chunk scan.")
        candidates = make_smoke_test_candidates()
    elif os.path.exists(CANDIDATES_CACHE):
        log.info(f"Phase 1: loading candidates from cache ({CANDIDATES_CACHE})...")
        with open(CANDIDATES_CACHE, "rb") as f:
            candidates = pickle.load(f)
        log.info(f"Loaded {len(candidates)} cached candidates.")
    else:
        log.info("Phase 1: scanning chunks 24-25 for valid tables (this takes ~13 min)...")
        t_scan = time.time()
        candidates = []
        for table_id, df, schema in iter_tables(CHUNK_IDS, T4_BASE):
            if is_valid_table(df, tokenizer, FORMATS, schema):
                candidates.append((table_id, df, schema))
        log.info(f"Phase 1 done in {(time.time()-t_scan)/60:.1f} min — {len(candidates)} valid candidates.")
        log.info(f"Saving cache to {CANDIDATES_CACHE}...")
        with open(CANDIDATES_CACHE, "wb") as f:
            pickle.dump(candidates, f)
        log.info("Cache saved.")

    n_select = 2 if args.smoke_test else N_TABLES
    rng = random.Random(RANDOM_SEED)
    selected = rng.sample(candidates, min(n_select, len(candidates)))
    log.info(f"Selected {len(selected)} tables for generation.")

    remaining = [t for t in selected if t[0] not in completed]
    log.info(
        f"Tables to process: {len(remaining)} "
        f"({len(selected) - len(remaining)} already completed in checkpoint)"
    )

    # ── Phase 2: generate + write ──────────────────────────────────────────────
    log.info("Phase 2: generating statement-distractor pairs and serializing all formats...")

    pair_files = {
        fmt: open(os.path.join(EVAL_DIR, f"pairs_{fmt}.jsonl"), "a")
        for fmt in FORMATS
    }
    gt_file = open(os.path.join(EVAL_DIR, "blimp_pairs.jsonl"), "a")

    written            = 0
    skipped_checkpoint = 0
    failed             = 0
    start_time         = time.time()

    try:
        for i, (table_id, df, schema) in enumerate(tqdm(selected, desc="tables")):

            if _SHUTDOWN:
                log.warning("Shutdown flag set — exiting loop early.")
                break

            if table_id in completed:
                skipped_checkpoint += 1
                continue

            # ── Progress ──────────────────────────────────────────────────────
            elapsed     = time.time() - start_time
            done_so_far = written + failed
            if done_so_far > 0:
                avg_secs        = elapsed / done_so_far
                remaining_count = len(remaining) - done_so_far
                eta_mins        = (avg_secs * remaining_count) / 60
                log.info(
                    f"[{i+1}/{len(selected)}] {table_id} | "
                    f"written={written} failed={failed} | "
                    f"avg={avg_secs:.1f}s/table | ETA≈{eta_mins:.0f}min"
                )
            else:
                log.info(f"[{i+1}/{len(selected)}] {table_id} | rows={len(df)} cols={len(df.columns)}")

            # ── Serialization ─────────────────────────────────────────────────
            log.info(f"  Serializing {len(df)} rows in {len(FORMATS)} formats...")
            t_ser = time.time()
            serialized = serialize_table_all_formats(df, schema, tokenizer)
            log.info(f"  Serialization done in {time.time()-t_ser:.2f}s")

            missing_fmts = [fmt for fmt, v in serialized.items() if v is None]
            if missing_fmts:
                log.warning(f"  Skipping {table_id}: serialization failed for {missing_fmts}")
                save_checkpoint(table_id)
                completed.add(table_id)
                failed += 1
                continue

            # ── Generation ────────────────────────────────────────────────────
            log.info(f"  Calling LLaMA generation...")
            t_gen_total = time.time()
            pairs = generate_pairs_for_table(df, table_id, tokenizer=tokenizer, pipe=pipe)
            log.info(f"  Total generation time: {time.time()-t_gen_total:.1f}s | pairs={len(pairs)}")

            if not pairs:
                log.warning(f"  Skipping {table_id}: no pairs generated after retries")
                save_checkpoint(table_id)
                completed.add(table_id)
                failed += 1
                continue

            # ── Write ─────────────────────────────────────────────────────────
            for fmt in FORMATS:
                for pair in pairs:
                    pair_files[fmt].write(json.dumps({
                        "table_id":         table_id,
                        "statement":        pair["statement"],
                        "distractor":       pair["distractor"],
                        "type":             pair["type"],
                        "table_serialized": serialized[fmt],
                    }, ensure_ascii=False) + "\n")
                pair_files[fmt].flush()

            for pair in pairs:
                gt_file.write(json.dumps({
                    "table_id":   table_id,
                    "statement":  pair["statement"],
                    "distractor": pair["distractor"],
                    "type":       pair["type"],
                }, ensure_ascii=False) + "\n")
            gt_file.flush()

            with open(os.path.join(df_dir, f"{table_id}.pkl"), "wb") as pf:
                pickle.dump(df, pf)

            save_checkpoint(table_id)
            completed.add(table_id)
            written += 1
            log.info(f"  ✓ {table_id} written successfully.")

    finally:
        for f in pair_files.values():
            f.close()
        gt_file.close()

    total_time = (time.time() - start_time) / 60
    log.info("=" * 60)
    log.info(
        f"Done. {written} tables written, "
        f"{skipped_checkpoint} skipped (checkpoint), "
        f"{failed} failed. "
        f"Total time: {total_time:.2f} min."
    )
    log.info(f"Output directory: {EVAL_DIR}")
    log.info("=" * 60)

    # ── Smoke test validation ──────────────────────────────────────────────────
    if args.smoke_test:
        log.info("--- Smoke test validation ---")
        errors = []

        # Check all output files exist and are non-empty
        for fmt in FORMATS:
            p = os.path.join(EVAL_DIR, f"pairs_{fmt}.jsonl")
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                errors.append(f"MISSING or EMPTY: pairs_{fmt}.jsonl")
            else:
                with open(p) as f:
                    lines = [json.loads(l) for l in f if l.strip()]
                expected_lines = written * 5  # 5 pairs per table
                if len(lines) != expected_lines:
                    errors.append(f"pairs_{fmt}.jsonl: expected {expected_lines} lines, got {len(lines)}")
                else:
                    log.info(f"  ✓ pairs_{fmt}.jsonl — {len(lines)} lines, keys: {list(lines[0].keys())}")

        gt_path = os.path.join(EVAL_DIR, "blimp_pairs.jsonl")
        if not os.path.exists(gt_path) or os.path.getsize(gt_path) == 0:
            errors.append("MISSING or EMPTY: blimp_pairs.jsonl")
        else:
            with open(gt_path) as f:
                gt_lines = [json.loads(l) for l in f if l.strip()]
            log.info(f"  ✓ blimp_pairs.jsonl — {len(gt_lines)} lines")

        ckpt_path = CHECKPOINT_FILE
        if not os.path.exists(ckpt_path):
            errors.append("MISSING: completed_table_ids.txt")
        else:
            with open(ckpt_path) as f:
                ckpt_ids = [l.strip() for l in f if l.strip()]
            log.info(f"  ✓ checkpoint — {len(ckpt_ids)} table IDs: {ckpt_ids}")

        pkl_files = os.listdir(os.path.join(EVAL_DIR, "dataframes"))
        if not pkl_files:
            errors.append("MISSING: no .pkl files in dataframes/")
        else:
            log.info(f"  ✓ dataframes/ — {len(pkl_files)} pickle files")

        if errors:
            log.error("SMOKE TEST FAILED:")
            for e in errors:
                log.error(f"  ✗ {e}")
            sys.exit(1)
        else:
            log.info("SMOKE TEST PASSED ✓  All output files present, non-empty, and correctly structured.")


if __name__ == "__main__":
    main()
