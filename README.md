# tabular_ablation

Serialization format ablation study for large-scale tabular pretraining. Part of a semester project at EPFL / CSCS.

**Goal:** Train 5 small proxy models (1.54B parameters, ~30B tokens each) on the [T4 dataset](https://huggingface.co/datasets/mlfoundations/t4), one per serialization format, and pick the best format for full-scale pretraining.

**Formats under test:** `csv`, `keyvalue`, `markdown`, `json`, `sql_schema`

---

## Pipeline overview

```
T4 parquet chunks (zip)
        │
        ▼
[Phase 2a] serialize_t4.py       → .jsonl files (one line per table segment)
        │
        ▼
[Phase 2b] submit_tokenize.sh    → .bin / .idx files (Megatron binary format)
        │
        ▼
[Phase 2c] submit_tabular_ablation.sh  → 5 proxy model checkpoints
        │
        ▼
[Phase 2d] Compare validation loss + downstream task accuracy → winning format
```

---

## Scripts

### `serialize_t4.py`

Reads T4 parquet files from a chunk zip archive, cleans and serializes every table using **Tabular Chunking** — splitting each table into multiple 3800-token segments so no row is discarded.

**Key design decisions:**
- **Binary search per segment**: finds the maximum number of rows fitting in the token budget without brute-force iteration.
- **Header/schema repetition**: every segment includes the column header or SQL schema so the model always has semantic context.
- **Deterministic shuffle**: rows are shuffled with a seed derived from the filename (MD5 hash), ensuring all 5 format runs see identical row orderings for a fair comparison.
- **Cleaning**: drops artifact columns (`Unnamed:*`, `index`), all-null columns, and columns with any cell exceeding 500 characters (avoids free-text blobs and base64 data swamping the token budget).

**Usage:**
```bash
python serialize_t4.py \
    --chunk_zip /path/to/chunk-0000.zip \
    --format csv \
    --output /path/to/output/chunk-0000.jsonl \
    --tokenizer /path/to/cached/tokenizer \
    --token_budget 3800
```

**Formats:**

| ID | Name | Description |
|---|---|---|
| A | `keyvalue` | `The col is val.` per cell, one row per line. Most verbose — column names repeat every row. Baseline from T4/TabLLM/UniPredict. |
| B | `csv` | Standard CSV with header. Most token-efficient — column names appear once. ~2× more rows per segment than keyvalue. |
| C | `markdown` | Markdown table with separator row. Matches format seen in GitHub READMEs and Wikipedia. |
| D | `json` | JSON array of records. Common on the web; similar verbosity to keyvalue. |
| F | `sql_schema` | `CREATE TABLE` schema with type/range annotations prepended to CSV data. Novel application: mirrors how a data scientist reads a new dataset. |

---

### `submit_serialize.sh`

SLURM array job script. Submits one job per chunk for a given format. The `--array` range is passed at submission time (not hardcoded) to allow flexible test or full runs.

**Phase A — test run (chunk 0 only):**
```bash
sbatch --array=0 submit_serialize.sh csv
# Check yield:
grep "Tokens written\|Done\." logs/serialize-*-0.out
```

**Phase B — full run (all 76 chunks, all formats):**
```bash
for fmt in csv sql_schema keyvalue markdown json; do
    sbatch --array=0-75 submit_serialize.sh $fmt
done
```

**For this ablation study (24 chunks):**
```bash
for fmt in csv sql_schema keyvalue markdown json; do
    sbatch --array=0-23 submit_serialize.sh $fmt
done
```

**SLURM config:** 1 node, 1 task, 16 CPUs, 12h wall time. No GPU needed for serialization.

**Important:** Compute nodes on Clariden have no internet access. The tokenizer must be pre-cached on the login node before submitting:
```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('swiss-ai/Apertus-70B-2509')
tok.save_pretrained('/iopsstor/scratch/cscs/djanjetovic/tokenizer_cache/apertus')
"
```

---

### `submit_tokenize.sh`

Converts `.jsonl` files to Megatron binary format (`.bin` + `.idx`) using datatrove's `MegatronDocumentTokenizer`.

**Why datatrove instead of Megatron's `preprocess_data.py`?**  
The swiss-ai Megatron fork explicitly recommends datatrove. `preprocess_data.py` was found to silently produce empty output on Clariden compute nodes because it tries to download the tokenizer at runtime (no internet on compute nodes). Datatrove with a pre-cached tokenizer path resolves this.

**Throughput:** ~70M tokens/second/node at 28 parallel workers (swiss-ai benchmark for Alps).

---

### `submit_tabular_ablation.sh`

Launches proxy model training on 8 GPU nodes (32 GH200s) using the Megatron-LM framework inside the provided container.

**Model:** 1.54B non-embedding parameters  
**Training:** ~30B tokens, 28,610 steps, global batch size 256 (≈1M tokens/step)

---

### `tabular_container.toml`

Container definition for GPU training jobs. Specifies `ngc-nemo:25.11.01-alps3`. Used by `submit_tabular_ablation.sh`.

---

## Data

**Source:** [mlfoundations/t4](https://huggingface.co/datasets/mlfoundations/t4) — a large-scale collection of tables scraped from the web, stored as parquet files organized in 76 chunks.
