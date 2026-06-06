# Tabular Ablation Study

Serialization format ablation study for large-scale tabular pretraining. Part of a semester project at EPFL / CSCS.

**Goal:** Train 5 small proxy models (1.54B parameters, ~30B tokens each) on the [T4 dataset](https://huggingface.co/datasets/mlf oundations/t4), one per serialization format, and evaluate which format leads to the best downstream tabular understanding — informing the format choice for full-scale Apertus pretraining.

**Formats under test:** `csv`, `keyvalue`, `markdown`, `json`, `sql_schema`

---

## Pipeline overview

```
T4 parquet chunks (zip)
│
▼
[Phase 1] serialize_t4.py        → .jsonl files (one line per table segment)
│
▼
[Phase 2] submit_tokenize.sh     → .bin / .idx files (Megatron binary format)
│
▼
[Phase 3] submit_tabular_ablation.sh  → 5 proxy model checkpoints
│
▼
[Phase 4] Evaluation             → winning format
├── prepare_eval_data.py       → prompts + ground truth from held-out T4 tables
└── run_eval.sh                → log-prob scoring → accuracy per format
```

---
---

## Scripts

### `serialize_t4.py`

Reads T4 parquet files from a chunk zip archive, cleans and serializes every table using **Tabular Chunking** — splitting each table into multiple 3800-token segments so no row is discarded.

**Key design decisions:**
- **Tabular Chunking**: each table is split into multiple 3800-token segments using greedy row accumulation with per-row token pre-computation. Writes one `{"text": "..."}` JSON line per segment.
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

**For a full run (all 76 chunks, all formats):**
```bash
sbatch --array=0-75 submit_serialize.sh csv
sbatch --array=0-75 submit_serialize.sh sql_schema
sbatch --array=0-75 submit_serialize.sh keyvalue
sbatch --array=0-75 submit_serialize.sh markdown
sbatch --array=0-75 submit_serialize.sh json
```

**For this ablation study (24 chunks):**
```bash
sbatch --array=0-23 submit_serialize.sh csv
sbatch --array=0-23 submit_serialize.sh sql_schema
sbatch --array=0-23 submit_serialize.sh keyvalue
sbatch --array=0-23 submit_serialize.sh markdown
sbatch --array=0-23 submit_serialize.sh json
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

## Evaluation

Validation loss curves during training are tracked via WandB. For downstream evaluation, we use a **BLiMP-style log-probability scoring** approach on held-out T4 data (chunks 24–25, not seen during training).

**Methodology:**
1. Sample 300 tables from the held-out chunks.
2. For each table and each format, build a context prompt from rows 0..k-1 (fitting within the 3800-token budget), holding out the last row.
3. Use LLaMA-3.1-8B to generate a *statement* (correct last-row completion) and a *distractor* (plausible but incorrect completion) for each table.
4. At inference time, compute `log P(statement | table_context)` and `log P(distractor | table_context)` using each of the 5 proxy models.
5. The model assigns higher probability to the correct statement → correct. Accuracy across 300 tables per format determines the winning serialization.

This evaluates whether the model has learned tabular structure well enough to prefer factually correct completions over plausible distractors.

### `Evaluation/prepare_eval_data.py`

Prepares the evaluation dataset. Scans T4 chunks 24–25, filters valid tables (≥2 rows, header fits token budget across all 5 formats), and samples 300 tables with a fixed seed. For each table, builds context prompts in all 5 formats simultaneously and saves the held-out last row as ground truth.

**Output** (written to `/iopsstor/scratch/cscs/djanjetovic/tabular_ablation/eval/`):
- `prompts_{format}.jsonl` — one `{table_id, prompt}` per table per format
- `ground_truth.jsonl` — `{table_id, ground_truth_dict, col_dtypes}` per table
- `dataframes/{table_id}.pkl` — cleaned DataFrames for cross-format consistency checks

**Usage** (login node, CPU only, ~10 min):
```bash
python Evaluation/prepare_eval_data.py
```

### `run_eval.sh`

SLURM job script that runs inference-time log-probability scoring on the prepared prompts. Requires 1 GPU, 12h wall time.

---

## Data

**Source:** [mlf oundations/t4](https://huggingface.co/datasets/mlf oundations/t4) — a large-scale collection of tables scraped from the web, stored as parquet files organized in 76 chunks. Chunks 0–23 used for training; chunks 24–25 held out for evaluation.
