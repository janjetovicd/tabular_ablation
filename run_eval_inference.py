# BLiMP-style inference eval for tabular format ablation.
#
# DESIGN: Megatron-native — loads the checkpoint directly using the same
# Megatron infrastructure used for training, without any HF conversion.
# Why: the swiss-megatron repo has no HF saver, and the proxy model
# architecture (RoPE, RMSNorm, GQA) may differ subtly from standard LLaMA,
# making a manual conversion fragile. Loading natively is the safe path.
#
# MUST be launched via submit_eval_inference.sh (uses srun + container + TP=2),
# NOT run directly with `python`. Megatron requires a proper distributed setup.
#
# For each record in pairs_{fmt}.jsonl it computes:
#   nll(statement)  = mean NLL of statement tokens given table_serialized prefix
#   nll(distractor) = mean NLL of distractor tokens given table_serialized prefix
#   correct = 1 if nll(statement) < nll(distractor)
#
# The table prefix tokens are masked (label = -100) so only hypothesis tokens
# contribute to the loss. This is mathematically equivalent to comparing full
# sequence log probs (the shared prefix cancels), but length-normalises the
# hypothesis so longer statements aren't penalised.
#
# Outputs (in --output-dir):
#   results_{fmt}.jsonl   — one record per pair with scores + correct flag
#   summary_{fmt}.json    — accuracy overall and by pair type
#   eval_{fmt}.log        — run log

import os
import sys
import json
import time
import logging
import argparse
from collections import defaultdict

# ── Parse our own args BEFORE Megatron hijacks sys.argv ──────────────────────
# Megatron re-parses sys.argv during initialize_megatron(), so we extract our
# args first, then replace sys.argv with the Megatron args below.

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--format', required=True,
                choices=['csv', 'json', 'keyvalue', 'markdown', 'sql_schema'])
_p.add_argument('--ckpt-dir', default='dummy',
                help='Path to checkpoint directory. Ignored in --smoke-test mode.')
_p.add_argument('--pairs-dir', required=True,
                help='Directory containing pairs_{fmt}.jsonl files.')
_p.add_argument('--output-dir', required=True)
_p.add_argument('--max-seq-len', type=int, default=4096)
_p.add_argument('--smoke-test', action='store_true',
                help='Skip Megatron and model entirely. Uses random scores to validate '
                     'I/O, pair loading, and output writing. Safe to run on CPU/login node.')
eval_args, _ = _p.parse_known_args()

# ── Paths ─────────────────────────────────────────────────────────────────────

MEGATRON_DIR   = os.environ.get('MEGATRON_LM_DIR',
                     '/iopsstor/scratch/cscs/djanjetovic/swiss-megatron')
TOKENIZER_PATH = '/iopsstor/scratch/cscs/djanjetovic/tokenizer_cache/apertus'
PAIRS_PATH     = os.path.join(eval_args.pairs_dir, f'pairs_{eval_args.format}.jsonl')
RESULTS_PATH   = os.path.join(eval_args.output_dir, f'results_{eval_args.format}.jsonl')
SUMMARY_PATH   = os.path.join(eval_args.output_dir, f'summary_{eval_args.format}.json')

os.makedirs(eval_args.output_dir, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(eval_args.output_dir, f'eval_{eval_args.format}.log')
        ),
    ],
)
log = logging.getLogger(__name__)

# ── Replace sys.argv with Megatron args ───────────────────────────────────────
# These must exactly match the training configuration in submit_tabular_ablation.sh.
# Megatron's arg parser will read these when initialize_megatron() is called.

sys.path.insert(0, MEGATRON_DIR)

sys.argv = [
    'eval_inference',
    # Architecture — must match training exactly
    '--num-layers',              '32',
    '--hidden-size',             '2048',
    '--ffn-hidden-size',         '6144',
    '--num-attention-heads',     '16',
    '--group-query-attention',
    '--num-query-groups',        '4',
    '--max-position-embeddings', str(eval_args.max_seq_len),
    '--seq-length',              str(eval_args.max_seq_len),
    '--position-embedding-type', 'rope',
    '--rotary-base',             '500000',
    '--normalization',           'RMSNorm',
    '--untie-embeddings-and-output-weights',
    '--make-vocab-size-divisible-by', '128',
    '--disable-bias-linear',
    # Precision
    '--bf16',
    # Tokenizer
    '--tokenizer-type',  'HuggingFaceTokenizer',
    '--tokenizer-model', TOKENIZER_PATH,
    '--trust-remote-code',
    # Checkpoint — load weights only, skip optimizer/rng state
    '--load',        eval_args.ckpt_dir,
    '--ckpt-format', 'torch_dist',
    '--no-load-optim',
    '--no-load-rng',
    # Parallelism — must match training (TP=2 PP=1)
    '--tensor-model-parallel-size', '2',
    '--pipeline-model-parallel-size', '1',
    # Minimal training args required by Megatron's arg parser
    '--micro-batch-size', '1',
    '--global-batch-size', '2',   # must be >= TP * MBS
    '--train-iters', '0',         # no training, eval only
    '--distributed-backend', 'nccl',
]

# ── Smoke test mode ───────────────────────────────────────────────────────────
# Run with: python3 run_eval_inference.py --smoke-test --format csv \
#               --pairs-dir /path/to/eval --output-dir /tmp/smoke_out
# No GPUs, no checkpoint, no Megatron needed. Validates I/O only.

if eval_args.smoke_test:
    import random
    log.info('SMOKE TEST MODE — skipping Megatron, model, and checkpoint.')
    log.info(f'Reading pairs from {PAIRS_PATH}')

    if not os.path.exists(PAIRS_PATH):
        log.error(f'Pairs file not found: {PAIRS_PATH}')
        sys.exit(1)

    with open(PAIRS_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]
    log.info(f'Loaded {len(records)} pairs.')

    results = []
    for record in records[:5]:   # only test 5 pairs
        nll_s = random.uniform(1.0, 5.0)
        nll_d = random.uniform(1.0, 5.0)
        results.append({
            'table_id':       record['table_id'],
            'type':           record.get('type', 'unknown'),
            'statement':      record['statement'],
            'distractor':     record['distractor'],
            'nll_statement':  nll_s,
            'nll_distractor': nll_d,
            'correct':        nll_s < nll_d,
        })

    with open(RESULTS_PATH, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    summary = {'smoke_test': True, 'n_pairs_tested': len(results), 'results_path': RESULTS_PATH}
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)

    log.info(f'SMOKE TEST PASSED — wrote {len(results)} results to {RESULTS_PATH}')
    log.info(f'Summary: {SUMMARY_PATH}')
    sys.exit(0)

# ── Megatron initialization ───────────────────────────────────────────────────

log.info('Initializing Megatron ...')
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args as megatron_get_args

initialize_megatron()
args = megatron_get_args()
log.info('Megatron initialized.')

import torch
from megatron.core import mpu

# Only rank 0 does I/O (file reading/writing)
global_rank = torch.distributed.get_rank()
is_io_rank  = (global_rank == 0)

# ── Model provider ────────────────────────────────────────────────────────────
# Mirrors what pretrain_gpt.py does. If this fails, check what model_provider
# looks like in your swiss-megatron/pretrain_gpt.py and replicate it here.

def model_provider(pre_process=True, post_process=True):
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.core.models.gpt import GPTModel

    config = core_transformer_config_from_args(args)

    # Get the transformer layer spec — try Transformer Engine first (used on A100/H100),
    # fall back to local spec if TE is not available.
    try:
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=False,
        )
        log.info('Using Transformer Engine layer spec.')
    except Exception as e:
        log.warning(f'TE spec failed ({e}), falling back to local spec.')
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
        transformer_layer_spec = get_gpt_layer_local_spec()

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=getattr(args, 'fp16_lm_cross_entropy', False),
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=getattr(args, 'rotary_percent', 1.0),
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=getattr(args, 'rotary_seq_len_interpolation_factor', None),
    )
    return model

# ── Load model ────────────────────────────────────────────────────────────────

from megatron.training import get_model
from megatron.training.checkpointing import load_checkpoint

log.info('Building model ...')
models = get_model(model_provider, wrap_with_ddp=False)
model  = models[0]
model.eval()
log.info('Model built.')

log.info(f'Loading checkpoint from {eval_args.ckpt_dir} ...')
load_checkpoint(models, None, None)
log.info('Checkpoint loaded.')

# ── Tokenizer ─────────────────────────────────────────────────────────────────

from megatron.training.tokenizer import build_tokenizer
tokenizer = build_tokenizer(args)

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_hypothesis(table_text: str, hypothesis: str) -> float | None:
    """
    Returns mean NLL of `hypothesis` tokens conditioned on `table_text` as prefix.
    Lower = model assigns higher probability to this hypothesis.
    Returns None if the combined sequence exceeds max_seq_len.
    """
    # Tokenize prefix and full sequence to find the boundary
    prefix_ids = tokenizer.tokenize(table_text)
    full_ids   = tokenizer.tokenize(table_text + '\n' + hypothesis)

    if len(full_ids) > eval_args.max_seq_len:
        return None

    n_hyp = len(full_ids) - len(prefix_ids)
    if n_hyp <= 0:
        log.warning('No hypothesis tokens after tokenization — skipping.')
        return None

    device = torch.cuda.current_device()

    tokens = torch.tensor([full_ids], dtype=torch.long, device=device)

    # Labels: -100 (masked) for prefix, actual ids for hypothesis.
    # Megatron's GPTModel.forward() with labels does the loss internally;
    # but we compute it ourselves here for full control over masking.
    labels = tokens.clone()
    labels[0, :len(prefix_ids)] = -100

    # Build attention mask and position ids (Megatron convention)
    from megatron.training.utils import get_ltor_masks_and_position_ids
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )

    with torch.no_grad():
        # GPTModel forward returns logits: [batch, seq, vocab]
        logits = model(tokens, position_ids, attention_mask)

    # Compute cross-entropy loss over hypothesis tokens only.
    # Shift by 1 for next-token prediction: logits[i] predicts token[i+1].
    shift_logits = logits[:, :-1, :].float().contiguous()   # [1, seq-1, vocab]
    shift_labels = labels[:, 1:].contiguous()               # [1, seq-1]

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction='mean',
    )

    return loss.item()


# ── Main eval loop ────────────────────────────────────────────────────────────

if not os.path.exists(PAIRS_PATH):
    log.error(f'Pairs file not found: {PAIRS_PATH}')
    torch.distributed.barrier()
    sys.exit(1)

# Load pairs (all ranks read, but only rank 0 writes results)
with open(PAIRS_PATH) as f:
    records = [json.loads(line) for line in f if line.strip()]

log.info(f'Loaded {len(records)} pairs from {PAIRS_PATH}')

# Resume support: skip already-scored pairs
scored_keys: set[tuple] = set()
if is_io_rank and os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                scored_keys.add((r['table_id'], r['statement']))
    log.info(f'Resuming: {len(scored_keys)} pairs already scored.')

# Broadcast scored_keys count to all ranks (they all need to stay in sync)
torch.distributed.barrier()

results_file = open(RESULTS_PATH, 'a') if is_io_rank else None

correct_by_type: dict[str, list[bool]] = defaultdict(list)
correct_overall: list[bool] = []
skipped  = 0
processed = 0
start_time = time.time()

for i, record in enumerate(records):
    table_id   = record['table_id']
    statement  = record['statement']
    distractor = record['distractor']
    pair_type  = record.get('type', 'unknown')
    table_text = record['table_serialized']

    if (table_id, statement) in scored_keys:
        continue

    if processed > 0 and processed % 50 == 0 and is_io_rank:
        elapsed = time.time() - start_time
        rate = processed / elapsed
        eta_min = ((len(records) - i) / rate) / 60 if rate > 0 else float('inf')
        log.info(
            f'[{i+1}/{len(records)}] processed={processed} skipped={skipped} '
            f'rate={rate:.1f}/s ETA≈{eta_min:.0f}min'
        )

    # All ranks must call forward (Megatron distributed model)
    nll_s = score_hypothesis(table_text, statement)
    nll_d = score_hypothesis(table_text, distractor)

    if nll_s is None or nll_d is None:
        if is_io_rank:
            log.warning(f'Skipping pair {i} ({table_id}): sequence too long.')
        skipped += 1
        continue

    is_correct = nll_s < nll_d
    correct_by_type[pair_type].append(is_correct)
    correct_overall.append(is_correct)

    if is_io_rank:
        result = {
            'table_id':       table_id,
            'type':           pair_type,
            'statement':      statement,
            'distractor':     distractor,
            'nll_statement':  nll_s,
            'nll_distractor': nll_d,
            'correct':        is_correct,
        }
        results_file.write(json.dumps(result, ensure_ascii=False) + '\n')
        results_file.flush()

    processed += 1

if results_file:
    results_file.close()

# ── Summary (rank 0 only) ──────────────────────────────────────────────────────

if is_io_rank:
    total_time = (time.time() - start_time) / 60
    overall_acc = sum(correct_overall) / len(correct_overall) if correct_overall else 0.0

    type_acc = {
        t: {'correct': sum(v), 'total': len(v), 'accuracy': round(sum(v) / len(v), 4)}
        for t, v in correct_by_type.items()
    }

    summary = {
        'format':             eval_args.format,
        'ckpt_dir':           eval_args.ckpt_dir,
        'n_pairs':            len(records),
        'n_processed':        processed,
        'n_skipped':          skipped,
        'overall_accuracy':   round(overall_acc, 4),
        'by_type':            type_acc,
        'total_time_min':     round(total_time, 2),
    }

    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)

    log.info('=' * 60)
    log.info(f'Done. processed={processed} skipped={skipped} time={total_time:.1f}min')
    log.info(f'Overall accuracy: {overall_acc:.4f} ({sum(correct_overall)}/{len(correct_overall)})')
    for t, v in type_acc.items():
        log.info(f'  {t:15s}: {v["accuracy"]:.4f}  ({v["correct"]}/{v["total"]})')
    log.info(f'Results  → {RESULTS_PATH}')
    log.info(f'Summary  → {SUMMARY_PATH}')
    log.info('=' * 60)

torch.distributed.barrier()
