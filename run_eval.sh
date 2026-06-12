#!/bin/bash
#SBATCH -A a139
#SBATCH --partition=normal
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --job-name=prepare_eval
#SBATCH --output=logs/prepare_eval_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "Running on $(hostname)"
echo "Start time: $(date)"

cd /iopsstor/scratch/cscs/djanjetovic/tabular_ablation

# Write generated eval pairs + candidates cache to permanent store so they
# survive the 14-day scratch cleanup. prepare_eval_data_v2.py otherwise defaults
# to a hardcoded scratch EVAL_DIR, so override it explicitly here.
STORE_DIR=/capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation

python prepare_eval_data_v2.py \
    --eval-dir $STORE_DIR/eval \
    --candidates-cache $STORE_DIR/eval/candidates_cache.pkl

echo "End time: $(date)"