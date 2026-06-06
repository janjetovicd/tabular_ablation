#!/bin/bash
#SBATCH -A a139
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --job-name=prepare_eval_cpu
#SBATCH --output=logs/prepare_eval_cpu_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "Running on $(hostname)"
echo "Start time: $(date)"

cd /iopsstor/scratch/cscs/djanjetovic/tabular_ablation

python prepare_eval_data_v2_cpu.py \
    --eval-dir /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/eval_cpu \
    --candidates-cache /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/eval/candidates_cache.pkl

echo "End time: $(date)"
