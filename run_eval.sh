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

python prepare_eval_data_v2.py

echo "End time: $(date)"