#!/bin/bash
# ============================================================
# SLURM job: Generate error maps for all models (1 GPU, short job)
# ============================================================
#SBATCH --job-name=error-maps
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/error_maps_%j.out
#SBATCH --error=logs/error_maps_%j.err

echo "============================================"
echo "Job ID: $SLURM_JOB_ID  [ERROR MAPS]"
echo "Date:   $(date)"
echo "============================================"

mkdir -p logs

module purge
module load Miniforge3/24.1.2-0

PROJECT_DIR=/projects/dsci435/NASA_REENTRY_SP26
ENV_DIR=$PROJECT_DIR/.conda/envs/mamba-cfd
export PATH=$ENV_DIR/bin:$PATH
export PYTHONPATH=$ENV_DIR/lib/python3.11/site-packages:$PYTHONPATH

torchrun --nproc_per_node=1 create_error_maps.py

echo ""
echo "Error maps completed: $(date)"
