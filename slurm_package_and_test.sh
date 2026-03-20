#!/bin/bash
# ============================================================
# SLURM job: Package best model and run inference tests
# ============================================================
#SBATCH --job-name=inference-test
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/inference_test_%j.out
#SBATCH --error=logs/inference_test_%j.err

echo "============================================"
echo "Job ID: $SLURM_JOB_ID  [PACKAGE + TEST]"
echo "Date:   $(date)"
echo "============================================"

mkdir -p logs

module purge
module load Miniforge3/24.1.2-0

PROJECT_DIR=/projects/dsci435/NASA_REENTRY_SP26
ENV_DIR=$PROJECT_DIR/.conda/envs/mamba-cfd
export PATH=$ENV_DIR/bin:$PATH
export PYTHONPATH=$ENV_DIR/lib/python3.11/site-packages:$PYTHONPATH

echo ""
echo "Step 1: Packaging best model..."
echo ""

torchrun --nproc_per_node=1 package_model.py \
    --checkpoint organized_results/full_model_long/best_model.pt \
    --output packaged_model \
    --split_seed 456

echo ""
echo "Step 2: Running inference tests..."
echo ""

torchrun --nproc_per_node=1 test_inference/run_tests.py

echo ""
echo "============================================"
echo "Completed: $(date)"
echo "============================================"
