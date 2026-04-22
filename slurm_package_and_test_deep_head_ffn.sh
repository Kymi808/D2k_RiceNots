#!/bin/bash
# ============================================================
# SLURM job: Package and test a deep-head + residual-FFN checkpoint
# Usage:
#   sbatch slurm_package_and_test_deep_head_ffn.sh <checkpoint_path> <train_frac> <val_frac> <split_seed> [output_dir]
# Example:
#   sbatch slurm_package_and_test_deep_head_ffn.sh checkpoints/deep_head_ffn_123/best_model.pt 0.60 0.20 123 packaged_model_deep_head_ffn_60_20
# ============================================================
#SBATCH --job-name=dhffn-pack
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/package_test_deep_head_ffn_%j.out
#SBATCH --error=logs/package_test_deep_head_ffn_%j.err

CKPT=${1:?Usage: sbatch slurm_package_and_test_deep_head_ffn.sh <checkpoint_path> <train_frac> <val_frac> <split_seed> [output_dir]}
TRAIN_FRAC=${2:?Usage: sbatch slurm_package_and_test_deep_head_ffn.sh <checkpoint_path> <train_frac> <val_frac> <split_seed> [output_dir]}
VAL_FRAC=${3:?Usage: sbatch slurm_package_and_test_deep_head_ffn.sh <checkpoint_path> <train_frac> <val_frac> <split_seed> [output_dir]}
SPLIT_SEED=${4:?Usage: sbatch slurm_package_and_test_deep_head_ffn.sh <checkpoint_path> <train_frac> <val_frac> <split_seed> [output_dir]}
OUTPUT_DIR=${5:-packaged_model_deep_head_ffn}

echo "============================================"
echo "Job ID: $SLURM_JOB_ID  [DEEP HEAD + FFN PACKAGE + TEST]"
echo "Checkpoint: $CKPT"
echo "Split: train=$TRAIN_FRAC val=$VAL_FRAC seed=$SPLIT_SEED"
echo "Output dir: $OUTPUT_DIR"
echo "Date:   $(date)"
echo "============================================"

mkdir -p logs

module purge
module load Miniforge3/24.1.2-0

PROJECT_DIR=/projects/dsci435/NASA_REENTRY_SP26
ENV_DIR=$PROJECT_DIR/.conda/envs/mamba-cfd
export PATH=$ENV_DIR/bin:$PATH
export PYTHONPATH=$ENV_DIR/lib/python3.11/site-packages:$PYTHONPATH

RESULTS_DIR=test_inference/results_$(basename "$OUTPUT_DIR")

echo ""
echo "Step 1: Packaging checkpoint..."
echo ""

torchrun --nproc_per_node=1 package_model.py \
    --checkpoint "$CKPT" \
    --output "$OUTPUT_DIR" \
    --train_frac "$TRAIN_FRAC" \
    --val_frac "$VAL_FRAC" \
    --split_seed "$SPLIT_SEED" \
    --pred_head_dims 128,64 \
    --pred_head_dropout 0.05 \
    --use_residual_ffn \
    --ffn_hidden_dim 128

echo ""
echo "Step 2: Running inference tests..."
echo ""

torchrun --nproc_per_node=1 test_inference/run_tests.py \
    --model_dir "$OUTPUT_DIR" \
    --results_dir "$RESULTS_DIR"

echo ""
echo "============================================"
echo "Completed: $(date)"
echo "============================================"
