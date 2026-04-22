#!/bin/bash
# ============================================================
# SLURM job: Evaluate a deep-head + residual-FFN checkpoint
# Usage:
#   sbatch slurm_eval_deep_head_ffn.sh <checkpoint_path> <train_frac> <val_frac> <split_seed>
# Example:
#   sbatch slurm_eval_deep_head_ffn.sh checkpoints/deep_head_ffn_123/best_model.pt 0.40 0.30 123
# ============================================================
#SBATCH --job-name=dhffn-eval
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_deep_head_ffn_%j.out
#SBATCH --error=logs/eval_deep_head_ffn_%j.err

CKPT=${1:?Usage: sbatch slurm_eval_deep_head_ffn.sh <checkpoint_path> <train_frac> <val_frac> <split_seed>}
TRAIN_FRAC=${2:?Usage: sbatch slurm_eval_deep_head_ffn.sh <checkpoint_path> <train_frac> <val_frac> <split_seed>}
VAL_FRAC=${3:?Usage: sbatch slurm_eval_deep_head_ffn.sh <checkpoint_path> <train_frac> <val_frac> <split_seed>}
SPLIT_SEED=${4:?Usage: sbatch slurm_eval_deep_head_ffn.sh <checkpoint_path> <train_frac> <val_frac> <split_seed>}

echo "============================================"
echo "Job ID: $SLURM_JOB_ID  [DEEP HEAD + FFN EVAL]"
echo "Checkpoint: $CKPT"
echo "Split: train=$TRAIN_FRAC val=$VAL_FRAC seed=$SPLIT_SEED"
echo "Date:   $(date)"
echo "============================================"

mkdir -p logs

module purge
module load Miniforge3/24.1.2-0

PROJECT_DIR=/projects/dsci435/NASA_REENTRY_SP26
ENV_DIR=$PROJECT_DIR/.conda/envs/mamba-cfd
export PATH=$ENV_DIR/bin:$PATH
export PYTHONPATH=$ENV_DIR/lib/python3.11/site-packages:$PYTHONPATH

torchrun --nproc_per_node=1 eval_checkpoint.py \
    --checkpoint "$CKPT" \
    --data data/apollo_cfd_database.csv \
    --train_frac "$TRAIN_FRAC" \
    --val_frac "$VAL_FRAC" \
    --split_seed "$SPLIT_SEED" \
    --pred_head_dims 128,64 \
    --pred_head_dropout 0.05 \
    --use_residual_ffn \
    --ffn_hidden_dim 128

echo ""
echo "Eval completed: $(date)"
