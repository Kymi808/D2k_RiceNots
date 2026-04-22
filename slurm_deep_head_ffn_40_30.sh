#!/bin/bash
# ============================================================
# SLURM job: Deep heads + residual FFN with 40/30/30 split
# ============================================================
#SBATCH --job-name=dhffn-40-30
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:lovelace:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/deep_head_ffn_40_30_%j.out
#SBATCH --error=logs/deep_head_ffn_40_30_%j.err
#SBATCH --mail-type=END,FAIL

echo "============================================"
echo "Job ID: $SLURM_JOB_ID  [DEEP HEAD + FFN 40/30/30]"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Date:   $(date)"
echo "============================================"

RUN_DIR=results/deep_head_ffn_${SLURM_JOB_ID}
CKPT_DIR=checkpoints/deep_head_ffn_${SLURM_JOB_ID}
mkdir -p logs "$RUN_DIR" "$CKPT_DIR"

module purge
module load Miniforge3/24.1.2-0

PROJECT_DIR=/projects/dsci435/NASA_REENTRY_SP26
ENV_DIR=$PROJECT_DIR/.conda/envs/mamba-cfd
export PATH=$ENV_DIR/bin:$PATH
export PYTHONPATH=$ENV_DIR/lib/python3.11/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "Starting deep-head + residual-FFN training (40/30/30 split) with 4 GPUs..."
echo ""

torchrun \
    --nproc_per_node=4 \
    train.py \
    --data data/apollo_cfd_database.csv \
    --train_frac 0.40 \
    --val_frac 0.30 \
    --pred_head_dims 128,64 \
    --pred_head_dropout 0.05 \
    --use_residual_ffn \
    --ffn_hidden_dim 128 \
    --save_dir "$RUN_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    --no_compile

echo ""
echo "============================================"
echo "Job completed: $(date)"
echo "============================================"

if [ -f "$RUN_DIR/config_summary.txt" ]; then
    echo ""
    echo "=== RESULTS SUMMARY ==="
    cat "$RUN_DIR/config_summary.txt"
    echo ""
    echo "Results in: $RUN_DIR"
fi
