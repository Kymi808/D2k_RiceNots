#!/bin/bash
# ============================================================
# SLURM job: normalized qw, deep head + FFN, no physics/recon, 80/10/10
# ============================================================
#SBATCH --job-name=nqw-dhffn-80
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:lovelace:4
#SBATCH --time=72:00:00
#SBATCH --output=logs/norm_qw_dhffn_nophys_norecon_80_%j.out
#SBATCH --error=logs/norm_qw_dhffn_nophys_norecon_80_%j.err
#SBATCH --mail-type=END,FAIL

SPLIT_SEED=${SPLIT_SEED:-456}
QW_WEIGHT=${QW_WEIGHT:-6.0}

echo "============================================"
echo "Job ID: $SLURM_JOB_ID  [NORM QW / DH+FFN / NO PHYS / NO RECON / 80/10/10]"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Seed:   $SPLIT_SEED"
echo "qw w:   $QW_WEIGHT"
echo "Date:   $(date)"
echo "============================================"

RUN_DIR=results/norm_qw_dhffn_nophys_norecon_80_${SLURM_JOB_ID}
CKPT_DIR=checkpoints/norm_qw_dhffn_nophys_norecon_80_${SLURM_JOB_ID}
mkdir -p logs "$RUN_DIR" "$CKPT_DIR"

module purge
module load Miniforge3/24.1.2-0

PROJECT_DIR=/projects/dsci435/NASA_REENTRY_SP26
ENV_DIR=$PROJECT_DIR/.conda/envs/mamba-cfd
export PATH=$ENV_DIR/bin:$PATH
export PYTHONPATH=$ENV_DIR/lib/python3.11/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
    --nproc_per_node=4 \
    train.py \
    --data data/apollo_cfd_database.csv \
    --train_frac 0.80 \
    --val_frac 0.10 \
    --split_seed "$SPLIT_SEED" \
    --pred_head_dims 128,64 \
    --pred_head_dropout 0.05 \
    --use_residual_ffn \
    --ffn_hidden_dim 128 \
    --normalize_qw_by_rhov3 \
    --w_qw "$QW_WEIGHT" \
    --no_physics \
    --no_reconstruction \
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
