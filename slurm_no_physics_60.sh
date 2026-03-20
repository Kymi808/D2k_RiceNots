#!/bin/bash
# ============================================================
# SLURM job: No-physics model with 60/20/20 split
# Tests if physics losses help more with less data
# ============================================================
#SBATCH --job-name=nophys-60
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:lovelace:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/nophys_60_%j.out
#SBATCH --error=logs/nophys_60_%j.err
#SBATCH --mail-type=END,FAIL

echo "============================================"
echo "Job ID: $SLURM_JOB_ID  [NO PHYSICS / 60/20/20]"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Date:   $(date)"
echo "============================================"

RUN_DIR=results/run_${SLURM_JOB_ID}
CKPT_DIR=checkpoints/run_${SLURM_JOB_ID}
mkdir -p logs $RUN_DIR $CKPT_DIR

module purge
module load Miniforge3/24.1.2-0

PROJECT_DIR=/projects/dsci435/NASA_REENTRY_SP26
ENV_DIR=$PROJECT_DIR/.conda/envs/mamba-cfd
export PATH=$ENV_DIR/bin:$PATH
export PYTHONPATH=$ENV_DIR/lib/python3.11/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "Starting training (no physics, 60/20/20) with 4 GPUs..."
echo ""

torchrun \
    --nproc_per_node=4 \
    train.py \
    --data data/apollo_cfd_database.csv \
    --train_frac 0.60 \
    --val_frac 0.20 \
    --no_physics \
    --save_dir $RUN_DIR \
    --checkpoint_dir $CKPT_DIR \
    --no_compile

echo ""
echo "============================================"
echo "Job completed: $(date)"
echo "============================================"

if [ -f $RUN_DIR/config_summary.txt ]; then
    echo ""
    echo "=== RESULTS SUMMARY ==="
    cat $RUN_DIR/config_summary.txt
    echo ""
    echo "Results in: $RUN_DIR"
fi
