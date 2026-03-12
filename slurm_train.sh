#!/bin/bash
# ============================================================
# SLURM job script for Mamba Autoencoder CFD on Rice NOTS
# 4x L40S GPUs (48GB each, 192GB total)
# ============================================================
#SBATCH --job-name=mamba-cfd
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:lovelace:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL

# NOTE: 'logs/' directory must exist BEFORE submitting this job.
#       Run: mkdir -p logs

# --- Setup ---
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Date:   $(date)"
echo "============================================"

# Per-job output dirs to avoid overwriting parallel runs
RUN_DIR=results/run_${SLURM_JOB_ID}
CKPT_DIR=checkpoints/run_${SLURM_JOB_ID}
mkdir -p logs $RUN_DIR $CKPT_DIR

# --- Environment ---
module purge
module load Miniforge3/24.1.2-0

# Use conda env's Python and add pip --target packages to path
PROJECT_DIR=/projects/dsci435/NASA_REENTRY_SP26
ENV_DIR=$PROJECT_DIR/.conda/envs/mamba-cfd
export PATH=$ENV_DIR/bin:$PATH
export PYTHONPATH=$ENV_DIR/lib/python3.11/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=1

# Verify GPU access
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
"

# --- Training ---
echo ""
echo "Starting training with $(python -c 'import torch; print(torch.cuda.device_count())') GPUs..."
echo ""

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train.py \
    --data data/apollo_cfd_database.csv \
    --save_dir $RUN_DIR \
    --checkpoint_dir $CKPT_DIR

echo ""
echo "============================================"
echo "Job completed: $(date)"
echo "============================================"

# Print results summary if it exists
if [ -f $RUN_DIR/config_summary.txt ]; then
    echo ""
    echo "=== RESULTS SUMMARY ==="
    cat $RUN_DIR/config_summary.txt
    echo ""
    echo "Results in: $RUN_DIR"
fi
