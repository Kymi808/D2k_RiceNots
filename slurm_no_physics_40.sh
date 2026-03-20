#!/bin/bash
# ============================================================
# SLURM job: No-physics model at 40/30/30
# ============================================================
#SBATCH --job-name=nophys-40
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:lovelace:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/nophys_40_%j.out
#SBATCH --error=logs/nophys_40_%j.err
#SBATCH --mail-type=END,FAIL

echo "============================================"
echo "Job ID: $SLURM_JOB_ID  [NO PHYSICS 40/30/30]"
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

torchrun --nproc_per_node=4 train.py \
    --data data/apollo_cfd_database.csv \
    --train_frac 0.40 \
    --val_frac 0.30 \
    --no_physics \
    --save_dir $RUN_DIR \
    --checkpoint_dir $CKPT_DIR \
    --no_compile

echo ""
echo "Job completed: $(date)"
if [ -f $RUN_DIR/config_summary.txt ]; then
    cat $RUN_DIR/config_summary.txt
fi
