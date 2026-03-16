#!/bin/bash
# ============================================================
# SLURM job: Evaluate a saved checkpoint (short job, 1 GPU)
# Usage: sbatch slurm_eval.sh <checkpoint_path> [extra_args]
# Example: sbatch slurm_eval.sh checkpoints/run_7516324/best_model.pt
# Example: sbatch slurm_eval.sh checkpoints/run_7516326/best_model.pt --qw_only
# ============================================================
#SBATCH --job-name=mamba-eval
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

CKPT=${1:?Usage: sbatch slurm_eval.sh <checkpoint_path> [extra_args]}
shift
EXTRA_ARGS="$@"

echo "============================================"
echo "Job ID: $SLURM_JOB_ID  [EVAL]"
echo "Checkpoint: $CKPT"
echo "Extra args: $EXTRA_ARGS"
echo "Date:   $(date)"
echo "============================================"

mkdir -p logs

module purge
module load Miniforge3/24.1.2-0

PROJECT_DIR=/projects/dsci435/NASA_REENTRY_SP26
ENV_DIR=$PROJECT_DIR/.conda/envs/mamba-cfd
export PATH=$ENV_DIR/bin:$PATH
export PYTHONPATH=$ENV_DIR/lib/python3.11/site-packages:$PYTHONPATH

$ENV_DIR/bin/python eval_checkpoint.py \
    --checkpoint $CKPT \
    --data data/apollo_cfd_database.csv \
    $EXTRA_ARGS

echo ""
echo "Eval completed: $(date)"
