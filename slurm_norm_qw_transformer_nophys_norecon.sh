#!/bin/bash
# ============================================================
# SLURM job: normalized qw Transformer, no physics/reconstruction
# Override example:
#   TRAIN_FRAC=0.60 VAL_FRAC=0.20 sbatch slurm_norm_qw_transformer_nophys_norecon.sh
# ============================================================
#SBATCH --job-name=nqw-transformer
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:lovelace:4
#SBATCH --time=72:00:00
#SBATCH --output=logs/norm_qw_transformer_nophys_norecon_%j.out
#SBATCH --error=logs/norm_qw_transformer_nophys_norecon_%j.err
#SBATCH --mail-type=END,FAIL

TRAIN_FRAC=${TRAIN_FRAC:-0.80}
VAL_FRAC=${VAL_FRAC:-0.10}
SPLIT_SEED=${SPLIT_SEED:-456}
QW_WEIGHT=${QW_WEIGHT:-6.0}
N_HEADS=${N_HEADS:-4}
TRANSFORMER_FFN_DIM=${TRANSFORMER_FFN_DIM:-128}
ATTENTION_DROPOUT=${ATTENTION_DROPOUT:-0.05}
FFN_DROPOUT=${FFN_DROPOUT:-0.05}

SPLIT_LABEL="${TRAIN_FRAC}_${VAL_FRAC}"
SPLIT_LABEL="${SPLIT_LABEL//./}"

echo "============================================"
echo "Job ID: $SLURM_JOB_ID  [NORM QW / TRANSFORMER / NO PHYS / NO RECON]"
echo "Split:  train=$TRAIN_FRAC val=$VAL_FRAC seed=$SPLIT_SEED"
echo "Heads:  $N_HEADS"
echo "FFN:    $TRANSFORMER_FFN_DIM"
echo "qw w:   $QW_WEIGHT"
echo "Date:   $(date)"
echo "============================================"

RUN_DIR=results/norm_qw_transformer_nophys_norecon_${SPLIT_LABEL}_${SLURM_JOB_ID}
CKPT_DIR=checkpoints/norm_qw_transformer_nophys_norecon_${SPLIT_LABEL}_${SLURM_JOB_ID}
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
    --block_type transformer \
    --train_frac "$TRAIN_FRAC" \
    --val_frac "$VAL_FRAC" \
    --split_seed "$SPLIT_SEED" \
    --n_heads "$N_HEADS" \
    --transformer_ffn_dim "$TRANSFORMER_FFN_DIM" \
    --attention_dropout "$ATTENTION_DROPOUT" \
    --ffn_dropout "$FFN_DROPOUT" \
    --pred_head_dims 128,64 \
    --pred_head_dropout 0.05 \
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
