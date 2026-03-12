#!/bin/bash
# ============================================================
# Setup conda environment on Rice NOTS cluster
# Run this ONCE before submitting SLURM jobs:
#   bash setup_env.sh
# ============================================================

module purge
module load Miniforge3/24.1.2-0

# Create environment
conda create -n mamba-cfd python=3.11 -y
conda activate mamba-cfd

# PyTorch with CUDA (check available CUDA version with: module spider CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install numpy pandas scikit-learn matplotlib

echo ""
echo "Environment setup complete."
echo "Activate with: conda activate mamba-cfd"
echo ""
echo "Before submitting jobs, verify GPU access:"
echo "  srun --partition=commons --gres=gpu:lovelace:1 --time=00:05:00 --pty bash"
echo "  conda activate mamba-cfd"
echo "  python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
