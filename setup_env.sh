#!/bin/bash
# ============================================================
# Setup conda environment on Rice NOTS cluster
# Installs to project directory to avoid home quota issues
# Run: bash setup_env.sh
# ============================================================

PROJECT_DIR=/projects/dsci435/NASA_REENTRY_SP26

module purge
module load Miniforge3/24.1.2-0

# Put conda envs and pip cache under project dir
export CONDA_ENVS_PATH=$PROJECT_DIR/.conda/envs
export CONDA_PKGS_DIRS=$PROJECT_DIR/.conda/pkgs
export PIP_CACHE_DIR=$PROJECT_DIR/.pip_cache
export PIP_TARGET=$PROJECT_DIR/.conda/envs/mamba-cfd/lib/python3.11/site-packages
mkdir -p $CONDA_ENVS_PATH $CONDA_PKGS_DIRS $PIP_CACHE_DIR

# Create environment in project dir
conda create -p $PROJECT_DIR/.conda/envs/mamba-cfd python=3.11 -y
conda activate $PROJECT_DIR/.conda/envs/mamba-cfd

# Install into the conda env directly (not ~/.local)
pip install --target=$PIP_TARGET --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install --target=$PIP_TARGET --no-cache-dir \
    numpy pandas scikit-learn matplotlib

echo ""
echo "Environment setup complete."
echo ""
echo "To activate:"
echo "  module load Miniforge3/24.1.2-0"
echo "  conda activate $PROJECT_DIR/.conda/envs/mamba-cfd"
