# Mamba-Enhanced Physics-Informed Autoencoder for Apollo Reentry CFD Surrogate Modeling

A deep learning surrogate model that predicts aerothermal surface quantities on the Apollo capsule during atmospheric reentry, replacing expensive CFD simulations with sub-second inference.

## Results

Evaluated on 19 held-out CFD solutions (~935,000 surface points):

| Output | No Physics ±5% | Full Model ±5% | QW Only ±5% | Median Error |
|--------|---------------|----------------|-------------|-------------|
| Heat Flux qw (W/m²) | **98.5%** | 98.0% | 98.9% | 0.74% |
| Pressure pw (Pa) | **95.3%** | 93.8% | — | 0.97% |
| Shear Stress τw (Pa) | **98.8%** | 98.0% | — | 0.76% |
| Edge Mach Me (-) | **99.8%** | 99.6% | — | 0.48% |
| Momentum Thickness θ (m) | **95.1%** | 94.8% | — | 1.13% |

Compared to previous approaches on the same dataset:

| Model | qw ±5% | Architecture |
|-------|--------|-------------|
| **This work (Mamba, full mesh)** | **98.5%** | Mamba-3 SSM, overlapping partitions |
| Simple Autoencoder | 94.5% | Pointwise dense layers |
| Mamba (downsampled) | 94.4% | Mamba-3 SSM, 4K subsample |
| Two-Stage NN (no leakage) | 79.6% | Physics-aware two-stage MLP |

## Architecture

**MambaAutoencoder** (244K parameters):
- Input projection: 7 features → 64-dim embedding
- Encoder: 4× Mamba-3 blocks with selective SSM, RoPE, trapezoidal discretization
- Latent bottleneck: 64 → 16 dims (regularizes via reconstruction loss)
- 5 prediction heads: branch from encoder output (64-dim), one per surface quantity
- Reconstruction head: from latent (16-dim), reconstructs input features

Key design choices:
- **Spatial sorting**: Mesh points ordered by geodesic spiral on capsule surface, converting 3D surface data into a meaningful 1D sequence
- **Overlapping partitions**: 50K-point solutions split into 8 windows (seq_len=8192, stride=6400, overlap=1792) — preserves full mesh resolution
- **Parallel scan**: O(L log L) computation via doubling trick, enabling 8K sequences on GPU
- **Mamba-3 extensions**: Data-dependent RoPE for non-uniform spatial encoding, trapezoidal discretization for sharp gradient capture

## Data Pipeline

1. **Input**: 185 Apollo reentry CFD solutions × 50,176 surface points × 7 features
2. **Cleaning**: Remove separated flow regions (theta < 0) — ~2% of points
3. **Spatial sort**: Geodesic spiral from stagnation point outward (64 latitude bands)
4. **Partitioning**: Overlapping sliding windows with padding for last partition
5. **Scaling**: log10 transform + StandardScaler on targets; StandardScaler on inputs
6. **Split**: 80/10/10 by solution (148 train, 18 val, 19 test)

## Training

- **Optimizer**: AdamW (lr=1e-3, weight_decay=8e-3)
- **Loss**: Weighted Huber (smooth L1) on standardized log10 targets + reconstruction loss + optional physics constraints
- **Gradient accumulation**: 4 steps (effective batch size 16 across 4 GPUs)
- **LR schedule**: 5-epoch linear warmup → ReduceLROnPlateau (factor=0.5, patience=10)
- **Early stopping**: patience=25 on validation loss
- **Hardware**: 4× NVIDIA L40S (48GB each) on Rice NOTS cluster via DDP

## Experiments

| Experiment | Description |
|-----------|-------------|
| `slurm_train.sh` | Full model — 5 outputs, physics losses, Mamba-3 |
| `slurm_no_physics.sh` | Ablation — same model, no physics loss terms |
| `slurm_qw_only.sh` | Ablation — single output (heat flux only) |
| `slurm_70_15.sh` | Data efficiency — 70/15/15 train/val/test split |
| `slurm_60_20.sh` | Data efficiency — 60/20/20 train/val/test split |
| `slurm_seed2_full.sh` | Cross-validation fold 2 — full model, seed 456 |
| `slurm_seed2_no_physics.sh` | Cross-validation fold 2 — no physics, seed 456 |

## Usage

### Training on NOTS
```bash
# Submit a training job
sbatch slurm_train.sh

# Monitor progress
squeue -u $USER
tail -f logs/train_*.out
```

### Evaluating a checkpoint
```bash
# Evaluate a saved model on the test set
sbatch slurm_eval.sh checkpoints/run_JOBID/best_model.pt

# For qw-only model
sbatch slurm_eval.sh checkpoints/run_JOBID/best_model.pt --qw_only

# For different split seed
sbatch slurm_eval.sh checkpoints/run_JOBID/best_model.pt --split_seed 456
```

### Inference
```python
import torch
from config import Config
from model import MambaAutoencoder

cfg = Config()
model = MambaAutoencoder(cfg)
model.load_state_dict(torch.load('best_model.pt', weights_only=True))
model.eval()

# x: (1, 8192, 7) — one partition of scaled input features
with torch.no_grad():
    out = model(x)
    qw_pred = out['qw']  # (1, 8192, 1) in standardized log10 space
```

## Project Structure

```
├── config.py              # Model and training configuration
├── model.py               # MambaAutoencoder architecture
├── dataset.py             # Data loading, spatial sorting, partitioning
├── train.py               # DDP training loop
├── evaluate.py            # Overlap-averaged evaluation and plotting
├── eval_checkpoint.py     # Standalone checkpoint evaluation
├── physics_losses.py      # Physics-informed loss constraints
├── slurm_*.sh             # SLURM job scripts for NOTS cluster
├── data/                  # Apollo CFD database (CSV)
└── organized_results/     # Training logs, checkpoints, and evaluation plots
```
