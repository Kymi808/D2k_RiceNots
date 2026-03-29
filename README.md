# Mamba-Enhanced Physics-Informed Autoencoder for Apollo Reentry CFD Surrogate Modeling

A deep learning surrogate model that predicts aerothermal surface quantities on the Apollo capsule during atmospheric reentry, replacing expensive CFD simulations with sub-second inference.

## Best Results

Fully converged strong physics model (seed 456, 300 epochs, 40h training) evaluated on 19 held-out CFD solutions (~935,000 surface points):

| Output | Within ±1% | Within ±3% | Within ±5% | Within ±10% | Median Error | 95th %ile |
|--------|-----------|-----------|-----------|------------|-------------|----------|
| Heat Flux qw (W/m²) | 75.6% | 98.1% | **99.8%** | 100.0% | 0.55% | 2.2% |
| Pressure pw (Pa) | 63.8% | 94.3% | **97.4%** | 99.5% | 0.71% | 3.4% |
| Shear Stress τw (Pa) | 72.6% | 97.3% | **99.4%** | 99.9% | 0.58% | 2.4% |
| Edge Mach Me (-) | 87.8% | 99.5% | **99.9%** | 100.0% | 0.34% | 1.5% |
| Momentum Thickness θ (m) | 59.1% | 94.5% | **98.3%** | 99.5% | 0.80% | 3.2% |

### Physics Ablation — Fully Converged (seed 456, 300 epochs, long partition)

| Output | No Physics | Normal Physics | Strong Physics (5x) |
|--------|-----------|---------------|---------------------|
| qw ±5% | 99.5% | 99.5% | **99.8%** |
| qw ±1% | 70.3% | 70.4% | **75.6%** |
| pw ±5% | 96.4% | 97.3% | **97.4%** |
| tw ±5% | 98.9% | 98.7% | **99.4%** |
| Me ±5% | 99.8% | 99.9% | **99.9%** |
| θ ±5% | 98.0% | 98.0% | **98.3%** |
| Val loss | 0.002194 | 0.002107 | **0.001900** |

With sufficient training time AND abundant data, strong physics (5x lambda) outperforms both no-physics and normal physics on every metric.

### Physics Ablation — All Splits (fully converged)

| Split | Converged? | No Physics | Normal Physics | Strong Physics (5x) | MLP Baseline |
|-------|-----------|-----------|---------------|---------------------|-------------|
| 80/10/10 (seed 456, 300ep) | Yes | 99.5% | 99.5% | **99.8%** | — |
| 80/10/10 (seed 123, ~155ep) | Cutoff | **98.5%** | 98.0% | 98.4% | 96.8% |
| 60/20/20 (early stop) | Yes | **96.8%** | 96.3% | 95.8% | — |
| 40/30/30 (300ep) | Yes | **92.9%** | 92.1% | 92.2% | — |

Physics constraints help when there is **both** abundant data (148 solutions) **and** sufficient training time (300 epochs). With less data (60-40%), no-physics wins even with full convergence — the constraints are too rigid for the model to fit limited data effectively.

### Comparison to Previous Approaches

| Model | qw ±5% | Architecture |
|-------|--------|-------------|
| **This work (best)** | **99.8%** | Mamba-3 SSM, strong physics, full mesh |
| MLP Baseline (same pipeline) | 96.8% | Pointwise MLP, same training pipeline |
| Simple Autoencoder | 94.5% | Pointwise dense layers |
| Mamba (downsampled) | 94.4% | Mamba-3 SSM, 4K subsample |
| Two-Stage NN (no leakage) | 79.6% | Physics-aware two-stage MLP |

## Data Efficiency

How many CFD simulations does NASA need? Accuracy degrades gracefully as training data decreases:

![Data Efficiency](partition_graph/accuracy_vs_data_all_metrics.png)

| Split | Train Solutions | qw ±5% | pw ±5% | tw ±5% | Me ±5% | θ ±5% | Test Solutions |
|-------|----------------|--------|--------|--------|--------|-------|---------------|
| 80/10/10 | 148 | 99.5% | 97.3% | 98.7% | 99.9% | 98.0% | 19 |
| 70/15/15 | 130 | 96.2% | 96.2% | 98.2% | 99.7% | 95.4% | 27 |
| 60/20/20 | 111 | 96.3% | 94.7% | 97.3% | 99.4% | 94.5% | 37 |
| 50/25/25 | 92 | 94.5% | 92.5% | 96.0% | 99.2% | 94.2% | 47 |
| 40/30/30 | 74 | 92.1% | 90.8% | 95.5% | 98.9% | 91.5% | 55 |

The model trained on 50% of the data matches the accuracy of the previous pointwise baseline trained on 80%. Even at 40% (74 solutions), all metrics remain above 90%.

![Data Efficiency Summary](partition_graph/data_efficiency_summary.png)

### Key Findings

- **Physics losses require both data and time**: Strong physics (5x) achieves the best results (99.8% qw ±5%) with 148 training solutions and 300 epochs. With less data (60-40%), no-physics wins even with full convergence — physics constraints become too rigid. With limited training time (24h), no-physics wins even at 80% data because the model hasn't recovered from the physics warmup disruption.
- **Architecture matters**: Mamba SSM (99.8%) outperforms MLP baseline (96.8%) by 3.0% — spatial context from sequential modeling captures patterns that pointwise models miss
- **Training pipeline contributes more than architecture**: MLP baseline (96.8%) beats the previous simple autoencoder (94.5%) using the same pointwise approach — the improvement comes from overlapping partitions, Huber loss, gradient accumulation, and the full training pipeline
- **Graceful degradation**: Reducing training data from 148 to 74 solutions costs only ~7% qw accuracy
- **Cross-validation consistency**: Two random seeds (123, 456) produce consistent results, confirming robustness

## Interactive Demo

A Streamlit web app provides interactive 3D visualization of predictions:

```bash
pip install streamlit plotly
streamlit run app.py
```

- Sliders for velocity, density, and angle of attack
- Switchable 3D interactive and 2D side views
- Hover over any mesh point to see exact values
- Switch between all 5 output quantities instantly (cached predictions)
- ~3s inference on GPU, ~60-120s on CPU

## Inference

The packaged model predicts all 5 surface quantities for any flight condition in ~3.3 seconds on a single GPU. Inference test suite passes **71/71** checks covering loading, physical plausibility, monotonicity, determinism, spatial patterns, and performance.

![Inference Surface Maps](test_inference/results/inference_surface_maps.png)

![Velocity Sweep](test_inference/results/velocity_sweep.png)

![Multi-Condition Comparison](test_inference/results/multi_condition_comparison.png)

### Quick Start

```bash
# Package the trained model (one time)
python package_model.py --checkpoint organized_results/full_model_long/best_model.pt \
                        --output packaged_model/ --split_seed 456
```

```python
from inference import MambaSurrogate

# Load once (~3s)
surrogate = MambaSurrogate('packaged_model/')

# Predict for any flight condition (~3s per prediction)
results = surrogate.predict(velocity=7500, density=0.003, aoa=155, dynamic_pressure=84375)

qw = results['qw']      # (49698,) heat flux in W/m^2
pw = results['pw']       # (49698,) pressure in Pa
tw = results['tw']       # (49698,) shear stress in Pa
me = results['me']       # (49698,) edge Mach number
theta = results['theta'] # (49698,) momentum thickness in m
xyz = results['xyz']     # (49698, 3) mesh coordinates for visualization

# Sweep a trajectory
for v in [4000, 6000, 8000, 10000]:
    r = surrogate.predict(velocity=v, density=0.003, aoa=155, dynamic_pressure=0.5*0.003*v**2)
    print(f"V={v}: max qw = {r['qw'].max():.0f} W/m^2")
```

## Installation

### Dependencies
```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and PyTorch 2.0+. GPU (CUDA) is optional — inference works on CPU (~60-120s) or GPU (~3s).

### Data Setup

The CFD database (`apollo_cfd_database.csv`, ~2.2GB) is not included in the repository due to its size. To set up:

1. Create the data directory: `mkdir -p data/`
2. Place `apollo_cfd_database.csv` in `data/`
3. The CSV must contain columns: `X, Y, Z, velocity (m/s), density (kg/m^3), aoa (degrees), dynamic_pressure (Pa), qw (W/m^2), pw (Pa), tauw (Pa), Me, theta (m), Re-theta`
4. Each solution must have exactly 50,176 mesh points

The pretrained model in `packaged_model/` can be used for inference without the training data.

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
6. **Split**: By solution (no data leakage — scalers fitted on training data only)

## Training

- **Optimizer**: AdamW (lr=1e-3, weight_decay=8e-3)
- **Loss**: Weighted Huber (smooth L1) on standardized log10 targets + reconstruction loss + optional physics constraints
- **Gradient accumulation**: 4 steps (effective batch size 16 across 4 GPUs)
- **LR schedule**: 5-epoch linear warmup → ReduceLROnPlateau (factor=0.5, patience=10)
- **Early stopping**: patience=25 on validation loss
- **Hardware**: 4× NVIDIA L40S (48GB each) on Rice NOTS cluster via DDP
- **Training time**: 20-40 hours depending on configuration

## Experiments

| Experiment | Description |
|-----------|-------------|
| `slurm_train.sh` | Full model — 5 outputs, physics losses, Mamba-3 |
| `slurm_no_physics.sh` | Ablation — same model, no physics loss terms |
| `slurm_qw_only.sh` | Ablation — single output (heat flux only) |
| `slurm_mlp.sh` | MLP baseline — pointwise blocks, no sequential modeling |
| `slurm_70_15.sh` | Data efficiency — 70/15/15 train/val/test split |
| `slurm_60_20.sh` | Data efficiency — 60/20/20 train/val/test split |
| `slurm_50_25.sh` | Data efficiency — 50/25/25 train/val/test split |
| `slurm_40_30.sh` | Data efficiency — 40/30/30 train/val/test split |
| `slurm_no_physics_60.sh` | No physics at 60/20/20 |
| `slurm_no_physics_40.sh` | No physics at 40/30/30 |
| `slurm_strong_physics_80.sh` | Strong physics (5x lambdas) at 80/10/10 |
| `slurm_strong_physics_60.sh` | Strong physics (5x lambdas) at 60/20/20 |
| `slurm_strong_physics_40.sh` | Strong physics (5x lambdas) at 40/30/30 |
| `slurm_seed2_full.sh` | Cross-validation fold 2 — full model, seed 456 |
| `slurm_strong_physics_80_seed2.sh` | Strong physics fold 2 — seed 456, long partition |
| `slurm_error_maps.sh` | Generate spatial error heatmaps for all models |
| `slurm_package_and_test.sh` | Package best model and run inference tests |

## Project Structure

```
├── config.py                  # Model and training configuration
├── model.py                   # MambaAutoencoder architecture
├── dataset.py                 # Data loading, spatial sorting, partitioning
├── train.py                   # DDP training loop
├── evaluate.py                # Overlap-averaged evaluation and plotting
├── eval_checkpoint.py         # Standalone checkpoint evaluation
├── physics_losses.py          # Physics-informed loss constraints
├── create_error_maps.py       # Spatial error heatmap generation
├── package_model.py           # Package model for production deployment
├── inference.py               # Production inference wrapper (MambaSurrogate class)
├── app.py                     # Interactive Streamlit 3D visualization demo
├── requirements.txt           # Python dependencies
├── TECHNICAL_REFERENCE.md     # Comprehensive technical documentation
├── INFERENCE_GUIDE.md         # Inference API and deployment guide
├── slurm_*.sh                 # SLURM job scripts for NOTS cluster
├── data/                      # Apollo CFD database (CSV)
├── packaged_model/            # Production-ready model (weights, scalers, mesh, config)
├── test_inference/            # Inference test suite (71/71 passing)
├── organized_results/         # Training logs, checkpoints, error maps, evaluation plots
│   ├── full_model/            # 80/10/10, physics, seed 123
│   ├── no_physics/            # 80/10/10, no physics, seed 123
│   ├── qw_only/               # 80/10/10, qw only, seed 123
│   ├── full_model_long/       # 80/10/10, physics, seed 456 (fully converged, best)
│   ├── mlp_baseline/          # 80/10/10, MLP blocks (no sequential modeling)
│   ├── no_physics_60/         # 60/20/20, no physics
│   ├── no_physics_40/         # 40/30/30, no physics
│   ├── strong_physics_80/     # 80/10/10, strong physics (5x lambdas)
│   ├── strong_physics_60/     # 60/20/20, strong physics (5x lambdas)
│   ├── strong_physics_40/     # 40/30/30, strong physics (5x lambdas)
│   ├── full_70_15/            # 70/15/15 split
│   ├── full_60_20/            # 60/20/20 split
│   ├── full_50_25/            # 50/25/25 split
│   └── full_40_30/            # 40/30/30 split
└── partition_graph/           # Data efficiency plots (accuracy vs training data)
```

## Limitations

- **Separated flow excluded**: ~2% of surface points (theta < 0) in wake/recirculation region
- **Interpolation only**: Trained on Mach 10-35, AoA 152-158° — no extrapolation guarantees
- **Fixed geometry**: Apollo capsule only — different vehicles require retraining (architecture is geometry-agnostic but untested on other shapes)
- **No uncertainty quantification**: Point predictions without confidence intervals
- **Cross-validation**: 2-fold (seeds 123 and 456) — sufficient for semester scope, additional folds planned
