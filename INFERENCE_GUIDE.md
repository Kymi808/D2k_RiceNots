# Inference & Model Guide

## Overview

The Mamba CFD Surrogate Model predicts 5 aerothermal surface quantities on the Apollo capsule during atmospheric reentry. Given 4 flight condition parameters, it returns predictions at ~49,700 surface mesh points in approximately 3 seconds on a single GPU.

| Input | Description | Unit | Training Range |
|-------|-------------|------|---------------|
| velocity | Freestream velocity | m/s | 3,000 – 11,000 |
| density | Atmospheric density | kg/m³ | 1.57e-5 – 8.21e-3 |
| aoa | Angle of attack | degrees | 152 – 158 |
| dynamic_pressure | ½ρV² | Pa | varies with V and ρ |

| Output | Description | Unit | Typical Range |
|--------|-------------|------|--------------|
| qw | Heat flux | W/m² | 1,500 – 6,400,000 |
| pw | Surface pressure | Pa | 1 – 96,000 |
| tw | Wall shear stress | Pa | 0.06 – 400 |
| me | Edge Mach number | dimensionless | 0.008 – 3.6 |
| theta | Momentum thickness | m | 1e-6 – 0.024 |

---

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- NumPy, scikit-learn
- GPU optional (CPU works, ~30s inference instead of ~3s)

### Loading the Model

```python
from inference import MambaSurrogate

# Load once — takes a few seconds
surrogate = MambaSurrogate('packaged_model/')
# Output: MambaSurrogate loaded: 244,256 params, 49698 mesh points, outputs: ['qw', 'pw', 'tw', 'me', 'theta'], device: cuda
```

### Making a Prediction

```python
results = surrogate.predict(
    velocity=7500.0,           # m/s
    density=0.003,             # kg/m³
    aoa=155.0,                 # degrees
    dynamic_pressure=84375.0   # Pa
)

# Access results — all numpy arrays in physical units
qw = results['qw']       # (49698,) heat flux in W/m²
pw = results['pw']        # (49698,) pressure in Pa
tw = results['tw']        # (49698,) shear stress in Pa
me = results['me']        # (49698,) edge Mach number
theta = results['theta']  # (49698,) momentum thickness in m
xyz = results['xyz']      # (49698, 3) mesh coordinates in meters
```

### Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

xyz = results['xyz']
r_perp = np.sqrt(xyz[:, 1]**2 + xyz[:, 2]**2)

# Side view of heat flux
plt.scatter(xyz[:, 0], r_perp, c=np.log10(results['qw']), cmap='jet', s=0.3)
plt.colorbar(label='log10(qw)')
plt.xlabel('X (m)')
plt.ylabel('r (m)')
plt.title('Heat Flux Distribution')
plt.axis('equal')
plt.show()

# Front view of pressure
plt.scatter(xyz[:, 1], xyz[:, 2], c=np.log10(results['pw']), cmap='jet', s=0.3)
plt.colorbar(label='log10(pw)')
plt.xlabel('Y (m)')
plt.ylabel('Z (m)')
plt.title('Pressure Distribution')
plt.axis('equal')
plt.show()
```

### Sweeping Multiple Conditions

```python
# Trajectory sweep: vary velocity at fixed altitude
import numpy as np

velocities = np.linspace(4000, 10000, 20)
density = 0.003
aoa = 155.0

max_qw = []
mean_qw = []
for v in velocities:
    q_inf = 0.5 * density * v**2
    r = surrogate.predict(velocity=v, density=density, aoa=aoa, dynamic_pressure=q_inf)
    max_qw.append(r['qw'].max())
    mean_qw.append(r['qw'].mean())

plt.plot(velocities, max_qw, 'r-o', label='Max qw')
plt.plot(velocities, mean_qw, 'b-s', label='Mean qw')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Heat Flux (W/m²)')
plt.legend()
plt.title('Heat Flux vs Velocity')
plt.show()
```

### Batch Prediction

```python
conditions = [
    {'velocity': 4000, 'density': 0.001, 'aoa': 154, 'dynamic_pressure': 8000},
    {'velocity': 7500, 'density': 0.003, 'aoa': 155, 'dynamic_pressure': 84375},
    {'velocity': 10000, 'density': 0.006, 'aoa': 156, 'dynamic_pressure': 300000},
]

results_list = surrogate.predict_batch(conditions)
for i, r in enumerate(results_list):
    print(f"Condition {i}: max qw = {r['qw'].max():.0f} W/m²")
```

### Command Line Usage

```bash
python inference.py --model_dir packaged_model/ \
    --velocity 7500 --density 0.003 --aoa 155 --dynamic_pressure 84375
```

---

## Packaged Model Contents

The `packaged_model/` directory contains everything needed for inference:

| File | Size | Description |
|------|------|-------------|
| `model_weights.pt` | 986 KB | Trained neural network parameters (244,256 weights) |
| `scaler_X.pkl` | ~600 B | Input feature scaler (StandardScaler fitted on training data) |
| `scaler_y.pkl` | ~600 B | Target scaler (log10 + StandardScaler fitted on training data) |
| `mesh_xyz_sorted.npy` | 583 KB | Pre-sorted Apollo capsule mesh (49,698 points × 3 coordinates) |
| `config.json` | ~800 B | Model architecture configuration |

**Total: ~1.6 MB** — the entire deployable model fits on a floppy disk.

### config.json

```json
{
  "n_features": 7,
  "d_model": 64,
  "d_state": 64,
  "d_conv": 4,
  "n_layers": 4,
  "latent_dim": 16,
  "expand": 2,
  "block_type": "mamba3",
  "use_rope": true,
  "use_trapezoidal": true,
  "seq_len": 8192,
  "partition_stride": 6400,
  "points_per_solution": 50176,
  "y_col_names": ["qw", "pw", "tw", "me", "theta"]
}
```

---

## How Inference Works Internally

When you call `surrogate.predict(velocity, density, aoa, dynamic_pressure)`, the following happens:

### Step 1: Build Input Array
The 4 freestream values are replicated across all 49,698 mesh points and concatenated with the pre-stored XYZ coordinates:
```
input[i] = [X_i, Y_i, Z_i, velocity, density, aoa, dynamic_pressure]
```
Result: `(49698, 7)` array

### Step 2: Partition
The mesh is already spatially sorted (geodesic spiral from stagnation point outward). It gets sliced into 8 overlapping windows:
```
Partition 1: points     0 –  8,191
Partition 2: points 6,400 – 14,591
Partition 3: points 12,800 – 20,991
...
Partition 8: remaining points (padded to 8,192)
```
Overlap = 1,792 points between adjacent windows.

### Step 3: Scale
Each partition is standardized using `scaler_X` (fitted on training data):
```
scaled = (raw - mean) / std
```

### Step 4: Forward Pass
Each of the 8 partitions is fed through the model independently:
```
(1, 8192, 7) → MambaAutoencoder → 5 outputs, each (1, 8192, 1)
```
The model processes the sequence through:
1. Input projection (7 → 64 dims)
2. 4× Mamba blocks (spatial context via selective SSM)
3. 5 prediction heads (64 → 1 dim each)

### Step 5: Inverse Transform
Predictions are in standardized log10 space. Convert back to physical units:
```
physical = 10^(prediction × std + mean)
```
Using `scaler_y` statistics from training.

### Step 6: Overlap Averaging
Points covered by multiple partitions get their predictions averaged:
```
final_prediction[point] = mean(predictions from all partitions covering this point)
```
Most points are covered by 1 partition. Points in overlap zones are covered by 2.

### Step 7: Return Results
A dictionary with 5 output arrays plus mesh coordinates, all in physical units.

---

## Repackaging for a Different Model

If you train a new model (different split, different config), package it with:

```bash
# Default (80/10/10, seed 123)
python package_model.py --checkpoint path/to/best_model.pt --output my_packaged_model/

# With config overrides
python package_model.py --checkpoint path/to/best_model.pt --output my_packaged_model/ \
    --split_seed 456 --train_frac 0.60 --val_frac 0.20

# QW-only model
python package_model.py --checkpoint path/to/best_model.pt --output my_packaged_model/ \
    --qw_only
```

The scalers and mesh are regenerated from the training data to match the model's configuration. The `--split_seed`, `--train_frac`, and `--val_frac` must match what was used during training, otherwise the scalers will be wrong.

---

## Retraining for a New Vehicle

To train a surrogate for a different capsule geometry:

### 1. Prepare Data
Create a CSV with the same column structure:
- `X, Y, Z` — mesh coordinates (meters)
- `velocity (m/s)`, `density (kg/m^3)`, `aoa (degrees)`, `dynamic_pressure (Pa)` — freestream conditions
- `qw (W/m^2)`, `pw (Pa)`, `tauw (Pa)`, `Me`, `theta (m)` — surface quantities
- Each solution must have the same number of mesh points
- Multiple solutions at different flight conditions

### 2. Update Config
In `config.py`, set `points_per_solution` to match your mesh. Adjust `seq_len` and `partition_stride` if needed (larger meshes may need larger windows).

### 3. Train
```bash
sbatch slurm_train.sh
```
Or with custom data path:
```bash
torchrun --nproc_per_node=4 train.py --data path/to/new_data.csv --no_compile
```

### 4. Package
```bash
python package_model.py --checkpoint checkpoints/run_XXXXX/best_model.pt \
    --data path/to/new_data.csv --output packaged_model_new/
```

### 5. Use
```python
surrogate = MambaSurrogate('packaged_model_new/')
results = surrogate.predict(velocity=..., density=..., aoa=..., dynamic_pressure=...)
```

The architecture, spatial sorting, and partitioning pipeline work for any convex capsule shape. For non-convex or winged vehicles, the geodesic spiral sorting may need adjustment.

---

## Performance

| Metric | Value |
|--------|-------|
| Inference time (L40S GPU) | ~3.3 seconds |
| Inference time (CPU) | ~30 seconds |
| Model size | 1.6 MB total |
| Parameters | 244,256 |
| Mesh points | 49,698 |
| Outputs | 5 (qw, pw, tw, Me, theta) |

### Accuracy (Best Model — 80/10/10, seed 456, fully converged)

| Output | Within ±5% | Median Error | 95th Percentile |
|--------|-----------|-------------|----------------|
| qw | 99.5% | 0.62% | 2.5% |
| pw | 97.3% | 0.87% | 3.8% |
| tw | 98.7% | 0.67% | 2.8% |
| Me | 99.9% | 0.35% | 1.5% |
| theta | 98.0% | 0.89% | 3.4% |

### Inference Test Suite

71/71 tests passing. Tests cover:
- Model loading and correct output shapes
- No NaN or Inf values in predictions
- Physical plausibility across 5 flight conditions
- Monotonicity (heating increases with velocity and density)
- Determinism (identical results on repeated runs)
- Spatial pattern checks (concentrated heating, spatial variation)
- Performance benchmarks

Run the test suite:
```bash
# On NOTS
sbatch slurm_package_and_test.sh

# Locally (slower on CPU)
python test_inference/run_tests.py
```

---

## Important Limitations

### Do NOT use this model for:

1. **Flight conditions outside the training envelope**
   - Velocity < 3,000 or > 11,000 m/s
   - Density < 1.5e-5 or > 8.2e-3 kg/m³
   - AoA < 152° or > 158°
   - The model will produce predictions without warning, but they will be unreliable

2. **Separated flow regions**
   - ~2% of the capsule surface (wake/recirculation region) is excluded
   - The model has no predictions for points with theta < 0

3. **Vehicle geometries other than Apollo**
   - The mesh and scalers are specific to this capsule
   - Different vehicles require retraining with new CFD data

4. **Safety-critical decisions without CFD validation**
   - This is a surrogate for rapid exploration, not a replacement for CFD
   - Critical design points should always be validated with full simulation
   - No uncertainty quantification — the model cannot tell you where it's uncertain

### The model IS suitable for:

- Rapid design space exploration
- Identifying critical flight conditions for targeted CFD analysis
- Sensitivity studies (how does heating change with velocity, altitude, AoA?)
- Preliminary TPS sizing with appropriate safety factors
- Trajectory optimization loops requiring fast aerothermal estimates

---

## API Reference

### `MambaSurrogate(model_dir, device=None)`

**Constructor.** Loads the packaged model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_dir` | str | required | Path to packaged model directory |
| `device` | str | auto | `'cuda'` or `'cpu'` (auto-detects GPU) |

### `surrogate.predict(velocity, density, aoa, dynamic_pressure)`

**Predict surface quantities for one flight condition.**

| Parameter | Type | Unit | Description |
|-----------|------|------|-------------|
| `velocity` | float | m/s | Freestream velocity |
| `density` | float | kg/m³ | Atmospheric density |
| `aoa` | float | degrees | Angle of attack |
| `dynamic_pressure` | float | Pa | Dynamic pressure (½ρV²) |

**Returns:** `dict` with keys `'qw'`, `'pw'`, `'tw'`, `'me'`, `'theta'`, `'xyz'`. Each value is a numpy array of shape `(N,)` except `'xyz'` which is `(N, 3)`. All values in physical units.

### `surrogate.predict_batch(conditions)`

**Predict for multiple flight conditions.**

| Parameter | Type | Description |
|-----------|------|-------------|
| `conditions` | list[dict] | List of dicts, each with keys: velocity, density, aoa, dynamic_pressure |

**Returns:** `list[dict]` — same format as `predict()` for each condition.

### `surrogate.sweep(velocities, densities, aoas, dynamic_pressures)`

**Predict for a grid of conditions.** All arguments must be arrays of the same length.

**Returns:** `list[dict]` — same format as `predict()` for each condition.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `surrogate.output_names` | list[str] | Active output names (e.g., `['qw', 'pw', 'tw', 'me', 'theta']`) |
| `surrogate.n_points` | int | Number of mesh points (49,698) |
| `surrogate.device` | torch.device | Device the model runs on |
| `surrogate.mesh_xyz` | np.ndarray | Mesh coordinates, shape `(N, 3)` |
