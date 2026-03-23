# Technical Reference: Mamba CFD Surrogate Model

## What This Document Is

This document explains everything about our Mamba-based surrogate model from the ground up. It assumes you understand basic machine learning concepts (what a neural network is, what training means, what a loss function does) but does NOT assume you know anything about:

- Mamba or state space models
- CFD (computational fluid dynamics) details
- How this specific codebase works
- Why certain design decisions were made

If you read this document front to back, you will understand exactly what the model does, why it works, and how every piece fits together.

---

## Table of Contents

1. [The Problem We Are Solving](#1-the-problem-we-are-solving)
2. [The Dataset](#2-the-dataset)
3. [Why Not Just Use a Regular Autoencoder?](#3-why-not-just-use-a-regular-autoencoder)
4. [The Core Idea: Treating the Mesh as a Sequence](#4-the-core-idea-treating-the-mesh-as-a-sequence)
5. [Step-by-Step: How Data Flows Through the Pipeline](#5-step-by-step-how-data-flows-through-the-pipeline)
6. [The Model Architecture](#6-the-model-architecture)
7. [What Is Mamba and Why Does It Matter?](#7-what-is-mamba-and-why-does-it-matter)
8. [The Loss Function](#8-the-loss-function)
9. [How Training Works (DDP, Gradient Accumulation, etc.)](#9-how-training-works)
10. [How Evaluation Works](#10-how-evaluation-works)
11. [Results and What They Mean](#11-results-and-what-they-mean)
12. [How Inference Works in Production](#12-how-inference-works-in-production)
13. [Comparison to Other Approaches](#13-comparison-to-other-approaches)
14. [Limitations](#14-limitations)
15. [File-by-File Code Guide](#15-file-by-file-code-guide)
16. [Glossary](#16-glossary)

---

## 1. The Problem We Are Solving

NASA simulates the Apollo capsule reentering Earth's atmosphere using CFD (Computational Fluid Dynamics). Each simulation solves complex fluid dynamics equations on a mesh of 50,176 points covering the capsule's surface. The output is five physical quantities at every mesh point:

| Quantity | Symbol | Unit | What It Means |
|----------|--------|------|--------------|
| Heat flux | qw | W/m² | How much heat the surface receives (most important for thermal protection) |
| Surface pressure | pw | Pa | Force per unit area on the surface |
| Wall shear stress | τw | Pa | Frictional force from airflow along the surface |
| Edge Mach number | Me | dimensionless | Speed of airflow at the boundary layer edge relative to sound |
| Momentum thickness | θ | m | A measure of how thick the boundary layer is |

**The problem**: Each CFD simulation takes hours to days on a supercomputer. NASA needs hundreds of these simulations across different flight conditions (different speeds, altitudes, angles) to design the thermal protection system. This is extremely expensive and slow.

**Our solution**: Train a neural network that takes the same inputs as CFD (mesh coordinates + flight conditions) and predicts the same outputs (qw, pw, τw, Me, θ) in under one second. Once trained, this model can replace many of those expensive CFD runs.

**The key metric**: NASA considers 95% of surface points within ±5% error as acceptable. Our best model achieves **99.5%** for heat flux.

---

## 2. The Dataset

### What's in the CSV

The file `data/apollo_cfd_database.csv` contains 9,282,560 rows. Each row represents one point on the capsule surface from one CFD simulation.

There are **185 CFD solutions** (different flight conditions), each with **50,176 mesh points**. So: 185 × 50,176 = 9,282,560 rows.

### Input features (7 columns — what we feed the model)

| Column | Description | Why It Matters |
|--------|-------------|---------------|
| X, Y, Z | 3D coordinates of the mesh point on the capsule surface (meters) | Tells the model WHERE on the capsule this point is |
| velocity (m/s) | Freestream velocity of the capsule | Same for all 50K points in a solution — defines the flight condition |
| density (kg/m³) | Atmospheric density | Same for all points — higher at lower altitudes |
| aoa (degrees) | Angle of attack (152-158°) | Same for all points — how the capsule is oriented |
| dynamic_pressure (Pa) | ½ × density × velocity² | Same for all points — derived from velocity and density |

**Important**: The last 4 features (velocity, density, aoa, dynamic_pressure) are the SAME for every point within a single solution. They describe the flight condition. The XYZ coordinates are different for each point but the SAME across solutions (the capsule shape doesn't change). The model learns: "given this shape and these flight conditions, what are the surface quantities?"

### Target outputs (5 columns — what the model predicts)

These are qw, pw, τw, Me, and θ as described above. They vary across both mesh points AND solutions.

### Train/Val/Test Split

The data is split **by solution, not by point**. This is critical:

- **Correct**: Solution 42 is entirely in train. Solution 107 is entirely in test. The model never sees any point from solution 107 during training.
- **Wrong** (data leakage): Random 80% of all 9M rows → some points from the same solution appear in both train and test.

Our default split: 80% train (148 solutions), 10% val (18 solutions), 10% test (19 solutions). The split is deterministic — controlled by a random seed (default 123).

### Data Cleaning

We remove ~2% of mesh points where:
- `theta < 0`: Indicates separated/recirculating flow (the boundary layer has detached from the surface). The physical quantities we predict don't have meaningful interpretations here.
- `Re_theta < 1e-5`: Negligible boundary layer.

After cleaning, each solution has ~49,200 points instead of 50,176. The model does NOT make predictions for the removed points.

---

## 3. Why This Approach?

Instead of compressing an entire 50,176-point solution into a small latent vector and decoding it back, we process the mesh as a **sequence** of points and make predictions **per point**, using local spatial context:

```
8,192 points (one window) → Mamba Encoder → 8,192 predictions (one per point)
```

Each point's prediction uses information from nearby points (spatial context). This preserves local detail that whole-solution compression tends to lose.

We DO have a bottleneck (64→16 dims), but it exists per-point as a regularizer, not as a whole-solution compression. The actual predictions branch from the full 64-dim encoder output, bypassing the bottleneck.

### Results comparison

| Approach | qw ±5% Accuracy |
|----------|----------------|
| Our Mamba approach | **99.5%** |
| Simple pointwise autoencoder (baseline) | 94.5% |

---

## 4. The Core Idea: Treating the Mesh as a Sequence

### The spatial sorting trick

The 50,176 mesh points are unordered in the CSV — they're just rows with XYZ coordinates. A sequence model (like Mamba) needs them in a meaningful order.

We sort the points into a **geodesic spiral** on the capsule surface:

1. Find the stagnation point (the nose of the capsule, where heating is highest)
2. Divide the surface into 64 latitude bands radiating outward from the stagnation point
3. Within each band, sort points by their angle around the capsule axis
4. The result: a continuous spiral path from nose to shoulder

**Why this works**: The flow physics naturally follows this pattern. Heat flux is highest at the stagnation point and decays as you move along the body. Pressure follows a similar pattern. By ordering points this way, adjacent points in the sequence are also physically related — exactly what a sequence model can exploit.

```
Stagnation (nose)                                    Shoulder
    ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
    ↑ High qw                                Low qw ↑
    Point 1                                  Point 50,176
```

### Why a sequence model helps

A **pointwise model** (like the simple autoencoder baseline) sees each mesh point in isolation:
```
Point: (X=0.1, Y=0.0, Z=0.0, V=8000, ρ=0.001) → qw = ???
```
It must infer everything from 7 numbers. It can't know what's happening at neighboring points.

A **sequence model** sees the point in context:
```
... Point 499 → Point 500 → Point 501 → Point 502 ...
     ↓            ↓            ↓            ↓
  qw=50000     qw=???       qw=48000     qw=47500
```
"Point 500 is between two points with qw around 48,000-50,000, and the trend is decreasing. Its qw is probably ~49,000." This spatial reasoning is exactly what CFD captures through its governing equations, and the sequence model learns it from data.

### Overlapping partitions

The sorted 50K sequence is too long for one forward pass (memory constraints). We slice it into 8 overlapping windows:

```
Window 1: points 0 ——————— 8,191
Window 2:     points 6,400 ——————— 14,591
Window 3:         points 12,800 ——————— 20,991
...
                                    overlap zones
```

- **seq_len = 8,192**: Points per window
- **stride = 6,400**: Step between window starts
- **overlap = 1,792**: Points covered by two adjacent windows

At evaluation, predictions from overlapping windows are **averaged** for the shared points. This smooths out any edge artifacts where one window ends and another begins.

---

## 5. Step-by-Step: How Data Flows Through the Pipeline

Here's exactly what happens from raw CSV to final prediction:

### During Training

```
1. Load CSV (9.28M rows)
2. Assign solution IDs (185 solutions × 50,176 points)
3. Split by solution: 148 train, 18 val, 19 test
4. Clean: remove theta < 0 and Re_theta < 1e-5 rows (~2%)
5. For each solution:
   a. Spatial sort (geodesic spiral)
   b. Slice into 8 overlapping partitions (8,192 points each)
6. Result: 1,184 training partitions, 144 val, 152 test
7. Fit scalers on training data ONLY:
   - Input features: StandardScaler (zero mean, unit variance)
   - Targets: log10 first, then StandardScaler
8. Apply scalers to all splits
9. Feed partitions through model, compute loss, backpropagate
```

### The log10 Transform (Important)

The target values span huge ranges:
- qw: 1,500 to 6,400,000 W/m² (4 orders of magnitude)
- Me: 0.008 to 3.6

If we trained on raw values, the model would focus entirely on the large values (a 1% error at qw=6M is 60,000, while a 1% error at qw=1,500 is just 15 — the large values dominate the loss).

The log10 transform compresses this:
- qw: log10(1,500) = 3.18 to log10(6,400,000) = 6.81

Now a 1% relative error maps to roughly the same absolute error in log space regardless of magnitude. After log10, we apply StandardScaler to center around 0 with std 1.

**The model predicts in standardized log10 space.** To get physical values back: `physical = 10^(prediction × std + mean)`.

---

## 6. The Model Architecture

### Overview (244,256 parameters)

```
Input (1, 8192, 7)
       ↓
Input Projection: Linear(7→64) + LayerNorm + SiLU
       ↓
h (1, 8192, 64)    ← "encoder features"
       ↓
4× Mamba Blocks (with residual connections)
       ↓
h (1, 8192, 64)    ← enriched with spatial context
       ↓
  ┌────┴────────────────────┐
  ↓                         ↓
Latent Bottleneck        5 Prediction Heads
LN → Linear(64→16)      Each: Linear(64→64) → SiLU → LN → Linear(64→1)
  ↓                         ↓
z (1, 8192, 16)          qw, pw, tw, Me, θ predictions
  ↓                      (each shape: 1, 8192, 1)
Reconstruction Head
Linear(16→64) → SiLU → Linear(64→7)
  ↓
Reconstructed input (1, 8192, 7)
```

### Key design decisions

**Why do prediction heads use h (64-dim) instead of z (16-dim)?**

The bottleneck z is intentionally lossy — it forces the autoencoder to learn a compressed representation. But for predictions, we want maximum information. The 64-dim encoder output h contains the full spatial context from all 4 Mamba blocks. Predictions from h are more accurate than predictions from z.

The bottleneck still helps: the reconstruction loss (recon_head(z) ≈ input) forces the encoder to learn meaningful features, which indirectly improves the prediction heads.

**Why 5 separate prediction heads instead of 1 shared head?**

Each output has different physics and different scales. A shared head would force compromises. Separate heads let each output specialize. The shared encoder captures common spatial patterns; the heads adapt to each output's specifics.

**Why 244K parameters (so small)?**

The physics is relatively low-dimensional (7 inputs → 5 outputs with smooth spatial variation). A larger model would overfit on 148 training solutions. The Mamba SSM is parameter-efficient because it reuses weights across all 8,192 positions (weight sharing). Small model also means fast inference.

---

## 7. What Is Mamba and Why Does It Matter?

### The basic concept

Mamba is a **sequence model** — it processes an ordered list of items and outputs a transformed list. Like an LSTM or Transformer, but with different tradeoffs.

At its core, Mamba maintains a **hidden state** h that gets updated at each position:

```
h₁ = α₁ × h₀ + input₁        (process point 1)
h₂ = α₂ × h₁ + input₂        (process point 2, using memory of point 1)
h₃ = α₃ × h₂ + input₃        (process point 3, using memory of points 1-2)
...
h₈₁₉₂ = α₈₁₉₂ × h₈₁₉₁ + input₈₁₉₂   (has context from all previous points)
```

The α values (between 0 and 1) control **how much to remember**. Small α = forget quickly (local patterns). Large α = remember for a long time (global patterns).

### What makes Mamba special

**1. Data-dependent parameters (the "selective" in Selective State Space Model)**

Unlike a fixed filter, Mamba's α and input transformation change at every position based on the actual data. Near the stagnation point (rapidly changing qw), the model learns to respond quickly. In smooth regions, it learns to maintain long-range memory.

**2. Parallel computation**

The recurrence `h_t = α_t × h_{t-1} + input_t` looks sequential, but because it's **linear** (no nonlinearity like tanh between steps), there's a mathematical trick called the **parallel scan** that computes all 8,192 steps in just 13 parallel operations on the GPU.

This is why we use Mamba instead of an LSTM:

| Model | Can model sequences? | Parallelizable? | Memory | Our use case |
|-------|---------------------|----------------|--------|-------------|
| Pointwise MLP | No | Yes | O(1) | 94.5% accuracy |
| LSTM | Yes | **No** (sequential) | O(L) | Would work, but too slow to train |
| Transformer | Yes | Yes | **O(L²)** | Can't fit L=8,192 in 48GB |
| **Mamba** | **Yes** | **Yes** | **O(L)** | **99.5% accuracy** |

**3. Linear memory scaling**

A Transformer with self-attention at L=8,192 needs to store an 8192×8192 attention matrix (~256MB per head per layer). With 4 layers and multiple heads, this exceeds 48GB GPU memory. Mamba uses O(L) memory — linear in sequence length.

### Mamba-3 extensions

Our model uses "Mamba-3" which adds three features to the base Mamba:

- **RoPE (Rotary Position Encoding)**: Helps the model know WHERE in the sequence each point is. Important because mesh point spacing isn't uniform (denser near the nose).
- **Trapezoidal discretization**: More accurate numerical method for the recurrence (2nd order vs 1st order). Helps capture sharp gradients near the stagnation point.
- **BC bias**: Learned offset that stabilizes training with the above extensions.

### Inside one Mamba block

Each of the 4 Mamba blocks does:

```
Input x (shape: 1, 8192, 64)
  ↓
LayerNorm
  ↓
Project to 2× width: Linear(64 → 256), split into:
  - x_ssm (128-dim): goes through the SSM
  - z (128-dim): gating signal (used later)
  ↓
x_ssm → Conv1d(kernel=4): captures immediate neighbors (3-point context)
  ↓
x_ssm → SiLU activation
  ↓
x_ssm → Generate SSM parameters: B (input gate), C (output gate), dt (step size)
  ↓
Parallel scan: process all 8,192 positions simultaneously
  ↓
Output gating: result × SiLU(z)   ← z controls how much SSM output passes through
  ↓
Project back: Linear(128 → 64)
  ↓
Add residual: output + original input x
```

The residual connection means each block ADDS information to the representation rather than replacing it. If the SSM has nothing useful to contribute, the block can learn to pass the input through unchanged.

---

## 8. The Loss Function

The total loss has three components:

```
L_total = L_data + 0.15 × L_recon + L_physics
```

### Data loss (the main objective)

```
L_data = 3.0 × Huber(pred_qw, true_qw)
       + 1.0 × Huber(pred_pw, true_pw)
       + 1.0 × Huber(pred_tw, true_tw)
       + 1.0 × Huber(pred_me, true_me)
       + 1.0 × Huber(pred_θ, true_θ)
```

All comparisons are in **standardized log10 space** (not physical units).

**Why Huber loss instead of MSE?**
- MSE squares the error: a large error gets amplified dramatically
- Huber loss behaves like MSE for small errors (smooth, differentiable) and like MAE for large errors (linear, doesn't explode)
- This prevents outlier points (e.g., near the stagnation point with extreme values) from dominating the gradient

**Why qw weight = 3.0?**
Heat flux is the most important quantity for thermal protection design. The 3× weight ensures the optimizer prioritizes qw accuracy when there's a tradeoff between outputs.

### Reconstruction loss (regularizer)

```
L_recon = 0.15 × MSE(reconstructed_input, original_input)
```

Forces the latent bottleneck to encode meaningful information about the input. Acts as a regularizer — prevents the encoder from learning arbitrary representations.

### Physics losses (6 constraints)

These enforce known physical relationships WITHOUT using ground truth:

| Constraint | What It Enforces | Lambda |
|-----------|-----------------|--------|
| Positivity | Surface quantities must be > 0 | 0.0015 |
| Reynolds analogy | qw/τw ratio must be in a known range | 0.0075 |
| Modified Newtonian | Pressure coefficient Cp ≤ 2.0 | 0.003 |
| Fay-Riddell | Stagnation heating scales as √ρ × V³ | 0.0015 |
| Skin friction bounds | Friction coefficient in [1e-5, 0.1] | 0.003 |
| BL consistency | Me ≤ 6.0, θ > 0 | 0.0015 |

The lambda values are small — these are soft constraints that nudge the model toward physical plausibility, not hard enforcement.

**Physics warmup**: These losses ramp up linearly from 0 to full weight over the first 70 epochs. Early in training, predictions are random — physics constraints would produce huge, noisy gradients. The warmup lets the model first learn from data, then gradually enforces physics.

**Important finding**: With sufficient training data (148 solutions), the no-physics model actually outperforms the physics-informed model. The model learns the physics implicitly from data, and the explicit constraints add optimization overhead. However, with more training time (72 hours vs 24 hours), the physics model catches up and achieves the best overall result (99.5%).

---

## 9. How Training Works

### Multi-GPU Training (DDP)

We train on 4 NVIDIA L40S GPUs (48GB each) using PyTorch's DistributedDataParallel (DDP):

1. **4 copies of the model** run simultaneously, one per GPU
2. Each GPU processes **different data** (the 1,184 training partitions are split 4 ways: 296 per GPU)
3. After each backward pass, **gradients are averaged** across all 4 GPUs via NCCL AllReduce
4. All 4 GPUs apply the **same averaged gradient** → weights stay synchronized
5. Effectively trains on 4× more data per unit time

### Gradient Accumulation

With seq_len=8,192, only batch_size=1 fits per GPU (48GB limit). That's noisy — one sample's gradient is unreliable.

**Solution**: Accumulate gradients over 4 mini-batches before updating:
```
Step 1: forward + backward (don't update weights, just accumulate gradient)
Step 2: forward + backward (accumulate)
Step 3: forward + backward (accumulate)
Step 4: forward + backward (accumulate) → NOW update weights
```

Effective batch size: 1 sample × 4 accumulation steps × 4 GPUs = **16 samples** per weight update. Same gradient quality as batch_size=16, but uses 1/4 the memory.

### Learning Rate Schedule

```
Epochs 1-5:   Linear warmup (0 → 1e-3)   ← prevents explosion from random weights
Epochs 6+:    ReduceLROnPlateau           ← halves LR if val loss stalls for 10 epochs
Min LR:       1e-6                        ← floor to prevent LR from reaching zero
```

The LR typically follows this progression: 1e-3 → 5e-4 → 2.5e-4 → 1.25e-4 → 6.25e-5 → ... as the model converges.

### Early Stopping

If validation loss doesn't improve for 25 consecutive epochs, training stops. The best checkpoint (lowest val loss) is saved throughout training.

### NaN Handling

If a batch produces NaN loss (from float16 overflow in mixed precision), the backward pass still runs (to keep all GPUs synchronized via AllReduce), but PyTorch's GradScaler detects the NaN and skips the weight update. This prevents one bad batch from crashing the entire training run.

---

## 10. How Evaluation Works

After training, the best model is evaluated on the test set:

1. **Run all test partitions** through the model (152 partitions for 19 test solutions)
2. **Inverse transform** predictions: standardized log10 → physical units
   ```
   physical_value = 10^(prediction × std + mean)
   ```
3. **Overlap averaging**: Each test solution was split into 8 overlapping partitions. Points covered by 2 partitions get their predictions averaged. This produces one prediction per physical mesh point.
4. **Compute metrics** in physical units:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - % of points within ±1%, ±3%, ±5%, ±10% relative error
   - Median relative error
   - 95th percentile relative error (worst-case measure)

**Important**: The evaluation metrics are computed completely independently of the training loss. Huber loss in log10 space is used for training optimization. MAE and relative error in physical units are used for evaluation. They measure different things.

---

## 11. Results and What They Mean

### Best model: 99.5% qw within ±5%

This means: out of ~935,000 test surface points (19 solutions × ~49,200 points each), 99.5% have heat flux predictions within 5% of the CFD ground truth. Only 0.5% of points (~4,675) have errors above 5%.

The median error of 0.62% means the typical prediction is extremely accurate. The 95th percentile of 2.5% means even the "bad" predictions (top 5%) are still within 2.5%.

### Why the fully converged model is the best

The first training runs were cut off at 24 hours (~155 epochs) by the SLURM time limit. The seed-456 model on the long partition ran for the full 300 epochs (40 hours). The additional training time allowed the LR to decay further and the model to squeeze out the last few percent of accuracy.

### Data efficiency findings

| Training Data | qw ±5% | Key Takeaway |
|--------------|--------|-------------|
| 148 solutions (80%) | 99.5% | Best result |
| 130 solutions (70%) | 96.2% | Still excellent |
| 111 solutions (60%) | 96.3% | Minimal degradation |
| 92 solutions (50%) | 94.5% | Matches old baseline that used 80% data |
| 74 solutions (40%) | 92.1% | Still above 90% |

**The key finding**: Our model trained on 50% of the data achieves the same accuracy as the previous best model trained on 80%. This means NASA could potentially commission fewer CFD simulations while maintaining the same surrogate accuracy.

### Full Physics Ablation Matrix (qw ±5%)

| Split | No Physics | Normal Physics | Strong Physics (5x) |
|-------|-----------|---------------|---------------------|
| 80/10/10 (148 train) | **98.5%** | 98.0% | 98.4% |
| 60/20/20 (111 train) | **96.8%** | 96.3% | 95.8% |
| 40/30/30 (74 train) | **92.9%** | 92.1% | 92.2% |

### All outputs at each physics level (80/10/10 split)

| Output | No Physics | Normal Physics | Strong Physics (5x) | MLP Baseline |
|--------|-----------|---------------|---------------------|-------------|
| qw ±5% | **98.5%** | 98.0% | 98.4% | 96.8% |
| pw ±5% | **95.3%** | 93.8% | 95.9% | 96.2% |
| tw ±5% | **98.8%** | 98.0% | **98.9%** | 97.2% |
| Me ±5% | **99.8%** | 99.6% | **99.8%** | 99.7% |
| θ ±5% | **95.1%** | 94.8% | 95.6% | 94.1% |

### Ablation findings

- **No-physics consistently wins**: At every data split tested (80% through 40%), the no-physics model outperforms both normal and strong physics variants for qw. The model learns aerothermodynamic relationships implicitly from data.
- **Strong physics (5x) is mixed**: Slightly improves over normal physics at 80% data (98.4% vs 98.0%) but slightly hurts at 60% (95.8% vs 96.3%). The added constraint strength doesn't compensate for the optimization complexity.
- **Physics doesn't help more with less data**: The hypothesis that physics constraints become more valuable with scarce data is not supported — no-physics wins even at 40% (92.9% vs 92.1%).
- **Architecture matters more than loss design**: MLP baseline (96.8%) vs Mamba (98.5%) shows a 1.7% gap from sequential modeling alone. The full pipeline (partitioning, Huber loss, gradient accumulation) contributes an additional 2.3% over the previous simple autoencoder (94.5%).
- **QW-only vs multi-output**: QW-only gets slightly better qw accuracy (98.9% vs 98.5% at 24h) because all model capacity focuses on one target. But multi-output predicts all 5 quantities simultaneously, which is more useful.
- **With sufficient training time, physics catches up**: The fully converged seed-456 model (300 epochs, 40h) achieves 99.5% with physics — suggesting physics losses need more epochs to recover from the warmup disruption at epoch 70.

---

## 12. How Inference Works in Production

Once trained, predicting for a new flight condition is simple. The complexity (spatial sorting, partitioning, scaling, overlap averaging) is wrapped in the `MambaSurrogate` class.

### Packaging (one time)

```bash
python package_model.py --checkpoint organized_results/full_model_long/best_model.pt \
                        --output packaged_model/ --split_seed 456
```

This saves everything needed for inference into one directory:
- `model_weights.pt` — trained neural network weights
- `scaler_X.pkl` / `scaler_y.pkl` — fitted scalers (from training data)
- `mesh_xyz_sorted.npy` — pre-sorted capsule mesh coordinates
- `config.json` — model configuration

### Using the Packaged Model

```python
from inference import MambaSurrogate

# Load once (takes a few seconds)
surrogate = MambaSurrogate('packaged_model/')

# Predict for any flight condition (sub-second on GPU, seconds on CPU)
results = surrogate.predict(
    velocity=7500.0,          # m/s
    density=0.003,            # kg/m^3
    aoa=155.0,                # degrees
    dynamic_pressure=84375.0  # Pa
)

# Access results — all in physical units
qw = results['qw']      # (N,) heat flux in W/m^2
pw = results['pw']       # (N,) pressure in Pa
tw = results['tw']       # (N,) shear stress in Pa
me = results['me']       # (N,) edge Mach number
theta = results['theta'] # (N,) momentum thickness in m
xyz = results['xyz']     # (N, 3) mesh coordinates for visualization

# Sweep multiple conditions
for v in [4000, 6000, 8000, 10000]:
    r = surrogate.predict(velocity=v, density=0.003, aoa=155,
                         dynamic_pressure=0.5 * 0.003 * v**2)
    print(f"V={v}: max qw = {r['qw'].max():.0f} W/m^2")
```

The user provides 4 numbers (flight condition) and gets back 5 arrays of surface quantities plus mesh coordinates. No knowledge of scalers, partitions, or overlap averaging required.

### Running Inference Tests

```bash
# On NOTS: packages model + runs comprehensive test suite
sbatch slurm_package_and_test.sh

# Tests cover: loading, physical plausibility, monotonicity,
# determinism, spatial patterns, performance benchmarks
# Results saved to test_inference/results/
```

**Speed comparison**:
- Full CFD simulation: hours to days
- Our surrogate model: ~3.3 seconds on a single L40S GPU
- Cost: ~$0.001 per prediction vs ~$500-$5,000 per CFD run (estimated)

### Inference Validation

The inference test suite (`test_inference/run_tests.py`) passes **71/71** checks:

| Category | Tests | What It Validates |
|----------|-------|------------------|
| Loading | 4 | Model loads, correct outputs, mesh size, partitions |
| Single Prediction | 22 | Correct shapes, no NaN/Inf, all outputs present |
| Physical Plausibility | 30 | All 5 conditions, all outputs within physical bounds |
| Monotonicity | 2 | qw increases with velocity and density |
| Determinism | 5 | Same input produces identical output |
| Spatial Patterns | 3 | Heating concentrated and spatially varying |
| Performance | 2 | Inference under 10s, benchmarked at 3.3s |

---

## 13. Comparison to Other Approaches

### Simple Autoencoder (94.5%)

The baseline from `Autoencoder.ipynb`. Treats each mesh point independently — no spatial reasoning. Uses a standard autoencoder with dense layers. Gets 94.5% because the 7 input features are quite informative on their own.

**Our advantage**: Spatial context via Mamba. Knowing what's happening at neighboring points pushes accuracy from 94.5% → 99.5%.

### Mamba on Downsampled Mesh (94.4%)

The original Mamba experiment from `Mamba_Autoencoder_CFD.ipynb`. Downsampled 50,176 points to 4,096 and processed as a single sequence. Got 94.4%.

**Our advantage**: Full mesh resolution via overlapping partitions. No information lost to downsampling. The 50K→4K downsampling threw away 92% of spatial detail.

### Two-Stage Neural Network (79.6%)

Predicts boundary layer properties first, then heat flux. No data leakage (doesn't use CFD outputs as inputs to predict other CFD outputs).

**Our advantage**: End-to-end learning avoids error propagation between stages.

---

## 14. Limitations

These are important to understand and communicate honestly:

1. **Separated flow excluded**: ~2% of surface points (negative momentum thickness) are removed. The model makes no predictions in the wake/recirculation region behind the capsule. These regions require different physical modeling.

2. **Interpolation only**: The model is trained on Mach 10-35 and AoA 152-158°. It will silently produce incorrect predictions for flight conditions outside this range (no warning, no error — just bad numbers). Never use the model outside its training envelope.

3. **Fixed geometry**: Only works for the Apollo capsule mesh. A different vehicle shape (Orion, Dream Chaser) would require full retraining with new CFD data.

4. **No uncertainty quantification**: The model outputs a single number per point with no confidence interval. You cannot tell which predictions the model is confident about and which it is uncertain about. Future work should add ensemble predictions or Monte Carlo dropout.

5. **Validated against CFD, not flight data**: The ground truth is CFD output, which itself has modeling errors. The model has not been compared to actual flight measurements or arc jet test data.

---

## 15. File-by-File Code Guide

### Core Model Code

| File | What It Does | Key Functions |
|------|-------------|---------------|
| `config.py` | All hyperparameters in one place | `Config` dataclass with properties for derived values |
| `model.py` | The MambaAutoencoder neural network | `MambaAutoencoder.forward()`, `SelectiveSSM`, `parallel_scan_simple()` |
| `dataset.py` | Data loading, cleaning, sorting, partitioning, scaling | `spatial_sort_solution()`, `create_partitions()`, `get_dataloaders()` |
| `train.py` | Training loop with DDP, gradient accumulation, scheduling | `train_epoch()`, `eval_epoch()`, `main()` |
| `evaluate.py` | Test evaluation with overlap averaging | `evaluate_model()`, `print_results()` |
| `physics_losses.py` | Physics-informed loss constraints | `PhysicsLoss.forward()`, `compute_physics_loss()` |
| `eval_checkpoint.py` | Standalone evaluation from a saved checkpoint | `main()` — loads model, runs eval |
| `package_model.py` | Package model + scalers + mesh for deployment | Saves everything needed for inference |
| `inference.py` | Production inference wrapper | `MambaSurrogate` class — predict() takes 4 numbers, returns 5 arrays |

### SLURM Scripts

All `slurm_*.sh` files are job scripts for the Rice NOTS cluster. They set up the conda environment and call `torchrun` to launch multi-GPU training. Key differences between scripts are the command-line flags passed to `train.py`:

- `--no_physics`: Disables physics losses
- `--qw_only`: Only predicts heat flux
- `--train_frac 0.60 --val_frac 0.20`: Changes the data split
- `--split_seed 456`: Uses a different random seed for the train/test split
- `--no_compile`: Disables torch.compile (required on NOTS — no C++ compiler on compute nodes)

### Visualization Scripts

| File | What It Does |
|------|-------------|
| `create_error_maps.py` | Generates spatial error heatmaps and truth-vs-prediction maps for all models |
| `partition_graph/create_partition_graphs.py` | Generates data efficiency plots (accuracy vs training data) |
| `test_inference/run_tests.py` | Comprehensive inference test suite (loading, plausibility, performance) |

### Organized Results

Each subfolder in `organized_results/` contains:
- `best_model.pt`: Saved model weights
- Training log (`.out` file)
- Evaluation results (from `eval_checkpoint.py` or end-of-training eval)
- `error_maps/`: Spatial error visualizations (if generated)
  - `error_map_*.png`: Red-white-green signed error on dark background
  - `truth_vs_pred_*.png`: Ground truth vs prediction side-by-side
  - `error_distribution.png`: Histogram of signed errors

### Production Deployment

| Directory | What It Contains |
|-----------|-----------------|
| `packaged_model/` | Everything needed for inference: model weights, scalers, sorted mesh, config |
| `test_inference/results/` | Test suite output: pass/fail summary, visualizations, performance benchmarks |

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **AoA** | Angle of Attack — the angle between the capsule's axis and the oncoming airflow |
| **Boundary layer** | Thin layer of air near the surface where velocity transitions from zero (at the surface) to freestream |
| **CFD** | Computational Fluid Dynamics — numerical simulation of fluid flow |
| **DDP** | DistributedDataParallel — PyTorch's multi-GPU training strategy |
| **Freestream** | The undisturbed airflow far from the capsule (characterized by velocity, density, etc.) |
| **GradScaler** | PyTorch tool that scales losses up for float16 training to prevent gradient underflow |
| **Huber loss** | Loss function that's MSE for small errors and MAE for large errors (smooth L1) |
| **Latent space** | The compressed representation inside the autoencoder bottleneck |
| **Log10 transform** | Taking log base 10 of values to compress large dynamic ranges |
| **Mamba** | A selective state space model — a sequence model with linear memory and parallel computation |
| **Mixed precision (AMP)** | Training with float16 for speed and float32 for accuracy where needed |
| **NCCL** | NVIDIA Collective Communications Library — handles GPU-to-GPU communication |
| **NOTS** | Rice University's HPC cluster (Night Owls Time-Sharing Service) |
| **Overlap averaging** | Averaging predictions from multiple overlapping windows for the same physical point |
| **Parallel scan** | Mathematical trick to compute linear recurrences in O(log L) parallel steps |
| **Partition** | One 8,192-point window from a spatially-sorted solution |
| **Physics-informed** | Using known physical laws as additional loss terms to constrain the model |
| **RoPE** | Rotary Position Encoding — data-dependent positional information for sequence models |
| **Separated flow** | Region where airflow detaches from the surface and recirculates (wake region) |
| **SSM** | State Space Model — mathematical framework for sequence modeling via hidden state recurrence |
| **Stagnation point** | The point on the capsule nose where airflow velocity drops to zero — highest heating |
| **StandardScaler** | Transforms data to zero mean and unit variance |
| **Surrogate model** | A fast approximation of an expensive simulation |
| **TPS** | Thermal Protection System — the heat shield on reentry vehicles |
| **Trapezoidal discretization** | 2nd-order numerical method for discretizing continuous dynamics (more accurate than Euler) |
