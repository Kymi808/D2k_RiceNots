# HPC Setup: Rice NOTS Cluster

Guide for reproducing all experiments on the Rice NOTS cluster.

## Cluster Overview

| Resource | Specification |
|----------|--------------|
| Cluster | Rice NOTS (Night Owls Time-Sharing Service) |
| GPUs | NVIDIA L40S (48GB VRAM each) |
| GPUs per job | 4 |
| Partitions used | `commons` (24h limit), `long` (72h limit), `scavenge` (preemptible) |
| Scheduler | SLURM |
| Conda environment | `/projects/dsci435/NASA_REENTRY_SP26/.conda/envs/mamba-cfd` |

## Environment Setup

The conda environment is shared at the project directory. SLURM scripts configure the environment automatically — no manual setup needed. The key environment variables set in each script:

```bash
module purge
module load Miniforge3/24.1.2-0

PROJECT_DIR=/projects/dsci435/NASA_REENTRY_SP26
ENV_DIR=$PROJECT_DIR/.conda/envs/mamba-cfd
export PATH=$ENV_DIR/bin:$PATH
export PYTHONPATH=$ENV_DIR/lib/python3.11/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=1
```

**Note**: `python` fails to import PyTorch on compute nodes due to module conflicts. All scripts use `torchrun` instead, which resolves the correct Python binary internally.

## Data Location

The CFD database is stored at the shared project directory:
```
/projects/dsci435/NASA_REENTRY_SP26/D2k_RiceNots/data/apollo_cfd_database.csv
```
Size: ~2.2GB (9,282,560 rows × 15 columns)

## Submitting Jobs

```bash
# SSH into NOTS
ssh <netid>@nots.rice.edu

# Navigate to project
cd /projects/dsci435/NASA_REENTRY_SP26/D2k_RiceNots

# Ensure logs directory exists
mkdir -p logs

# Submit a job
sbatch slurm_train.sh

# Monitor
squeue -u $USER
squeue -u $USER --start    # estimated start time for pending jobs

# Check job output
tail -f logs/train_*.out

# Check GPU usage on a running job's node
ssh <nodename>             # e.g. ssh bb6u21g1
nvidia-smi
```

## SLURM Scripts Reference

### Training

| Script | Description | Partition | Time |
|--------|-------------|-----------|------|
| `slurm_train.sh` | Full model (physics, 80/10/10) | commons | 24h |
| `slurm_no_physics.sh` | No physics losses (80/10/10) | commons | 24h |
| `slurm_qw_only.sh` | Heat flux only (80/10/10) | commons | 24h |
| `slurm_mlp.sh` | MLP baseline (80/10/10) | commons | 24h |
| `slurm_70_15.sh` | 70/15/15 data split | commons | 24h |
| `slurm_60_20.sh` | 60/20/20 data split | commons | 24h |
| `slurm_50_25.sh` | 50/25/25 data split | commons | 24h |
| `slurm_40_30.sh` | 40/30/30 data split | commons | 24h |
| `slurm_no_physics_60.sh` | No physics, 60/20/20 | commons | 24h |
| `slurm_no_physics_40.sh` | No physics, 40/30/30 | commons | 24h |
| `slurm_strong_physics_80.sh` | Strong physics 5x (80/10/10) | commons | 24h |
| `slurm_strong_physics_60.sh` | Strong physics 5x (60/20/20) | commons | 24h |
| `slurm_strong_physics_40.sh` | Strong physics 5x (40/30/30) | commons | 24h |
| `slurm_seed2_full.sh` | Full model, seed 456 | long | 72h |
| `slurm_strong_physics_80_seed2.sh` | Strong physics 5x, seed 456 | long | 72h |

### Evaluation & Utilities

| Script | Description | Partition | Time |
|--------|-------------|-----------|------|
| `slurm_eval.sh <checkpoint> [args]` | Evaluate a checkpoint | scavenge | 30min |
| `slurm_error_maps.sh` | Generate error heatmaps for all models | scavenge | 1h |
| `slurm_package_and_test.sh` | Package model + run inference tests | scavenge | 30min |

### Evaluation Examples

```bash
# Basic evaluation
sbatch slurm_eval.sh checkpoints/run_JOBID/best_model.pt

# With config overrides (must match training config)
sbatch slurm_eval.sh checkpoints/run_JOBID/best_model.pt --qw_only
sbatch slurm_eval.sh checkpoints/run_JOBID/best_model.pt --train_frac 0.60 --val_frac 0.20
sbatch slurm_eval.sh checkpoints/run_JOBID/best_model.pt --split_seed 456
```

## Resource Usage

### Per-Job Requirements

```
#SBATCH --nodes=1
#SBATCH --ntasks=4              # 4 GPU processes
#SBATCH --cpus-per-task=8       # 8 CPU cores per GPU
#SBATCH --mem=192G              # total RAM
#SBATCH --gres=gpu:lovelace:4   # 4x L40S GPUs
```

### Training Time by Configuration

| Configuration | Epoch Time | Typical Epochs | Total Time |
|--------------|-----------|---------------|------------|
| 80/10/10 (1184 partitions) | ~475s | 150-300 | 20-40h |
| 70/15/15 (1040 partitions) | ~426s | 150-200 | 18-24h |
| 60/20/20 (888 partitions) | ~375s | 150-180 | 16-19h |
| 50/25/25 (736 partitions) | ~320s | 200-260 | 18-23h |
| 40/30/30 (592 partitions) | ~270s | 250-300 | 19-23h |
| MLP baseline (1184 partitions) | ~3.7s | 300 | ~19min |

### GPU Memory

- Steady-state: ~12.6 GB / 46 GB per GPU (27% utilization)
- Epoch 1 may spike higher due to random weight initialization and GradScaler calibration
- batch_size=2 causes OOM on first epoch despite fitting in steady-state

### Total Compute Used

~1,600 GPU-hours across all experiments (~$2,400 at cloud rates, free on NOTS commons partition).

## Output Structure

Each training job creates:
```
results/run_JOBID/
├── config_summary.txt      # Hyperparameters and final metrics
├── training_curves.png     # Loss curves
└── evaluation_plots.png    # Pred vs true scatter + error distributions

checkpoints/run_JOBID/
└── best_model.pt           # Best validation loss checkpoint

logs/
├── train_JOBID.out         # stdout (training progress)
└── train_JOBID.err         # stderr (warnings/errors)
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `python` fails to import torch | Module conflict on compute nodes | Use `torchrun` instead of `python` |
| `expandable_segments not supported` | L40S doesn't support this CUDA allocator feature | Warning only — ignore |
| NCCL timeout / deadlock | DDP rank desync (NaN on one GPU, not others) | Fixed in train.py — NaN backward still triggers allreduce |
| Jobs stuck in `(Priority)` | Commons partition is busy | Wait, or try `long`/`scavenge` partition |
| Disk quota exceeded | Too many checkpoints | Remove `checkpoints/` and `results/` (keep `organized_results/`) |
| 24h cutoff before evaluation | Training didn't finish in time | Run `sbatch slurm_eval.sh` separately, or use `long` partition |
