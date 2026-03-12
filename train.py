"""
Multi-GPU training script for Mamba Autoencoder CFD surrogate.
Supports DDP (torchrun) and single-GPU execution.

Usage:
  Multi-GPU:  torchrun --nproc_per_node=4 train.py
  Single-GPU: python train.py
  With args:  torchrun --nproc_per_node=4 train.py --epochs 200 --seq_len 4096
"""
import os
import sys
import time
import math
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import Config
from model import MambaAutoencoder
from dataset import get_dataloaders
from physics_losses import PhysicsLoss, compute_physics_loss
from evaluate import (evaluate_model, print_results,
                      save_training_curves, save_evaluation_plots)

warnings.filterwarnings('ignore')

# ============================================================
# DDP Setup
# ============================================================

def setup_ddp():
    """Initialize distributed training if launched with torchrun."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


# ============================================================
# Loss Computation
# ============================================================

def compute_loss(out, X_batch, Y_batch, cfg, y_col_names, y_weights,
                 physics_loss_fn, scaler_X):
    """Compute weighted data loss + reconstruction loss + physics losses."""
    loss_dict = {}
    total_data_loss = torch.tensor(0.0, device=X_batch.device)
    preds_dict = {}

    for i, name in enumerate(y_col_names):
        if name in out:
            target = Y_batch[:, :, i:i+1]
            pred = out[name]
            data_loss = F.mse_loss(pred, target)
            weighted = y_weights[i] * data_loss
            loss_dict[f'{name}_mse'] = data_loss.item()
            total_data_loss = total_data_loss + weighted
            preds_dict[name] = pred

    recon_loss = F.mse_loss(out['recon'], X_batch)
    loss_dict['recon'] = recon_loss.item()

    # Physics losses
    phys_raw = physics_loss_fn(preds_dict, X_batch, scaler_X)
    phys_total, phys_weighted = compute_physics_loss(phys_raw, cfg)
    for k, v in phys_weighted.items():
        loss_dict[f'phys_{k}'] = v

    total = total_data_loss + cfg.lambda_recon * recon_loss + phys_total
    loss_dict['total'] = total.item()
    return total, loss_dict


# ============================================================
# Training & Evaluation Epochs
# ============================================================

def train_epoch(model, dl, optimizer, scaler_amp, cfg, y_col_names, y_weights,
                physics_loss_fn, scaler_X, device, use_amp):
    model.train()
    epoch_losses = {}
    for X_batch, Y_batch in dl:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(X_batch)
            loss, loss_dict = compute_loss(
                out, X_batch, Y_batch, cfg, y_col_names, y_weights,
                physics_loss_fn, scaler_X
            )

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0) + v

    n = len(dl)
    return {k: v / n for k, v in epoch_losses.items()}


@torch.no_grad()
def eval_epoch(model, dl, cfg, y_col_names, y_weights,
               physics_loss_fn, scaler_X, device, use_amp):
    model.eval()
    epoch_losses = {}
    for X_batch, Y_batch in dl:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(X_batch)
            _, loss_dict = compute_loss(
                out, X_batch, Y_batch, cfg, y_col_names, y_weights,
                physics_loss_fn, scaler_X
            )
        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0) + v

    n = len(dl)
    return {k: v / n for k, v in epoch_losses.items()}


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='apollo_cfd_database.csv')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--d_state', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--latent_dim', type=int, default=None)
    parser.add_argument('--block_type', type=str, default=None)
    parser.add_argument('--no_compile', action='store_true')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    return parser.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size, distributed = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    # Config with CLI overrides
    cfg = Config()
    if distributed:
        cfg.num_gpus = world_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.d_model is not None:
        cfg.d_model = args.d_model
    if args.d_state is not None:
        cfg.d_state = args.d_state
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    if args.latent_dim is not None:
        cfg.latent_dim = args.latent_dim
    if args.block_type is not None:
        cfg.block_type = args.block_type

    if is_main(rank):
        print(f"{'=' * 60}")
        print(f"  Mamba Autoencoder CFD — {'DDP' if distributed else 'Single GPU'}")
        print(f"{'=' * 60}")
        print(f"  Device: {device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
        print(f"  World size: {world_size}")
        print(f"  Block type: {cfg.block_type}")
        print(f"  seq_len: {cfg.seq_len}, partitions: {cfg.n_partitions}")
        print(f"  d_model: {cfg.d_model}, d_state: {cfg.d_state}, n_layers: {cfg.n_layers}")
        print(f"  Targets: {cfg.y_col_names}")
        print(f"  Target weights: {dict(zip(cfg.y_col_names, cfg.y_weights))}")
        print(f"  Batch size: {cfg.batch_size} (per GPU: {cfg.batch_per_gpu})")
        print(f"  Epochs: {cfg.epochs}, LR: {cfg.lr}")
        print()

    # Data
    (train_dl, val_dl, test_dl, scaler_X, scaler_y,
     Y_test_raw, meta_test, train_sampler) = get_dataloaders(
        cfg, args.data, distributed=distributed, rank=rank, world_size=world_size
    )

    y_col_names = cfg.y_col_names
    y_weights = cfg.y_weights

    # Model
    model = MambaAutoencoder(cfg).to(device)

    if is_main(rank):
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {n_params:,} trainable parameters")
        print(f"Prediction heads: {list(model.pred_heads.keys())}")

    # torch.compile
    if not args.no_compile and torch.cuda.is_available():
        model = torch.compile(model, mode="reduce-overhead")
        if is_main(rank):
            print("torch.compile applied")

    # DDP wrap
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # Physics loss
    physics_loss_fn = PhysicsLoss(scaler_y, y_col_names, cfg).to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # AMP
    use_amp = device.type == 'cuda'
    scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Dirs
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # === Training Loop ===
    if is_main(rank):
        header = (f"{'Ep':>4} | {'Train':>9} | {'Val':>9} | {'qw MSE':>9} | "
                  f"{'pw MSE':>9} | {'tw MSE':>9} | {'LR':>9}")
        print(f"\n{header}")
        print("-" * 80)

    history = {
        'train_loss': [], 'val_loss': [], 'val_qw_mse': [],
        'val_pw_mse': [], 'val_tw_mse': [], 'val_me_mse': [],
        'val_theta_mse': [],
    }
    best_val_loss = float('inf')
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_losses = train_epoch(
            model, train_dl, optimizer, scaler_amp, cfg,
            y_col_names, y_weights, physics_loss_fn, scaler_X, device, use_amp
        )
        val_losses = eval_epoch(
            model, val_dl, cfg, y_col_names, y_weights,
            physics_loss_fn, scaler_X, device, use_amp
        )
        elapsed = time.time() - t0

        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_losses['total'])
        for name in y_col_names:
            key = f'val_{name}_mse'
            if key in history:
                history[key].append(val_losses.get(f'{name}_mse', 0))

        scheduler.step(val_losses['total'])
        lr = optimizer.param_groups[0]['lr']

        # Save best
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            if is_main(rank):
                state = model.module.state_dict() if distributed else model.state_dict()
                torch.save(state, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            patience_counter = 0
            marker = ' *'
        else:
            patience_counter += 1
            marker = ''

        if is_main(rank) and (epoch <= 5 or epoch % 10 == 0 or marker):
            qw_m = val_losses.get('qw_mse', 0)
            pw_m = val_losses.get('pw_mse', 0)
            tw_m = val_losses.get('tw_mse', 0)
            print(f"{epoch:4d} | {train_losses['total']:9.4f} | {val_losses['total']:9.4f} | "
                  f"{qw_m:9.4f} | {pw_m:9.4f} | {tw_m:9.4f} | {lr:9.2e}{marker}  ({elapsed:.1f}s)")

        if patience_counter >= cfg.patience:
            if is_main(rank):
                print(f"\nEarly stopping at epoch {epoch}")
            break

    total_time = time.time() - t_start
    if is_main(rank):
        print(f"\nTraining complete in {total_time / 60:.1f} minutes. Best val loss: {best_val_loss:.4f}")

    # === Load Best & Evaluate ===
    if is_main(rank):
        # Load best checkpoint on a fresh (non-compiled, non-DDP) model for eval
        eval_model = MambaAutoencoder(cfg).to(device)
        eval_model.load_state_dict(
            torch.load(os.path.join(args.checkpoint_dir, 'best_model.pt'),
                       weights_only=True, map_location=device)
        )

        print("\n" + "=" * 60)
        print("  EVALUATION ON TEST SET")
        print("=" * 60)

        test_results = evaluate_model(
            eval_model, test_dl, scaler_y, y_col_names, Y_test_raw, device
        )
        print_results(test_results, y_col_names, cfg.block_type)

        # Save plots
        save_training_curves(history, args.save_dir)
        save_evaluation_plots(test_results, y_col_names, args.save_dir)

        # Save config summary
        with open(os.path.join(args.save_dir, 'config_summary.txt'), 'w') as f:
            f.write(f"block_type: {cfg.block_type}\n")
            f.write(f"seq_len: {cfg.seq_len}\n")
            f.write(f"n_partitions: {cfg.n_partitions}\n")
            f.write(f"d_model: {cfg.d_model}\n")
            f.write(f"d_state: {cfg.d_state}\n")
            f.write(f"n_layers: {cfg.n_layers}\n")
            f.write(f"latent_dim: {cfg.latent_dim}\n")
            f.write(f"targets: {y_col_names}\n")
            f.write(f"target_weights: {dict(zip(y_col_names, y_weights))}\n")
            f.write(f"batch_size: {cfg.batch_size}\n")
            f.write(f"epochs_run: {epoch}\n")
            f.write(f"best_val_loss: {best_val_loss:.6f}\n")
            f.write(f"training_time_min: {total_time / 60:.1f}\n")
            f.write(f"world_size: {world_size}\n")
            for name in y_col_names:
                r = test_results[name]
                f.write(f"\n{name}:\n")
                f.write(f"  MAE: {r['mae']:,.0f}\n")
                f.write(f"  RMSE: {r['rmse']:,.0f}\n")
                f.write(f"  pct_1: {r['pct_1']:.1f}%\n")
                f.write(f"  pct_5: {r['pct_5']:.1f}%\n")
                f.write(f"  pct_10: {r['pct_10']:.1f}%\n")
                f.write(f"  median_err: {r['median_err']:.2f}%\n")
                f.write(f"  q95_err: {r['q95_err']:.1f}%\n")

        print(f"\nResults saved to {args.save_dir}/")

    cleanup_ddp()


if __name__ == '__main__':
    main()
