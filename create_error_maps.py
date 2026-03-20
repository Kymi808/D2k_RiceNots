"""
Generate spatial error heatmaps on the Apollo capsule mesh.
Color: red (under-predict) -> white (perfect) -> green (over-predict)
Creates maps for each model in organized_results/ for each output metric.
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from config import Config
from model import MambaAutoencoder
from dataset import (load_and_clean, build_partition_dataset,
                     fit_scalers, apply_scalers, spatial_sort_solution)

RESULTS_DIR = 'organized_results'
DATA_PATH = 'data/apollo_cfd_database.csv'

# Model configs: (folder, kwargs for Config overrides, extra flags)
MODEL_CONFIGS = [
    ('full_model',      dict(),                                    dict()),
    ('no_physics',      dict(),                                    dict()),
    ('qw_only',         dict(),                                    dict(qw_only=True)),
    ('full_model_long', dict(split_seed=456),                      dict()),
    ('full_70_15',      dict(train_frac=0.70, val_frac=0.15),      dict()),
    ('full_60_20',      dict(train_frac=0.60, val_frac=0.20),      dict()),
    ('full_50_25',      dict(train_frac=0.50, val_frac=0.25),      dict()),
    ('full_40_30',      dict(train_frac=0.40, val_frac=0.30),      dict()),
]

# Custom red-white-green diverging colormap
colors_rwg = ['#E53935', '#EF5350', '#FFCDD2', '#FFFFFF', '#C8E6C9', '#66BB6A', '#4CAF50']
cmap_rwg = mcolors.LinearSegmentedColormap.from_list('RedWhiteGreen', colors_rwg, N=256)


def load_data_and_coords(cfg):
    """Load data, return test partitions + raw XYZ coords per test solution."""
    print(f"  Loading data (seed={cfg.split_seed}, train={cfg.train_frac})...")
    df = load_and_clean(cfg, DATA_PATH)

    # Build partitioned datasets
    X_train_raw, Y_train_raw, _ = build_partition_dataset(df, 'train', cfg)
    X_test_raw, Y_test_raw, meta_test = build_partition_dataset(df, 'test', cfg)

    # Fit scalers on training data
    scaler_X, scaler_y = fit_scalers(X_train_raw, Y_train_raw, cfg)

    # Scale test data
    from dataset import apply_scalers
    X_test_s, Y_test_s = apply_scalers(X_test_raw, Y_test_raw, scaler_X, scaler_y, cfg)

    # Get raw XYZ coords per test solution (pre-partition, pre-scale)
    df_test = df[df['split'] == 'test']
    loc_ids = sorted(df_test['location_id'].unique())
    solution_coords = {}
    for lid in loc_ids:
        sol = df_test[df_test['location_id'] == lid]
        sort_idx = spatial_sort_solution(sol)
        sol_sorted = sol.iloc[sort_idx]
        solution_coords[lid] = {
            'X': sol_sorted['X'].values,
            'Y': sol_sorted['Y'].values,
            'Z': sol_sorted['Z'].values,
        }

    return X_test_s, Y_test_raw, meta_test, scaler_y, solution_coords, cfg.y_col_names


def run_model(model, X_test_s, device):
    """Run model on all test partitions."""
    model.eval()
    all_preds = {}
    with torch.no_grad():
        for i in range(X_test_s.shape[0]):
            x = torch.tensor(X_test_s[i:i+1], dtype=torch.float32).to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(x)
            for name in out:
                if name in ('recon', 'latent'):
                    continue
                if name not in all_preds:
                    all_preds[name] = []
                all_preds[name].append(out[name].cpu().numpy())

    for name in all_preds:
        all_preds[name] = np.concatenate(all_preds[name], axis=0)
    return all_preds


def reconstruct_solutions(all_preds, Y_test_raw, meta_test, scaler_y, y_col_names):
    """Overlap-average predictions and compute signed relative errors per solution."""
    # Inverse transform predictions
    pred_phys = {}
    for i, name in enumerate(y_col_names):
        if name not in all_preds:
            continue
        pred_std = all_preds[name]
        pred_log = pred_std * scaler_y.scale_[i] + scaler_y.mean_[i]
        pred_phys[name] = np.power(10.0, pred_log)

    # Group partitions by solution
    sol_parts = {}
    for pidx, m in enumerate(meta_test):
        lid = m['location_id']
        if lid not in sol_parts:
            sol_parts[lid] = []
        sol_parts[lid].append(pidx)

    # Reconstruct per solution
    solution_errors = {}
    for lid, part_indices in sol_parts.items():
        n_pts = meta_test[part_indices[0]]['n_points_orig']
        solution_errors[lid] = {}

        for i, name in enumerate(y_col_names):
            if name not in pred_phys:
                continue
            pred_sum = np.zeros(n_pts)
            pred_count = np.zeros(n_pts)
            true_vals = np.zeros(n_pts)

            for pidx in part_indices:
                m = meta_test[pidx]
                start = m['start']
                n_valid = m['n_valid']
                pred_sum[start:start + n_valid] += pred_phys[name][pidx, :n_valid, 0]
                pred_count[start:start + n_valid] += 1
                true_vals[start:start + n_valid] = Y_test_raw[pidx, :n_valid, i]

            mask = pred_count > 0
            pred_avg = np.zeros(n_pts)
            pred_avg[mask] = pred_sum[mask] / pred_count[mask]

            # Signed relative error: positive = over-predict, negative = under-predict
            signed_err = np.zeros(n_pts)
            signed_err[mask] = (pred_avg[mask] - true_vals[mask]) / (true_vals[mask] + 1e-9) * 100

            solution_errors[lid][name] = {
                'signed_error': signed_err,
                'mask': mask,
                'true': true_vals,
                'pred': pred_avg,
            }

    return solution_errors


def plot_error_maps(solution_errors, solution_coords, y_col_names, save_dir, model_name,
                    n_solutions=3, error_range=10):
    """Create spatial error maps for each output on selected test solutions."""
    os.makedirs(save_dir, exist_ok=True)

    # Pick first n solutions
    lids = sorted(solution_errors.keys())[:n_solutions]
    active_names = [n for n in y_col_names if n in solution_errors[lids[0]]]

    output_labels = {
        'qw': 'Heat Flux qw (W/m\u00b2)',
        'pw': 'Pressure pw (Pa)',
        'tw': 'Shear Stress \u03c4w (Pa)',
        'me': 'Edge Mach Me',
        'theta': 'Momentum Thickness \u03b8 (m)',
    }

    # --- Per-output: show n_solutions side by side ---
    for name in active_names:
        fig, axes = plt.subplots(2, n_solutions, figsize=(6 * n_solutions, 10))
        if n_solutions == 1:
            axes = axes.reshape(-1, 1)
        fig.suptitle(f'{model_name} — {output_labels.get(name, name)} Error Map',
                     fontsize=16, fontweight='bold', y=0.98)

        for j, lid in enumerate(lids):
            errs = solution_errors[lid][name]
            coords = solution_coords[lid]
            mask = errs['mask']
            signed = errs['signed_error'][mask]
            x_coord = coords['X'][mask]
            y_coord = coords['Y'][mask]
            z_coord = coords['Z'][mask]
            r_perp = np.sqrt(y_coord**2 + z_coord**2)

            # Side view: X vs r_perp
            ax = axes[0, j]
            sc = ax.scatter(x_coord, r_perp, c=signed, cmap=cmap_rwg,
                           vmin=-error_range, vmax=error_range,
                           s=0.3, alpha=0.8, edgecolors='none')
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('r (m)', fontsize=10)
            ax.set_title(f'Solution {lid} — Side View', fontsize=11)
            ax.set_aspect('equal')
            plt.colorbar(sc, ax=ax, label='Signed Error (%)', shrink=0.8)

            # Front view: Y vs Z
            ax = axes[1, j]
            sc = ax.scatter(y_coord, z_coord, c=signed, cmap=cmap_rwg,
                           vmin=-error_range, vmax=error_range,
                           s=0.3, alpha=0.8, edgecolors='none')
            ax.set_xlabel('Y (m)', fontsize=10)
            ax.set_ylabel('Z (m)', fontsize=10)
            ax.set_title(f'Solution {lid} — Front View', fontsize=11)
            ax.set_aspect('equal')
            plt.colorbar(sc, ax=ax, label='Signed Error (%)', shrink=0.8)

        plt.tight_layout()
        out_path = os.path.join(save_dir, f'error_map_{name}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    Saved {out_path}")

    # --- Combined overview: all outputs for solution 0, side view ---
    lid = lids[0]
    fig, axes = plt.subplots(1, len(active_names), figsize=(5 * len(active_names), 4))
    if len(active_names) == 1:
        axes = [axes]
    fig.suptitle(f'{model_name} — Solution {lid} — All Outputs (Side View)',
                 fontsize=14, fontweight='bold', y=1.02)

    for i, name in enumerate(active_names):
        errs = solution_errors[lid][name]
        coords = solution_coords[lid]
        mask = errs['mask']
        signed = errs['signed_error'][mask]
        x_coord = coords['X'][mask]
        y_coord = coords['Y'][mask]
        z_coord = coords['Z'][mask]
        r_perp = np.sqrt(y_coord**2 + z_coord**2)

        ax = axes[i]
        sc = ax.scatter(x_coord, r_perp, c=signed, cmap=cmap_rwg,
                       vmin=-error_range, vmax=error_range,
                       s=0.5, alpha=0.8, edgecolors='none')
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('r (m)', fontsize=9)
        ax.set_title(output_labels.get(name, name), fontsize=10)
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, label='Error (%)', shrink=0.8)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'error_map_overview.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved {out_path}")

    # --- Error distribution histograms ---
    lid = lids[0]
    fig, axes = plt.subplots(1, len(active_names), figsize=(4.5 * len(active_names), 3.5))
    if len(active_names) == 1:
        axes = [axes]
    fig.suptitle(f'{model_name} — Signed Error Distribution (Solution {lid})',
                 fontsize=14, fontweight='bold', y=1.02)

    for i, name in enumerate(active_names):
        errs = solution_errors[lid][name]
        mask = errs['mask']
        signed = errs['signed_error'][mask]

        ax = axes[i]
        ax.hist(signed, bins=100, range=(-15, 15), color='steelblue',
                alpha=0.7, edgecolor='white', linewidth=0.3)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.axvline(np.median(signed), color='orange', linestyle='-', linewidth=1.5,
                   label=f'Median: {np.median(signed):.2f}%')
        ax.set_xlabel('Signed Error (%)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(output_labels.get(name, name), fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'error_distribution.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved {out_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Cache loaded data by config key to avoid reloading CSV
    data_cache = {}

    for folder, cfg_overrides, flags in MODEL_CONFIGS:
        ckpt_path = os.path.join(RESULTS_DIR, folder, 'best_model.pt')
        if not os.path.exists(ckpt_path):
            print(f"Skipping {folder} — no checkpoint found")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing: {folder}")
        print(f"{'='*60}")

        # Build config
        cfg = Config()
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
        if flags.get('qw_only'):
            cfg.predict_pw = False
            cfg.predict_tw = False
            cfg.predict_me = False
            cfg.predict_theta = False

        # Cache key
        cache_key = (cfg.train_frac, cfg.val_frac, cfg.split_seed,
                     cfg.predict_qw, cfg.predict_pw, cfg.predict_tw,
                     cfg.predict_me, cfg.predict_theta)

        if cache_key not in data_cache:
            data_cache[cache_key] = load_data_and_coords(cfg)
        X_test_s, Y_test_raw, meta_test, scaler_y, solution_coords, y_col_names = data_cache[cache_key]

        # Load model
        print(f"  Loading model from {ckpt_path}...")
        model = MambaAutoencoder(cfg).to(device)
        # Fix expanded A_log parameters (shared memory view from .expand())
        for name, param in model.named_parameters():
            if 'A_log' in name:
                param.data = param.data.clone()
        ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(cleaned, strict=False)

        # Run inference
        print(f"  Running inference on {X_test_s.shape[0]} test partitions...")
        all_preds = run_model(model, X_test_s, device)

        # Reconstruct solutions with overlap averaging
        print(f"  Reconstructing solutions and computing errors...")
        solution_errors = reconstruct_solutions(
            all_preds, Y_test_raw, meta_test, scaler_y, y_col_names
        )

        # Plot
        save_dir = os.path.join(RESULTS_DIR, folder, 'error_maps')
        print(f"  Generating error maps...")
        plot_error_maps(solution_errors, solution_coords, y_col_names,
                       save_dir, folder, n_solutions=3, error_range=10)

        # Free memory
        del model, all_preds, solution_errors
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("  All error maps generated!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
