"""
Generate spatial error heatmaps on the Apollo capsule mesh.
Error maps: red (under-predict) -> white (perfect) -> green (over-predict) on dark bg
Ground truth maps: blue (low) -> red (high) on white bg
Prediction maps: same colorscale as ground truth for direct comparison
"""
import os
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

MODEL_CONFIGS = [
    ('full_model',      dict(),                                    dict()),
    ('no_physics',      dict(),                                    dict()),
    ('qw_only',         dict(),                                    dict(qw_only=True)),
    ('full_model_long', dict(split_seed=456),                      dict()),
    ('full_70_15',      dict(train_frac=0.70, val_frac=0.15),      dict()),
    ('full_60_20',      dict(train_frac=0.60, val_frac=0.20),      dict()),
    ('full_50_25',      dict(train_frac=0.50, val_frac=0.25),      dict()),
    ('full_40_30',      dict(train_frac=0.40, val_frac=0.30),      dict()),
    ('mlp_baseline',    dict(),                                    dict(block_type='mlp')),
    ('no_physics_60',   dict(train_frac=0.60, val_frac=0.20),      dict()),
    ('no_physics_40',   dict(train_frac=0.40, val_frac=0.30),      dict()),
    ('strong_physics_80', dict(),                                  dict()),
    ('strong_physics_60', dict(train_frac=0.60, val_frac=0.20),    dict()),
    ('strong_physics_40', dict(train_frac=0.40, val_frac=0.30),    dict()),
    ('strong_physics_80_seed2', dict(split_seed=456),              dict()),
    ('no_physics_long', dict(split_seed=456),                      dict()),
]

# Custom red-white-green diverging colormap (for error maps on dark bg)
colors_rwg = ['#E53935', '#EF5350', '#FFCDD2', '#FFFFFF', '#C8E6C9', '#66BB6A', '#4CAF50']
cmap_rwg = mcolors.LinearSegmentedColormap.from_list('RedWhiteGreen', colors_rwg, N=256)

DARK_BG = '#1A1A2E'
DARK_AXES = '#16213E'

OUTPUT_LABELS = {
    'qw': 'Heat Flux qw (W/m\u00b2)',
    'pw': 'Pressure pw (Pa)',
    'tw': 'Shear Stress \u03c4w (Pa)',
    'me': 'Edge Mach Me',
    'theta': 'Momentum Thickness \u03b8 (m)',
}


def style_ax_dark(ax):
    """Style axis for dark background (error maps)."""
    ax.set_facecolor(DARK_AXES)
    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('#444444')


def style_ax_light(ax):
    """Style axis for light background (ground truth / prediction maps)."""
    ax.set_facecolor('#F8F8F8')
    ax.tick_params(colors='#333333', labelsize=8)
    ax.xaxis.label.set_color('#333333')
    ax.yaxis.label.set_color('#333333')
    ax.title.set_color('#222222')
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')


def load_data_and_coords(cfg):
    """Load data, return test partitions + raw XYZ coords per test solution."""
    print(f"  Loading data (seed={cfg.split_seed}, train={cfg.train_frac})...")
    df = load_and_clean(cfg, DATA_PATH)

    X_train_raw, Y_train_raw, _ = build_partition_dataset(df, 'train', cfg)
    X_test_raw, Y_test_raw, meta_test = build_partition_dataset(df, 'test', cfg)

    scaler_X, scaler_y = fit_scalers(X_train_raw, Y_train_raw, cfg)

    from dataset import apply_scalers
    X_test_s, Y_test_s = apply_scalers(X_test_raw, Y_test_raw, scaler_X, scaler_y, cfg)

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
    pred_phys = {}
    for i, name in enumerate(y_col_names):
        if name not in all_preds:
            continue
        pred_std = all_preds[name]
        pred_log = pred_std * scaler_y.scale_[i] + scaler_y.mean_[i]
        pred_phys[name] = np.power(10.0, pred_log)

    sol_parts = {}
    for pidx, m in enumerate(meta_test):
        lid = m['location_id']
        if lid not in sol_parts:
            sol_parts[lid] = []
        sol_parts[lid].append(pidx)

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
    """Create all visualizations for a model."""
    os.makedirs(save_dir, exist_ok=True)

    lids = sorted(solution_errors.keys())[:n_solutions]
    active_names = [n for n in y_col_names if n in solution_errors[lids[0]]]

    # ========================================
    # 1. ERROR MAPS (dark background, red-white-green)
    # ========================================
    for name in active_names:
        fig, axes = plt.subplots(2, n_solutions, figsize=(6 * n_solutions, 10))
        fig.patch.set_facecolor(DARK_BG)
        if n_solutions == 1:
            axes = axes.reshape(-1, 1)
        fig.suptitle(f'{model_name} \u2014 {OUTPUT_LABELS.get(name, name)} Signed Error',
                     fontsize=16, fontweight='bold', color='white', y=0.98)

        for j, lid in enumerate(lids):
            errs = solution_errors[lid][name]
            coords = solution_coords[lid]
            mask = errs['mask']
            signed = errs['signed_error'][mask]
            x_coord = coords['X'][mask]
            y_coord = coords['Y'][mask]
            z_coord = coords['Z'][mask]
            r_perp = np.sqrt(y_coord**2 + z_coord**2)

            # Side view
            ax = axes[0, j]
            style_ax_dark(ax)
            sc = ax.scatter(x_coord, r_perp, c=signed, cmap=cmap_rwg,
                           vmin=-error_range, vmax=error_range,
                           s=0.3, alpha=0.9, edgecolors='none')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('r (m)')
            ax.set_title(f'Solution {lid} \u2014 Side View')
            ax.set_aspect('equal')
            cb = plt.colorbar(sc, ax=ax, shrink=0.8)
            cb.set_label('Signed Error (%)', color='white')
            cb.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

            # Front view
            ax = axes[1, j]
            style_ax_dark(ax)
            sc = ax.scatter(y_coord, z_coord, c=signed, cmap=cmap_rwg,
                           vmin=-error_range, vmax=error_range,
                           s=0.3, alpha=0.9, edgecolors='none')
            ax.set_xlabel('Y (m)')
            ax.set_ylabel('Z (m)')
            ax.set_title(f'Solution {lid} \u2014 Front View')
            ax.set_aspect('equal')
            cb = plt.colorbar(sc, ax=ax, shrink=0.8)
            cb.set_label('Signed Error (%)', color='white')
            cb.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

        plt.tight_layout()
        out_path = os.path.join(save_dir, f'error_map_{name}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=DARK_BG)
        plt.close()
        print(f"    Saved {out_path}")

    # ========================================
    # 2. ERROR OVERVIEW (dark bg, all outputs, 1 solution)
    # ========================================
    lid = lids[0]
    fig, axes = plt.subplots(1, len(active_names), figsize=(5 * len(active_names), 4))
    fig.patch.set_facecolor(DARK_BG)
    if len(active_names) == 1:
        axes = [axes]
    fig.suptitle(f'{model_name} \u2014 Solution {lid} \u2014 Error Overview (Side View)',
                 fontsize=14, fontweight='bold', color='white', y=1.02)

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
        style_ax_dark(ax)
        sc = ax.scatter(x_coord, r_perp, c=signed, cmap=cmap_rwg,
                       vmin=-error_range, vmax=error_range,
                       s=0.5, alpha=0.9, edgecolors='none')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('r (m)')
        ax.set_title(OUTPUT_LABELS.get(name, name))
        ax.set_aspect('equal')
        cb = plt.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label('Error (%)', color='white')
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'error_map_overview.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"    Saved {out_path}")

    # ========================================
    # 3. GROUND TRUTH vs PREDICTION (light bg, blue-red, side by side)
    # ========================================
    lid = lids[0]
    for name in active_names:
        errs = solution_errors[lid][name]
        coords = solution_coords[lid]
        mask = errs['mask']
        true_vals = errs['true'][mask]
        pred_vals = errs['pred'][mask]
        x_coord = coords['X'][mask]
        y_coord = coords['Y'][mask]
        z_coord = coords['Z'][mask]
        r_perp = np.sqrt(y_coord**2 + z_coord**2)

        # Use log10 for display (values span orders of magnitude)
        true_log = np.log10(np.clip(true_vals, 1e-10, None))
        pred_log = np.log10(np.clip(pred_vals, 1e-10, None))
        vmin = min(true_log.min(), pred_log.min())
        vmax = max(true_log.max(), pred_log.max())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('white')
        fig.suptitle(f'{model_name} \u2014 {OUTPUT_LABELS.get(name, name)} \u2014 Solution {lid}',
                     fontsize=16, fontweight='bold', color='#222222', y=0.98)

        # Top row: side views
        for col, (vals, label) in enumerate([(true_log, 'Ground Truth (CFD)'),
                                              (pred_log, 'Model Prediction')]):
            ax = axes[0, col]
            style_ax_light(ax)
            sc = ax.scatter(x_coord, r_perp, c=vals, cmap='jet',
                           vmin=vmin, vmax=vmax,
                           s=0.3, alpha=0.9, edgecolors='none')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('r (m)')
            ax.set_title(f'{label} \u2014 Side View')
            ax.set_aspect('equal')
            cb = plt.colorbar(sc, ax=ax, shrink=0.8)
            cb.set_label(f'log10({name})')

        # Bottom row: front views
        for col, (vals, label) in enumerate([(true_log, 'Ground Truth (CFD)'),
                                              (pred_log, 'Model Prediction')]):
            ax = axes[1, col]
            style_ax_light(ax)
            sc = ax.scatter(y_coord, z_coord, c=vals, cmap='jet',
                           vmin=vmin, vmax=vmax,
                           s=0.3, alpha=0.9, edgecolors='none')
            ax.set_xlabel('Y (m)')
            ax.set_ylabel('Z (m)')
            ax.set_title(f'{label} \u2014 Front View')
            ax.set_aspect('equal')
            cb = plt.colorbar(sc, ax=ax, shrink=0.8)
            cb.set_label(f'log10({name})')

        plt.tight_layout()
        out_path = os.path.join(save_dir, f'truth_vs_pred_{name}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved {out_path}")

    # ========================================
    # 4. GROUND TRUTH vs PREDICTION OVERVIEW (all outputs, side view only)
    # ========================================
    lid = lids[0]
    n_out = len(active_names)
    fig, axes = plt.subplots(2, n_out, figsize=(5 * n_out, 8))
    fig.patch.set_facecolor('white')
    if n_out == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle(f'{model_name} \u2014 Solution {lid} \u2014 Ground Truth vs Prediction',
                 fontsize=15, fontweight='bold', color='#222222', y=1.0)

    for i, name in enumerate(active_names):
        errs = solution_errors[lid][name]
        coords = solution_coords[lid]
        mask = errs['mask']
        true_vals = errs['true'][mask]
        pred_vals = errs['pred'][mask]
        x_coord = coords['X'][mask]
        y_coord = coords['Y'][mask]
        z_coord = coords['Z'][mask]
        r_perp = np.sqrt(y_coord**2 + z_coord**2)

        true_log = np.log10(np.clip(true_vals, 1e-10, None))
        pred_log = np.log10(np.clip(pred_vals, 1e-10, None))
        vmin = min(true_log.min(), pred_log.min())
        vmax = max(true_log.max(), pred_log.max())

        # Row 0: Ground truth
        ax = axes[0, i]
        style_ax_light(ax)
        sc = ax.scatter(x_coord, r_perp, c=true_log, cmap='jet',
                       vmin=vmin, vmax=vmax, s=0.4, alpha=0.9, edgecolors='none')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('r (m)')
        ax.set_title(f'{OUTPUT_LABELS.get(name, name)}\nGround Truth', fontsize=10)
        ax.set_aspect('equal')
        cb = plt.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label(f'log10({name})', fontsize=8)

        # Row 1: Prediction
        ax = axes[1, i]
        style_ax_light(ax)
        sc = ax.scatter(x_coord, r_perp, c=pred_log, cmap='jet',
                       vmin=vmin, vmax=vmax, s=0.4, alpha=0.9, edgecolors='none')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('r (m)')
        ax.set_title(f'{OUTPUT_LABELS.get(name, name)}\nPrediction', fontsize=10)
        ax.set_aspect('equal')
        cb = plt.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label(f'log10({name})', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'truth_vs_pred_overview.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved {out_path}")

    # ========================================
    # 5. ERROR DISTRIBUTION (dark bg)
    # ========================================
    lid = lids[0]
    fig, axes = plt.subplots(1, len(active_names), figsize=(4.5 * len(active_names), 3.5))
    fig.patch.set_facecolor(DARK_BG)
    if len(active_names) == 1:
        axes = [axes]
    fig.suptitle(f'{model_name} \u2014 Signed Error Distribution (Solution {lid})',
                 fontsize=14, fontweight='bold', color='white', y=1.02)

    for i, name in enumerate(active_names):
        errs = solution_errors[lid][name]
        mask = errs['mask']
        signed = errs['signed_error'][mask]

        ax = axes[i]
        style_ax_dark(ax)
        ax.hist(signed, bins=100, range=(-15, 15), color='#1E88E5',
                alpha=0.8, edgecolor='#0D47A1', linewidth=0.3)
        ax.axvline(0, color='#FF5252', linestyle='--', linewidth=1.2, label='Zero')
        ax.axvline(np.median(signed), color='#FFB74D', linestyle='-', linewidth=1.5,
                   label=f'Median: {np.median(signed):.2f}%')
        pct5 = (np.abs(signed) <= 5).mean() * 100
        ax.set_xlabel('Signed Error (%)')
        ax.set_ylabel('Count')
        ax.set_title(f'{OUTPUT_LABELS.get(name, name)}\n(\u00b15%: {pct5:.1f}%)')
        leg = ax.legend(fontsize=8, facecolor=DARK_AXES, edgecolor='#444444',
                       labelcolor='white')

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'error_distribution.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"    Saved {out_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_cache = {}

    for folder, cfg_overrides, flags in MODEL_CONFIGS:
        ckpt_path = os.path.join(RESULTS_DIR, folder, 'best_model.pt')
        if not os.path.exists(ckpt_path):
            print(f"Skipping {folder} \u2014 no checkpoint found")
            continue

        error_maps_dir = os.path.join(RESULTS_DIR, folder, 'error_maps')
        if os.path.exists(error_maps_dir) and len(os.listdir(error_maps_dir)) > 0:
            print(f"Skipping {folder} \u2014 error maps already exist")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing: {folder}")
        print(f"{'='*60}")

        cfg = Config()
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
        if flags.get('qw_only'):
            cfg.predict_pw = False
            cfg.predict_tw = False
            cfg.predict_me = False
            cfg.predict_theta = False
        if flags.get('block_type'):
            cfg.block_type = flags['block_type']

        cache_key = (cfg.train_frac, cfg.val_frac, cfg.split_seed,
                     cfg.predict_qw, cfg.predict_pw, cfg.predict_tw,
                     cfg.predict_me, cfg.predict_theta)

        if cache_key not in data_cache:
            data_cache[cache_key] = load_data_and_coords(cfg)
        X_test_s, Y_test_raw, meta_test, scaler_y, solution_coords, y_col_names = data_cache[cache_key]

        print(f"  Loading model from {ckpt_path}...")
        model = MambaAutoencoder(cfg).to(device)
        ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(cleaned)

        print(f"  Running inference on {X_test_s.shape[0]} test partitions...")
        all_preds = run_model(model, X_test_s, device)

        print(f"  Reconstructing solutions and computing errors...")
        solution_errors = reconstruct_solutions(
            all_preds, Y_test_raw, meta_test, scaler_y, y_col_names
        )

        save_dir = os.path.join(RESULTS_DIR, folder, 'error_maps')
        print(f"  Generating error maps...")
        plot_error_maps(solution_errors, solution_coords, y_col_names,
                       save_dir, folder, n_solutions=3, error_range=10)

        del model, all_preds, solution_errors
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("  All error maps generated!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
