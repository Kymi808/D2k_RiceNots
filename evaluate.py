"""
Evaluation and visualization for Mamba Autoencoder CFD.
Computes per-output metrics in physical units and saves plots.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


OUTPUT_LABELS = {
    'qw': 'Heat Flux qw (W/m²)',
    'pw': 'Pressure pw (Pa)',
    'tw': 'Shear Stress τw (Pa)',
    'me': 'Edge Mach Me (-)',
    'theta': 'Momentum Thickness θ (m)',
}


def _add_qw_physics_log_if_needed(pred_log, X_std_all, scaler_X, y_col_names, name, cfg):
    """Convert normalized qw log predictions back to log10(qw) when requested."""
    if not getattr(cfg, 'normalize_qw_by_rhov3', False) or name != 'qw':
        return pred_log

    X_phys = scaler_X.inverse_transform(
        X_std_all.reshape(-1, cfg.n_features)
    ).reshape(X_std_all.shape)
    velocity = X_phys[:, :, 3:4]
    density = X_phys[:, :, 4:5]
    phys_log = np.log10(np.clip(density * np.power(velocity, 3), 1e-12, None))
    return pred_log + phys_log


@torch.no_grad()
def evaluate_model(model, dl, scaler_y, y_col_names, Y_raw, meta, device,
                   scaler_X=None, cfg=None):
    """
    Evaluate model with overlap averaging across partitions.
    Predictions from overlapping partitions are averaged per physical point,
    so each surface point gets a single prediction regardless of how many
    partitions cover it.

    Y_raw: raw (unscaled) target array, shape (n_partitions, seq_len, n_outputs).
    meta: list of dicts with location_id, start, end, n_points_orig, n_valid.
    """
    model.eval()
    all_preds = {name: [] for name in y_col_names}
    all_inputs = []

    for X_batch, Y_batch in dl:
        X_batch = X_batch.to(device)
        all_inputs.append(X_batch.cpu().numpy())
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out = model(X_batch)
        for name in y_col_names:
            if name in out:
                all_preds[name].append(out[name].cpu().numpy())

    # (n_partitions, seq_len, 1) per output
    for name in y_col_names:
        all_preds[name] = np.concatenate(all_preds[name], axis=0)
    X_std_all = np.concatenate(all_inputs, axis=0)

    # Inverse transform predictions to physical units
    pred_phys_all = {}
    for i, name in enumerate(y_col_names):
        pred_std = all_preds[name]
        pred_log = pred_std * scaler_y.scale_[i] + scaler_y.mean_[i]
        if cfg is not None and scaler_X is not None:
            pred_log = _add_qw_physics_log_if_needed(
                pred_log, X_std_all, scaler_X, y_col_names, name, cfg
            )
        pred_phys_all[name] = np.power(10.0, pred_log)

    # Group partitions by solution
    sol_parts = {}
    for pidx, m in enumerate(meta):
        lid = m['location_id']
        if lid not in sol_parts:
            sol_parts[lid] = []
        sol_parts[lid].append(pidx)

    # Per-output: reconstruct full solutions with overlap averaging
    results = {}
    for i, name in enumerate(y_col_names):
        all_true, all_pred = [], []

        for lid, part_indices in sol_parts.items():
            n_pts = meta[part_indices[0]]['n_points_orig']
            pred_sum = np.zeros(n_pts)
            pred_count = np.zeros(n_pts)
            true_vals = np.zeros(n_pts)

            for pidx in part_indices:
                m = meta[pidx]
                start = m['start']
                n_valid = m['n_valid']

                pred_sum[start:start + n_valid] += pred_phys_all[name][pidx, :n_valid, 0]
                pred_count[start:start + n_valid] += 1
                true_vals[start:start + n_valid] = Y_raw[pidx, :n_valid, i]

            mask = pred_count > 0
            all_pred.append(pred_sum[mask] / pred_count[mask])
            all_true.append(true_vals[mask])

        true_phys = np.concatenate(all_true)
        pred_phys = np.concatenate(all_pred)

        rel_errors = np.abs(true_phys - pred_phys) / (true_phys + 1e-9) * 100
        mae = mean_absolute_error(true_phys, pred_phys)
        rmse = np.sqrt(mean_squared_error(true_phys, pred_phys))

        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'pct_1': (rel_errors <= 1).mean() * 100,
            'pct_3': (rel_errors <= 3).mean() * 100,
            'pct_5': (rel_errors <= 5).mean() * 100,
            'pct_10': (rel_errors <= 10).mean() * 100,
            'median_err': float(np.median(rel_errors)),
            'q95_err': float(np.percentile(rel_errors, 95)),
            'rel_errors': rel_errors,
            'pred_phys': pred_phys,
            'true_phys': true_phys,
        }

    return results


def print_results(results, y_col_names, block_type='MAMBA3'):
    """Print formatted evaluation results."""
    for name in y_col_names:
        r = results[name]
        label = OUTPUT_LABELS.get(name, name)
        print(f"\n{'=' * 60}")
        print(f"  {label} — {block_type.upper()} RESULTS")
        print(f"{'=' * 60}")
        print(f"  MAE:          {r['mae']:,.0f}")
        print(f"  RMSE:         {r['rmse']:,.0f}")
        print(f"  Within ±1%:   {r['pct_1']:.1f}%")
        print(f"  Within ±3%:   {r['pct_3']:.1f}%")
        print(f"  Within ±5%:   {r['pct_5']:.1f}%")
        print(f"  Within ±10%:  {r['pct_10']:.1f}%")
        print(f"  Median error:  {r['median_err']:.2f}%")
        print(f"  95th %%ile:    {r['q95_err']:.1f}%")


def save_training_curves(history, save_dir='results'):
    """Save training loss curves to disk."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train', linewidth=1)
    axes[0].plot(history['val_loss'], label='Val', linewidth=1)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Plot qw MSE specifically (primary target)
    if any(v > 0 for v in history.get('val_qw_mse', [])):
        axes[1].plot(history['val_qw_mse'], label='Val qw MSE', color='tab:orange', linewidth=1)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('qw MSE (standardized)')
        axes[1].set_title('Heat Flux Prediction')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Saved training curves to {save_dir}/training_curves.png")


def save_evaluation_plots(results, y_col_names, save_dir='results'):
    """Save predicted-vs-true scatter and error distribution plots."""
    os.makedirs(save_dir, exist_ok=True)
    n_outputs = len(y_col_names)
    fig, axes = plt.subplots(2, n_outputs, figsize=(6 * n_outputs, 10))
    if n_outputs == 1:
        axes = axes.reshape(-1, 1)

    for i, name in enumerate(y_col_names):
        r = results[name]
        label = OUTPUT_LABELS.get(name, name)

        # Scatter: predicted vs true
        ax = axes[0, i]
        n_points = len(r['true_phys'])
        sample_idx = np.random.choice(n_points, min(50000, n_points), replace=False)
        ax.scatter(r['true_phys'][sample_idx], r['pred_phys'][sample_idx],
                   alpha=0.1, s=1, color='steelblue')
        lims = [min(r['true_phys'].min(), r['pred_phys'].min()),
                max(r['true_phys'].max(), r['pred_phys'].max())]
        ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect')
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{label}\n\u00b15%: {r["pct_5"]:.1f}%')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()

        # Error distribution
        ax = axes[1, i]
        ax.hist(r['rel_errors'], bins=100, range=(0, 30),
                color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(5, color='red', linestyle='--', label='5% threshold')
        ax.set_xlabel('Relative Error (%)')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Error Distribution\nMedian: {r["median_err"]:.2f}%')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_plots.png'), dpi=150)
    plt.close()
    print(f"Saved evaluation plots to {save_dir}/evaluation_plots.png")
