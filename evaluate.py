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


@torch.no_grad()
def evaluate_model(model, dl, scaler_y, y_col_names, Y_raw, device):
    """
    Evaluate model on a DataLoader. Returns per-output metrics in physical units.
    Y_raw: raw (unscaled) target array for ground truth comparison.
    """
    model.eval()
    all_preds = {name: [] for name in y_col_names}

    for X_batch, Y_batch in dl:
        X_batch = X_batch.to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out = model(X_batch)
        for name in y_col_names:
            if name in out:
                all_preds[name].append(out[name].cpu().numpy())

    results = {}
    for i, name in enumerate(y_col_names):
        pred_std = np.concatenate(all_preds[name], axis=0)

        # Inverse transform: standardized → log10 → physical
        pred_flat = pred_std.reshape(-1, 1)
        pred_log = pred_flat * scaler_y.scale_[i] + scaler_y.mean_[i]
        pred_phys = np.power(10.0, pred_log).ravel()

        true_phys = Y_raw[:, :, i].ravel()

        # Truncate to matching length (in case of distributed padding)
        min_len = min(len(pred_phys), len(true_phys))
        pred_phys = pred_phys[:min_len]
        true_phys = true_phys[:min_len]

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
