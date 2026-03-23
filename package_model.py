"""
Package a trained model for production inference.
Saves model weights, scalers, mesh data, and config into a single directory.

Usage:
    python package_model.py --checkpoint organized_results/full_model_long/best_model.pt \
                            --output packaged_model/

    # For different split configs:
    python package_model.py --checkpoint organized_results/full_50_25/best_model.pt \
                            --output packaged_model_50/ \
                            --train_frac 0.50 --val_frac 0.25
"""
import os
import argparse
import json
import numpy as np
import torch
import pickle
from config import Config
from model import MambaAutoencoder
from dataset import (load_and_clean, build_partition_dataset,
                     fit_scalers, spatial_sort_solution)


def main():
    """Package model weights, scalers, mesh, and config for production inference."""
    parser = argparse.ArgumentParser(description='Package trained model for inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, default='data/apollo_cfd_database.csv')
    parser.add_argument('--output', type=str, default='packaged_model')
    parser.add_argument('--split_seed', type=int, default=None)
    parser.add_argument('--train_frac', type=float, default=None)
    parser.add_argument('--val_frac', type=float, default=None)
    parser.add_argument('--qw_only', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Build config
    cfg = Config()
    if args.split_seed is not None:
        cfg.split_seed = args.split_seed
    if args.train_frac is not None:
        cfg.train_frac = args.train_frac
    if args.val_frac is not None:
        cfg.val_frac = args.val_frac
    if args.qw_only:
        cfg.predict_pw = False
        cfg.predict_tw = False
        cfg.predict_me = False
        cfg.predict_theta = False

    print("Loading data to extract scalers and mesh...")
    df = load_and_clean(cfg, args.data)

    # Fit scalers on training data
    X_train_raw, Y_train_raw, _ = build_partition_dataset(df, 'train', cfg)
    scaler_X, scaler_y = fit_scalers(X_train_raw, Y_train_raw, cfg)

    # Extract the canonical mesh (from any solution — geometry is the same)
    # Use the first solution in the dataset before cleaning to get the full mesh
    first_sol = df[df['location_id'] == df['location_id'].iloc[0]]
    sort_idx = spatial_sort_solution(first_sol)
    sorted_sol = first_sol.iloc[sort_idx]
    mesh_xyz = sorted_sol[['X', 'Y', 'Z']].values.astype(np.float32)

    # Save scalers
    with open(os.path.join(args.output, 'scaler_X.pkl'), 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(args.output, 'scaler_y.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)

    # Save sorted mesh
    np.save(os.path.join(args.output, 'mesh_xyz_sorted.npy'), mesh_xyz)

    # Save model weights (cleaned of compile prefixes)
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
    torch.save(cleaned, os.path.join(args.output, 'model_weights.pt'))

    # Save config as JSON
    config_dict = {
        'n_features': cfg.n_features,
        'd_model': cfg.d_model,
        'd_state': cfg.d_state,
        'd_conv': cfg.d_conv,
        'n_layers': cfg.n_layers,
        'latent_dim': cfg.latent_dim,
        'expand': cfg.expand,
        'block_type': cfg.block_type,
        'use_rope': cfg.use_rope,
        'use_trapezoidal': cfg.use_trapezoidal,
        'seq_len': cfg.seq_len,
        'partition_stride': cfg.partition_stride,
        'points_per_solution': cfg.points_per_solution,
        'y_col_names': cfg.y_col_names,
        'target_config': [(name, csv_col, weight) for name, csv_col, weight in cfg.target_config],
        'x_cols': cfg.x_cols,
    }
    with open(os.path.join(args.output, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Verify model loads correctly
    print("Verifying model loads...")
    model = MambaAutoencoder(cfg)
    for name, param in model.named_parameters():
        if 'A_log' in name:
            param.data = param.data.clone()
    model.load_state_dict(cleaned, strict=False)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nPackaged model saved to {args.output}/")
    print(f"  model_weights.pt    — {n_params:,} parameters")
    print(f"  scaler_X.pkl        — input feature scaler")
    print(f"  scaler_y.pkl        — target scaler (log10 + standardized)")
    print(f"  mesh_xyz_sorted.npy — sorted capsule mesh ({mesh_xyz.shape[0]} points)")
    print(f"  config.json         — model configuration")
    print(f"\nOutputs: {cfg.y_col_names}")
    print(f"Ready for inference with inference.py")


if __name__ == '__main__':
    main()
