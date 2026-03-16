"""
Standalone evaluation script — loads a saved checkpoint and evaluates on test set.
Usage: python eval_checkpoint.py --checkpoint checkpoints/run_XXXXX/best_model.pt
"""
import argparse
import torch
import numpy as np
from config import Config
from model import MambaAutoencoder
from dataset import get_dataloaders
from evaluate import evaluate_model, print_results, save_evaluation_plots

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, default='data/apollo_cfd_database.csv')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--block_type', type=str, default=None)
    parser.add_argument('--no_physics', action='store_true')
    parser.add_argument('--qw_only', action='store_true')
    parser.add_argument('--train_frac', type=float, default=None)
    parser.add_argument('--val_frac', type=float, default=None)
    parser.add_argument('--split_seed', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = Config()
    if args.block_type is not None:
        cfg.block_type = args.block_type
    if args.train_frac is not None:
        cfg.train_frac = args.train_frac
    if args.val_frac is not None:
        cfg.val_frac = args.val_frac
    if args.split_seed is not None:
        cfg.split_seed = args.split_seed
    if args.qw_only:
        cfg.predict_pw = False
        cfg.predict_tw = False
        cfg.predict_me = False
        cfg.predict_theta = False

    # Use save_dir from checkpoint path if not specified
    if args.save_dir is None:
        import os
        args.save_dir = os.path.dirname(args.checkpoint).replace('checkpoints', 'results')
        os.makedirs(args.save_dir, exist_ok=True)

    print("Loading data...")
    (_, _, test_dl, scaler_X, scaler_y,
     Y_test_raw, meta_test, _) = get_dataloaders(
        cfg, args.data, distributed=False, rank=0, world_size=1
    )

    y_col_names = cfg.y_col_names

    # Load model
    model = MambaAutoencoder(cfg).to(device)
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
    cleaned_ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(cleaned_ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model: {n_params:,} parameters, block_type={cfg.block_type}")
    print(f"Targets: {y_col_names}")

    # Evaluate
    print(f"\n{'=' * 60}")
    print("  EVALUATION ON TEST SET")
    print(f"{'=' * 60}")

    test_results = evaluate_model(
        model, test_dl, scaler_y, y_col_names, Y_test_raw,
        meta_test, device
    )
    print_results(test_results, y_col_names, cfg.block_type)
    save_evaluation_plots(test_results, y_col_names, args.save_dir)
    print(f"\nPlots saved to {args.save_dir}/")


if __name__ == '__main__':
    main()
