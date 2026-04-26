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


def parse_head_dims(spec):
    """Parse a comma-separated list of hidden dims for prediction heads."""
    dims = [int(part.strip()) for part in spec.split(',') if part.strip()]
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(f"Invalid prediction head dims: {spec!r}")
    return dims

def main():
    """Load a saved checkpoint and evaluate on the test set."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, default='data/apollo_cfd_database.csv')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--block_type', type=str, default=None)
    parser.add_argument('--pred_head_dims', type=str, default=None)
    parser.add_argument('--pred_head_dropout', type=float, default=None)
    parser.add_argument('--use_residual_ffn', action='store_true')
    parser.add_argument('--ffn_hidden_dim', type=int, default=None)
    parser.add_argument('--ffn_dropout', type=float, default=None)
    parser.add_argument('--normalize_qw_by_rhov3', action='store_true')
    parser.add_argument('--n_heads', type=int, default=None)
    parser.add_argument('--transformer_ffn_dim', type=int, default=None)
    parser.add_argument('--attention_dropout', type=float, default=None)
    parser.add_argument('--moe_num_experts', type=int, default=None)
    parser.add_argument('--moe_top_k', type=int, default=None)
    parser.add_argument('--no_physics', action='store_true')
    parser.add_argument('--no_reconstruction', action='store_true')
    parser.add_argument('--lambda_recon', type=float, default=None)
    parser.add_argument('--w_qw', type=float, default=None)
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
    if args.pred_head_dims is not None:
        cfg.pred_head_hidden_dims = parse_head_dims(args.pred_head_dims)
    if args.pred_head_dropout is not None:
        cfg.pred_head_dropout = args.pred_head_dropout
    if args.use_residual_ffn:
        cfg.use_residual_ffn = True
    if args.ffn_hidden_dim is not None:
        cfg.ffn_hidden_dim = args.ffn_hidden_dim
    if args.ffn_dropout is not None:
        cfg.ffn_dropout = args.ffn_dropout
    if args.normalize_qw_by_rhov3:
        cfg.normalize_qw_by_rhov3 = True
    if args.n_heads is not None:
        cfg.n_heads = args.n_heads
    if args.transformer_ffn_dim is not None:
        cfg.transformer_ffn_dim = args.transformer_ffn_dim
    if args.attention_dropout is not None:
        cfg.attention_dropout = args.attention_dropout
    if args.moe_num_experts is not None:
        cfg.moe_num_experts = args.moe_num_experts
    if args.moe_top_k is not None:
        cfg.moe_top_k = args.moe_top_k
    if args.w_qw is not None:
        cfg.w_qw = args.w_qw
    if args.lambda_recon is not None:
        cfg.lambda_recon = args.lambda_recon
    if args.no_reconstruction:
        cfg.lambda_recon = 0.0
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
    print(f"Prediction head dims: {cfg.pred_head_hidden_dims}, dropout={cfg.pred_head_dropout}")
    print(f"Residual FFN: {cfg.use_residual_ffn}, hidden_dim={cfg.ffn_hidden_dim}, dropout={cfg.ffn_dropout}")
    if cfg.block_type in ('transformer', 'transformer_moe'):
        print(f"Transformer heads: {cfg.n_heads}, ffn_dim={cfg.transformer_ffn_dim}, attention_dropout={cfg.attention_dropout}")
    if cfg.block_type == 'transformer_moe':
        print(f"MoE experts: {cfg.moe_num_experts}, top_k={cfg.moe_top_k}")
    print(f"Normalize qw by rho*V^3: {cfg.normalize_qw_by_rhov3}")
    print(f"Reconstruction weight: {cfg.lambda_recon}")
    print(f"Target weights: {dict(zip(y_col_names, cfg.y_weights))}")
    print(f"Targets: {y_col_names}")

    # Evaluate
    print(f"\n{'=' * 60}")
    print("  EVALUATION ON TEST SET")
    print(f"{'=' * 60}")

    test_results = evaluate_model(
        model, test_dl, scaler_y, y_col_names, Y_test_raw,
        meta_test, device, scaler_X=scaler_X, cfg=cfg
    )
    print_results(test_results, y_col_names, cfg.block_type)
    save_evaluation_plots(test_results, y_col_names, args.save_dir)
    print(f"\nPlots saved to {args.save_dir}/")


if __name__ == '__main__':
    main()
