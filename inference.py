"""
Production inference wrapper for the Mamba CFD Surrogate Model.

Usage:
    from inference import MambaSurrogate

    # Load once
    surrogate = MambaSurrogate('packaged_model/')

    # Predict for any flight condition (sub-second)
    results = surrogate.predict(
        velocity=7500.0,       # m/s
        density=0.003,         # kg/m^3
        aoa=155.0,             # degrees
        dynamic_pressure=84375.0  # Pa
    )

    # results is a dict:
    #   results['qw']    — (N,) array of heat flux in W/m^2
    #   results['pw']    — (N,) array of pressure in Pa
    #   results['tw']    — (N,) array of shear stress in Pa
    #   results['me']    — (N,) array of edge Mach number
    #   results['theta'] — (N,) array of momentum thickness in m
    #   results['xyz']   — (N, 3) array of mesh coordinates

    # Get the mesh coordinates for visualization
    X, Y, Z = results['xyz'][:, 0], results['xyz'][:, 1], results['xyz'][:, 2]
"""
import os
import json
import pickle
import numpy as np
import torch
from config import Config
from model import MambaAutoencoder


class MambaSurrogate:
    """
    Production inference wrapper for the Mamba CFD Surrogate Model.

    Handles all preprocessing (scaling, partitioning, overlap averaging)
    internally. Users only need to provide flight conditions.

    Args:
        model_dir: Path to packaged model directory (from package_model.py)
        device: 'cuda' or 'cpu' (auto-detects if not specified)
    """

    def __init__(self, model_dir, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Load config
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            self.config_dict = json.load(f)

        # Build model config
        self.cfg = Config()
        for key in ['d_model', 'd_state', 'd_conv', 'n_layers', 'latent_dim',
                     'expand', 'block_type', 'use_rope', 'use_trapezoidal',
                     'seq_len', 'partition_stride', 'points_per_solution']:
            if key in self.config_dict:
                setattr(self.cfg, key, self.config_dict[key])

        # Handle target config
        active_targets = [t[0] for t in self.config_dict['target_config']]
        self.cfg.predict_qw = 'qw' in active_targets
        self.cfg.predict_pw = 'pw' in active_targets
        self.cfg.predict_tw = 'tw' in active_targets
        self.cfg.predict_me = 'me' in active_targets
        self.cfg.predict_theta = 'theta' in active_targets
        self.output_names = self.cfg.y_col_names

        # Load model
        self.model = MambaAutoencoder(self.cfg).to(self.device)
        for name, param in self.model.named_parameters():
            if 'A_log' in name:
                param.data = param.data.clone()
        weights = torch.load(os.path.join(model_dir, 'model_weights.pt'),
                           weights_only=True, map_location=self.device)
        self.model.load_state_dict(weights, strict=False)
        self.model.eval()

        # Load scalers
        with open(os.path.join(model_dir, 'scaler_X.pkl'), 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open(os.path.join(model_dir, 'scaler_y.pkl'), 'rb') as f:
            self.scaler_y = pickle.load(f)

        # Load sorted mesh
        self.mesh_xyz = np.load(os.path.join(model_dir, 'mesh_xyz_sorted.npy'))
        self.n_points = len(self.mesh_xyz)

        # Precompute partition boundaries
        self.partitions = self._compute_partitions()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"MambaSurrogate loaded: {n_params:,} params, "
              f"{self.n_points} mesh points, "
              f"outputs: {self.output_names}, "
              f"device: {self.device}")

    def _compute_partitions(self):
        """Precompute partition start/end indices."""
        partitions = []
        start = 0
        while start < self.n_points:
            end = min(start + self.cfg.seq_len, self.n_points)
            partitions.append((start, end))
            if end >= self.n_points:
                break
            start += self.cfg.partition_stride
        return partitions

    def _build_input(self, velocity, density, aoa, dynamic_pressure):
        """Build the full input array from flight conditions."""
        n = self.n_points
        freestream = np.array([velocity, density, aoa, dynamic_pressure],
                             dtype=np.float32)
        freestream_tiled = np.tile(freestream, (n, 1))
        raw_input = np.hstack([self.mesh_xyz, freestream_tiled])  # (N, 7)
        return raw_input

    def _partition_and_scale(self, raw_input):
        """Partition the input and apply scaling."""
        scaled = self.scaler_X.transform(raw_input).astype(np.float32)
        partitions = []
        for start, end in self.partitions:
            part = scaled[start:end]
            # Pad last partition if needed
            if len(part) < self.cfg.seq_len:
                pad_len = self.cfg.seq_len - len(part)
                part = np.concatenate([part, np.tile(part[-1:], (pad_len, 1))], axis=0)
            partitions.append(part)
        return np.stack(partitions)  # (n_partitions, seq_len, 7)

    @torch.no_grad()
    def predict(self, velocity, density, aoa, dynamic_pressure):
        """
        Predict surface quantities for a given flight condition.

        Args:
            velocity: Freestream velocity in m/s
            density: Atmospheric density in kg/m^3
            aoa: Angle of attack in degrees
            dynamic_pressure: Dynamic pressure in Pa

        Returns:
            dict with keys for each output (e.g., 'qw', 'pw', 'tw', 'me', 'theta')
            plus 'xyz' containing the mesh coordinates.
            Each value is a numpy array of shape (n_points,) in physical units.
        """
        # Build input
        raw_input = self._build_input(velocity, density, aoa, dynamic_pressure)

        # Partition and scale
        X_partitions = self._partition_and_scale(raw_input)

        # Run model on all partitions
        all_preds = {name: [] for name in self.output_names}
        for i in range(len(X_partitions)):
            x = torch.tensor(X_partitions[i:i+1], dtype=torch.float32).to(self.device)
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                out = self.model(x)
            for name in self.output_names:
                if name in out:
                    all_preds[name].append(out[name].cpu().numpy()[0])  # (seq_len, 1)

        # Inverse transform and overlap average
        results = {}
        for idx, name in enumerate(self.output_names):
            pred_parts = all_preds[name]  # list of (seq_len, 1) arrays

            # Inverse transform: standardized -> log10 -> physical
            pred_phys_parts = []
            for part in pred_parts:
                pred_log = part[:, 0] * self.scaler_y.scale_[idx] + self.scaler_y.mean_[idx]
                pred_phys_parts.append(np.power(10.0, pred_log))

            # Overlap averaging
            pred_sum = np.zeros(self.n_points)
            pred_count = np.zeros(self.n_points)
            for part_idx, (start, end) in enumerate(self.partitions):
                n_valid = end - start
                pred_sum[start:end] += pred_phys_parts[part_idx][:n_valid]
                pred_count[start:end] += 1

            mask = pred_count > 0
            result = np.zeros(self.n_points)
            result[mask] = pred_sum[mask] / pred_count[mask]
            results[name] = result

        results['xyz'] = self.mesh_xyz.copy()
        return results

    def predict_batch(self, conditions):
        """
        Predict for multiple flight conditions.

        Args:
            conditions: list of dicts, each with keys:
                        velocity, density, aoa, dynamic_pressure

        Returns:
            list of result dicts (same format as predict())
        """
        return [self.predict(**cond) for cond in conditions]

    def sweep(self, velocities, densities, aoas, dynamic_pressures):
        """
        Predict for a grid of flight conditions.
        All arguments should be arrays of the same length.

        Returns:
            list of result dicts
        """
        conditions = [
            {'velocity': v, 'density': d, 'aoa': a, 'dynamic_pressure': q}
            for v, d, a, q in zip(velocities, densities, aoas, dynamic_pressures)
        ]
        return self.predict_batch(conditions)


# ============================================================
# Example usage (runs if executed directly)
# ============================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run inference with the Mamba CFD Surrogate')
    parser.add_argument('--model_dir', type=str, default='packaged_model',
                       help='Path to packaged model directory')
    parser.add_argument('--velocity', type=float, default=7500.0,
                       help='Freestream velocity (m/s)')
    parser.add_argument('--density', type=float, default=0.003,
                       help='Atmospheric density (kg/m^3)')
    parser.add_argument('--aoa', type=float, default=155.0,
                       help='Angle of attack (degrees)')
    parser.add_argument('--dynamic_pressure', type=float, default=84375.0,
                       help='Dynamic pressure (Pa)')
    args = parser.parse_args()

    # Load model
    surrogate = MambaSurrogate(args.model_dir)

    # Predict
    print(f"\nPredicting for: V={args.velocity} m/s, rho={args.density} kg/m^3, "
          f"AoA={args.aoa} deg, q_inf={args.dynamic_pressure} Pa")
    print("-" * 60)

    results = surrogate.predict(
        velocity=args.velocity,
        density=args.density,
        aoa=args.aoa,
        dynamic_pressure=args.dynamic_pressure,
    )

    # Print summary
    for name in surrogate.output_names:
        vals = results[name]
        print(f"\n{name}:")
        print(f"  Min:    {vals.min():,.2f}")
        print(f"  Max:    {vals.max():,.2f}")
        print(f"  Mean:   {vals.mean():,.2f}")
        print(f"  Median: {np.median(vals):,.2f}")

    print(f"\nMesh points: {len(results['xyz']):,}")
    print("Done.")
