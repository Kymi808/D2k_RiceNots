"""
Physics-informed loss functions for Apollo reentry CFD surrogate.
All physics constraints operate in physical units (inverse-transformed).
No data leakage — only enforces known aerothermodynamic relationships.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLoss(nn.Module):
    """
    Physics coupling losses between predicted surface quantities.

    Constraints:
      1. Reynolds analogy: qw/tauw ratio bounded and smooth
      2. Modified Newtonian: Cp <= Cp_max for hypersonic blunt body
      3. Fay-Riddell scaling: qw_stag ~ sqrt(rho) * V^3
      4. Skin friction bounds: Cf in [1e-5, 0.1]
      5. BL consistency: delta/theta ratio bounded
      6. Positivity: all surface quantities > 0
    """

    def __init__(self, scaler_y, y_col_names, cfg):
        super().__init__()
        self.y_col_names = y_col_names
        self.cfg = cfg
        self.idx_map = {name: i for i, name in enumerate(y_col_names)}
        self.register_buffer('y_mean', torch.tensor(scaler_y.mean_, dtype=torch.float32))
        self.register_buffer('y_std', torch.tensor(scaler_y.scale_, dtype=torch.float32))

    def _to_physical(self, pred_std, name):
        """Convert standardized log10 prediction back to physical units."""
        idx = self.idx_map[name]
        log_val = pred_std * self.y_std[idx] + self.y_mean[idx]
        # Clamp to prevent overflow (10^20 is already way beyond physical range)
        log_val = torch.clamp(log_val, min=-10.0, max=20.0)
        return torch.pow(10.0, log_val)

    def _get_freestream(self, X_batch, scaler_X):
        """
        Extract freestream conditions from input features.
        X_batch is in standardized space — inverse transform the relevant columns.
        Returns dict of physical values: velocity, density, dynamic_pressure.
        """
        # Column indices in x_cols: velocity=3, density=4, dynamic_pressure=6
        x_mean = torch.tensor(scaler_X.mean_, device=X_batch.device, dtype=X_batch.dtype)
        x_std = torch.tensor(scaler_X.scale_, device=X_batch.device, dtype=X_batch.dtype)
        X_phys = X_batch * x_std + x_mean

        return {
            'velocity': X_phys[:, :, 3:4],      # (B, L, 1)
            'density': X_phys[:, :, 4:5],        # (B, L, 1)
            'q_inf': X_phys[:, :, 6:7],          # (B, L, 1) dynamic_pressure
        }

    def forward(self, preds, X_batch, scaler_X):
        """
        preds: dict of {name: (B, L, 1)} in standardized log10 space.
        X_batch: (B, L, n_features) in standardized space.
        scaler_X: fitted StandardScaler for input features.

        Returns dict of individual loss terms.
        """
        losses = {}
        has = lambda k: k in preds

        # === 1. POSITIVITY ===
        # Penalize predictions that map to very small physical values
        pos_loss = torch.tensor(0.0, device=X_batch.device)
        for name in self.y_col_names:
            if has(name):
                # In standardized log space, very negative = near-zero physical
                pos_loss = pos_loss + F.relu(-preds[name] - 3.0).pow(2).mean()
        if pos_loss > 0:
            losses['positivity'] = pos_loss

        # === 2. REYNOLDS ANALOGY: qw/tauw ratio ===
        if has('qw') and has('tw'):
            qw_phys = self._to_physical(preds['qw'], 'qw')
            tw_phys = self._to_physical(preds['tw'], 'tw')
            ratio = (qw_phys + 1e-8) / (tw_phys + 1e-8)
            log_ratio = torch.log10(ratio.clamp(min=1e-8))

            # Ratio should be in [10^2.5, 10^6.7] for Apollo reentry
            bound_loss = (
                F.relu(2.5 - log_ratio).pow(2).mean() +
                F.relu(log_ratio - 6.7).pow(2).mean()
            )
            losses['reynolds_bound'] = bound_loss

            # Spatial smoothness of the ratio (within each sequence)
            if log_ratio.shape[1] > 1:
                smoothness = (log_ratio[:, 1:] - log_ratio[:, :-1]).pow(2).mean()
                losses['reynolds_smooth'] = smoothness

        # === 3. MODIFIED NEWTONIAN PRESSURE BOUND ===
        if has('pw'):
            freestream = self._get_freestream(X_batch, scaler_X)
            pw_phys = self._to_physical(preds['pw'], 'pw')
            q_inf = freestream['q_inf']

            # Cp = (pw - p_inf) / q_inf; for high Mach, p_inf << pw
            # Upper bound: Cp_max ≈ 1.84 for gamma=1.4
            Cp = pw_phys / (q_inf + 1e-8)
            Cp_max = 1.84
            losses['newtonian'] = F.relu(Cp - Cp_max).pow(2).mean()

        # === 4. FAY-RIDDELL SCALING ===
        if has('qw'):
            freestream = self._get_freestream(X_batch, scaler_X)
            qw_phys = self._to_physical(preds['qw'], 'qw')
            V = freestream['velocity']
            rho = freestream['density']

            # Per-solution max qw should scale as sqrt(rho) * V^3
            qw_max = qw_phys.max(dim=1, keepdim=True).values  # (B, 1, 1)
            scaling = torch.sqrt(rho[:, :1, :].abs() + 1e-8) * V[:, :1, :].abs().pow(3)

            log_qw_max = torch.log10(qw_max.clamp(min=1e-8))
            log_scaling = torch.log10(scaling.clamp(min=1e-8))
            log_ratio = log_qw_max - log_scaling
            # Variance across batch — should be small if scaling holds
            if log_ratio.shape[0] > 1:
                losses['fay_riddell'] = log_ratio.var()

        # === 5. SKIN FRICTION BOUNDS ===
        if has('tw'):
            freestream = self._get_freestream(X_batch, scaler_X)
            tw_phys = self._to_physical(preds['tw'], 'tw')
            q_inf = freestream['q_inf']
            Cf = tw_phys / (q_inf + 1e-8)
            log_Cf = torch.log10(Cf.clamp(min=1e-10))

            losses['cf_bounds'] = (
                F.relu(-5.0 - log_Cf).pow(2).mean() +  # Cf > 1e-5
                F.relu(log_Cf - (-1.0)).pow(2).mean()   # Cf < 0.1
            )

        # === 6. BOUNDARY LAYER CONSISTENCY ===
        if has('theta') and has('me'):
            theta_phys = self._to_physical(preds['theta'], 'theta')
            me_phys = self._to_physical(preds['me'], 'me')

            # theta should be positive (attached flow after cleaning)
            losses['theta_pos'] = F.relu(-preds['theta'] - 2.0).pow(2).mean()

            # Me should be bounded [0, ~6] for hypersonic reentry
            me_log = preds['me'] * self.y_std[self.idx_map['me']] + self.y_mean[self.idx_map['me']]
            me_log = torch.clamp(me_log, min=-10.0, max=20.0)
            me_val = torch.pow(10.0, me_log)
            losses['me_bound'] = F.relu(me_val - 6.0).pow(2).mean()

        return losses


def compute_physics_loss(losses_dict, cfg, epoch=1):
    """
    Weighted sum of physics loss terms with warmup and NaN guard.
    Returns (total_physics_loss, individual_weighted_terms).
    """
    # Linear warmup: 0 at epoch 1, full weight at physics_warmup_epochs
    warmup = min(1.0, epoch / max(1, cfg.physics_warmup_epochs))

    weight_map = {
        'positivity': cfg.lambda_positivity,
        'reynolds_bound': cfg.lambda_reynolds,
        'reynolds_smooth': cfg.lambda_reynolds,
        'newtonian': cfg.lambda_newtonian,
        'fay_riddell': cfg.lambda_fay_riddell,
        'cf_bounds': cfg.lambda_cf_bounds,
        'theta_pos': cfg.lambda_bl_consistency,
        'me_bound': cfg.lambda_bl_consistency,
    }

    total = torch.tensor(0.0)
    weighted = {}
    for name, loss in losses_dict.items():
        w = weight_map.get(name, 0.01)
        if torch.is_tensor(loss):
            # NaN/Inf guard — skip this term if it's bad
            if torch.isnan(loss) or torch.isinf(loss):
                weighted[name] = 0.0
                continue
            total = total.to(loss.device)
            weighted_loss = warmup * w * loss
            total = total + weighted_loss
            weighted[name] = weighted_loss.item()
        else:
            weighted[name] = 0.0

    return total, weighted
