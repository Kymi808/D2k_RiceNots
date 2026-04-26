"""
Configuration for Mamba-Enhanced Physics-Informed Multi-Output Autoencoder.
Sized for 4x L40S (48GB each, 192GB total) on Rice NOTS cluster.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Configuration for the Mamba CFD Surrogate model, data pipeline, and training."""
    # --- Data ---
    points_per_solution: int = 50176
    seq_len: int = 8192
    partition_stride: int = 6400        # overlap = seq_len - stride = 1792 points
    train_frac: float = 0.80
    val_frac: float = 0.10

    # --- Model ---
    block_type: str = "mamba3"
    d_model: int = 64
    d_state: int = 64
    d_conv: int = 4
    n_layers: int = 4
    latent_dim: int = 16
    expand: int = 2
    use_rope: bool = True
    use_trapezoidal: bool = True
    pred_head_hidden_dims: List[int] = field(default_factory=lambda: [64])
    pred_head_dropout: float = 0.0
    use_residual_ffn: bool = False
    ffn_hidden_dim: int = 128
    ffn_dropout: float = 0.0
    normalize_qw_by_rhov3: bool = False
    n_heads: int = 4
    transformer_ffn_dim: int = 128
    attention_dropout: float = 0.0
    moe_num_experts: int = 4
    moe_top_k: int = 1

    # --- Input features ---
    x_cols: List[str] = field(default_factory=lambda: [
        'X', 'Y', 'Z',
        'velocity (m/s)', 'density (kg/m^3)',
        'aoa (degrees)', 'dynamic_pressure (Pa)',
    ])

    # --- Outputs (all enabled for physics-informed training) ---
    predict_qw: bool = True
    predict_pw: bool = True
    predict_tw: bool = True
    predict_me: bool = True
    predict_theta: bool = True

    # --- Loss weights ---
    # Data loss weights (qw prioritized 3x)
    w_qw: float = 3.0
    w_pw: float = 1.0
    w_tw: float = 1.0
    w_me: float = 1.0
    w_theta: float = 1.0

    lambda_recon: float = 0.15
    lambda_reynolds: float = 0.0075
    lambda_newtonian: float = 0.003
    lambda_fay_riddell: float = 0.0015
    lambda_cf_bounds: float = 0.003
    lambda_bl_consistency: float = 0.0015
    lambda_positivity: float = 0.0015

    # Physics loss warmup: linearly ramp from 0 to full weight over this many epochs
    physics_warmup_epochs: int = 70

    # --- Training ---
    batch_size: int = 4                 # total across GPUs (1 per GPU x 4 GPUs)
    lr: float = 1e-3
    weight_decay: float = 8e-3
    epochs: int = 300
    patience: int = 25

    # --- Data split ---
    split_seed: int = 123

    # --- Cluster ---
    num_gpus: int = 4
    num_workers: int = 4                # dataloader workers per GPU

    @property
    def n_features(self):
        return len(self.x_cols)

    @property
    def target_config(self):
        """Returns list of (name, csv_col, loss_weight) for active targets."""
        targets = []
        if self.predict_qw:
            targets.append(('qw', 'qw (W/m^2)', self.w_qw))
        if self.predict_pw:
            targets.append(('pw', 'pw (Pa)', self.w_pw))
        if self.predict_tw:
            targets.append(('tw', 'tauw (Pa)', self.w_tw))
        if self.predict_me:
            targets.append(('me', 'Me', self.w_me))
        if self.predict_theta:
            targets.append(('theta', 'theta (m)', self.w_theta))
        return targets

    @property
    def y_col_names(self):
        return [t[0] for t in self.target_config]

    @property
    def y_csv_cols(self):
        return [t[1] for t in self.target_config]

    @property
    def y_weights(self):
        return [t[2] for t in self.target_config]

    @property
    def n_outputs(self):
        return len(self.target_config)

    @property
    def n_partitions(self):
        """Number of overlapping partitions to cover all mesh points."""
        import math
        return max(1, math.ceil(
            (self.points_per_solution - self.seq_len)
            / self.partition_stride
        ) + 1)

    @property
    def batch_per_gpu(self):
        return max(1, self.batch_size // self.num_gpus)
