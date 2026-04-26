"""
Mamba-Enhanced Physics-Informed Multi-Output Autoencoder.
Architecture from the original notebook, extended with 5 prediction heads.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ============================================================
# Mamba SSM Core with Parallel Scan + Mamba-3 Extensions
# ============================================================

def rotate_half(x):
    """Rotate the second half of the last dimension for RoPE."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """Apply rotary position encoding to tensor x."""
    return x * cos + rotate_half(x) * sin


def parallel_scan_simple(alpha, inp):
    """
    Parallel scan via doubling trick for linear recurrence:
        h_t = alpha_t * h_{t-1} + inp_t
    O(log L) parallel steps instead of O(L) sequential.
    """
    B, L, D, N = alpha.shape
    a = alpha
    b = inp
    for k in range(math.ceil(math.log2(L))):
        stride = 2 ** k
        a_shifted = F.pad(a[:, :-stride], (0, 0, 0, 0, stride, 0), value=0.0)
        b_shifted = F.pad(b[:, :-stride], (0, 0, 0, 0, stride, 0), value=0.0)
        a_shifted[:, :stride] = 1.0
        b = a * b_shifted + b
        a = a * a_shifted
    return b


class SelectiveSSM(nn.Module):
    """
    Selective SSM with Mamba-3 extensions + parallel scan.
    - Complex-valued RoPE on B,C for data-dependent positional encoding
    - Trapezoidal discretization for 2nd-order accuracy
    - BC bias (learned, init=1)
    """

    def __init__(self, d_model, d_state=64, d_conv=4, expand=2,
                 use_rope=False, use_trapezoidal=False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.use_rope = use_rope
        self.use_trapezoidal = use_trapezoidal

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1))

        # Mamba-3: BC bias
        if use_rope or use_trapezoidal:
            self.B_bias = nn.Parameter(torch.ones(d_state))
            self.C_bias = nn.Parameter(torch.ones(d_state))

        if use_rope:
            self.theta_proj = nn.Linear(self.d_inner, d_state // 2, bias=False)
        if use_trapezoidal:
            self.lambda_proj = nn.Linear(self.d_inner, 1, bias=True)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def _compute_rope(self, x, B, C):
        theta = self.theta_proj(x)
        theta_cum = torch.cumsum(theta, dim=1)
        cos_t = torch.cos(theta_cum).repeat(1, 1, 2)
        sin_t = torch.sin(theta_cum).repeat(1, 1, 2)
        return apply_rope(B, cos_t, sin_t), apply_rope(C, cos_t, sin_t)

    def _parallel_selective_scan(self, x, B, C, delta, A):
        batch, seq_len, d_inner = x.shape

        alpha = torch.exp(delta.unsqueeze(-1) * A)
        x_db = x.unsqueeze(-1)
        B_exp = B.unsqueeze(2)
        delta_exp = delta.unsqueeze(-1)
        Bx = B_exp * x_db

        if self.use_trapezoidal:
            lam = torch.sigmoid(self.lambda_proj(x)).unsqueeze(-1)
            Bx_prev = F.pad(Bx[:, :-1], (0, 0, 0, 0, 1, 0), value=0.0)
            gamma = lam * delta_exp
            beta = (1 - lam) * delta_exp * alpha
            inp = gamma * Bx + beta * Bx_prev
        else:
            inp = delta_exp * Bx

        h_all = parallel_scan_simple(alpha, inp)

        C_exp = C.unsqueeze(2)
        y = (h_all * C_exp).sum(dim=-1)
        return y

    def forward(self, x):
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :seq_len]
        x_ssm = F.silu(x_conv.transpose(1, 2))

        ssm_params = self.x_proj(x_ssm)
        B = ssm_params[:, :, :self.d_state]
        C = ssm_params[:, :, self.d_state:2*self.d_state]
        dt = ssm_params[:, :, -1:]

        if hasattr(self, 'B_bias'):
            B = B + self.B_bias
            C = C + self.C_bias

        if self.use_rope:
            B, C = self._compute_rope(x_ssm, B, C)

        delta = F.softplus(self.dt_proj(dt).squeeze(-1))
        A = -torch.exp(self.A_log)

        y = self._parallel_selective_scan(x_ssm, B, C, delta, A)
        y = y + x_ssm * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.out_proj(y)


# ============================================================
# Building Blocks
# ============================================================

class MambaBlock(nn.Module):
    """Pre-norm Mamba block with residual connection."""
    def __init__(self, d_model, d_state, d_conv, expand, use_rope, use_trapezoidal,
                 use_residual_ffn=False, ffn_hidden_dim=128, ffn_dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, use_rope=use_rope, use_trapezoidal=use_trapezoidal
        )
        if ffn_hidden_dim <= 0:
            raise ValueError(f"FFN hidden dim must be positive, got {ffn_hidden_dim}")
        if not 0.0 <= ffn_dropout < 1.0:
            raise ValueError(f"FFN dropout must be in [0, 1), got {ffn_dropout}")
        self.ffn_norm = nn.LayerNorm(d_model) if use_residual_ffn else None
        if use_residual_ffn:
            ffn_layers = [
                nn.Linear(d_model, ffn_hidden_dim),
                nn.SiLU(),
            ]
            if ffn_dropout > 0.0:
                ffn_layers.append(nn.Dropout(ffn_dropout))
            ffn_layers.append(nn.Linear(ffn_hidden_dim, d_model))
            self.ffn = nn.Sequential(*ffn_layers)
        else:
            self.ffn = None

    def forward(self, x):
        x = x + self.ssm(self.norm(x))
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x))
        return x


class MLPBlock(nn.Module):
    """Simple MLP block for ablation comparison."""
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.SiLU(),
            nn.Linear(d_model * expand, d_model)
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class MoEFeedForward(nn.Module):
    """Sparse top-k routed feed-forward layer for Transformer MoE ablations."""
    def __init__(self, d_model, hidden_dim, num_experts=4, top_k=1, dropout=0.0):
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"MoE num_experts must be positive, got {num_experts}")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"MoE top_k must be in [1, num_experts], got {top_k}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"MoE dropout must be in [0, 1), got {dropout}")

        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, d_model),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        orig_shape = x.shape
        flat_x = x.reshape(-1, orig_shape[-1])

        gate_probs = F.softmax(self.gate(flat_x), dim=-1)
        topk_weights, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        flat_out = torch.zeros_like(flat_x)
        for expert_idx, expert in enumerate(self.experts):
            expert_weights = (topk_weights * (topk_idx == expert_idx)).sum(dim=-1)
            token_mask = expert_weights > 0
            if token_mask.any():
                flat_out[token_mask] += (
                    expert(flat_x[token_mask])
                    * expert_weights[token_mask].unsqueeze(-1)
                )

        return flat_out.reshape(orig_shape)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block with optional MoE feed-forward layer."""
    def __init__(self, d_model, n_heads=4, ffn_hidden_dim=128,
                 attention_dropout=0.0, ffn_dropout=0.0,
                 use_moe=False, moe_num_experts=4, moe_top_k=1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if ffn_hidden_dim <= 0:
            raise ValueError(f"Transformer FFN hidden dim must be positive, got {ffn_hidden_dim}")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError(f"Attention dropout must be in [0, 1), got {attention_dropout}")
        if not 0.0 <= ffn_dropout < 1.0:
            raise ValueError(f"Transformer FFN dropout must be in [0, 1), got {ffn_dropout}")

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity()
        self.ffn_norm = nn.LayerNorm(d_model)

        if use_moe:
            self.ffn = MoEFeedForward(
                d_model=d_model,
                hidden_dim=ffn_hidden_dim,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                dropout=ffn_dropout,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_hidden_dim),
                nn.SiLU(),
                nn.Dropout(ffn_dropout) if ffn_dropout > 0.0 else nn.Identity(),
                nn.Linear(ffn_hidden_dim, d_model),
            )

    def forward(self, x):
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_dropout(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class PredictionHead(nn.Module):
    """Per-output prediction head from latent features."""
    def __init__(self, d_in, hidden_dims=None, dropout=0.0, n_outputs=1):
        super().__init__()
        hidden_dims = list(hidden_dims or [64])
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError(f"Prediction head dims must be positive, got {hidden_dims}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"Prediction head dropout must be in [0, 1), got {dropout}")

        # Preserve the exact legacy layout so existing checkpoints still load.
        if hidden_dims == [64] and dropout == 0.0:
            self.net = nn.Sequential(
                nn.Linear(d_in, 64),
                nn.SiLU(),
                nn.LayerNorm(64),
                nn.Linear(64, n_outputs)
            )
            return

        layers = []
        prev_dim = d_in
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            ])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Full Autoencoder
# ============================================================

class MambaAutoencoder(nn.Module):
    """
    Physics-informed multi-output autoencoder with Mamba encoder.
    5 prediction heads: qw, pw, tauw, Me, theta(m)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model

        self.input_proj = nn.Sequential(
            nn.Linear(config.n_features, d),
            nn.LayerNorm(d),
            nn.SiLU()
        )
        self.pos_embed = None

        if config.block_type in ('mamba2', 'mamba3'):
            use_rope = config.use_rope and config.block_type == 'mamba3'
            use_trap = config.use_trapezoidal and config.block_type == 'mamba3'
            self.encoder = nn.ModuleList([
                MambaBlock(
                    d_model=d, d_state=config.d_state,
                    d_conv=config.d_conv, expand=config.expand,
                    use_rope=use_rope, use_trapezoidal=use_trap,
                    use_residual_ffn=config.use_residual_ffn,
                    ffn_hidden_dim=config.ffn_hidden_dim,
                    ffn_dropout=config.ffn_dropout,
                ) for _ in range(config.n_layers)
            ])
        elif config.block_type == 'mlp':
            self.encoder = nn.ModuleList([
                MLPBlock(d_model=d, expand=config.expand)
                for _ in range(config.n_layers)
            ])
        elif config.block_type in ('transformer', 'transformer_moe'):
            self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_len, d))
            nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
            self.encoder = nn.ModuleList([
                TransformerBlock(
                    d_model=d,
                    n_heads=config.n_heads,
                    ffn_hidden_dim=config.transformer_ffn_dim,
                    attention_dropout=config.attention_dropout,
                    ffn_dropout=config.ffn_dropout,
                    use_moe=(config.block_type == 'transformer_moe'),
                    moe_num_experts=config.moe_num_experts,
                    moe_top_k=config.moe_top_k,
                )
                for _ in range(config.n_layers)
            ])
        else:
            raise ValueError(f"Unknown block_type: {config.block_type}")

        self.to_latent = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, config.latent_dim)
        )

        self.recon_head = nn.Sequential(
            nn.Linear(config.latent_dim, d),
            nn.SiLU(),
            nn.Linear(d, config.n_features)
        )

        # Multi-output prediction heads
        self.pred_heads = nn.ModuleDict()
        for name, _, _ in config.target_config:
            self.pred_heads[name] = PredictionHead(
                config.d_model,
                hidden_dims=config.pred_head_hidden_dims,
                dropout=config.pred_head_dropout,
                n_outputs=1,
            )

    def forward(self, x):
        h = self.input_proj(x)
        if self.pos_embed is not None:
            h = h + self.pos_embed[:, :h.shape[1], :]
        for layer in self.encoder:
            if self.training:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        z = self.to_latent(h)

        out = {'recon': self.recon_head(z), 'latent': z}
        for name, head in self.pred_heads.items():
            out[name] = head(h)
        return out
