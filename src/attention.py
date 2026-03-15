"""Attention modules for QCPC: standard learnable PE and decoupled RoPE variants."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QCPCConfig
from .latent import truncated_normal_


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------
def build_rope_cache(
    seq_len: int, dim: int, theta: float = 1000000.0, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE sin/cos tables.

    Returns:
        cos, sin: (seq_len, dim) tensors
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (seq_len, dim//2)
    cos = freqs.cos()  # (seq_len, dim//2)
    sin = freqs.sin()  # (seq_len, dim//2)
    # Duplicate for pairing: [cos0, cos0, cos1, cos1, ...] → (seq_len, dim)
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to tensor x.

    Args:
        x: (..., seq_len, dim) tensor
        cos, sin: (seq_len, dim) precomputed tables

    Returns:
        Rotated tensor with same shape as x
    """
    # Rotate pairs: [x0, x1, x2, x3, ...] → [-x1, x0, -x3, x2, ...]
    x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
    return x * cos + x_rot * sin


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network: gate * silu(gate) * up, then down."""

    def __init__(self, dim: int, intermediate_dim: int, init_scale: float = 0.02):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=False)

        truncated_normal_(self.gate_proj.weight, std=init_scale)
        truncated_normal_(self.up_proj.weight, std=init_scale)
        truncated_normal_(self.down_proj.weight, std=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Mode A: Standard Multi-Head Attention (no positional encoding inside)
# ---------------------------------------------------------------------------
class StandardAttention(nn.Module):
    """Standard multi-head attention without internal positional encoding.

    Used with learnable PE injected at input level (Mode A).
    Works for both cross-attention and self-attention depending on inputs.
    """

    def __init__(self, config: QCPCConfig):
        super().__init__()
        D = config.hidden_dim
        n_h = config.num_heads
        d_h = config.head_dim
        self.n_h = n_h
        self.d_h = d_h
        self.scale = 1.0 / math.sqrt(d_h)

        self.W_Q = nn.Linear(D, n_h * d_h, bias=False)
        self.W_K = nn.Linear(D, n_h * d_h, bias=False)
        self.W_V = nn.Linear(D, n_h * d_h, bias=False)
        self.W_O = nn.Linear(n_h * d_h, D, bias=False)

        for w in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            truncated_normal_(w.weight, std=config.init_scale)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, M, D) query tokens
            key_value: (B, N, D) key/value tokens (same as query for self-attn)
            key_padding_mask: (B, N) mask where True=ignore, False=attend

        Returns:
            output: (B, M, D)
        """
        B, M, _ = query.shape
        N = key_value.shape[1]

        Q = self.W_Q(query).view(B, M, self.n_h, self.d_h).transpose(1, 2)   # (B, n_h, M, d_h)
        K = self.W_K(key_value).view(B, N, self.n_h, self.d_h).transpose(1, 2)  # (B, n_h, N, d_h)
        V = self.W_V(key_value).view(B, N, self.n_h, self.d_h).transpose(1, 2)  # (B, n_h, N, d_h)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, n_h, M, N)

        if key_padding_mask is not None:
            # key_padding_mask: (B, N), True = ignore
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, n_h, M, d_h)
        out = out.transpose(1, 2).contiguous().view(B, M, -1)  # (B, M, n_h*d_h)
        return self.W_O(out)


# ---------------------------------------------------------------------------
# Mode B: Decoupled RoPE Attention
# ---------------------------------------------------------------------------
class DecoupledRoPEAttention(nn.Module):
    """Multi-head attention with decoupled content and RoPE position channels.

    Q = [Q_C ; Q_R], K = [K_C ; K_R]
    score = (Q_C @ K_C^T + Q_R @ K_R^T) / sqrt(d_h + d_R)
    V is standard projection.
    """

    def __init__(self, config: QCPCConfig):
        super().__init__()
        D = config.hidden_dim
        n_h = config.num_heads
        d_h = config.head_dim
        d_R = config.rope_dim
        self.n_h = n_h
        self.d_h = d_h
        self.d_R = d_R
        self.scale = 1.0 / math.sqrt(d_h + d_R)

        # Content projections (no RoPE)
        self.W_QC = nn.Linear(D, n_h * d_h, bias=False)
        self.W_KC = nn.Linear(D, n_h * d_h, bias=False)

        # Position projections (with RoPE)
        self.W_QR = nn.Linear(D, n_h * d_R, bias=False)
        self.W_KR = nn.Linear(D, n_h * d_R, bias=False)

        # Value and output projections
        self.W_V = nn.Linear(D, n_h * d_h, bias=False)
        self.W_O = nn.Linear(n_h * d_h, D, bias=False)

        self.rope_theta = config.rope_theta

        for w in [self.W_QC, self.W_KC, self.W_QR, self.W_KR, self.W_V, self.W_O]:
            truncated_normal_(w.weight, std=config.init_scale)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_positions: torch.Tensor | None = None,
        kv_positions: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, M, D)
            key_value: (B, N, D)
            query_positions: (M,) or None, defaults to 0..M-1
            kv_positions: (N,) or None, defaults to 0..N-1
            key_padding_mask: (B, N) where True=ignore

        Returns:
            output: (B, M, D)
        """
        B, M, _ = query.shape
        N = key_value.shape[1]
        device = query.device

        # Content channels
        Q_C = self.W_QC(query).view(B, M, self.n_h, self.d_h).transpose(1, 2)   # (B, n_h, M, d_h)
        K_C = self.W_KC(key_value).view(B, N, self.n_h, self.d_h).transpose(1, 2)  # (B, n_h, N, d_h)

        # Position channels
        Q_R = self.W_QR(query).view(B, M, self.n_h, self.d_R).transpose(1, 2)   # (B, n_h, M, d_R)
        K_R = self.W_KR(key_value).view(B, N, self.n_h, self.d_R).transpose(1, 2)  # (B, n_h, N, d_R)

        # Build RoPE tables
        max_pos = max(M, N)
        cos, sin = build_rope_cache(max_pos, self.d_R, self.rope_theta, device)

        # Apply RoPE to position channels
        q_cos = cos[:M].unsqueeze(0).unsqueeze(0)  # (1, 1, M, d_R)
        q_sin = sin[:M].unsqueeze(0).unsqueeze(0)
        k_cos = cos[:N].unsqueeze(0).unsqueeze(0)  # (1, 1, N, d_R)
        k_sin = sin[:N].unsqueeze(0).unsqueeze(0)

        Q_R = apply_rope(Q_R, q_cos, q_sin)
        K_R = apply_rope(K_R, k_cos, k_sin)

        # Attention scores: content + position
        score_C = torch.matmul(Q_C, K_C.transpose(-2, -1))  # (B, n_h, M, N)
        score_R = torch.matmul(Q_R, K_R.transpose(-2, -1))  # (B, n_h, M, N)
        scores = (score_C + score_R) * self.scale

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)

        # Value projection (standard, no position encoding)
        V = self.W_V(key_value).view(B, N, self.n_h, self.d_h).transpose(1, 2)  # (B, n_h, N, d_h)
        out = torch.matmul(attn, V)  # (B, n_h, M, d_h)
        out = out.transpose(1, 2).contiguous().view(B, M, -1)
        return self.W_O(out)


# ---------------------------------------------------------------------------
# Unified Attention Block (Pre-Norm + Attention + FFN)
# ---------------------------------------------------------------------------
class AttentionBlock(nn.Module):
    """Pre-Norm attention block with SwiGLU FFN.

    Dispatches to StandardAttention or DecoupledRoPEAttention based on config.
    Can operate as cross-attention (query != kv) or self-attention (query == kv).
    """

    def __init__(self, config: QCPCConfig):
        super().__init__()
        D = config.hidden_dim

        self.norm_q = RMSNorm(D)
        self.norm_kv = RMSNorm(D)
        self.norm_ffn = RMSNorm(D)

        self.use_decoupled_rope = config.use_decoupled_rope
        if config.use_decoupled_rope:
            self.attn = DecoupledRoPEAttention(config)
        else:
            self.attn = StandardAttention(config)

        self.ffn = SwiGLUFFN(D, config.ffn_intermediate_dim, config.init_scale)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, M, D)
            key_value: (B, N, D) (same as query for self-attention)
            key_padding_mask: (B, N)

        Returns:
            output: (B, M, D)
        """
        # Pre-norm
        q_normed = self.norm_q(query)
        kv_normed = self.norm_kv(key_value)

        # Attention with residual
        if self.use_decoupled_rope:
            attn_out = self.attn(q_normed, kv_normed, key_padding_mask=key_padding_mask)
        else:
            attn_out = self.attn(q_normed, kv_normed, key_padding_mask=key_padding_mask)

        query = query + attn_out

        # FFN with residual
        query = query + self.ffn(self.norm_ffn(query))

        return query
