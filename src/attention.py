"""Attention modules for QCPC: standard learnable PE variant."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QCPCConfig
from .latent import truncated_normal_


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
# Standard Multi-Head Attention (no positional encoding inside)
# ---------------------------------------------------------------------------
class StandardAttention(nn.Module):
    """Standard multi-head attention without internal positional encoding.

    Position info is injected at input level via learnable PE.
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
        # When all keys are masked (e.g. padded chunks), softmax over all -inf
        # produces NaN. Replace with zeros so padded positions contribute nothing.
        attn = attn.nan_to_num(0.0)
        out = torch.matmul(attn, V)  # (B, n_h, M, d_h)
        out = out.transpose(1, 2).contiguous().view(B, M, -1)  # (B, M, n_h*d_h)
        return self.W_O(out)


# ---------------------------------------------------------------------------
# Attention Block (Pre-Norm + Attention + FFN)
# ---------------------------------------------------------------------------
class AttentionBlock(nn.Module):
    """Pre-Norm attention block with SwiGLU FFN.

    Uses StandardAttention with learnable PE injected at input level.
    Can operate as cross-attention (query != kv) or self-attention (query == kv).
    """

    def __init__(self, config: QCPCConfig):
        super().__init__()
        D = config.hidden_dim

        self.norm_q = RMSNorm(D)
        self.norm_kv = RMSNorm(D)
        self.norm_ffn = RMSNorm(D)

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
        attn_out = self.attn(q_normed, kv_normed, key_padding_mask=key_padding_mask)
        query = query + attn_out

        # FFN with residual
        query = query + self.ffn(self.norm_ffn(query))

        return query
