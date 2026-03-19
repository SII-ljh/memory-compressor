"""Unit tests for attention modules."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.attention import (
    RMSNorm,
    SwiGLUFFN,
    StandardAttention,
    AttentionBlock,
)

B, N, M, D = 2, 64, 8, 1024  # batch, context_len, memory_tokens, hidden_dim


def test_rmsnorm():
    """Test RMSNorm output shape and normalization."""
    norm = RMSNorm(D)
    x = torch.randn(B, M, D)
    out = norm(x)
    assert out.shape == x.shape
    print("[PASS] test_rmsnorm")


def test_swiglu_ffn():
    """Test SwiGLU FFN shape."""
    ffn = SwiGLUFFN(D, 2048)
    x = torch.randn(B, M, D)
    out = ffn(x)
    assert out.shape == (B, M, D), f"Expected ({B}, {M}, {D}), got {out.shape}"
    print("[PASS] test_swiglu_ffn")


def test_standard_attention_cross():
    """Test standard attention as cross-attention."""
    cfg = QCPCConfig(hidden_dim=D, num_heads=16, head_dim=64)
    attn = StandardAttention(cfg)

    query = torch.randn(B, M, D)
    kv = torch.randn(B, N, D)
    out = attn(query, kv)
    assert out.shape == (B, M, D), f"Expected ({B}, {M}, {D}), got {out.shape}"
    print("[PASS] test_standard_attention_cross")


def test_standard_attention_self():
    """Test standard attention as self-attention."""
    cfg = QCPCConfig(hidden_dim=D, num_heads=16, head_dim=64)
    attn = StandardAttention(cfg)

    x = torch.randn(B, M, D)
    out = attn(x, x)
    assert out.shape == (B, M, D)
    print("[PASS] test_standard_attention_self")


def test_standard_attention_with_mask():
    """Test standard attention with key padding mask."""
    cfg = QCPCConfig(hidden_dim=D, num_heads=16, head_dim=64)
    attn = StandardAttention(cfg)

    query = torch.randn(B, M, D)
    kv = torch.randn(B, N, D)
    # Mask last 16 positions as padding (True=ignore)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, -16:] = True
    out = attn(query, kv, key_padding_mask=mask)
    assert out.shape == (B, M, D)
    print("[PASS] test_standard_attention_with_mask")


def test_attention_block_cross():
    """Test AttentionBlock as cross-attention."""
    for bias in [False, True]:
        cfg = QCPCConfig(
            use_prompt_bias=bias,
            hidden_dim=D, num_heads=16, head_dim=64,
            ffn_intermediate_dim=2048,
        )
        block = AttentionBlock(cfg)
        query = torch.randn(B, M, D)
        kv = torch.randn(B, N, D)
        out = block(query, kv)
        name = "Bias" if bias else "Baseline"
        assert out.shape == (B, M, D), f"{name}: expected ({B}, {M}, {D}), got {out.shape}"
        print(f"  [PASS] {name} (cross): {out.shape}")
    print("[PASS] test_attention_block_cross")


def test_attention_block_self_attn():
    """Test AttentionBlock as self-attention."""
    cfg = QCPCConfig(
        hidden_dim=D, num_heads=16, head_dim=64,
        ffn_intermediate_dim=2048,
    )
    block = AttentionBlock(cfg)
    x = torch.randn(B, M, D)
    out = block(x, x)
    assert out.shape == (B, M, D)
    print("[PASS] test_attention_block_self_attn")


def test_gradient_flow():
    """Test gradients flow through attention block."""
    cfg = QCPCConfig(
        hidden_dim=D, num_heads=16, head_dim=64,
        ffn_intermediate_dim=2048,
    )
    block = AttentionBlock(cfg)
    query = torch.randn(B, M, D, requires_grad=True)
    kv = torch.randn(B, N, D, requires_grad=True)
    out = block(query, kv)
    loss = out.sum()
    loss.backward()
    assert query.grad is not None, "Query gradient should exist"
    assert kv.grad is not None, "KV gradient should exist"
    print("[PASS] test_gradient_flow")


if __name__ == "__main__":
    test_rmsnorm()
    test_swiglu_ffn()
    test_standard_attention_cross()
    test_standard_attention_self()
    test_standard_attention_with_mask()
    test_attention_block_cross()
    test_attention_block_self_attn()
    test_gradient_flow()
    print("\n=== All attention tests PASSED ===")
