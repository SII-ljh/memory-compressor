"""Unit tests for attention modules."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.attention import (
    build_rope_cache,
    apply_rope,
    RMSNorm,
    SwiGLUFFN,
    StandardAttention,
    DecoupledRoPEAttention,
    AttentionBlock,
)

B, N, M, D = 2, 64, 8, 1024  # batch, context_len, memory_tokens, hidden_dim


def test_rope_cache():
    """Test RoPE cache shapes and properties."""
    dim = 64
    seq_len = 128
    cos, sin = build_rope_cache(seq_len, dim)
    assert cos.shape == (seq_len, dim), f"cos shape: {cos.shape}"
    assert sin.shape == (seq_len, dim), f"sin shape: {sin.shape}"
    # cos^2 + sin^2 should be 1
    assert torch.allclose(cos ** 2 + sin ** 2, torch.ones_like(cos), atol=1e-5)
    print("[PASS] test_rope_cache")


def test_apply_rope():
    """Test RoPE application preserves shape and norms approximately."""
    dim = 64
    x = torch.randn(B, 4, 16, dim)  # (B, heads, seq, dim)
    cos, sin = build_rope_cache(16, dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape
    # RoPE is a rotation, norms should be approximately preserved
    x_norm = x.norm(dim=-1)
    out_norm = out.norm(dim=-1)
    assert torch.allclose(x_norm, out_norm, atol=1e-4), "RoPE should preserve vector norms"
    print("[PASS] test_apply_rope")


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
    cfg = QCPCConfig(use_decoupled_rope=False, hidden_dim=D, num_heads=16, head_dim=64)
    attn = StandardAttention(cfg)

    query = torch.randn(B, M, D)
    kv = torch.randn(B, N, D)
    out = attn(query, kv)
    assert out.shape == (B, M, D), f"Expected ({B}, {M}, {D}), got {out.shape}"
    print("[PASS] test_standard_attention_cross")


def test_standard_attention_self():
    """Test standard attention as self-attention."""
    cfg = QCPCConfig(use_decoupled_rope=False, hidden_dim=D, num_heads=16, head_dim=64)
    attn = StandardAttention(cfg)

    x = torch.randn(B, M, D)
    out = attn(x, x)
    assert out.shape == (B, M, D)
    print("[PASS] test_standard_attention_self")


def test_standard_attention_with_mask():
    """Test standard attention with key padding mask."""
    cfg = QCPCConfig(use_decoupled_rope=False, hidden_dim=D, num_heads=16, head_dim=64)
    attn = StandardAttention(cfg)

    query = torch.randn(B, M, D)
    kv = torch.randn(B, N, D)
    # Mask last 16 positions as padding (True=ignore)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, -16:] = True
    out = attn(query, kv, key_padding_mask=mask)
    assert out.shape == (B, M, D)
    print("[PASS] test_standard_attention_with_mask")


def test_decoupled_rope_attention_cross():
    """Test decoupled RoPE attention as cross-attention."""
    cfg = QCPCConfig(use_decoupled_rope=True, hidden_dim=D, num_heads=16, head_dim=64, rope_dim=64)
    attn = DecoupledRoPEAttention(cfg)

    query = torch.randn(B, M, D)
    kv = torch.randn(B, N, D)
    out = attn(query, kv)
    assert out.shape == (B, M, D), f"Expected ({B}, {M}, {D}), got {out.shape}"
    print("[PASS] test_decoupled_rope_attention_cross")


def test_decoupled_rope_attention_self():
    """Test decoupled RoPE attention as self-attention."""
    cfg = QCPCConfig(use_decoupled_rope=True, hidden_dim=D, num_heads=16, head_dim=64, rope_dim=64)
    attn = DecoupledRoPEAttention(cfg)

    x = torch.randn(B, M, D)
    out = attn(x, x)
    assert out.shape == (B, M, D)
    print("[PASS] test_decoupled_rope_attention_self")


def test_decoupled_rope_attention_with_mask():
    """Test decoupled RoPE attention with key padding mask."""
    cfg = QCPCConfig(use_decoupled_rope=True, hidden_dim=D, num_heads=16, head_dim=64, rope_dim=64)
    attn = DecoupledRoPEAttention(cfg)

    query = torch.randn(B, M, D)
    kv = torch.randn(B, N, D)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, -16:] = True
    out = attn(query, kv, key_padding_mask=mask)
    assert out.shape == (B, M, D)
    print("[PASS] test_decoupled_rope_attention_with_mask")


def test_attention_block_all_four_combos():
    """Test AttentionBlock for all 4 mode combinations (cross-attention)."""
    combos = [
        (False, False, "Baseline (cross)"),
        (False, True, "PE + Bias (cross)"),
        (True, False, "RoPE (cross)"),
        (True, True, "Full (cross)"),
    ]
    for rope, bias, name in combos:
        cfg = QCPCConfig(
            use_decoupled_rope=rope,
            use_prompt_bias=bias,
            hidden_dim=D, num_heads=16, head_dim=64,
            rope_dim=64, ffn_intermediate_dim=2048,
        )
        block = AttentionBlock(cfg)
        query = torch.randn(B, M, D)
        kv = torch.randn(B, N, D)
        out = block(query, kv)
        assert out.shape == (B, M, D), f"{name}: expected ({B}, {M}, {D}), got {out.shape}"
        print(f"  [PASS] {name}: {out.shape}")
    print("[PASS] test_attention_block_all_four_combos")


def test_attention_block_self_attn():
    """Test AttentionBlock as self-attention."""
    for rope in [True, False]:
        cfg = QCPCConfig(
            use_decoupled_rope=rope,
            hidden_dim=D, num_heads=16, head_dim=64,
            rope_dim=64, ffn_intermediate_dim=2048,
        )
        block = AttentionBlock(cfg)
        x = torch.randn(B, M, D)
        out = block(x, x)
        assert out.shape == (B, M, D)
        mode = "RoPE" if rope else "Standard"
        print(f"  [PASS] Self-attn ({mode}): {out.shape}")
    print("[PASS] test_attention_block_self_attn")


def test_gradient_flow():
    """Test gradients flow through attention block."""
    cfg = QCPCConfig(
        use_decoupled_rope=True,
        hidden_dim=D, num_heads=16, head_dim=64,
        rope_dim=64, ffn_intermediate_dim=2048,
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
    test_rope_cache()
    test_apply_rope()
    test_rmsnorm()
    test_swiglu_ffn()
    test_standard_attention_cross()
    test_standard_attention_self()
    test_standard_attention_with_mask()
    test_decoupled_rope_attention_cross()
    test_decoupled_rope_attention_self()
    test_decoupled_rope_attention_with_mask()
    test_attention_block_all_four_combos()
    test_attention_block_self_attn()
    test_gradient_flow()
    print("\n=== All attention tests PASSED ===")
