"""Unit tests for LatentArray and QueryMapper."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.latent import LatentArray, QueryMapper, truncated_normal_

B, M, D, L = 2, 8, 1024, 16  # batch, memory_tokens, hidden_dim, prompt_len


def test_truncated_normal():
    """Test truncated normal initialization is within bounds."""
    t = torch.empty(1000, 1000)
    truncated_normal_(t, std=0.02)
    assert t.min() >= -0.04, f"Min {t.min()} below -2*std"
    assert t.max() <= 0.04, f"Max {t.max()} above 2*std"
    assert abs(t.mean()) < 0.01, f"Mean {t.mean()} too far from 0"
    print("[PASS] test_truncated_normal")


def test_query_mapper_zero_init():
    """Test QueryMapper second layer is zero-initialized."""
    cfg = QCPCConfig(num_memory_tokens=M, hidden_dim=D)
    mapper = QueryMapper(cfg)

    assert torch.all(mapper.fc2.weight == 0), "fc2 weight should be zero"
    assert torch.all(mapper.fc2.bias == 0), "fc2 bias should be zero"

    # Forward should produce zeros at initialization
    x = torch.randn(B, D)
    out = mapper(x)
    assert out.shape == (B, M, D), f"Expected ({B}, {M}, {D}), got {out.shape}"
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7), \
        "Zero-init mapper should output zeros"
    print("[PASS] test_query_mapper_zero_init")


def test_latent_no_bias():
    """Test latent array without prompt bias (use_prompt_bias=False)."""
    cfg = QCPCConfig(use_prompt_bias=False, num_memory_tokens=M, hidden_dim=D)
    latent = LatentArray(cfg)

    Z = latent(batch_size=B)
    assert Z.shape == (B, M, D), f"Expected ({B}, {M}, {D}), got {Z.shape}"

    # All samples in batch should be identical (same Z_base expanded)
    assert torch.equal(Z[0], Z[1]), "Without bias, batch elements should be identical"
    print("[PASS] test_latent_no_bias")


def test_latent_with_bias():
    """Test latent array with prompt bias (use_prompt_bias=True)."""
    cfg = QCPCConfig(use_prompt_bias=True, num_memory_tokens=M, hidden_dim=D)
    latent = LatentArray(cfg)

    prompt_embeds = torch.randn(B, L, D)
    Z = latent(batch_size=B, prompt_embeds=prompt_embeds)
    assert Z.shape == (B, M, D), f"Expected ({B}, {M}, {D}), got {Z.shape}"

    # At init, alpha=0, so Z should equal Z_base
    Z_base_expanded = latent.z_base.unsqueeze(0).expand(B, -1, -1)
    assert torch.allclose(Z, Z_base_expanded, atol=1e-6), \
        "At init (alpha=0), Z should equal Z_base"
    print("[PASS] test_latent_with_bias")


def test_latent_with_bias_and_mask():
    """Test latent array with prompt bias and attention mask."""
    cfg = QCPCConfig(use_prompt_bias=True, num_memory_tokens=M, hidden_dim=D)
    latent = LatentArray(cfg)

    prompt_embeds = torch.randn(B, L, D)
    # Mask: first sample has 10 valid tokens, second has 5
    prompt_mask = torch.zeros(B, L)
    prompt_mask[0, :10] = 1
    prompt_mask[1, :5] = 1

    Z = latent(batch_size=B, prompt_embeds=prompt_embeds, prompt_mask=prompt_mask)
    assert Z.shape == (B, M, D)
    print("[PASS] test_latent_with_bias_and_mask")


def test_latent_gradient_flow():
    """Test gradients flow through trainable parameters."""
    cfg = QCPCConfig(use_prompt_bias=True, num_memory_tokens=M, hidden_dim=D)
    latent = LatentArray(cfg)

    # Manually set alpha to non-zero to enable gradient flow through mapper
    with torch.no_grad():
        latent.alpha.fill_(1.0)

    prompt_embeds = torch.randn(B, L, D)
    Z = latent(batch_size=B, prompt_embeds=prompt_embeds)
    loss = Z.sum()
    loss.backward()

    assert latent.z_base.grad is not None, "z_base should have gradient"
    assert latent.alpha.grad is not None, "alpha should have gradient"
    assert latent.query_mapper.fc1.weight.grad is not None, "fc1 should have gradient"
    print("[PASS] test_latent_gradient_flow")


def test_both_combos():
    """Test both config combinations produce correct output shapes."""
    combos = [
        (False, "Baseline"),
        (True, "Prompt Bias"),
    ]
    for bias, name in combos:
        cfg = QCPCConfig(
            use_prompt_bias=bias,
            num_memory_tokens=M,
            hidden_dim=D,
        )
        latent = LatentArray(cfg)

        prompt_embeds = torch.randn(B, L, D) if bias else None
        Z = latent(batch_size=B, prompt_embeds=prompt_embeds)
        assert Z.shape == (B, M, D), f"{name}: shape mismatch"
        print(f"  [PASS] {name}: Z shape = {Z.shape}")
    print("[PASS] test_both_combos")


if __name__ == "__main__":
    test_truncated_normal()
    test_query_mapper_zero_init()
    test_latent_no_bias()
    test_latent_with_bias()
    test_latent_with_bias_and_mask()
    test_latent_gradient_flow()
    test_both_combos()
    print("\n=== All latent tests PASSED ===")
