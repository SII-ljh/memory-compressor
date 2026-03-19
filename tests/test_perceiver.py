"""Unit tests for Perceiver IO main body."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.perceiver import PerceiverIO

B, N, M, D, L = 2, 64, 8, 1024, 16


def _make_config(bias: bool) -> QCPCConfig:
    return QCPCConfig(
        use_prompt_bias=bias,
        hidden_dim=D,
        num_heads=16,
        head_dim=64,
        num_memory_tokens=M,
        num_process_layers=2,  # use 2 for fast testing
        query_mapper_mid_dim=512,
        ffn_intermediate_dim=2048,
        max_position_embeddings=128,
        init_scale=0.02,
    )


def test_perceiver_baseline():
    """Test baseline mode: no prompt bias."""
    cfg = _make_config(bias=False)
    model = PerceiverIO(cfg)

    text_embeds = torch.randn(B, N, D)
    out = model(text_embeds)
    assert out.shape == (B, M, D), f"Expected ({B}, {M}, {D}), got {out.shape}"
    print("[PASS] test_perceiver_baseline")


def test_perceiver_bias():
    """Test prompt bias mode."""
    cfg = _make_config(bias=True)
    model = PerceiverIO(cfg)

    text_embeds = torch.randn(B, N, D)
    prompt_embeds = torch.randn(B, L, D)
    out = model(text_embeds, prompt_embeds=prompt_embeds)
    assert out.shape == (B, M, D)
    print("[PASS] test_perceiver_bias")


def test_perceiver_with_masks():
    """Test with both text and prompt masks."""
    cfg = _make_config(bias=True)
    model = PerceiverIO(cfg)

    text_embeds = torch.randn(B, N, D)
    text_mask = torch.ones(B, N)
    text_mask[0, -10:] = 0  # pad last 10 for first sample
    text_mask[1, -20:] = 0  # pad last 20 for second sample

    prompt_embeds = torch.randn(B, L, D)
    prompt_mask = torch.ones(B, L)
    prompt_mask[0, -3:] = 0

    out = model(text_embeds, text_mask=text_mask,
                prompt_embeds=prompt_embeds, prompt_mask=prompt_mask)
    assert out.shape == (B, M, D)
    assert not torch.isnan(out).any(), "Output should not contain NaN"
    print("[PASS] test_perceiver_with_masks")


def test_perceiver_gradient_flow():
    """Test gradients flow through the entire Perceiver IO."""
    cfg = _make_config(bias=True)
    model = PerceiverIO(cfg)

    # Set alpha to non-zero for gradient flow through prompt bias
    with torch.no_grad():
        model.latent.alpha.fill_(1.0)

    text_embeds = torch.randn(B, N, D, requires_grad=True)
    prompt_embeds = torch.randn(B, L, D, requires_grad=True)
    out = model(text_embeds, prompt_embeds=prompt_embeds)
    loss = out.sum()
    loss.backward()

    assert text_embeds.grad is not None, "Text gradients should flow"
    assert prompt_embeds.grad is not None, "Prompt gradients should flow"
    assert model.latent.z_base.grad is not None, "Z_base gradient should exist"
    assert model.latent.alpha.grad is not None, "Alpha gradient should exist"
    print("[PASS] test_perceiver_gradient_flow")


def test_perceiver_both_combos():
    """Test both config combinations."""
    combos = [
        (False, "Baseline"),
        (True, "Prompt Bias"),
    ]
    for bias, name in combos:
        cfg = _make_config(bias)
        model = PerceiverIO(cfg)

        text_embeds = torch.randn(B, N, D)
        prompt_embeds = torch.randn(B, L, D) if bias else None

        out = model(text_embeds, prompt_embeds=prompt_embeds)
        assert out.shape == (B, M, D), f"{name}: shape mismatch"
        assert not torch.isnan(out).any(), f"{name}: NaN detected"
        print(f"  [PASS] {name}: {out.shape}")
    print("[PASS] test_perceiver_both_combos")


def test_perceiver_param_count():
    """Print param count for each mode (informational)."""
    for bias, name in [(False, "Baseline"), (True, "Prompt Bias")]:
        cfg = _make_config(bias)
        model = PerceiverIO(cfg)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name}: {total:,} total params, {trainable:,} trainable")
    print("[PASS] test_perceiver_param_count")


if __name__ == "__main__":
    test_perceiver_baseline()
    test_perceiver_bias()
    test_perceiver_with_masks()
    test_perceiver_gradient_flow()
    test_perceiver_both_combos()
    test_perceiver_param_count()
    print("\n=== All perceiver tests PASSED ===")
