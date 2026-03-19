"""End-to-end forward pass tests for the full QCPC model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.model import QCPC

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B")
B, N, M, L, T = 2, 64, 8, 16, 10  # batch, context, memory, prompt, target
D = 1024


def _make_config(bias: bool) -> QCPCConfig:
    return QCPCConfig(
        use_prompt_bias=bias,
        qwen3_model_path=MODEL_PATH,
        hidden_dim=D,
        num_heads=16,
        head_dim=64,
        num_memory_tokens=M,
        num_process_layers=2,
        query_mapper_mid_dim=512,
        ffn_intermediate_dim=2048,
        max_position_embeddings=128,
    )


def test_e2e_baseline_inference():
    """Test full pipeline: Baseline mode, inference (no target)."""
    cfg = _make_config(bias=False)
    model = QCPC(cfg)

    context_ids = torch.randint(0, 1000, (B, N))
    result = model(context_ids=context_ids)

    assert "logits" in result
    assert "memory_tokens" in result
    assert result["memory_tokens"].shape == (B, M, D)
    # Inference: no prompt, so total = 1 + M + 1 = M+2
    assert result["logits"].shape[1] == M + 2
    print(f"[PASS] test_e2e_baseline_inference: logits={result['logits'].shape}")


def test_e2e_baseline_training():
    """Test full pipeline: Baseline mode, training (with target)."""
    cfg = _make_config(bias=False)
    model = QCPC(cfg)
    model.set_stage(1)

    context_ids = torch.randint(0, 1000, (B, N))
    target_ids = torch.randint(0, 1000, (B, T))

    result = model(context_ids=context_ids, target_ids=target_ids)
    assert "loss" in result
    assert result["loss"].ndim == 0
    assert not torch.isnan(result["loss"])

    # Backward should work
    result["loss"].backward()

    # Check gradients flow to perceiver but not embedding/decoder
    assert model.perceiver.latent.z_base.grad is not None, "z_base should have grad"
    for param in model.embedding.parameters():
        assert param.grad is None, "Embedding should be frozen"

    print(f"[PASS] test_e2e_baseline_training: loss={result['loss'].item():.4f}")


def test_e2e_bias_inference():
    """Test full pipeline: Prompt Bias mode, inference."""
    cfg = _make_config(bias=True)
    model = QCPC(cfg)

    context_ids = torch.randint(0, 1000, (B, N))
    prompt_ids = torch.randint(0, 1000, (B, L))

    result = model(context_ids=context_ids, prompt_ids=prompt_ids)
    assert result["memory_tokens"].shape == (B, M, D)
    # Total = 1 + M + 1 + L = M + L + 2
    assert result["logits"].shape[1] == M + L + 2
    print(f"[PASS] test_e2e_bias_inference: logits={result['logits'].shape}")


def test_e2e_bias_training():
    """Test full pipeline: Prompt Bias mode, training with QA data."""
    cfg = _make_config(bias=True)
    model = QCPC(cfg)
    model.set_stage(2)

    context_ids = torch.randint(0, 1000, (B, N))
    prompt_ids = torch.randint(0, 1000, (B, L))
    target_ids = torch.randint(0, 1000, (B, T))

    result = model(
        context_ids=context_ids,
        prompt_ids=prompt_ids,
        target_ids=target_ids,
    )
    assert "loss" in result
    result["loss"].backward()

    assert model.perceiver.latent.z_base.grad is not None
    print(f"[PASS] test_e2e_bias_training: loss={result['loss'].item():.4f}")


def test_e2e_both_combos():
    """Test both config combinations in training mode."""
    combos = [
        (False, "Baseline"),
        (True, "Prompt Bias"),
    ]
    for bias, name in combos:
        cfg = _make_config(bias)
        model = QCPC(cfg)
        stage = 2 if bias else 1
        model.set_stage(stage)

        context_ids = torch.randint(0, 1000, (B, N))
        prompt_ids = torch.randint(0, 1000, (B, L)) if bias else None
        target_ids = torch.randint(0, 1000, (B, T))

        result = model(
            context_ids=context_ids,
            prompt_ids=prompt_ids,
            target_ids=target_ids,
        )
        assert "loss" in result
        assert not torch.isnan(result["loss"]), f"{name}: NaN loss"
        result["loss"].backward()
        print(f"  [PASS] {name}: loss={result['loss'].item():.4f}")
    print("[PASS] test_e2e_both_combos")


def test_e2e_with_masks():
    """Test with context and prompt masks."""
    cfg = _make_config(bias=True)
    model = QCPC(cfg)
    model.set_stage(2)

    context_ids = torch.randint(0, 1000, (B, N))
    context_mask = torch.ones(B, N)
    context_mask[0, -10:] = 0

    prompt_ids = torch.randint(0, 1000, (B, L))
    prompt_mask = torch.ones(B, L)
    prompt_mask[1, -3:] = 0

    target_ids = torch.randint(0, 1000, (B, T))
    target_mask = torch.ones(B, T)
    target_mask[0, -2:] = 0

    result = model(
        context_ids=context_ids,
        context_mask=context_mask,
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        target_ids=target_ids,
        target_mask=target_mask,
    )
    assert not torch.isnan(result["loss"])
    print(f"[PASS] test_e2e_with_masks: loss={result['loss'].item():.4f}")


def test_stage_freeze_logic():
    """Test that set_stage correctly freezes/unfreezes parameters."""
    cfg = _make_config(bias=True)
    model = QCPC(cfg)

    # Stage 1: QueryMapper and alpha should be frozen
    model.set_stage(1)
    for name, param in model.perceiver.named_parameters():
        if "query_mapper" in name or "alpha" in name:
            assert not param.requires_grad, f"Stage 1: {name} should be frozen"
        else:
            assert param.requires_grad, f"Stage 1: {name} should be trainable"

    # Embedding and decoder always frozen
    for param in model.embedding.parameters():
        assert not param.requires_grad
    for param in model.decoder.parameters():
        assert not param.requires_grad

    # Stage 2: Everything in perceiver should be trainable
    model.set_stage(2)
    for name, param in model.perceiver.named_parameters():
        assert param.requires_grad, f"Stage 2: {name} should be trainable"
    print("[PASS] test_stage_freeze_logic")


def test_param_count():
    """Print parameter counts for informational purposes."""
    cfg = _make_config(bias=True)
    model = QCPC(cfg)
    model.set_stage(1)
    counts = model.count_params()
    print(f"  Embedding: {counts['embedding']['total']:,} (trainable: {counts['embedding']['trainable']:,})")
    print(f"  Perceiver: {counts['perceiver']['total']:,} (trainable: {counts['perceiver']['trainable']:,})")
    print(f"  Decoder:   {counts['decoder']['total']:,} (trainable: {counts['decoder']['trainable']:,})")
    print(f"  Total:     {counts['total']:,} (trainable: {counts['total_trainable']:,})")
    print("[PASS] test_param_count")


def test_multi_chunk_padded_no_nan():
    """Test that padded chunks in multi-chunk mode do NOT produce NaN."""
    cfg = _make_config(bias=False)
    model = QCPC(cfg)
    model.set_stage(1)

    K, chunk_N = 4, N  # 4 chunks of length N

    chunk_ids = torch.randint(0, 1000, (B, K, chunk_N))
    chunk_mask = torch.ones(B, K, chunk_N, dtype=torch.long)

    # Simulate padding: second sample only has 2 valid chunks out of 4
    chunk_ids[1, 2:] = 0
    chunk_mask[1, 2:] = 0

    target_ids = torch.randint(0, 1000, (B, T))
    target_mask = torch.ones(B, T, dtype=torch.long)

    result = model(
        chunk_ids=chunk_ids,
        chunk_mask=chunk_mask,
        target_ids=target_ids,
        target_mask=target_mask,
    )

    assert not torch.isnan(result["loss"]), (
        "NaN loss from padded multi-chunk! "
        "This means softmax over all-masked keys produced NaN."
    )
    assert not torch.isinf(result["loss"]), "Inf loss"

    result["loss"].backward()

    # Perceiver grads should be finite
    for name, p in model.perceiver.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"

    print(f"[PASS] test_multi_chunk_padded_no_nan: loss={result['loss'].item():.4f}")


def test_all_masked_context_no_nan():
    """Test that a fully-masked context (edge case) does not produce NaN."""
    cfg = _make_config(bias=False)
    model = QCPC(cfg)
    model.set_stage(1)

    context_ids = torch.randint(0, 1000, (B, N))
    context_mask = torch.ones(B, N, dtype=torch.long)
    # Sample 0 has no valid context tokens at all
    context_mask[0, :] = 0

    target_ids = torch.randint(0, 1000, (B, T))
    target_mask = torch.ones(B, T, dtype=torch.long)

    result = model(
        context_ids=context_ids,
        context_mask=context_mask,
        target_ids=target_ids,
        target_mask=target_mask,
    )
    assert not torch.isnan(result["loss"]), "NaN loss from all-masked context!"
    result["loss"].backward()
    print(f"[PASS] test_all_masked_context_no_nan: loss={result['loss'].item():.4f}")


if __name__ == "__main__":
    test_e2e_baseline_inference()
    test_e2e_baseline_training()
    test_e2e_bias_inference()
    test_e2e_bias_training()
    test_e2e_both_combos()
    test_e2e_with_masks()
    test_stage_freeze_logic()
    test_param_count()
    test_multi_chunk_padded_no_nan()
    test_all_masked_context_no_nan()
    print("\n=== All E2E tests PASSED ===")
