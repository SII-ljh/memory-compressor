"""Unit tests for FrozenDecoder."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.decoder import FrozenDecoder

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B")
B, M, D, L, T = 2, 8, 1024, 16, 10  # batch, memory, dim, prompt_len, target_len


def _make_decoder():
    cfg = QCPCConfig(qwen3_model_path=MODEL_PATH)
    return FrozenDecoder(cfg)


def test_decoder_special_tokens():
    """Test special tokens <MEM> and </MEM> are registered."""
    dec = _make_decoder()
    assert dec.mem_start_id != dec.tokenizer.unk_token_id, "<MEM> should not be UNK"
    assert dec.mem_end_id != dec.tokenizer.unk_token_id, "</MEM> should not be UNK"
    assert dec.mem_start_id != dec.mem_end_id, "<MEM> and </MEM> should be different"
    print(f"[PASS] test_decoder_special_tokens: <MEM>={dec.mem_start_id}, </MEM>={dec.mem_end_id}")


def test_decoder_frozen():
    """Test all LLM parameters are frozen."""
    dec = _make_decoder()
    for name, param in dec.lm.named_parameters():
        assert not param.requires_grad, f"LLM param {name} should be frozen"
    print("[PASS] test_decoder_frozen")


def test_decoder_forward_inference():
    """Test forward pass in inference mode (no target)."""
    dec = _make_decoder()
    memory = torch.randn(B, M, D)
    prompt_ids = torch.randint(0, 1000, (B, L))

    result = dec(memory, prompt_ids=prompt_ids)
    # Total length: 1 (<MEM>) + M + 1 (</MEM>) + L = M + L + 2
    expected_len = M + L + 2
    assert result["logits"].shape == (B, expected_len, len(dec.tokenizer)), \
        f"Expected logits shape (B, {expected_len}, vocab), got {result['logits'].shape}"
    assert result["input_length"] == expected_len
    assert "loss" not in result
    print(f"[PASS] test_decoder_forward_inference: logits shape = {result['logits'].shape}")


def test_decoder_forward_training():
    """Test forward pass in training mode (with target, computes loss)."""
    dec = _make_decoder()
    memory = torch.randn(B, M, D)
    prompt_ids = torch.randint(0, 1000, (B, L))
    target_ids = torch.randint(0, 1000, (B, T))

    result = dec(memory, prompt_ids=prompt_ids, target_ids=target_ids)
    # Total length: M + L + 2 + T
    expected_len = M + L + 2 + T
    assert result["logits"].shape[1] == expected_len, \
        f"Expected {expected_len} positions, got {result['logits'].shape[1]}"
    assert "loss" in result
    assert result["loss"].ndim == 0, "Loss should be scalar"
    assert not torch.isnan(result["loss"]), "Loss should not be NaN"
    print(f"[PASS] test_decoder_forward_training: loss = {result['loss'].item():.4f}")


def test_decoder_with_prompt_embeds():
    """Test forward with prompt embeddings instead of IDs."""
    dec = _make_decoder()
    memory = torch.randn(B, M, D)
    prompt_embeds = torch.randn(B, L, D)

    result = dec(memory, prompt_embeds=prompt_embeds)
    expected_len = M + L + 2
    assert result["logits"].shape[1] == expected_len
    print("[PASS] test_decoder_with_prompt_embeds")


def test_decoder_with_masks():
    """Test forward with attention masks on prompt and target."""
    dec = _make_decoder()
    memory = torch.randn(B, M, D)
    prompt_ids = torch.randint(0, 1000, (B, L))
    target_ids = torch.randint(0, 1000, (B, T))
    prompt_mask = torch.ones(B, L)
    prompt_mask[0, -3:] = 0
    target_mask = torch.ones(B, T)
    target_mask[1, -2:] = 0

    result = dec(
        memory,
        prompt_ids=prompt_ids,
        target_ids=target_ids,
        prompt_mask=prompt_mask,
        target_mask=target_mask,
    )
    assert "loss" in result
    assert not torch.isnan(result["loss"])
    print(f"[PASS] test_decoder_with_masks: loss = {result['loss'].item():.4f}")


def test_decoder_gradient_through_memory():
    """Test gradient flows through memory tokens (trainable compressor)."""
    dec = _make_decoder()
    memory = torch.randn(B, M, D, requires_grad=True)
    prompt_ids = torch.randint(0, 1000, (B, L))
    target_ids = torch.randint(0, 1000, (B, T))

    result = dec(memory, prompt_ids=prompt_ids, target_ids=target_ids)
    result["loss"].backward()

    assert memory.grad is not None, "Gradient should flow to memory tokens"
    assert memory.grad.abs().sum() > 0, "Memory gradient should be non-zero"
    print("[PASS] test_decoder_gradient_through_memory")


if __name__ == "__main__":
    test_decoder_special_tokens()
    test_decoder_frozen()
    test_decoder_forward_inference()
    test_decoder_forward_training()
    test_decoder_with_prompt_embeds()
    test_decoder_with_masks()
    test_decoder_gradient_through_memory()
    print("\n=== All decoder tests PASSED ===")
