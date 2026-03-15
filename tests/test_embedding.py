"""Unit tests for FrozenEmbedding."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.embedding import FrozenEmbedding

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B")


def test_embedding_shape():
    """Test output shape is correct."""
    cfg = QCPCConfig(qwen3_model_path=MODEL_PATH)
    emb = FrozenEmbedding(cfg)

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    out = emb(input_ids)

    assert out.shape == (batch_size, seq_len, cfg.hidden_dim), \
        f"Expected ({batch_size}, {seq_len}, {cfg.hidden_dim}), got {out.shape}"
    print(f"[PASS] test_embedding_shape: {out.shape}")


def test_embedding_frozen():
    """Test all parameters are frozen."""
    cfg = QCPCConfig(qwen3_model_path=MODEL_PATH)
    emb = FrozenEmbedding(cfg)

    for name, param in emb.named_parameters():
        assert not param.requires_grad, f"Param {name} should be frozen"
    print("[PASS] test_embedding_frozen")


def test_embedding_no_grad():
    """Test that frozen embedding output has no grad_fn (not part of autograd graph)."""
    cfg = QCPCConfig(qwen3_model_path=MODEL_PATH)
    emb = FrozenEmbedding(cfg)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = emb(input_ids)

    # Frozen embedding output should not be attached to the computation graph
    assert out.grad_fn is None, "Frozen embedding output should have no grad_fn"
    assert not out.requires_grad, "Frozen embedding output should not require grad"
    print("[PASS] test_embedding_no_grad")


def test_embedding_deterministic():
    """Test same input produces same output."""
    cfg = QCPCConfig(qwen3_model_path=MODEL_PATH)
    emb = FrozenEmbedding(cfg)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
    out1 = emb(input_ids)
    out2 = emb(input_ids)

    assert torch.equal(out1, out2), "Same input should give same output"
    print("[PASS] test_embedding_deterministic")


if __name__ == "__main__":
    test_embedding_shape()
    test_embedding_frozen()
    test_embedding_no_grad()
    test_embedding_deterministic()
    print("\n=== All embedding tests PASSED ===")
