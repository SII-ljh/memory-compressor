"""Unit tests for QCPC inference."""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.model import QCPC
from src.inference import QCPCInference

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B")
B, M, D = 1, 8, 1024


def _make_config(**kwargs):
    defaults = dict(
        qwen3_model_path=MODEL_PATH,
        hidden_dim=D,
        num_heads=16,
        head_dim=64,
        num_memory_tokens=M,
        num_process_layers=2,
        query_mapper_mid_dim=512,
        ffn_intermediate_dim=2048,
        max_position_embeddings=128,
        use_prompt_bias=True,
        stage2_chunk_len=64,
    )
    defaults.update(kwargs)
    return QCPCConfig(**defaults)


def _save_dummy_checkpoint(config):
    """Create and save a dummy checkpoint for testing."""
    model = QCPC(config)
    perceiver_state = model.perceiver.state_dict()
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save({"model": perceiver_state, "epoch": 0, "global_step": 0}, tmp.name)
    return tmp.name


def test_compress():
    """Test compression produces correct shape (multi-chunk)."""
    cfg = _make_config(stage2_chunk_len=64)
    ckpt_path = _save_dummy_checkpoint(cfg)
    inferencer = QCPCInference(cfg, ckpt_path, device=torch.device("cpu"))

    context = "The quick brown fox jumps over the lazy dog. " * 50
    memory, memory_mask = inferencer.compress(context, question="What does the fox do?")
    # With chunk_len=64, context will be split into K chunks → K*M memory tokens
    K = memory.shape[1] // M
    assert memory.shape == (1, K * M, D), f"Expected (1, {K*M}, {D}), got {memory.shape}"
    assert memory_mask.shape == (1, K * M), f"Expected (1, {K*M}), got {memory_mask.shape}"
    assert K >= 1

    Path(ckpt_path).unlink()
    print(f"[PASS] test_compress: {memory.shape} (K={K} chunks)")


def test_generate():
    """Test generation produces non-empty string."""
    cfg = _make_config()
    ckpt_path = _save_dummy_checkpoint(cfg)
    inferencer = QCPCInference(cfg, ckpt_path, device=torch.device("cpu"))

    context = "Paris is the capital of France. It is known for the Eiffel Tower."
    answer = inferencer.generate(
        context=context,
        question="What is the capital of France?",
        max_new_tokens=20,
        temperature=0.0,
    )
    assert isinstance(answer, str)
    # With random weights, output may not make sense but should not crash
    print(f"[PASS] test_generate: '{answer[:50]}...'")

    Path(ckpt_path).unlink()


def test_generate_no_question():
    """Test generation without question (Stage 1 mode)."""
    cfg = _make_config(use_prompt_bias=False)
    ckpt_path = _save_dummy_checkpoint(cfg)
    inferencer = QCPCInference(cfg, ckpt_path, device=torch.device("cpu"))

    context = "The quick brown fox jumps over the lazy dog."
    answer = inferencer.generate(
        context=context,
        question=None,
        max_new_tokens=20,
        temperature=0.0,
    )
    assert isinstance(answer, str)
    print(f"[PASS] test_generate_no_question: '{answer[:50]}...'")

    Path(ckpt_path).unlink()


def test_generate_batch():
    """Test batch generation."""
    cfg = _make_config()
    ckpt_path = _save_dummy_checkpoint(cfg)
    inferencer = QCPCInference(cfg, ckpt_path, device=torch.device("cpu"))

    contexts = ["Paris is the capital of France.", "Tokyo is the capital of Japan."]
    questions = ["What is the capital of France?", "What is the capital of Japan?"]
    answers = inferencer.generate_batch(contexts, questions, max_new_tokens=10)
    assert len(answers) == 2
    assert all(isinstance(a, str) for a in answers)
    print(f"[PASS] test_generate_batch: {len(answers)} answers")

    Path(ckpt_path).unlink()


def test_both_combos_inference():
    """Test inference works for both mode combinations."""
    combos = [
        (False, "Baseline"),
        (True, "Prompt Bias"),
    ]
    for bias, name in combos:
        cfg = _make_config(use_prompt_bias=bias)
        ckpt_path = _save_dummy_checkpoint(cfg)
        inferencer = QCPCInference(cfg, ckpt_path, device=torch.device("cpu"))

        context = "Test context for inference."
        question = "Test question?" if bias else None
        answer = inferencer.generate(context, question, max_new_tokens=5)
        assert isinstance(answer, str)
        print(f"  [PASS] {name}")

        Path(ckpt_path).unlink()
    print("[PASS] test_both_combos_inference")


if __name__ == "__main__":
    test_compress()
    test_generate()
    test_generate_no_question()
    test_generate_batch()
    test_both_combos_inference()
    print("\n=== All inference tests PASSED ===")
