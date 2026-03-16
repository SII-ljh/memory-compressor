"""Unit tests for auto batch size probing and accumulation calculation."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.auto_batch import compute_accumulation_steps, _make_dummy_batch


MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B")


def _make_config(**kwargs):
    defaults = dict(
        qwen3_model_path=MODEL_PATH,
        hidden_dim=1024,
        num_heads=16,
        head_dim=64,
        rope_dim=64,
        num_memory_tokens=8,
        num_process_layers=2,
        query_mapper_mid_dim=512,
        ffn_intermediate_dim=2048,
        max_position_embeddings=128,
        use_decoupled_rope=True,
        use_prompt_bias=False,
        stage1_max_context_len=64,
        stage1_max_cont_len=16,
        stage2_max_context_len=64,
        stage2_max_prompt_len=16,
        stage2_max_answer_len=16,
    )
    defaults.update(kwargs)
    return QCPCConfig(**defaults)


# --- compute_accumulation_steps tests ---

def test_compute_accum_exact_division():
    """Target EBS divides evenly into per_gpu_bs * num_gpus."""
    accum, actual = compute_accumulation_steps(per_gpu_batch_size=4, num_gpus=8, target_ebs=256)
    assert accum == 8, f"Expected 8, got {accum}"
    assert actual == 256, f"Expected 256, got {actual}"
    print(f"[PASS] test_compute_accum_exact_division: accum={accum}, actual_ebs={actual}")


def test_compute_accum_rounds_up():
    """Target EBS does not divide evenly — should round up."""
    accum, actual = compute_accumulation_steps(per_gpu_batch_size=3, num_gpus=8, target_ebs=256)
    # 3 * 8 = 24 per step, ceil(256/24) = 11, actual = 264
    assert accum == 11, f"Expected 11, got {accum}"
    assert actual == 264, f"Expected 264, got {actual}"
    print(f"[PASS] test_compute_accum_rounds_up: accum={accum}, actual_ebs={actual}")


def test_compute_accum_single_gpu():
    """Single GPU scenario."""
    accum, actual = compute_accumulation_steps(per_gpu_batch_size=16, num_gpus=1, target_ebs=256)
    assert accum == 16, f"Expected 16, got {accum}"
    assert actual == 256, f"Expected 256, got {actual}"
    print(f"[PASS] test_compute_accum_single_gpu: accum={accum}, actual_ebs={actual}")


def test_compute_accum_large_bs():
    """Per-step total already >= target — accum should be 1."""
    accum, actual = compute_accumulation_steps(per_gpu_batch_size=32, num_gpus=8, target_ebs=256)
    assert accum == 1, f"Expected 1, got {accum}"
    assert actual == 256, f"Expected 256, got {actual}"
    print(f"[PASS] test_compute_accum_large_bs: accum={accum}, actual_ebs={actual}")


def test_compute_accum_large_bs_over_target():
    """Per-step total > target — accum should still be 1."""
    accum, actual = compute_accumulation_steps(per_gpu_batch_size=64, num_gpus=8, target_ebs=256)
    assert accum == 1, f"Expected 1, got {accum}"
    assert actual == 512, f"Expected 512, got {actual}"
    print(f"[PASS] test_compute_accum_large_bs_over_target: accum={accum}, actual_ebs={actual}")


def test_compute_accum_invalid_inputs():
    """Should raise ValueError on non-positive inputs."""
    import pytest
    for args in [(0, 8, 256), (4, 0, 256), (4, 8, 0), (-1, 8, 256)]:
        try:
            compute_accumulation_steps(*args)
            assert False, f"Should have raised ValueError for {args}"
        except ValueError:
            pass
    print("[PASS] test_compute_accum_invalid_inputs")


# --- _make_dummy_batch shape tests ---

def test_dummy_batch_stage1a_shapes():
    """Dummy batch for stage 1a has correct shapes."""
    config = _make_config(stage1a_max_context_len=64, stage1a_max_cont_len=16)
    batch = _make_dummy_batch(4, config, stage="1a", device=torch.device("cpu"))

    assert batch["context_ids"].shape == (4, 64), f"Got {batch['context_ids'].shape}"
    assert batch["context_mask"].shape == (4, 64)
    assert batch["target_ids"].shape == (4, 16), f"Got {batch['target_ids'].shape}"
    assert batch["target_mask"].shape == (4, 16)
    assert "prompt_ids" not in batch
    print(f"[PASS] test_dummy_batch_stage1a_shapes")


def test_dummy_batch_stage1b_shapes():
    """Dummy batch for stage 1b (multi-chunk) has correct shapes."""
    config = _make_config(stage1b_num_chunks=4, stage1b_chunk_len=32, stage1b_max_cont_len=8)
    batch = _make_dummy_batch(2, config, stage="1b", device=torch.device("cpu"))

    assert batch["chunk_ids"].shape == (2, 4, 32), f"Got {batch['chunk_ids'].shape}"
    assert batch["chunk_mask"].shape == (2, 4, 32)
    assert batch["target_ids"].shape == (2, 8), f"Got {batch['target_ids'].shape}"
    assert batch["target_mask"].shape == (2, 8)
    assert "context_ids" not in batch
    print(f"[PASS] test_dummy_batch_stage1b_shapes")


def test_dummy_batch_stage2_shapes():
    """Dummy batch for stage 2 has correct shapes including prompt."""
    config = _make_config(
        stage2_max_context_len=64,
        stage2_max_prompt_len=16,
        stage2_max_answer_len=32,
    )
    batch = _make_dummy_batch(2, config, stage="2", device=torch.device("cpu"))

    assert batch["context_ids"].shape == (2, 64)
    assert batch["context_mask"].shape == (2, 64)
    assert batch["prompt_ids"].shape == (2, 16), f"Got {batch['prompt_ids'].shape}"
    assert batch["prompt_mask"].shape == (2, 16)
    assert batch["target_ids"].shape == (2, 32), f"Got {batch['target_ids'].shape}"
    assert batch["target_mask"].shape == (2, 32)
    print(f"[PASS] test_dummy_batch_stage2_shapes")


def test_dummy_batch_values():
    """Dummy batch should use token ID 1 (not 0) and all-ones masks."""
    config = _make_config()
    batch = _make_dummy_batch(1, config, stage="1a", device=torch.device("cpu"))

    assert (batch["context_ids"] == 1).all(), "context_ids should all be 1"
    assert (batch["context_mask"] == 1).all(), "context_mask should all be 1"
    assert (batch["target_ids"] == 1).all(), "target_ids should all be 1"
    print("[PASS] test_dummy_batch_values")


if __name__ == "__main__":
    # Arithmetic tests (no GPU needed)
    test_compute_accum_exact_division()
    test_compute_accum_rounds_up()
    test_compute_accum_single_gpu()
    test_compute_accum_large_bs()
    test_compute_accum_large_bs_over_target()
    test_compute_accum_invalid_inputs()

    # Dummy batch shape tests (CPU only)
    test_dummy_batch_stage1a_shapes()
    test_dummy_batch_stage1b_shapes()
    test_dummy_batch_stage2_shapes()
    test_dummy_batch_values()

    print("\n=== All auto_batch tests PASSED ===")
