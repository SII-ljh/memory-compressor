"""Unit tests for training pipeline: data loaders + training loop."""

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.model import QCPC
from src.data import (
    PretrainDataset, QADataset, collate_fn, collate_qa_chunk_fn,
    create_pretrain_dataloader, create_qa_dataloader,
)
from transformers import AutoTokenizer

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B")
PRETRAIN_DATA = str(Path(__file__).resolve().parent.parent / "data" / "pretrain" / "fineweb_sampled.jsonl")
B, M, D = 2, 8, 1024


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
        use_prompt_bias=False,
    )
    defaults.update(kwargs)
    return QCPCConfig(**defaults)


def test_pretrain_dataset():
    """Test pretrain dataset loads and returns correct structure."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    ds = PretrainDataset(
        PRETRAIN_DATA, tokenizer, max_context_len=128, max_cont_len=32,
    )
    assert len(ds) > 0, f"Dataset should not be empty, found {len(ds)} samples"

    item = ds[0]
    assert "context_ids" in item
    assert "target_ids" in item
    assert len(item["context_ids"]) > 0
    assert len(item["target_ids"]) > 0
    print(f"[PASS] test_pretrain_dataset: {len(ds)} samples, ctx={len(item['context_ids'])}, tgt={len(item['target_ids'])}")


def test_pretrain_collate():
    """Test collate_fn pads correctly."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    ds = PretrainDataset(PRETRAIN_DATA, tokenizer, max_context_len=128, max_cont_len=32)

    batch = [ds[i] for i in range(min(B, len(ds)))]
    collated = collate_fn(batch)

    assert "context_ids" in collated
    assert "context_mask" in collated
    assert "target_ids" in collated
    assert "target_mask" in collated
    assert collated["context_ids"].shape[0] == len(batch)
    assert collated["context_mask"].shape == collated["context_ids"].shape
    print(f"[PASS] test_pretrain_collate: ctx={collated['context_ids'].shape}, tgt={collated['target_ids'].shape}")


def test_qa_dataset():
    """Test QA dataset with synthetic data."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Create temp QA data
    records = [
        {"context": "The capital of France is Paris.", "question": "What is the capital of France?", "answer": "Paris", "source": "test"},
        {"context": "Water boils at 100 degrees Celsius.", "question": "At what temperature does water boil?", "answer": "100 degrees Celsius", "source": "test"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(records, f)
        tmp_path = f.name

    ds = QADataset(tmp_path, tokenizer, max_context_len=128, chunk_len=64, max_prompt_len=32, max_answer_len=32)
    assert len(ds) == 2

    item = ds[0]
    assert "chunk_ids" in item
    assert "chunk_masks" in item
    assert "num_chunks" in item
    assert "prompt_ids" in item
    assert "target_ids" in item

    batch = [ds[i] for i in range(2)]
    collated = collate_qa_chunk_fn(batch)
    assert "chunk_ids" in collated
    assert "chunk_mask" in collated
    assert "prompt_ids" in collated
    assert "prompt_mask" in collated

    Path(tmp_path).unlink()
    print(f"[PASS] test_qa_dataset: chunks={item['num_chunks']}, prompt={len(item['prompt_ids'])}, tgt={len(item['target_ids'])}")


def test_stage1_training_step():
    """Test a single stage 1 training step end-to-end."""
    cfg = _make_config(use_prompt_bias=False)
    model = QCPC(cfg)
    model.set_stage(1)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # Create a mini batch
    tokenizer = model.decoder.tokenizer
    ds = PretrainDataset(PRETRAIN_DATA, tokenizer, max_context_len=64, max_cont_len=16)
    batch_items = [ds[i] for i in range(min(B, len(ds)))]
    batch = collate_fn(batch_items)

    # Forward
    result = model(
        context_ids=batch["context_ids"],
        context_mask=batch["context_mask"],
        target_ids=batch["target_ids"],
        target_mask=batch["target_mask"],
    )
    loss = result["loss"]
    assert not torch.isnan(loss)

    # Backward + step
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
    optimizer.zero_grad()

    print(f"[PASS] test_stage1_training_step: loss={loss.item():.4f}")


def test_stage2_training_step():
    """Test a single stage 2 training step with QA data."""
    cfg = _make_config(use_prompt_bias=True)
    model = QCPC(cfg)
    model.set_stage(2)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)

    # Create a mini QA batch
    tokenizer = model.decoder.tokenizer
    records = [
        {"context": "The capital of France is Paris.", "question": "What is the capital of France?", "answer": "Paris", "source": "test"},
        {"context": "Water boils at 100 degrees Celsius.", "question": "At what temperature does water boil?", "answer": "100 degrees Celsius", "source": "test"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(records, f)
        tmp_path = f.name

    ds = QADataset(tmp_path, tokenizer, max_context_len=64, chunk_len=32, max_prompt_len=32, max_answer_len=32)
    batch_items = [ds[i] for i in range(2)]
    batch = collate_qa_chunk_fn(batch_items)

    # Forward
    result = model(
        chunk_ids=batch["chunk_ids"],
        chunk_mask=batch["chunk_mask"],
        prompt_ids=batch["prompt_ids"],
        prompt_mask=batch["prompt_mask"],
        target_ids=batch["target_ids"],
        target_mask=batch["target_mask"],
    )
    loss = result["loss"]
    assert not torch.isnan(loss)

    # Backward + step
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
    optimizer.zero_grad()

    Path(tmp_path).unlink()
    print(f"[PASS] test_stage2_training_step: loss={loss.item():.4f}")


def test_checkpoint_save_load():
    """Test saving and loading perceiver checkpoint."""
    cfg = _make_config()
    model = QCPC(cfg)
    model.set_stage(1)

    # Save perceiver state
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    perceiver_state = model.perceiver.state_dict()
    torch.save({"model": perceiver_state, "epoch": 0, "global_step": 100}, tmp_path)

    # Load into new model
    model2 = QCPC(cfg)
    ckpt = torch.load(tmp_path, map_location="cpu")
    model2.perceiver.load_state_dict(ckpt["model"])

    # Verify weights match
    for (n1, p1), (n2, p2) in zip(
        model.perceiver.named_parameters(), model2.perceiver.named_parameters()
    ):
        assert torch.equal(p1, p2), f"Mismatch in {n1}"

    Path(tmp_path).unlink()
    print("[PASS] test_checkpoint_save_load")


if __name__ == "__main__":
    test_pretrain_dataset()
    test_pretrain_collate()
    test_qa_dataset()
    test_stage1_training_step()
    test_stage2_training_step()
    test_checkpoint_save_load()
    print("\n=== All training tests PASSED ===")
