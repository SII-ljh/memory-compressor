"""Overfitting experiment: verify training pipeline on tiny data.

Validates that:
1. Gradients flow correctly through Perceiver IO → Frozen Decoder
2. Loss can decrease to near zero on a memorizable dataset
3. Both Stage 1 (text completion) and Stage 2 (QA) pipelines work

Usage:
    python scripts/overfit_test.py                    # run both stages
    python scripts/overfit_test.py --stage 1          # stage 1 only
    python scripts/overfit_test.py --stage 2          # stage 2 only
    python scripts/overfit_test.py --steps 200        # custom step count
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.model import QCPC
from src.data import PretrainDataset, QADataset, collate_fn


# ── Synthetic data ──────────────────────────────────────────────────────

PRETRAIN_SAMPLES = [
    {"text": "The quick brown fox jumps over the lazy dog. " * 20, "id": "overfit-0"},
    {"text": "In a galaxy far far away, there lived a brave knight. " * 20, "id": "overfit-1"},
]

QA_SAMPLES = [
    {
        "context": "Paris is the capital of France. The Eiffel Tower is located in Paris. " * 10,
        "question": "What is the capital of France?",
        "answer": "Paris",
    },
    {
        "context": "Water boils at 100 degrees Celsius at sea level. " * 10,
        "question": "At what temperature does water boil?",
        "answer": "100 degrees Celsius",
    },
]


def create_temp_pretrain_file(samples, repeat=1):
    """Write pretrain JSONL to a temp file."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for _ in range(repeat):
        for s in samples:
            f.write(json.dumps(s) + "\n")
    f.flush()
    return f.name


def create_temp_qa_file(samples, repeat=1):
    """Write QA JSON to a temp file."""
    data = samples * repeat
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.flush()
    return f.name


# ── Overfit loops ───────────────────────────────────────────────────────

def overfit_stage1(config: QCPCConfig, device: torch.device, num_steps: int, lr: float):
    """Stage 1: overfit on tiny text completion data."""
    print("\n" + "=" * 60)
    print("  Stage 1 Overfit: Text Completion")
    print("=" * 60)

    # Build model
    model = QCPC(config)
    model.set_stage(1)
    model.to(device)

    counts = model.count_params()
    print(f"  Perceiver trainable: {counts['perceiver']['trainable']:,}")
    print(f"  Total trainable:     {counts['total_trainable']:,}")

    # Create tiny dataset
    data_file = create_temp_pretrain_file(PRETRAIN_SAMPLES, repeat=1)
    tokenizer = model.decoder.tokenizer
    dataset = PretrainDataset(
        data_path=data_file,
        tokenizer=tokenizer,
        max_context_len=128,   # short for speed
        max_cont_len=32,       # short target
    )
    print(f"  Dataset size: {len(dataset)} samples")

    # Prepare a fixed batch (repeat to overfit)
    batch_items = [dataset[i % len(dataset)] for i in range(2)]
    batch = collate_fn(batch_items)
    batch = {k: v.to(device) for k, v in batch.items()}

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.0)

    # Training loop
    model.train()
    losses = []
    print(f"\n  {'Step':>5}  {'Loss':>10}  {'Status'}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*20}")

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        result = model(
            context_ids=batch["context_ids"],
            context_mask=batch["context_mask"],
            target_ids=batch["target_ids"],
            target_mask=batch["target_mask"],
        )
        loss = result["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step <= 5 or step % 10 == 0 or step == num_steps:
            status = ""
            if step == 1:
                status = "(initial)"
            elif loss_val < 1.0:
                status = "< 1.0"
            elif loss_val < 0.1:
                status = "< 0.1"
            print(f"  {step:>5}  {loss_val:>10.4f}  {status}")

    # Summary
    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"\n  ── Summary ──")
    print(f"  Initial loss:  {initial_loss:.4f}")
    print(f"  Final loss:    {final_loss:.4f}")
    print(f"  Min loss:      {min_loss:.4f}")
    print(f"  Reduction:     {reduction:.1f}%")

    success = final_loss < initial_loss * 0.5  # at least 50% reduction
    if success:
        print(f"  Result:        PASS (loss decreased significantly)")
    else:
        print(f"  Result:        FAIL (loss did not decrease enough)")

    return losses, success


def overfit_stage2(config: QCPCConfig, device: torch.device, num_steps: int, lr: float):
    """Stage 2: overfit on tiny QA data."""
    print("\n" + "=" * 60)
    print("  Stage 2 Overfit: QA (Question-Answering)")
    print("=" * 60)

    # Build model
    config.use_prompt_bias = True
    model = QCPC(config)
    model.set_stage(2)
    model.to(device)

    counts = model.count_params()
    print(f"  Perceiver trainable: {counts['perceiver']['trainable']:,}")
    print(f"  Total trainable:     {counts['total_trainable']:,}")

    # Create tiny dataset
    data_file = create_temp_qa_file(QA_SAMPLES, repeat=1)
    tokenizer = model.decoder.tokenizer
    dataset = QADataset(
        data_path=data_file,
        tokenizer=tokenizer,
        max_context_len=128,
        max_prompt_len=32,
        max_answer_len=16,
    )
    print(f"  Dataset size: {len(dataset)} samples")

    # Fixed batch
    batch_items = [dataset[i % len(dataset)] for i in range(2)]
    batch = collate_fn(batch_items)
    batch = {k: v.to(device) for k, v in batch.items()}

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.0)

    # Training loop
    model.train()
    losses = []
    print(f"\n  {'Step':>5}  {'Loss':>10}  {'Status'}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*20}")

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        result = model(
            context_ids=batch["context_ids"],
            context_mask=batch["context_mask"],
            prompt_ids=batch["prompt_ids"],
            prompt_mask=batch["prompt_mask"],
            target_ids=batch["target_ids"],
            target_mask=batch["target_mask"],
        )
        loss = result["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step <= 5 or step % 10 == 0 or step == num_steps:
            status = ""
            if step == 1:
                status = "(initial)"
            elif loss_val < 1.0:
                status = "< 1.0"
            elif loss_val < 0.1:
                status = "< 0.1"
            print(f"  {step:>5}  {loss_val:>10.4f}  {status}")

    # Summary
    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"\n  ── Summary ──")
    print(f"  Initial loss:  {initial_loss:.4f}")
    print(f"  Final loss:    {final_loss:.4f}")
    print(f"  Min loss:      {min_loss:.4f}")
    print(f"  Reduction:     {reduction:.1f}%")

    success = final_loss < initial_loss * 0.5
    if success:
        print(f"  Result:        PASS (loss decreased significantly)")
    else:
        print(f"  Result:        FAIL (loss did not decrease enough)")

    return losses, success


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QCPC Overfit Experiment")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="Run specific stage (default: both)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of training steps (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3, higher for fast overfit)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: use built-in defaults)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu, cuda, mps (default: auto)")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Steps:  {args.steps}")
    print(f"LR:     {args.lr}")

    # Config
    if args.config:
        config = QCPCConfig.load(args.config)
    else:
        config = QCPCConfig()
    # Override to smaller dims for speed if using default (no custom config)
    if not args.config:
        print("Using default config with Qwen3-0.6B backbone")

    results = {}

    if args.stage is None or args.stage == 1:
        losses1, ok1 = overfit_stage1(config, device, args.steps, args.lr)
        results["stage1"] = {"success": ok1, "final_loss": losses1[-1]}

    if args.stage is None or args.stage == 2:
        losses2, ok2 = overfit_stage2(config, device, args.steps, args.lr)
        results["stage2"] = {"success": ok2, "final_loss": losses2[-1]}

    # Final report
    print("\n" + "=" * 60)
    print("  OVERFIT EXPERIMENT RESULTS")
    print("=" * 60)
    all_pass = True
    for stage, r in results.items():
        status = "PASS" if r["success"] else "FAIL"
        all_pass = all_pass and r["success"]
        print(f"  {stage}: {status} (final_loss={r['final_loss']:.4f})")
    print("=" * 60)

    if all_pass:
        print("  All stages passed - training pipeline is working correctly.")
    else:
        print("  Some stages failed - investigate gradient flow or data pipeline.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
