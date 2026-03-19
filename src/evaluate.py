"""Standalone evaluation script for QCPC experiments.

Evaluates stage1b (multi-chunk pretrain) and/or stage2 (QA) checkpoints
on eval datasets with multi-GPU support via Accelerate.

Metrics:
  stage1b: loss, perplexity
  stage2:  loss, perplexity, ROUGE-L, token-F1, exact match

Usage:
    # Stage 1b (loss + PPL)
    accelerate launch --num_processes 8 --multi_gpu \
      src/evaluate.py --checkpoint outputs/qwen06b_m128/stage1b/best.pt \
      --stage 1b --auto_config

    # Stage 2 (loss + PPL + generation metrics)
    accelerate launch --num_processes 8 --multi_gpu \
      src/evaluate.py --checkpoint outputs/qwen06b_m128/stage2/best.pt \
      --stage 2 --auto_config --gen_samples 200

    # With explicit config overrides
    accelerate launch --num_processes 8 --multi_gpu \
      src/evaluate.py --config config/default.yaml \
      --checkpoint outputs/qwen06b_m128/stage2/best.pt --stage 2 \
      --override num_memory_tokens=128 hidden_dim=1024
"""

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.model import QCPC
from src.data import create_multi_chunk_dataloader, create_qa_dataloader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Config inference from checkpoint
# ──────────────────────────────────────────────────────────────

HIDDEN_DIM_TO_QWEN = {
    1024: "./models/Qwen3-0.6B",
    2048: "./models/Qwen3-1.7B",
    2560: "./models/Qwen3-4B",
}


def infer_config_from_state_dict(state: dict, config: QCPCConfig) -> QCPCConfig:
    """Auto-detect model config from perceiver state dict shapes.

    Infers: num_memory_tokens, hidden_dim, num_heads, ffn_intermediate_dim,
            query_mapper_mid_dim, use_decoupled_rope, use_prompt_bias,
            num_process_layers, qwen3_model_path.
    """
    z_base = state.get("latent_array.z_base")
    if z_base is not None:
        M, D = z_base.shape
        config.num_memory_tokens = M
        config.hidden_dim = D
        config.head_dim = 64
        config.num_heads = D // 64
        config.ffn_intermediate_dim = D * 2
        config.query_mapper_mid_dim = D // 2
        if D in HIDDEN_DIM_TO_QWEN:
            config.qwen3_model_path = HIDDEN_DIM_TO_QWEN[D]
        logger.info(f"Inferred from checkpoint: M={M}, D={D}, model={config.qwen3_model_path}")

    config.use_decoupled_rope = any("slot_pe" in k for k in state)
    config.use_prompt_bias = any("query_mapper" in k for k in state)

    proc_indices = [int(k.split(".")[1]) for k in state if k.startswith("process_blocks.")]
    if proc_indices:
        config.num_process_layers = max(proc_indices) + 1

    return config


# ──────────────────────────────────────────────────────────────
# Text metrics (ROUGE-L, F1, EM)
# ──────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Longest common subsequence length (space-optimized DP)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def compute_rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score."""
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    prec = lcs / len(pred_tokens)
    rec = lcs / len(ref_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_f1(prediction: str, reference: str) -> float:
    """Token-level F1 (SQuAD-style with normalization)."""
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if common == 0:
        return 0.0
    prec = common / len(pred_tokens)
    rec = common / len(ref_tokens)
    return 2 * prec * rec / (prec + rec)


def compute_em(prediction: str, reference: str) -> float:
    """Exact match after normalization."""
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


# ──────────────────────────────────────────────────────────────
# Loss / PPL evaluation (all GPUs)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss_ppl(model, eval_loader, accelerator, stage: str):
    """Compute average loss and PPL, aggregated across all GPUs."""
    model.eval()
    local_loss = 0.0
    local_count = 0

    for batch in eval_loader:
        if stage == "1b":
            result = model(
                chunk_ids=batch["chunk_ids"],
                chunk_mask=batch["chunk_mask"],
                target_ids=batch["target_ids"],
                target_mask=batch["target_mask"],
            )
        else:
            result = model(
                chunk_ids=batch["chunk_ids"],
                chunk_mask=batch["chunk_mask"],
                prompt_ids=batch["prompt_ids"],
                prompt_mask=batch["prompt_mask"],
                target_ids=batch["target_ids"],
                target_mask=batch["target_mask"],
            )
        local_loss += result["loss"].item()
        local_count += 1

    # All-reduce across GPUs
    stats = torch.tensor([local_loss, float(local_count)], device=accelerator.device)
    if accelerator.num_processes > 1 and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

    total_loss = stats[0].item()
    total_count = int(stats[1].item())
    avg_loss = total_loss / max(total_count, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl, total_count


# ──────────────────────────────────────────────────────────────
# Generation evaluation (Stage 2 only, all GPUs)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_generation(model, config, eval_data_path, accelerator, max_samples=200):
    """Generate answers and compute ROUGE-L / F1 / EM.

    Data is sharded across GPUs for parallel generation.
    Metrics are all-reduced for the final result.
    """
    from src.inference import QCPCInference

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    device = accelerator.device

    # Reuse loaded model via QCPCInference wrapper (skip __init__ to avoid reloading)
    inferencer = QCPCInference.__new__(QCPCInference)
    inferencer.config = config
    inferencer.device = device
    inferencer.model = unwrapped
    inferencer.tokenizer = unwrapped.decoder.tokenizer

    # Load and shard eval data across GPUs
    with open(eval_data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    eval_data = eval_data[:max_samples]

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    my_shard = eval_data[rank::world_size]

    local_rouge = 0.0
    local_f1 = 0.0
    local_em = 0.0
    local_n = 0

    for i, item in enumerate(my_shard):
        pred = inferencer.generate(
            context=item["context"],
            question=item.get("question"),
            max_new_tokens=config.stage2_max_answer_len,
        )
        ref = item["answer"]

        local_rouge += compute_rouge_l(pred, ref)
        local_f1 += compute_f1(pred, ref)
        local_em += compute_em(pred, ref)
        local_n += 1

        if rank == 0 and (i + 1) % 20 == 0:
            logger.info(f"  Generation: {i + 1}/{len(my_shard)} samples")

    # All-reduce
    stats = torch.tensor(
        [local_rouge, local_f1, local_em, float(local_n)], device=device
    )
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

    n = max(stats[3].item(), 1)
    return {
        "rouge_l": stats[0].item() / n,
        "f1": stats[1].item() / n,
        "em": stats[2].item() / n,
        "gen_samples": int(stats[3].item()),
    }


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="QCPC Evaluation")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stage", type=str, required=True, choices=["1b", "2"])
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument(
        "--auto_config", action="store_true",
        help="Infer model config (M, D, mode switches, backbone) from checkpoint",
    )
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Override eval data path (default: from config)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gen_samples", type=int, default=200,
                        help="Max samples for generation eval (stage 2 only)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save metrics JSON to this path")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load base config
    config = QCPCConfig.load(args.config)
    if args.override:
        from src.train import apply_overrides
        config = apply_overrides(config, args.override)

    # Load checkpoint (once — reused for config inference and weight loading)
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Auto-infer config from checkpoint state dict
    if args.auto_config:
        config = infer_config_from_state_dict(ckpt["model"], config)

    logger.info(
        f"Config: model={config.qwen3_model_path}, M={config.num_memory_tokens}, "
        f"D={config.hidden_dim}, L_proc={config.num_process_layers}, "
        f"rope={config.use_decoupled_rope}, bias={config.use_prompt_bias}"
    )

    # Build model and load weights
    model = QCPC(config)
    stage_num = 1 if args.stage == "1b" else 2
    model.set_stage(stage_num)
    model.perceiver.load_state_dict(ckpt["model"], strict=False)
    del ckpt  # free memory

    # Accelerator (no mixed precision for eval)
    accelerator = Accelerator(mixed_precision="no")
    tokenizer = model.decoder.tokenizer

    # Resolve eval data path
    if args.eval_data:
        eval_data_path = args.eval_data
    elif args.stage == "1b":
        eval_data_path = config.pretrain_eval_data_path
    else:
        eval_data_path = config.sft_eval_data_path

    if accelerator.is_main_process:
        logger.info(f"Eval data: {eval_data_path}")

    # Create eval dataloader
    if args.stage == "1b":
        eval_loader = create_multi_chunk_dataloader(
            data_path=eval_data_path,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_chunks=config.stage1b_max_chunks,
            min_chunks=config.stage1b_min_chunks,
            chunk_len=config.stage1b_chunk_len,
            cont_len=config.stage1b_max_cont_len,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
        )
    else:
        eval_loader = create_qa_dataloader(
            data_path=eval_data_path,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_context_len=config.stage2_max_context_len,
            chunk_len=config.stage2_chunk_len,
            max_prompt_len=config.stage2_max_prompt_len,
            max_answer_len=config.stage2_max_answer_len,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
        )

    # Distribute model and dataloader across GPUs
    model, eval_loader = accelerator.prepare(model, eval_loader)

    # ── Phase 1: Loss / PPL ──────────────────────────────────
    if accelerator.is_main_process:
        logger.info("Phase 1: Computing loss and PPL...")
    avg_loss, ppl, n_batches = eval_loss_ppl(model, eval_loader, accelerator, args.stage)

    metrics = {
        "checkpoint": args.checkpoint,
        "stage": args.stage,
        "loss": round(avg_loss, 6),
        "ppl": round(ppl, 4),
        "eval_batches": n_batches,
    }

    if accelerator.is_main_process:
        logger.info(f"  loss={avg_loss:.6f}  ppl={ppl:.4f}  ({n_batches} batches)")

    # ── Phase 2: Generation metrics (stage 2 only) ───────────
    if args.stage == "2":
        if accelerator.is_main_process:
            logger.info(f"Phase 2: Generation eval ({args.gen_samples} samples across {accelerator.num_processes} GPUs)...")
        gen_metrics = eval_generation(
            model, config, eval_data_path,
            accelerator, max_samples=args.gen_samples,
        )
        metrics.update({
            "rouge_l": round(gen_metrics["rouge_l"], 4),
            "f1": round(gen_metrics["f1"], 4),
            "em": round(gen_metrics["em"], 4),
            "gen_samples": gen_metrics["gen_samples"],
        })
        if accelerator.is_main_process:
            logger.info(
                f"  ROUGE-L={gen_metrics['rouge_l']:.4f}  "
                f"F1={gen_metrics['f1']:.4f}  "
                f"EM={gen_metrics['em']:.4f}"
            )

    # ── Output ────────────────────────────────────────────────
    if accelerator.is_main_process:
        metrics_json = json.dumps(metrics, indent=2, ensure_ascii=False)
        print(metrics_json)
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                f.write(metrics_json + "\n")
            logger.info(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
