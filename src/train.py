"""Training script for QCPC: Stage 1a/1b (pretrain) + Stage 2 (QA finetune).

Supports multi-GPU via Accelerate (wraps FSDP/DDP).
Frozen LLM Decoder is excluded from FSDP sharding and optimizer states.

Stage 1 is split into two phases:
  1a — Short window warmup  (single-chunk compression, first ~100M tokens)
  1b — Long window training (multi-chunk concatenation, remaining tokens)

When auto_batch_size is enabled (default), the script:
1. Builds the model on a single GPU
2. Binary-searches for the max per-GPU batch size
3. Computes gradient_accumulation_steps to reach target_effective_batch_size
4. Creates the Accelerator with the computed accumulation steps

Usage:
    # Stage 1: both phases sequentially
    accelerate launch --num_processes 8 --multi_gpu \
      src/train.py --config config/default.yaml --stage 1

    # Stage 1a only (warmup)
    accelerate launch --num_processes 8 --multi_gpu \
      src/train.py --config config/default.yaml --stage 1a

    # Stage 1b only (multi-chunk, loads 1a checkpoint)
    accelerate launch --num_processes 8 --multi_gpu \
      src/train.py --config config/default.yaml --stage 1b

    # Stage 2: QA finetune (loads stage 1b checkpoint)
    accelerate launch --num_processes 8 --multi_gpu \
      src/train.py --config config/default.yaml --stage 2 --resume outputs/full/stage1b/best.pt

    # Disable auto batch, use config values:
    accelerate launch src/train.py --config config/default.yaml --stage 1a \
      --override auto_batch_size=false

    # Single GPU:
    python src/train.py --config config/default.yaml --stage 1a
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.model import QCPC
from src.data import (
    create_pretrain_dataloader,
    create_multi_chunk_dataloader,
    create_qa_dataloader,
)
from src.auto_batch import find_max_batch_size, compute_accumulation_steps

logger = logging.getLogger(__name__)


def _mode_name(config: QCPCConfig) -> str:
    """Derive a human-readable mode name from config switches."""
    if config.use_decoupled_rope and config.use_prompt_bias:
        return "full"
    elif config.use_decoupled_rope:
        return "rope"
    elif config.use_prompt_bias:
        return "bias"
    else:
        return "baseline"


def _find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Find the latest checkpoint in a directory.

    Priority: final.pt > latest step_*.pt > best.pt
    """
    final = ckpt_dir / "final.pt"
    if final.exists():
        return final
    step_ckpts = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
        reverse=True,
    )
    if step_ckpts:
        return step_ckpts[0]
    best = ckpt_dir / "best.pt"
    if best.exists():
        return best
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="QCPC Training")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--stage", type=str, choices=["1", "1a", "1b", "2"], required=True,
        help="Training stage: 1 (both 1a+1b), 1a (warmup), 1b (multi-chunk), 2 (QA)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--override", nargs="*", default=[], help="Override config: key=value pairs")
    return parser.parse_args()


def apply_overrides(config: QCPCConfig, overrides: list[str]) -> QCPCConfig:
    """Apply command-line overrides to config."""
    import dataclasses
    d = dataclasses.asdict(config)
    for item in overrides:
        key, val = item.split("=", 1)
        if key not in d:
            raise ValueError(f"Unknown config key: {key}")
        # Type-cast based on existing type
        orig_type = type(d[key])
        if orig_type == bool:
            d[key] = val.lower() in ("true", "1", "yes")
        else:
            d[key] = orig_type(val)
    return QCPCConfig.from_dict(d)


def _resolve_batch_params(
    model: nn.Module,
    config: QCPCConfig,
    stage: str,
    world_size: int,
    rank: int,
) -> tuple[int, int]:
    """Resolve per-GPU batch size and gradient accumulation steps.

    If auto_batch_size is enabled, probes GPU memory on rank 0 and broadcasts
    the result. Otherwise falls back to config values.

    Args:
        stage: "1a", "1b", or "2"

    Returns:
        (per_gpu_batch_size, gradient_accumulation_steps)
    """
    if not config.auto_batch_size:
        if stage in ("1a", "1b"):
            return config.stage1_batch_size, config.stage1_gradient_accumulation_steps
        else:
            return config.stage2_batch_size, config.stage2_gradient_accumulation_steps

    # --- Auto batch probing ---
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to config batch size")
        if stage in ("1a", "1b"):
            return config.stage1_batch_size, config.stage1_gradient_accumulation_steps
        else:
            return config.stage2_batch_size, config.stage2_gradient_accumulation_steps

    # Only rank 0 probes; others wait for the broadcast
    if rank == 0:
        model.to(device)
        per_gpu_bs = find_max_batch_size(model, config, stage, device)
        model.to("cpu")
        torch.cuda.empty_cache()
        bs_tensor = torch.tensor([per_gpu_bs], dtype=torch.long, device=device)
    else:
        bs_tensor = torch.tensor([0], dtype=torch.long, device=device)

    # Broadcast from rank 0
    if world_size > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.distributed.broadcast(bs_tensor, src=0)

    per_gpu_bs = bs_tensor.item()
    accum_steps, actual_ebs = compute_accumulation_steps(
        per_gpu_bs, world_size, config.target_effective_batch_size
    )
    logger.info(
        f"Auto batch: per_gpu_bs={per_gpu_bs}, accum_steps={accum_steps}, "
        f"actual_ebs={actual_ebs} (target={config.target_effective_batch_size}), "
        f"world_size={world_size}"
    )
    return per_gpu_bs, accum_steps


@torch.no_grad()
def evaluate_stage1(model, eval_loader, accelerator):
    """Run evaluation for Stage 1a (single-chunk text completion). Returns eval loss and PPL."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in eval_loader:
        result = model(
            context_ids=batch["context_ids"],
            context_mask=batch["context_mask"],
            target_ids=batch["target_ids"],
            target_mask=batch["target_mask"],
        )
        total_loss += result["loss"].item()
        num_batches += 1
    model.train()
    if num_batches == 0:
        logger.warning("Eval loader yielded 0 batches — check eval_samples vs batch_size")
    avg_loss = total_loss / max(num_batches, 1)
    ppl = math.exp(min(avg_loss, 20))  # clamp to avoid overflow
    return avg_loss, ppl


@torch.no_grad()
def evaluate_stage1b(model, eval_loader, accelerator):
    """Run evaluation for Stage 1b (multi-chunk). Returns eval loss and PPL."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in eval_loader:
        result = model(
            chunk_ids=batch["chunk_ids"],
            chunk_mask=batch["chunk_mask"],
            target_ids=batch["target_ids"],
            target_mask=batch["target_mask"],
        )
        total_loss += result["loss"].item()
        num_batches += 1
    model.train()
    if num_batches == 0:
        logger.warning("Eval loader yielded 0 batches — check eval_samples vs batch_size")
    avg_loss = total_loss / max(num_batches, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


@torch.no_grad()
def evaluate_stage2(model, eval_loader, accelerator):
    """Run evaluation for Stage 2 (QA). Returns eval loss and PPL."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in eval_loader:
        result = model(
            context_ids=batch["context_ids"],
            context_mask=batch["context_mask"],
            prompt_ids=batch["prompt_ids"],
            prompt_mask=batch["prompt_mask"],
            target_ids=batch["target_ids"],
            target_mask=batch["target_mask"],
        )
        total_loss += result["loss"].item()
        num_batches += 1
    model.train()
    if num_batches == 0:
        logger.warning("Eval loader yielded 0 batches — check eval_samples vs batch_size")
    avg_loss = total_loss / max(num_batches, 1)
    ppl = math.exp(min(avg_loss, 20))  # clamp to avoid overflow
    return avg_loss, ppl


# ──────────────────────────────────────────────────────────────
# Stage 1a: Short window warmup (single-chunk compression)
# ──────────────────────────────────────────────────────────────

def train_stage1a(config: QCPCConfig, resume_path: str | None = None):
    """Stage 1a: Short window warmup — learn single-chunk compression.

    Data:  warmup_train.jsonl (first ~100M tokens)
    Input: 512 tokens context → Perceiver → M memory tokens
    Target: 128 tokens continuation, loss on target only
    """
    logger.info("=== Stage 1a: Short Window Warmup (Single-Chunk) ===")
    logger.info(f"use_decoupled_rope={config.use_decoupled_rope}")
    logger.info(f"context_len={config.stage1a_max_context_len}, cont_len={config.stage1a_max_cont_len}")

    # Build model
    model = QCPC(config)
    model.set_stage(1)

    counts = model.count_params()
    logger.info(f"Perceiver: {counts['perceiver']['trainable']:,} trainable params")
    logger.info(f"Decoder: {counts['decoder']['total']:,} frozen params")
    logger.info(f"Total trainable: {counts['total_trainable']:,}")

    # Resolve batch parameters
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    per_gpu_bs, accum_steps = _resolve_batch_params(
        model, config, stage="1a", world_size=world_size, rank=rank,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=accum_steps,
        mixed_precision="no",
    )

    mode = _mode_name(config)
    if accelerator.is_main_process:
        os.environ["WANDB_MODE"] = "offline"
        qwen_name = Path(config.qwen3_model_path).name
        wandb.init(
            project=config.wandb_project,
            group=mode,
            name=(
                f"s1a_{mode}"
                f"_{qwen_name}"
                f"_M{config.num_memory_tokens}"
                f"_ctx{config.stage1a_max_context_len}"
                f"_lr{config.stage1_lr}"
                f"_bs{per_gpu_bs}x{accum_steps}"
                f"_ep{config.stage1a_max_epochs}"
            ),
            config={
                "stage": "1a",
                "mode": mode,
                "backbone": qwen_name,
                "use_decoupled_rope": config.use_decoupled_rope,
                "use_prompt_bias": False,
                "hidden_dim": config.hidden_dim,
                "num_memory_tokens": config.num_memory_tokens,
                "num_process_layers": config.num_process_layers,
                "lr": config.stage1_lr,
                "batch_size": per_gpu_bs,
                "accum_steps": accum_steps,
                "world_size": accelerator.num_processes,
                "effective_batch_size": per_gpu_bs * accum_steps * accelerator.num_processes,
                "max_epochs": config.stage1a_max_epochs,
                "max_context_len": config.stage1a_max_context_len,
                "max_cont_len": config.stage1a_max_cont_len,
                "total_trainable": counts["total_trainable"],
            },
        )

    if accelerator.is_main_process:
        logger.info(f"Config: {config}")
        logger.info(f"Devices: {accelerator.num_processes}")
        logger.info(f"per_gpu_batch_size={per_gpu_bs}, gradient_accumulation_steps={accum_steps}")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.stage1_lr, weight_decay=config.stage1_weight_decay)

    # Data loaders
    tokenizer = model.decoder.tokenizer
    train_loader = create_pretrain_dataloader(
        data_path=config.stage1a_train_data_path,
        tokenizer=tokenizer,
        batch_size=per_gpu_bs,
        max_context_len=config.stage1a_max_context_len,
        max_cont_len=config.stage1a_max_cont_len,
        num_workers=config.num_workers,
    )
    eval_loader = create_pretrain_dataloader(
        data_path=config.pretrain_eval_data_path,
        tokenizer=tokenizer,
        batch_size=per_gpu_bs,
        max_context_len=config.stage1a_max_context_len,
        max_cont_len=config.stage1a_max_cont_len,
        num_workers=config.num_workers,
        shuffle=False,
        max_samples=config.eval_samples,
        drop_last=False,
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    eval_loader = accelerator.prepare(eval_loader)

    # LR scheduler
    max_epochs = config.stage1a_max_epochs
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = int(total_steps * config.stage1_warmup_ratio)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=config.stage1_lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    if accelerator.is_main_process:
        logger.info(
            f"Batches/epoch (per GPU): {len(train_loader)}, "
            f"optimizer steps/epoch: {steps_per_epoch}, "
            f"total optimizer steps: {total_steps}, warmup: {warmup_steps}, "
            f"epochs: {max_epochs}"
        )

    output_dir = Path(config.output_dir) / "stage1a"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect checkpoint
    if not resume_path:
        auto_ckpt = _find_latest_checkpoint(output_dir)
        if auto_ckpt:
            resume_path = str(auto_ckpt)
            logger.info(f"Auto-detected checkpoint: {resume_path}")
        else:
            logger.info("No checkpoint found, training from scratch")

    start_epoch = 0
    global_step = 0
    if resume_path:
        logger.info(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.perceiver.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        if global_step > 0:
            for _ in range(global_step):
                scheduler.step()
            if accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"Resumed scheduler to step {global_step}, lr={lr:.2e}")

    if start_epoch >= max_epochs:
        if accelerator.is_main_process:
            logger.warning(
                f"Checkpoint epoch ({start_epoch}) >= max_epochs ({max_epochs}). "
                f"Increase stage1a_max_epochs to continue training."
            )
            wandb.finish()
        return

    # Training loop
    model.train()
    best_eval_loss = float("inf")

    for epoch in range(start_epoch, max_epochs):
        epoch_loss = 0.0
        num_micro_batches = 0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                result = model(
                    context_ids=batch["context_ids"],
                    context_mask=batch["context_mask"],
                    target_ids=batch["target_ids"],
                    target_mask=batch["target_mask"],
                )
                loss = result["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, config.stage1_grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_micro_batches += 1

            if accelerator.sync_gradients:
                scheduler.step()
                global_step += 1

                if global_step % config.log_interval == 0 and accelerator.is_main_process:
                    avg_loss = epoch_loss / num_micro_batches
                    train_ppl = math.exp(min(avg_loss, 20))
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"[1a] Epoch {epoch+1}/{max_epochs} "
                        f"Step {global_step}/{total_steps} | "
                        f"loss={loss.item():.4f} avg_loss={avg_loss:.4f} ppl={train_ppl:.2f} lr={lr:.2e}"
                    )
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/ppl": train_ppl,
                        "train/lr": lr,
                        "train/epoch": epoch + 1,
                    }, step=global_step)

                if global_step % config.eval_interval == 0 and accelerator.is_main_process:
                    eval_loss, eval_ppl = evaluate_stage1(model, eval_loader, accelerator)
                    logger.info(
                        f"  [Eval 1a] Step {global_step}/{total_steps} | "
                        f"eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}"
                    )
                    wandb.log({"eval/loss": eval_loss, "eval/ppl": eval_ppl}, step=global_step)
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        _save_checkpoint(accelerator, model, optimizer, epoch, global_step, output_dir / "best.pt")

                if global_step % config.save_interval == 0 and accelerator.is_main_process:
                    _save_checkpoint(accelerator, model, optimizer, epoch, global_step, output_dir / f"step_{global_step}.pt")

        avg_epoch_loss = epoch_loss / max(num_micro_batches, 1)
        epoch_ppl = math.exp(min(avg_epoch_loss, 20))
        if accelerator.is_main_process:
            logger.info(
                f"[1a] Epoch {epoch+1}/{max_epochs} done | "
                f"step {global_step}/{total_steps} | "
                f"avg_loss={avg_epoch_loss:.4f} ppl={epoch_ppl:.2f}"
            )

    if accelerator.is_main_process:
        _save_checkpoint(accelerator, model, optimizer, max_epochs, global_step, output_dir / "final.pt")
        wandb.finish()
    logger.info(f"Stage 1a training complete. Total optimizer steps: {global_step}")


# ──────────────────────────────────────────────────────────────
# Stage 1b: Long window multi-chunk concatenation
# ──────────────────────────────────────────────────────────────

def train_stage1b(config: QCPCConfig, resume_path: str | None = None):
    """Stage 1b: Long window training — learn cross-chunk reading.

    Data:  multichunk_train.jsonl (remaining tokens after warmup)
    Input: K x 512 tokens → K independent Perceiver compressions → concatenate → K*M memory tokens
           K is dynamic per sample (min_chunks..max_chunks), determined by document length.
    Target: 128 tokens continuation, loss on target only
    """
    K_max = config.stage1b_max_chunks
    K_min = config.stage1b_min_chunks
    N = config.stage1b_chunk_len
    T = config.stage1b_max_cont_len

    logger.info("=== Stage 1b: Long Window Multi-Chunk Training ===")
    logger.info(f"use_decoupled_rope={config.use_decoupled_rope}")
    logger.info(f"chunks=[{K_min},{K_max}], chunk_len={N}, cont_len={T}")

    # Build model
    model = QCPC(config)
    model.set_stage(1)

    counts = model.count_params()
    logger.info(f"Perceiver: {counts['perceiver']['trainable']:,} trainable params")
    logger.info(f"Total trainable: {counts['total_trainable']:,}")

    # Auto-detect stage1a checkpoint for init
    s1b_dir = Path(config.output_dir) / "stage1b"
    s1a_dir = Path(config.output_dir) / "stage1a"
    s1b_ckpt_path = None
    s1a_init_path = None

    s1b_auto = _find_latest_checkpoint(s1b_dir)
    if s1b_auto:
        s1b_ckpt_path = str(s1b_auto)
        logger.info(f"Auto-detected stage1b checkpoint: {s1b_ckpt_path}")
    elif resume_path:
        s1a_init_path = resume_path
    else:
        for name in ["best.pt", "final.pt"]:
            p = s1a_dir / name
            if p.exists():
                s1a_init_path = str(p)
                logger.info(f"Auto-detected stage1a checkpoint for init: {s1a_init_path}")
                break
        if not s1a_init_path:
            logger.warning("No stage1a checkpoint found! Training 1b from scratch.")

    # Load stage1a weights (only if NOT doing stage1b resume)
    if s1a_init_path and not s1b_ckpt_path:
        logger.info(f"Loading stage 1a weights from {s1a_init_path}")
        ckpt = torch.load(s1a_init_path, map_location="cpu")
        model.perceiver.load_state_dict(ckpt["model"], strict=False)
        model.set_stage(1)

    # Resolve batch parameters
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    per_gpu_bs, accum_steps = _resolve_batch_params(
        model, config, stage="1b", world_size=world_size, rank=rank,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=accum_steps,
        mixed_precision="no",
    )

    mode = _mode_name(config)
    if accelerator.is_main_process:
        os.environ["WANDB_MODE"] = "offline"
        qwen_name = Path(config.qwen3_model_path).name
        wandb.init(
            project=config.wandb_project,
            group=mode,
            name=(
                f"s1b_{mode}"
                f"_{qwen_name}"
                f"_M{config.num_memory_tokens}"
                f"_K{K_min}-{K_max}x{N}"
                f"_lr{config.stage1_lr}"
                f"_bs{per_gpu_bs}x{accum_steps}"
                f"_ep{config.stage1b_max_epochs}"
            ),
            config={
                "stage": "1b",
                "mode": mode,
                "backbone": qwen_name,
                "use_decoupled_rope": config.use_decoupled_rope,
                "use_prompt_bias": False,
                "hidden_dim": config.hidden_dim,
                "num_memory_tokens": config.num_memory_tokens,
                "num_process_layers": config.num_process_layers,
                "lr": config.stage1_lr,
                "batch_size": per_gpu_bs,
                "accum_steps": accum_steps,
                "world_size": accelerator.num_processes,
                "effective_batch_size": per_gpu_bs * accum_steps * accelerator.num_processes,
                "max_epochs": config.stage1b_max_epochs,
                "max_chunks": K_max,
                "min_chunks": K_min,
                "chunk_len": N,
                "cont_len": T,
                "total_trainable": counts["total_trainable"],
            },
        )

    if accelerator.is_main_process:
        logger.info(f"Config: {config}")
        logger.info(f"Devices: {accelerator.num_processes}")
        logger.info(f"per_gpu_batch_size={per_gpu_bs}, gradient_accumulation_steps={accum_steps}")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.stage1_lr, weight_decay=config.stage1_weight_decay)

    # Data loaders
    tokenizer = model.decoder.tokenizer
    train_loader = create_multi_chunk_dataloader(
        data_path=config.stage1b_train_data_path,
        tokenizer=tokenizer,
        batch_size=per_gpu_bs,
        max_chunks=K_max,
        min_chunks=K_min,
        chunk_len=N,
        cont_len=T,
        num_workers=config.num_workers,
    )
    eval_loader = create_multi_chunk_dataloader(
        data_path=config.pretrain_eval_data_path,
        tokenizer=tokenizer,
        batch_size=per_gpu_bs,
        max_chunks=K_max,
        min_chunks=K_min,
        chunk_len=N,
        cont_len=T,
        num_workers=config.num_workers,
        shuffle=False,
        max_samples=config.eval_samples,
        drop_last=False,
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    eval_loader = accelerator.prepare(eval_loader)

    # LR scheduler
    max_epochs = config.stage1b_max_epochs
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = int(total_steps * config.stage1_warmup_ratio)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=config.stage1_lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    if accelerator.is_main_process:
        logger.info(
            f"Batches/epoch (per GPU): {len(train_loader)}, "
            f"optimizer steps/epoch: {steps_per_epoch}, "
            f"total optimizer steps: {total_steps}, warmup: {warmup_steps}, "
            f"epochs: {max_epochs}"
        )

    output_dir = Path(config.output_dir) / "stage1b"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full stage1b resume
    start_epoch = 0
    global_step = 0
    if s1b_ckpt_path:
        logger.info(f"Resuming stage1b from {s1b_ckpt_path}")
        ckpt = torch.load(s1b_ckpt_path, map_location="cpu")
        model.perceiver.load_state_dict(ckpt["model"], strict=False)
        model.set_stage(1)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        if global_step > 0:
            for _ in range(global_step):
                scheduler.step()
            if accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"Resumed scheduler to step {global_step}, lr={lr:.2e}")

    if start_epoch >= max_epochs:
        if accelerator.is_main_process:
            logger.warning(
                f"Checkpoint epoch ({start_epoch}) >= max_epochs ({max_epochs}). "
                f"Increase stage1b_max_epochs to continue training."
            )
            wandb.finish()
        return

    # Training loop
    model.train()
    best_eval_loss = float("inf")

    for epoch in range(start_epoch, max_epochs):
        epoch_loss = 0.0
        num_micro_batches = 0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                result = model(
                    chunk_ids=batch["chunk_ids"],
                    chunk_mask=batch["chunk_mask"],
                    target_ids=batch["target_ids"],
                    target_mask=batch["target_mask"],
                )
                loss = result["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, config.stage1_grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_micro_batches += 1

            if accelerator.sync_gradients:
                scheduler.step()
                global_step += 1

                if global_step % config.log_interval == 0 and accelerator.is_main_process:
                    avg_loss = epoch_loss / num_micro_batches
                    train_ppl = math.exp(min(avg_loss, 20))
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"[1b] Epoch {epoch+1}/{max_epochs} "
                        f"Step {global_step}/{total_steps} | "
                        f"loss={loss.item():.4f} avg_loss={avg_loss:.4f} ppl={train_ppl:.2f} lr={lr:.2e}"
                    )
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/ppl": train_ppl,
                        "train/lr": lr,
                        "train/epoch": epoch + 1,
                    }, step=global_step)

                if global_step % config.eval_interval == 0 and accelerator.is_main_process:
                    eval_loss, eval_ppl = evaluate_stage1b(model, eval_loader, accelerator)
                    logger.info(
                        f"  [Eval 1b] Step {global_step}/{total_steps} | "
                        f"eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}"
                    )
                    wandb.log({"eval/loss": eval_loss, "eval/ppl": eval_ppl}, step=global_step)
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        _save_checkpoint(accelerator, model, optimizer, epoch, global_step, output_dir / "best.pt")

                if global_step % config.save_interval == 0 and accelerator.is_main_process:
                    _save_checkpoint(accelerator, model, optimizer, epoch, global_step, output_dir / f"step_{global_step}.pt")

        avg_epoch_loss = epoch_loss / max(num_micro_batches, 1)
        epoch_ppl = math.exp(min(avg_epoch_loss, 20))
        if accelerator.is_main_process:
            logger.info(
                f"[1b] Epoch {epoch+1}/{max_epochs} done | "
                f"step {global_step}/{total_steps} | "
                f"avg_loss={avg_epoch_loss:.4f} ppl={epoch_ppl:.2f}"
            )

    if accelerator.is_main_process:
        _save_checkpoint(accelerator, model, optimizer, max_epochs, global_step, output_dir / "final.pt")
        wandb.finish()
    logger.info(f"Stage 1b training complete. Total optimizer steps: {global_step}")


# ──────────────────────────────────────────────────────────────
# Stage 2: QA Instruction Finetuning
# ──────────────────────────────────────────────────────────────

def train_stage2(config: QCPCConfig, resume_path: str | None = None):
    """Stage 2: QA instruction finetuning."""
    logger.info("=== Stage 2: QA Instruction Finetuning ===")
    logger.info(f"use_decoupled_rope={config.use_decoupled_rope}, use_prompt_bias={config.use_prompt_bias}")

    # Build model with prompt bias enabled
    config.use_prompt_bias = True
    model = QCPC(config)
    model.set_stage(2)

    # Auto-detect: stage2 checkpoint (full resume) or stage1b/stage1a checkpoint (init)
    s2_dir = Path(config.output_dir) / "stage2"
    s1b_dir = Path(config.output_dir) / "stage1b"
    s1a_dir = Path(config.output_dir) / "stage1a"
    # Legacy fallback
    s1_dir = Path(config.output_dir) / "stage1"
    s2_ckpt_path = None
    s1_init_path = None

    s2_auto = _find_latest_checkpoint(s2_dir)
    if s2_auto:
        s2_ckpt_path = str(s2_auto)
        logger.info(f"Auto-detected stage2 checkpoint: {s2_ckpt_path}")
    elif resume_path:
        s1_init_path = resume_path
    else:
        # Search order: stage1b > stage1a > stage1 (legacy)
        for search_dir in [s1b_dir, s1a_dir, s1_dir]:
            for name in ["best.pt", "final.pt"]:
                p = search_dir / name
                if p.exists():
                    s1_init_path = str(p)
                    logger.info(f"Auto-detected checkpoint for init: {s1_init_path}")
                    break
            if s1_init_path:
                break
        if not s1_init_path:
            logger.warning("No stage1 checkpoint found for initialization!")

    # Load weights for init (only if NOT doing stage2 resume)
    if s1_init_path and not s2_ckpt_path:
        logger.info(f"Loading stage 1 weights from {s1_init_path}")
        ckpt = torch.load(s1_init_path, map_location="cpu")
        model.perceiver.load_state_dict(ckpt["model"], strict=False)
        model.set_stage(2)

    counts = model.count_params()
    logger.info(f"Perceiver: {counts['perceiver']['trainable']:,} trainable params")
    logger.info(f"Total trainable: {counts['total_trainable']:,}")

    # Resolve batch parameters
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    per_gpu_bs, accum_steps = _resolve_batch_params(
        model, config, stage="2", world_size=world_size, rank=rank,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=accum_steps,
        mixed_precision="no",
    )

    mode = _mode_name(config)
    if accelerator.is_main_process:
        os.environ["WANDB_MODE"] = "offline"
        qwen_name = Path(config.qwen3_model_path).name
        wandb.init(
            project=config.wandb_project,
            group=mode,
            name=(
                f"s2_{mode}"
                f"_{qwen_name}"
                f"_M{config.num_memory_tokens}"
                f"_L{config.num_process_layers}"
                f"_D{config.hidden_dim}"
                f"_lr{config.stage2_lr}"
                f"_bs{per_gpu_bs}x{accum_steps}"
                f"_ep{config.stage2_max_epochs}"
            ),
            config={
                "stage": 2,
                "mode": mode,
                "backbone": qwen_name,
                "use_decoupled_rope": config.use_decoupled_rope,
                "use_prompt_bias": True,
                "hidden_dim": config.hidden_dim,
                "num_memory_tokens": config.num_memory_tokens,
                "num_process_layers": config.num_process_layers,
                "lr": config.stage2_lr,
                "batch_size": per_gpu_bs,
                "accum_steps": accum_steps,
                "world_size": accelerator.num_processes,
                "effective_batch_size": per_gpu_bs * accum_steps * accelerator.num_processes,
                "max_epochs": config.stage2_max_epochs,
                "max_context_len": config.stage2_max_context_len,
                "max_prompt_len": config.stage2_max_prompt_len,
                "max_answer_len": config.stage2_max_answer_len,
                "total_trainable": counts["total_trainable"],
            },
        )

    if accelerator.is_main_process:
        logger.info(f"Config: {config}")
        logger.info(f"Devices: {accelerator.num_processes}")
        logger.info(f"per_gpu_batch_size={per_gpu_bs}, gradient_accumulation_steps={accum_steps}")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.stage2_lr, weight_decay=config.stage2_weight_decay)

    # Data loaders
    tokenizer = model.decoder.tokenizer
    train_loader = create_qa_dataloader(
        data_path=config.sft_train_data_path,
        tokenizer=tokenizer,
        batch_size=per_gpu_bs,
        max_context_len=config.stage2_max_context_len,
        max_prompt_len=config.stage2_max_prompt_len,
        max_answer_len=config.stage2_max_answer_len,
        num_workers=config.num_workers,
    )
    eval_loader = create_qa_dataloader(
        data_path=config.sft_eval_data_path,
        tokenizer=tokenizer,
        batch_size=per_gpu_bs,
        max_context_len=config.stage2_max_context_len,
        max_prompt_len=config.stage2_max_prompt_len,
        max_answer_len=config.stage2_max_answer_len,
        num_workers=config.num_workers,
        shuffle=False,
        max_samples=config.eval_samples,
        drop_last=False,
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    eval_loader = accelerator.prepare(eval_loader)

    # LR scheduler
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = steps_per_epoch * config.stage2_max_epochs
    warmup_steps = int(total_steps * config.stage2_warmup_ratio)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=config.stage2_lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    if accelerator.is_main_process:
        logger.info(
            f"Batches/epoch (per GPU): {len(train_loader)}, "
            f"optimizer steps/epoch: {steps_per_epoch}, "
            f"total optimizer steps: {total_steps}, warmup: {warmup_steps}, "
            f"epochs: {config.stage2_max_epochs}"
        )

    output_dir = Path(config.output_dir) / "stage2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full stage2 resume
    start_epoch = 0
    global_step = 0
    if s2_ckpt_path:
        logger.info(f"Resuming stage2 from {s2_ckpt_path}")
        ckpt = torch.load(s2_ckpt_path, map_location="cpu")
        model.perceiver.load_state_dict(ckpt["model"], strict=False)
        model.set_stage(2)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        if global_step > 0:
            for _ in range(global_step):
                scheduler.step()
            if accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"Resumed scheduler to step {global_step}, lr={lr:.2e}")

    if start_epoch >= config.stage2_max_epochs:
        if accelerator.is_main_process:
            logger.warning(
                f"Checkpoint epoch ({start_epoch}) >= max_epochs ({config.stage2_max_epochs}). "
                f"Increase stage2_max_epochs to continue training."
            )
            wandb.finish()
        return

    # Training loop
    model.train()
    best_eval_loss = float("inf")

    for epoch in range(start_epoch, config.stage2_max_epochs):
        epoch_loss = 0.0
        num_micro_batches = 0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                result = model(
                    context_ids=batch["context_ids"],
                    context_mask=batch["context_mask"],
                    prompt_ids=batch["prompt_ids"],
                    prompt_mask=batch["prompt_mask"],
                    target_ids=batch["target_ids"],
                    target_mask=batch["target_mask"],
                )
                loss = result["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, config.stage2_grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_micro_batches += 1

            if accelerator.sync_gradients:
                scheduler.step()
                global_step += 1

                if global_step % config.log_interval == 0 and accelerator.is_main_process:
                    avg_loss = epoch_loss / num_micro_batches
                    train_ppl = math.exp(min(avg_loss, 20))
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Epoch {epoch+1}/{config.stage2_max_epochs} "
                        f"Step {global_step}/{total_steps} | "
                        f"loss={loss.item():.4f} avg_loss={avg_loss:.4f} ppl={train_ppl:.2f} lr={lr:.2e}"
                    )
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/ppl": train_ppl,
                        "train/lr": lr,
                        "train/epoch": epoch + 1,
                    }, step=global_step)

                if global_step % config.eval_interval == 0 and accelerator.is_main_process:
                    eval_loss, eval_ppl = evaluate_stage2(model, eval_loader, accelerator)
                    logger.info(
                        f"  [Eval] Step {global_step}/{total_steps} | "
                        f"eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}"
                    )
                    wandb.log({"eval/loss": eval_loss, "eval/ppl": eval_ppl}, step=global_step)
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        _save_checkpoint(accelerator, model, optimizer, epoch, global_step, output_dir / "best.pt")

                if global_step % config.save_interval == 0 and accelerator.is_main_process:
                    _save_checkpoint(accelerator, model, optimizer, epoch, global_step, output_dir / f"step_{global_step}.pt")

        avg_epoch_loss = epoch_loss / max(num_micro_batches, 1)
        epoch_ppl = math.exp(min(avg_epoch_loss, 20))
        if accelerator.is_main_process:
            logger.info(
                f"Epoch {epoch+1}/{config.stage2_max_epochs} done | "
                f"step {global_step}/{total_steps} | "
                f"avg_loss={avg_epoch_loss:.4f} ppl={epoch_ppl:.2f}"
            )

    if accelerator.is_main_process:
        _save_checkpoint(accelerator, model, optimizer, config.stage2_max_epochs, global_step, output_dir / "final.pt")
        wandb.finish()
    logger.info(f"Stage 2 training complete. Total optimizer steps: {global_step}")


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────

def _save_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: Path,
):
    """Save checkpoint with only the Perceiver IO (trainable) state dict."""
    unwrapped = accelerator.unwrap_model(model)
    # Only save perceiver weights (trainable part) — not the frozen LLM
    perceiver_state = unwrapped.perceiver.state_dict()
    torch.save({
        "model": perceiver_state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, path)
    logger.info(f"Saved checkpoint: {path}")


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load config
    config = QCPCConfig.load(args.config)
    if args.override:
        config = apply_overrides(config, args.override)

    # Set seed
    set_seed(config.seed)

    if args.stage == "1":
        # Run both phases sequentially
        train_stage1a(config, resume_path=args.resume)
        train_stage1b(config)  # auto-loads 1a checkpoint
    elif args.stage == "1a":
        train_stage1a(config, resume_path=args.resume)
    elif args.stage == "1b":
        train_stage1b(config, resume_path=args.resume)
    elif args.stage == "2":
        train_stage2(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
