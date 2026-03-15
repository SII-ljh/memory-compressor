"""Training script for QCPC: Stage 1 (pretrain) + Stage 2 (QA finetune).

Supports multi-GPU via Accelerate (wraps FSDP/DDP).
Frozen LLM Decoder is excluded from FSDP sharding and optimizer states.

When auto_batch_size is enabled (default), the script:
1. Builds the model on a single GPU
2. Binary-searches for the max per-GPU batch size
3. Computes gradient_accumulation_steps to reach target_effective_batch_size
4. Creates the Accelerator with the computed accumulation steps

Usage:
    # Stage 1: pretrain (auto batch size)
    accelerate launch --num_processes 8 --multi_gpu \
      src/train.py --config config/default.yaml --stage 1

    # Stage 2: QA finetune (loads stage 1 checkpoint)
    accelerate launch --num_processes 8 --multi_gpu \
      src/train.py --config config/default.yaml --stage 2 --resume outputs/full/stage1/best.pt

    # Disable auto batch, use config values:
    accelerate launch src/train.py --config config/default.yaml --stage 1 \
      --override auto_batch_size=false

    # Single GPU:
    python src/train.py --config config/default.yaml --stage 1
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
from src.data import create_pretrain_dataloader, create_qa_dataloader
from src.auto_batch import find_max_batch_size, compute_accumulation_steps

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="QCPC Training")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
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
    stage: int,
    world_size: int,
    rank: int,
) -> tuple[int, int]:
    """Resolve per-GPU batch size and gradient accumulation steps.

    If auto_batch_size is enabled, probes GPU memory on rank 0 and broadcasts
    the result. Otherwise falls back to config values.

    Returns:
        (per_gpu_batch_size, gradient_accumulation_steps)
    """
    if not config.auto_batch_size:
        if stage == 1:
            return config.stage1_batch_size, config.stage1_gradient_accumulation_steps
        else:
            return config.stage2_batch_size, config.stage2_gradient_accumulation_steps

    # --- Auto batch probing ---
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to config batch size")
        if stage == 1:
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
    """Run evaluation for Stage 1 (text completion). Returns eval loss and PPL."""
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


def train_stage1(config: QCPCConfig, resume_path: str | None = None):
    """Stage 1: Text completion pretraining."""
    logger.info("=== Stage 1: Text Completion Pretraining ===")
    logger.info(f"use_decoupled_rope={config.use_decoupled_rope}")

    # Build model
    model = QCPC(config)
    model.set_stage(1)

    # Log parameter counts
    counts = model.count_params()
    logger.info(f"Perceiver: {counts['perceiver']['trainable']:,} trainable params")
    logger.info(f"Decoder: {counts['decoder']['total']:,} frozen params")
    logger.info(f"Total trainable: {counts['total_trainable']:,}")

    # Resolve batch parameters (auto-probe or config)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    per_gpu_bs, accum_steps = _resolve_batch_params(model, config, stage=1, world_size=world_size, rank=rank)

    # Create Accelerator with resolved accumulation steps
    accelerator = Accelerator(
        gradient_accumulation_steps=accum_steps,
        mixed_precision="no",
    )

    # Init wandb (offline mode, main process only)
    if accelerator.is_main_process:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project=config.wandb_project,
            name=f"stage1_rope{config.use_decoupled_rope}",
            config={
                "stage": 1,
                "use_decoupled_rope": config.use_decoupled_rope,
                "use_prompt_bias": False,
                "hidden_dim": config.hidden_dim,
                "num_memory_tokens": config.num_memory_tokens,
                "num_process_layers": config.num_process_layers,
                "lr": config.stage1_lr,
                "batch_size": per_gpu_bs,
                "accum_steps": accum_steps,
                "max_epochs": config.stage1_max_epochs,
                "max_context_len": config.stage1_max_context_len,
                "max_cont_len": config.stage1_max_cont_len,
                "total_trainable": counts["total_trainable"],
            },
        )

    if accelerator.is_main_process:
        logger.info(f"Config: {config}")
        logger.info(f"Devices: {accelerator.num_processes}")
        logger.info(f"per_gpu_batch_size={per_gpu_bs}, gradient_accumulation_steps={accum_steps}")

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.stage1_lr,
        weight_decay=config.stage1_weight_decay,
    )

    # Data loaders
    tokenizer = model.decoder.tokenizer
    train_loader = create_pretrain_dataloader(
        data_path=config.pretrain_data_path,
        tokenizer=tokenizer,
        batch_size=per_gpu_bs,
        max_context_len=config.stage1_max_context_len,
        max_cont_len=config.stage1_max_cont_len,
        num_workers=config.num_workers,
    )

    # Eval loader: last eval_samples from pretrain data, no shuffle
    eval_loader = create_pretrain_dataloader(
        data_path=config.pretrain_data_path,
        tokenizer=tokenizer,
        batch_size=per_gpu_bs,
        max_context_len=config.stage1_max_context_len,
        max_cont_len=config.stage1_max_cont_len,
        num_workers=config.num_workers,
        shuffle=False,
        max_samples=config.eval_samples,
        drop_last=False,
    )

    # LR scheduler
    total_steps = len(train_loader) * config.stage1_max_epochs // accum_steps
    warmup_steps = int(total_steps * config.stage1_warmup_ratio)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=config.stage1_lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    if accelerator.is_main_process:
        logger.info(f"Total training steps: {total_steps}, warmup: {warmup_steps}")

    # Prepare with accelerator
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    eval_loader = accelerator.prepare(eval_loader)

    # Resume checkpoint (perceiver-only state dict)
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

    # Output dir
    output_dir = Path(config.output_dir) / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    best_eval_loss = float("inf")

    for epoch in range(start_epoch, config.stage1_max_epochs):
        epoch_loss = 0.0
        num_batches = 0

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
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % config.log_interval == 0 and accelerator.is_main_process:
                avg_loss = epoch_loss / num_batches
                train_ppl = math.exp(min(avg_loss, 20))
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch+1} Step {global_step}/{total_steps} | "
                    f"loss={loss.item():.4f} avg_loss={avg_loss:.4f} ppl={train_ppl:.2f} lr={lr:.2e}"
                )
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/ppl": train_ppl,
                    "train/lr": lr,
                    "train/epoch": epoch + 1,
                }, step=global_step)

            # Eval
            if global_step % config.eval_interval == 0 and accelerator.is_main_process:
                eval_loss, eval_ppl = evaluate_stage1(model, eval_loader, accelerator)
                logger.info(
                    f"  [Eval] Step {global_step}/{total_steps} | "
                    f"eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}"
                )
                wandb.log({
                    "eval/loss": eval_loss,
                    "eval/ppl": eval_ppl,
                }, step=global_step)

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    _save_checkpoint(
                        accelerator, model, optimizer, epoch, global_step,
                        output_dir / "best.pt"
                    )

            if global_step % config.save_interval == 0 and accelerator.is_main_process:
                _save_checkpoint(
                    accelerator, model, optimizer, epoch, global_step,
                    output_dir / f"step_{global_step}.pt"
                )

        # End of epoch
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        epoch_ppl = math.exp(min(avg_epoch_loss, 20))
        if accelerator.is_main_process:
            logger.info(
                f"Epoch {epoch+1} done | avg_loss={avg_epoch_loss:.4f} ppl={epoch_ppl:.2f}"
            )

    # Save final
    if accelerator.is_main_process:
        _save_checkpoint(
            accelerator, model, optimizer, config.stage1_max_epochs, global_step,
            output_dir / "final.pt"
        )
        wandb.finish()
    logger.info("Stage 1 training complete.")


def train_stage2(config: QCPCConfig, resume_path: str | None = None):
    """Stage 2: QA instruction finetuning."""
    logger.info("=== Stage 2: QA Instruction Finetuning ===")
    logger.info(f"use_decoupled_rope={config.use_decoupled_rope}, use_prompt_bias={config.use_prompt_bias}")

    # Build model with prompt bias enabled
    config.use_prompt_bias = True
    model = QCPC(config)
    model.set_stage(2)

    # Load stage 1 checkpoint (perceiver-only state dict)
    if resume_path:
        logger.info(f"Loading stage 1 weights from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.perceiver.load_state_dict(ckpt["model"], strict=False)
        # Re-set stage 2 after loading (ensure prompt bias params are unfrozen)
        model.set_stage(2)

    counts = model.count_params()
    logger.info(f"Perceiver: {counts['perceiver']['trainable']:,} trainable params")
    logger.info(f"Total trainable: {counts['total_trainable']:,}")

    # Resolve batch parameters (auto-probe or config)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    per_gpu_bs, accum_steps = _resolve_batch_params(model, config, stage=2, world_size=world_size, rank=rank)

    # Create Accelerator with resolved accumulation steps
    accelerator = Accelerator(
        gradient_accumulation_steps=accum_steps,
        mixed_precision="no",
    )

    # Init wandb (offline mode, main process only)
    if accelerator.is_main_process:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project=config.wandb_project,
            name=f"stage2_rope{config.use_decoupled_rope}_bias{config.use_prompt_bias}",
            config={
                "stage": 2,
                "use_decoupled_rope": config.use_decoupled_rope,
                "use_prompt_bias": True,
                "hidden_dim": config.hidden_dim,
                "num_memory_tokens": config.num_memory_tokens,
                "num_process_layers": config.num_process_layers,
                "lr": config.stage2_lr,
                "batch_size": per_gpu_bs,
                "accum_steps": accum_steps,
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
    optimizer = AdamW(
        trainable_params,
        lr=config.stage2_lr,
        weight_decay=config.stage2_weight_decay,
    )

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

    # Eval loader: dev set, capped at eval_samples
    eval_loader = create_qa_dataloader(
        data_path=config.sft_dev_data_path,
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

    # LR scheduler
    total_steps = len(train_loader) * config.stage2_max_epochs // accum_steps
    warmup_steps = int(total_steps * config.stage2_warmup_ratio)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=config.stage2_lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    if accelerator.is_main_process:
        logger.info(f"Total training steps: {total_steps}, warmup: {warmup_steps}")

    # Prepare
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    eval_loader = accelerator.prepare(eval_loader)

    # Output dir
    output_dir = Path(config.output_dir) / "stage2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    best_eval_loss = float("inf")
    global_step = 0

    for epoch in range(config.stage2_max_epochs):
        epoch_loss = 0.0
        num_batches = 0

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
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % config.log_interval == 0 and accelerator.is_main_process:
                avg_loss = epoch_loss / num_batches
                train_ppl = math.exp(min(avg_loss, 20))
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch+1} Step {global_step}/{total_steps} | "
                    f"loss={loss.item():.4f} avg_loss={avg_loss:.4f} ppl={train_ppl:.2f} lr={lr:.2e}"
                )
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/ppl": train_ppl,
                    "train/lr": lr,
                    "train/epoch": epoch + 1,
                }, step=global_step)

            # Eval
            if global_step % config.eval_interval == 0 and accelerator.is_main_process:
                eval_loss, eval_ppl = evaluate_stage2(model, eval_loader, accelerator)
                logger.info(
                    f"  [Eval] Step {global_step}/{total_steps} | "
                    f"eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}"
                )
                wandb.log({
                    "eval/loss": eval_loss,
                    "eval/ppl": eval_ppl,
                }, step=global_step)

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    _save_checkpoint(
                        accelerator, model, optimizer, epoch, global_step,
                        output_dir / "best.pt"
                    )

            if global_step % config.save_interval == 0 and accelerator.is_main_process:
                _save_checkpoint(
                    accelerator, model, optimizer, epoch, global_step,
                    output_dir / f"step_{global_step}.pt"
                )

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        epoch_ppl = math.exp(min(avg_epoch_loss, 20))
        if accelerator.is_main_process:
            logger.info(
                f"Epoch {epoch+1} done | avg_loss={avg_epoch_loss:.4f} ppl={epoch_ppl:.2f}"
            )

    if accelerator.is_main_process:
        _save_checkpoint(
            accelerator, model, optimizer, config.stage2_max_epochs, global_step,
            output_dir / "final.pt"
        )
        wandb.finish()
    logger.info("Stage 2 training complete.")


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

    if args.stage == 1:
        train_stage1(config, resume_path=args.resume)
    elif args.stage == 2:
        train_stage2(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
