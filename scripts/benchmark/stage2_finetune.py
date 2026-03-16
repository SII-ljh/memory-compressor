"""Stage 2 Upper Bound: LoRA fine-tune Qwen3 on SFT QA data.

Fine-tunes Qwen3 with full context + question → answer using LoRA.
This establishes the ceiling for QCPC Stage 2.

Uses data from data_processor.py:
  - Train: config.sft_train_data_path
  - Eval:  config.sft_eval_data_path

Format: [{"context": str, "question": str, "answer": str, "source": str}]

Usage:
    # Multi-GPU (auto batch size):
    accelerate launch --multi_gpu \
      scripts/benchmark/stage2_finetune.py --config config/default.yaml

    # Override batch size:
    accelerate launch --multi_gpu \
      scripts/benchmark/stage2_finetune.py --config config/default.yaml --batch_size 8

Requires: pip install peft
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    print("Error: peft not installed. Run: pip install peft")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import QCPCConfig

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"


def _probe_max_batch_size(model, max_seq_len, device, train_mode=False,
                          upper_bound=128, safety_margin=0.85):
    """Binary search for max batch size that fits in GPU memory.

    Simulates a full training step (forward + backward + optimizer.step) to
    account for optimizer states and gradient buffers. Applies safety_margin
    to leave room for DDP overhead not captured in single-GPU probing.
    """
    lo, hi = 1, upper_bound
    best = 1
    logger.info(f"Auto batch probe: searching [{lo}, {hi}] on {device} "
                f"(seq_len={max_seq_len}, train={train_mode})")

    was_training = model.training
    if train_mode:
        model.train()
    else:
        model.eval()

    # Create a temporary optimizer to capture optimizer state memory
    probe_optimizer = None
    if train_mode:
        trainable = [p for p in model.parameters() if p.requires_grad]
        if trainable:
            probe_optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    while lo <= hi:
        mid = (lo + hi) // 2
        outputs = None
        dummy = None
        try:
            dummy = {
                "input_ids": torch.ones(mid, max_seq_len, dtype=torch.long, device=device),
                "attention_mask": torch.ones(mid, max_seq_len, dtype=torch.long, device=device),
                "labels": torch.ones(mid, max_seq_len, dtype=torch.long, device=device),
            }
            if train_mode:
                outputs = model(**dummy)
                outputs.loss.backward()
                if probe_optimizer is not None:
                    probe_optimizer.step()
                    probe_optimizer.zero_grad(set_to_none=True)
                else:
                    model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    model(**dummy)
            best = mid
            lo = mid + 1
            logger.info(f"  bs={mid} OK (best={best})")
        except torch.cuda.OutOfMemoryError:
            if train_mode:
                if probe_optimizer is not None:
                    probe_optimizer.zero_grad(set_to_none=True)
                else:
                    model.zero_grad(set_to_none=True)
            hi = mid - 1
            logger.info(f"  bs={mid} OOM (shrink to {hi})")
        finally:
            del outputs, dummy
            gc.collect()
            torch.cuda.empty_cache()

    # Clean up probe optimizer
    del probe_optimizer
    gc.collect()
    torch.cuda.empty_cache()

    if was_training:
        model.train()
    else:
        model.eval()

    safe_bs = max(1, int(best * safety_margin))
    logger.info(f"Auto batch probe done: max_bs={best}, safe_bs={safe_bs} (margin={safety_margin})")
    return safe_bs


class QAFineTuneDataset(Dataset):
    """QA dataset for direct Qwen3 fine-tuning."""

    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_seq_len: int = 4096):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(data_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        prompt = PROMPT_TEMPLATE.format(context=rec["context"], question=rec["question"])
        answer = " " + rec["answer"]

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            answer_ids = answer_ids + [eos_id]

        max_prompt_len = self.max_seq_len - len(answer_ids) - 1
        if max_prompt_len < 10:
            answer_ids = answer_ids[:self.max_seq_len // 4]
            max_prompt_len = self.max_seq_len - len(answer_ids) - 1
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
    """Pad sequences and labels to max length in batch."""
    max_len = max(len(item["input_ids"]) for item in batch)
    B = len(batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask[i, :seq_len] = 1
        labels[i, :seq_len] = torch.tensor(item["labels"], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def evaluate(model, dataloader, accelerator):
    """Evaluate loss and PPL on dev set (distributed-aware)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in dataloader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        valid_mask = batch["labels"] != -100
        n_tokens = valid_mask.sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    loss_tensor = torch.tensor([total_loss], device=accelerator.device)
    tokens_tensor = torch.tensor([total_tokens], device=accelerator.device)
    loss_tensor = accelerator.reduce(loss_tensor, reduction="sum")
    tokens_tensor = accelerator.reduce(tokens_tensor, reduction="sum")

    model.train()
    avg_loss = loss_tensor.item() / max(tokens_tensor.item(), 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Upper Bound: LoRA Fine-tune Qwen3")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override qwen3_model_path from config")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Per-GPU batch size (0 = auto-detect)")
    parser.add_argument("--target_ebs", type=int, default=256,
                        help="Target effective batch size (used when batch_size=0)")
    parser.add_argument("--grad_accum", type=int, default=0,
                        help="Gradient accumulation steps (0 = auto from target_ebs)")
    parser.add_argument("--max_seq_len", type=int, default=4096,
                        help="Max sequence length (context + question + answer)")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./outputs/benchmark/stage2_finetuned")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )

    torch.manual_seed(args.seed)
    config = QCPCConfig.load(args.config)
    if args.model_path:
        config.qwen3_model_path = args.model_path

    train_path = config.sft_train_data_path
    dev_path = config.sft_eval_data_path

    # Load model & tokenizer
    logger.info(f"Loading Qwen3 from {config.qwen3_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.qwen3_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.qwen3_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA trainable params: {trainable_params:,} / {total_params:,} "
                f"({trainable_params/total_params*100:.2f}%)")

    # --- Create datasets early so we can scan actual max sequence length ---
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = QAFineTuneDataset(train_path, tokenizer, args.max_seq_len)
    dev_dataset = QAFineTuneDataset(dev_path, tokenizer, args.max_seq_len)

    # Scan dataset for actual max sequence length (rank 0 only, others get broadcast)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        actual_max_len = 0
        for i in range(len(train_dataset)):
            actual_max_len = max(actual_max_len, len(train_dataset[i]["input_ids"]))
        # Round up to multiple of 64 for memory-aligned probe
        probe_seq_len = min(args.max_seq_len, ((actual_max_len + 63) // 64) * 64)
        logger.info(f"Dataset actual max_len={actual_max_len}, "
                    f"probe_seq_len={probe_seq_len} (config max={args.max_seq_len})")
    else:
        probe_seq_len = args.max_seq_len  # will be overridden by broadcast

    # --- Auto batch size probe (rank 0 only, before Accelerator) ---
    per_gpu_bs = args.batch_size

    if per_gpu_bs <= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        if rank == 0:
            model.to(device)
            per_gpu_bs = _probe_max_batch_size(
                model, probe_seq_len, device, train_mode=True,
            )
            model.to("cpu")
            torch.cuda.empty_cache()
            bs_tensor = torch.tensor([per_gpu_bs], dtype=torch.long, device=device)
        else:
            bs_tensor = torch.tensor([0], dtype=torch.long, device=device)

        if world_size > 1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
            torch.distributed.broadcast(bs_tensor, src=0)
        per_gpu_bs = bs_tensor.item()
    elif per_gpu_bs <= 0:
        per_gpu_bs = 4  # CPU fallback

    # Compute grad_accum
    if args.grad_accum > 0:
        accum_steps = args.grad_accum
    else:
        per_step_total = per_gpu_bs * world_size
        accum_steps = max(1, math.ceil(args.target_ebs / per_step_total))

    actual_ebs = per_gpu_bs * world_size * accum_steps
    logger.info(f"Batch config: per_gpu_bs={per_gpu_bs}, accum={accum_steps}, "
                f"world_size={world_size}, effective_bs={actual_ebs}")

    # Create Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=accum_steps,
        mixed_precision="no",  # model already bf16; autocast would promote cross_entropy to fp32
    )

    if accelerator.is_main_process:
        logger.info(f"Train data: {train_path}")
        logger.info(f"Dev data:   {dev_path}")
        logger.info(f"Devices: {accelerator.num_processes}")
        logger.info(f"Train samples: {len(train_dataset)}, Dev samples: {len(dev_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=per_gpu_bs, shuffle=True,
        collate_fn=collate_fn, num_workers=4, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=per_gpu_bs, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
    )

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Prepare with Accelerator
    model, optimizer, train_loader, dev_loader = accelerator.prepare(
        model, optimizer, train_loader, dev_loader,
    )

    # Scheduler
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=max(warmup_steps, 1))
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=args.lr * 0.01,
    )
    scheduler = SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps],
    )
    if accelerator.is_main_process:
        logger.info(f"Total optimizer steps: {total_steps}, warmup: {warmup_steps}")

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    global_step = 0
    best_eval_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_micro_batches = 0
        pbar = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                accelerator.backward(outputs.loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.grad_clip,
                    )

                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += outputs.loss.item()
            num_micro_batches += 1

            if accelerator.sync_gradients:
                scheduler.step()
                global_step += 1
                pbar.update(1)

                if global_step % args.log_interval == 0 and accelerator.is_main_process:
                    avg_loss = epoch_loss / num_micro_batches
                    train_ppl = math.exp(min(avg_loss, 20))
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Epoch {epoch+1} Step {global_step}/{total_steps} | "
                        f"loss={outputs.loss.item():.4f} avg={avg_loss:.4f} ppl={train_ppl:.2f} lr={lr:.2e}"
                    )

                if global_step % args.eval_interval == 0:
                    eval_loss, eval_ppl = evaluate(model, dev_loader, accelerator)
                    if accelerator.is_main_process:
                        logger.info(f"  [Eval] Step {global_step} | loss={eval_loss:.4f} ppl={eval_ppl:.2f}")
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            unwrapped = accelerator.unwrap_model(model)
                            unwrapped.save_pretrained(output_dir / "best_lora")
                            tokenizer.save_pretrained(output_dir / "best_lora")
                            logger.info(f"  Saved best model (loss={eval_loss:.4f})")

        pbar.close()

        # End of epoch eval
        eval_loss, eval_ppl = evaluate(model, dev_loader, accelerator)
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1} done | eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}")
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(output_dir / "best_lora")
                tokenizer.save_pretrained(output_dir / "best_lora")

    # Save final
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir / "final_lora")
        tokenizer.save_pretrained(output_dir / "final_lora")

        info = {
            "benchmark": "stage2_finetune",
            "model": config.qwen3_model_path,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": per_gpu_bs,
            "grad_accum": accum_steps,
            "num_gpus": accelerator.num_processes,
            "effective_batch_size": actual_ebs,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "best_eval_loss": best_eval_loss,
            "best_eval_ppl": math.exp(min(best_eval_loss, 20)),
            "train_samples": len(train_dataset),
            "dev_samples": len(dev_dataset),
        }
        with open(output_dir / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print("\n" + "=" * 60)
        print("Stage 2 Fine-tuning Complete")
        print("=" * 60)
        print(f"  Best eval loss: {best_eval_loss:.4f}")
        print(f"  Best eval PPL:  {math.exp(min(best_eval_loss, 20)):.2f}")
        print(f"  LoRA adapter:   {output_dir / 'best_lora'}")
        print("=" * 60)


if __name__ == "__main__":
    main()
