"""Stage 2 Upper Bound: LoRA fine-tune Qwen3 on SFT QA data.

Fine-tunes Qwen3 with full context + question → answer using LoRA.
This establishes the ceiling for QCPC Stage 2.

Supports both single-GPU and multi-GPU:
    # Single GPU:
    python scripts/benchmark/stage2_finetune.py \
      --config config/default.yaml --model_path ./models/Qwen3-0.6B

    # Multi-GPU:
    torchrun --nproc_per_node=8 scripts/benchmark/stage2_finetune.py \
      --config config/default.yaml --model_path ./models/Qwen3-0.6B

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
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    print("Error: peft not installed. Run: pip install peft")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import QCPCConfig

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"


# ---------------------------------------------------------------------------
# Dataset & collate
# ---------------------------------------------------------------------------
class QAFineTuneDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 4096):
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
            answer_ids = answer_ids[: self.max_seq_len // 4]
            max_prompt_len = self.max_seq_len - len(answer_ids) - 1
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
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


# ---------------------------------------------------------------------------
# Auto batch probe — uses the real optimizer so memory matches training
# ---------------------------------------------------------------------------
def _probe_max_batch_size(model, optimizer, max_seq_len, device,
                          upper_bound=128, safety_margin=0.85):
    """Binary search for max batch size (forward + backward + optimizer.step)."""
    lo, hi = 1, upper_bound
    best = 1
    model.train()
    logger.info(f"Auto batch probe: [{lo}, {hi}], seq_len={max_seq_len}, device={device}")

    while lo <= hi:
        mid = (lo + hi) // 2
        dummy = None
        try:
            dummy = {
                "input_ids": torch.ones(mid, max_seq_len, dtype=torch.long, device=device),
                "attention_mask": torch.ones(mid, max_seq_len, dtype=torch.long, device=device),
                "labels": torch.ones(mid, max_seq_len, dtype=torch.long, device=device),
            }
            outputs = model(**dummy)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            best = mid
            lo = mid + 1
            logger.info(f"  bs={mid} OK (best={best})")
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            hi = mid - 1
            logger.info(f"  bs={mid} OOM (shrink to {hi})")
        finally:
            del dummy
            gc.collect()
            torch.cuda.empty_cache()

    safe_bs = max(1, int(best * safety_margin))
    logger.info(f"Auto batch probe done: max={best}, safe={safe_bs} (margin={safety_margin})")
    return safe_bs



# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def _setup_distributed():
    """Init distributed if launched via torchrun. Returns (rank, local_rank, world_size)."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def _cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        # unwrap DDP for cleaner forward
        m = model.module if isinstance(model, DDP) else model
        outputs = m(**batch)
        n_tokens = (batch["labels"] != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    # aggregate across ranks
    if dist.is_initialized():
        stats = torch.tensor([total_loss, total_tokens], device=device)
        dist.all_reduce(stats)
        total_loss, total_tokens = stats[0].item(), stats[1].item()

    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 2 Upper Bound: LoRA Fine-tune Qwen3")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=0, help="Per-GPU batch size (0 = auto)")
    parser.add_argument("--grad_accum", type=int, default=0, help="0 = auto (target 256 EBS)")
    parser.add_argument("--max_seq_len", type=int, default=4096)
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

    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    torch.manual_seed(args.seed)

    rank, local_rank, world_size = _setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = _is_main()

    config = QCPCConfig.load(args.config)
    if args.model_path:
        config.qwen3_model_path = args.model_path

    # ---- Model ----
    if is_main:
        logger.info(f"Loading Qwen3 from {config.qwen3_model_path} (world_size={world_size})")

    tokenizer = AutoTokenizer.from_pretrained(config.qwen3_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.qwen3_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info(f"LoRA: {trainable_params:,} / {total_params:,} "
                    f"({trainable_params / total_params * 100:.2f}%)")

    model.to(device)

    # ---- Optimizer (before probe, so probe includes optimizer states) ----
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---- Auto batch size probe (rank 0, before dataset/DDP) ----
    per_gpu_bs = args.batch_size
    if per_gpu_bs <= 0 and torch.cuda.is_available():
        if is_main:
            per_gpu_bs = _probe_max_batch_size(
                model, optimizer, args.max_seq_len, device)
            bs_tensor = torch.tensor([per_gpu_bs], device=device)
        else:
            bs_tensor = torch.tensor([0], device=device)

        if dist.is_initialized():
            dist.broadcast(bs_tensor, src=0)
        per_gpu_bs = bs_tensor.item()
    elif per_gpu_bs <= 0:
        per_gpu_bs = 4

    # ---- Dataset (after probe, so user sees probe result fast) ----
    train_dataset = QAFineTuneDataset(config.sft_train_data_path, tokenizer, args.max_seq_len)
    dev_dataset = QAFineTuneDataset(config.sft_eval_data_path, tokenizer, args.max_seq_len)
    if is_main:
        logger.info(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")

    # ---- Grad accum ----
    if args.grad_accum > 0:
        accum_steps = args.grad_accum
    else:
        accum_steps = max(1, math.ceil(256 / (per_gpu_bs * world_size)))

    actual_ebs = per_gpu_bs * world_size * accum_steps
    if is_main:
        logger.info(f"Batch: bs={per_gpu_bs}, accum={accum_steps}, "
                    f"world_size={world_size}, effective={actual_ebs}")

    # ---- Wrap DDP (after probe, so probe runs on raw model) ----
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ---- DataLoaders ----
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist.is_initialized() else None
    dev_sampler = DistributedSampler(dev_dataset, shuffle=False) if dist.is_initialized() else None

    train_loader = DataLoader(
        train_dataset, batch_size=per_gpu_bs,
        shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=collate_fn, num_workers=4, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=per_gpu_bs,
        shuffle=False, sampler=dev_sampler,
        collate_fn=collate_fn, num_workers=4,
    )

    # ---- Scheduler ----
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.01, total_iters=max(warmup_steps, 1)),
        CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1),
                          eta_min=args.lr * 0.01),
    ], milestones=[warmup_steps])
    if is_main:
        logger.info(f"Steps: {total_steps}, warmup: {warmup_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training ----
    model.train()
    global_step = 0
    best_eval_loss = float("inf")

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_micro = 0
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{args.epochs}",
                    disable=not is_main)

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / accum_steps
            loss.backward()

            epoch_loss += outputs.loss.item()
            num_micro += 1

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                pbar.update(1)

                if global_step % args.log_interval == 0 and is_main:
                    avg = epoch_loss / num_micro
                    logger.info(
                        f"Ep {epoch+1} Step {global_step}/{total_steps} | "
                        f"loss={outputs.loss.item():.4f} avg={avg:.4f} "
                        f"ppl={math.exp(min(avg, 20)):.2f} lr={optimizer.param_groups[0]['lr']:.2e}"
                    )

                if global_step % args.eval_interval == 0:
                    eval_loss, eval_ppl = evaluate(model, dev_loader, device)
                    if is_main:
                        logger.info(f"  [Eval] step {global_step} | "
                                    f"loss={eval_loss:.4f} ppl={eval_ppl:.2f}")
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            _save_lora(model, tokenizer, output_dir / "best_lora")
                            logger.info(f"  Saved best (loss={eval_loss:.4f})")

        pbar.close()

        # End-of-epoch eval
        eval_loss, eval_ppl = evaluate(model, dev_loader, device)
        if is_main:
            logger.info(f"Epoch {epoch+1} done | eval_loss={eval_loss:.4f} ppl={eval_ppl:.2f}")
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                _save_lora(model, tokenizer, output_dir / "best_lora")

    # ---- Save final ----
    if is_main:
        _save_lora(model, tokenizer, output_dir / "final_lora")

        info = {
            "benchmark": "stage2_finetune",
            "model": config.qwen3_model_path,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": per_gpu_bs,
            "grad_accum": accum_steps,
            "num_gpus": world_size,
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

        print(f"\n{'=' * 60}")
        print("Stage 2 Fine-tuning Complete")
        print(f"{'=' * 60}")
        print(f"  Best eval loss: {best_eval_loss:.4f}")
        print(f"  Best eval PPL:  {math.exp(min(best_eval_loss, 20)):.2f}")
        print(f"  LoRA adapter:   {output_dir / 'best_lora'}")
        print(f"{'=' * 60}")

    _cleanup_distributed()


def _save_lora(model, tokenizer, path):
    m = model.module if isinstance(model, DDP) else model
    m.save_pretrained(path)
    tokenizer.save_pretrained(path)


if __name__ == "__main__":
    main()
