"""Stage 1 Upper Bound: Qwen3 direct NTP on long text (no compression).

Measures the best possible next-token prediction performance by feeding
the full context directly to the Qwen3 model (no Perceiver compression).

This establishes the ceiling for QCPC Stage 1:
  QCPC_loss >= upper_bound_loss  (more compression → higher loss)

Supports three substage conditions matching actual training:
  --substage 1a     → context=512, cont=128   (matches Stage 1a short-window warmup)
  --substage 1b     → context=K*512, cont=128 (matches Stage 1b multi-chunk)
  --substage legacy → context=4096, cont=256  (original single-stage setting)

Input: pretrain eval split (data/stage1/eval.jsonl).
Method: Feed [context + continuation] as one sequence to Qwen3, compute NTP loss
        on the continuation tokens only (same target region as QCPC).

Usage:
    # Stage 1a upper bound (short window: 512→128)
    python scripts/benchmark/stage1_upper_bound.py --config config/default.yaml --substage 1a

    # Stage 1b upper bound (multi-chunk: K*512→128)
    python scripts/benchmark/stage1_upper_bound.py --config config/default.yaml --substage 1b

    # Legacy (4096→256)
    python scripts/benchmark/stage1_upper_bound.py --config config/default.yaml --substage legacy

    # Custom lengths
    python scripts/benchmark/stage1_upper_bound.py --substage 1a --context_len 1024 --cont_len 128
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import QCPCConfig

logger = logging.getLogger(__name__)


class DirectNTPDataset(Dataset):
    """Dataset for direct NTP evaluation (no compression).

    Same splitting logic as PretrainDataset: splits text into
    context prefix and continuation suffix, returns them as
    a single concatenated sequence for direct LM evaluation.
    """

    def __init__(self, data_path: str, tokenizer, max_context_len: int = 4096,
                 max_cont_len: int = 256):
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.max_cont_len = max_cont_len
        self.records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        text = self.records[idx]["text"]
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_max = self.max_context_len + self.max_cont_len
        if len(tokens) > total_max:
            tokens = tokens[:total_max]
        if len(tokens) < self.max_cont_len + 10:
            split_point = max(1, len(tokens) - self.max_cont_len)
        else:
            split_point = min(len(tokens) - self.max_cont_len, self.max_context_len)
        return {
            "input_ids": tokens[:split_point + self.max_cont_len],
            "split_point": split_point,
        }


def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    B = len(batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    split_points = torch.zeros(B, dtype=torch.long)
    for i, item in enumerate(batch):
        seq = item["input_ids"]
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :len(seq)] = 1
        split_points[i] = item["split_point"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "split_points": split_points}


@torch.no_grad()
def evaluate_direct_ntp(model, dataloader, device):
    """Evaluate Qwen3 direct NTP, loss on continuation tokens only.

    Forward WITHOUT labels to avoid HF's internal full-vocab cross_entropy
    which OOMs at large batch sizes. Instead, extract logits only at
    continuation positions and compute loss on that smaller tensor.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Stage 1 Upper Bound"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        split_points = batch["split_points"]
        B, T = input_ids.shape

        # Build labels: -100 for context/padding, real token ids for continuation
        labels = torch.full_like(input_ids, -100)
        for i in range(B):
            sp = split_points[i].item()
            seq_len = int(attention_mask[i].sum().item())
            if seq_len > sp:
                labels[i, sp:seq_len] = input_ids[i, sp:seq_len]

        # Forward WITHOUT labels — avoids the huge internal cross_entropy
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits  # (B, T, V) in bfloat16

        # Causal LM shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # Select only valid (continuation) token positions
        valid_mask = shift_labels != -100
        n_tokens = valid_mask.sum().item()
        if n_tokens == 0:
            del logits, outputs
            continue

        valid_logits = shift_logits[valid_mask]   # (N_valid, V)
        valid_labels = shift_labels[valid_mask]    # (N_valid,)

        # Free large tensors before the float32 cross_entropy
        del logits, shift_logits, outputs

        loss = nn.functional.cross_entropy(
            valid_logits, valid_labels, reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += n_tokens

        del valid_logits, valid_labels, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return {"loss": avg_loss, "ppl": ppl, "total_tokens": int(total_tokens)}


def _probe_max_batch_size(model, max_seq_len, device, upper_bound=256):
    """Binary search for max batch size that fits in GPU memory.

    Probes forward pass WITHOUT labels (matching evaluate_direct_ntp),
    then applies 0.85x safety margin for the cross_entropy computation.
    """
    lo, hi = 1, upper_bound
    best = 1
    logger.info(f"Auto batch probe: searching [{lo}, {hi}] on {device} (seq_len={max_seq_len})")
    model.eval()
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            dummy_ids = torch.ones(mid, max_seq_len, dtype=torch.long, device=device)
            dummy_mask = torch.ones(mid, max_seq_len, dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_ids=dummy_ids, attention_mask=dummy_mask, use_cache=False)
            best = mid
            lo = mid + 1
            logger.info(f"  bs={mid} OK (best={best})")
        except torch.cuda.OutOfMemoryError:
            hi = mid - 1
            logger.info(f"  bs={mid} OOM (shrink to {hi})")
        finally:
            torch.cuda.empty_cache()
    # Safety margin for cross_entropy on valid tokens
    safe = max(1, int(best * 0.85))
    logger.info(f"Auto batch probe done: max_bs={best}, safe={safe}")
    return safe


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Upper Bound: Qwen3 Direct NTP")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override qwen3_model_path from config")
    parser.add_argument("--substage", type=str, choices=["1a", "1b", "legacy"], default="legacy",
                        help="Which stage conditions to use: "
                             "1a (512/128), 1b (K*512/128), legacy (4096/256)")
    parser.add_argument("--context_len", type=int, default=None,
                        help="Override context length (default: from config based on substage)")
    parser.add_argument("--cont_len", type=int, default=None,
                        help="Override continuation length (default: from config based on substage)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Batch size (0 = auto-detect)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto from substage)")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    config = QCPCConfig.load(args.config)
    if args.model_path:
        config.qwen3_model_path = args.model_path

    # Resolve context/cont lengths based on substage
    if args.substage == "1a":
        context_len = args.context_len or config.stage1a_max_context_len
        cont_len = args.cont_len or config.stage1a_max_cont_len
    elif args.substage == "1b":
        # Stage 1b: K_max chunks of chunk_len as total context
        context_len = args.context_len or (config.stage1b_max_chunks * config.stage1b_chunk_len)
        cont_len = args.cont_len or config.stage1b_max_cont_len
    else:
        context_len = args.context_len or config.stage1_max_context_len
        cont_len = args.cont_len or config.stage1_max_cont_len

    # Default output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"./outputs/benchmark/stage1_upper_bound_{args.substage}.json")

    eval_path = config.pretrain_eval_data_path
    logger.info(f"Eval data: {eval_path}")
    logger.info(f"Substage: {args.substage}, context_len={context_len}, cont_len={cont_len}")

    # Load model
    logger.info(f"Loading Qwen3 from {config.qwen3_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.qwen3_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.qwen3_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Device: {device}")

    # Auto batch size
    batch_size = args.batch_size
    if batch_size <= 0 and device.type == "cuda":
        max_seq_len = context_len + cont_len
        batch_size = _probe_max_batch_size(model, max_seq_len, device)
    elif batch_size <= 0:
        batch_size = 4
    logger.info(f"Using batch_size={batch_size}")

    dataset = DirectNTPDataset(eval_path, tokenizer, context_len, cont_len)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))
    logger.info(f"Eval samples: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    results = evaluate_direct_ntp(model, dataloader, device)

    substage_label = {"1a": "Stage 1a (512→128)", "1b": "Stage 1b (K*512→128)", "legacy": "Legacy (4096→256)"}
    print(f"\n{'='*60}")
    print(f"Stage 1 Upper Bound: Qwen3 Direct NTP (no compression)")
    print(f"  Condition: {substage_label.get(args.substage, args.substage)}")
    print(f"{'='*60}")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  PPL:  {results['ppl']:.2f}")
    print(f"  Tokens evaluated: {results['total_tokens']}")
    print(f"  Context len: {context_len}, Cont len: {cont_len}")
    print(f"{'='*60}")
    print(f"QCPC Stage 1 loss should be >= {results['loss']:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "benchmark": "stage1_upper_bound",
        "substage": args.substage,
        "model": config.qwen3_model_path,
        "context_len": context_len,
        "continuation_len": cont_len,
        "num_samples": len(dataset),
        **results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
