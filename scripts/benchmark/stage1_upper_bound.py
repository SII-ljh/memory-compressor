"""Stage 1 Upper Bound: Qwen3 direct NTP on long text (no compression).

Measures the best possible next-token prediction performance by feeding
the full context directly to the Qwen3 model (no Perceiver compression).

This establishes the ceiling for QCPC Stage 1:
  QCPC_loss >= upper_bound_loss  (more compression → higher loss)

Input: pretrain eval split (data/stage1/eval.jsonl).
Method: Feed [context + continuation] as one sequence to Qwen3, compute NTP loss
        on the continuation tokens only (same target region as QCPC).

Usage:
    python scripts/benchmark/stage1_upper_bound.py --config config/default.yaml
    python scripts/benchmark/stage1_upper_bound.py --config config/default.yaml --batch_size 4
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
    """Evaluate Qwen3 direct NTP, loss on continuation tokens only."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Stage 1 Upper Bound"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        split_points = batch["split_points"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        B, T, V = logits.shape

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        per_token_loss = loss_fct(shift_logits.reshape(-1, V), shift_labels.reshape(-1)).view(B, T - 1)

        cont_mask = torch.zeros(B, T - 1, device=device)
        for i in range(B):
            sp = split_points[i].item()
            seq_len = attention_mask[i].sum().item()
            start = max(sp - 1, 0)
            end = int(seq_len) - 1
            if end > start:
                cont_mask[i, start:end] = 1.0

        total_loss += (per_token_loss * cont_mask).sum().item()
        total_tokens += cont_mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return {"loss": avg_loss, "ppl": ppl, "total_tokens": int(total_tokens)}


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Upper Bound: Qwen3 Direct NTP")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="./outputs/benchmark/stage1_upper_bound.json")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    config = QCPCConfig.load(args.config)

    eval_path = config.pretrain_eval_data_path
    logger.info(f"Eval data: {eval_path}")

    # Load model
    logger.info(f"Loading Qwen3 from {config.qwen3_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.qwen3_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.qwen3_model_path, trust_remote_code=True, torch_dtype=torch.float32,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Device: {device}")

    dataset = DirectNTPDataset(eval_path, tokenizer, config.stage1_max_context_len, config.stage1_max_cont_len)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))
    logger.info(f"Eval samples: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    results = evaluate_direct_ntp(model, dataloader, device)

    print(f"\n{'='*60}")
    print(f"Stage 1 Upper Bound: Qwen3 Direct NTP (no compression)")
    print(f"{'='*60}")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  PPL:  {results['ppl']:.2f}")
    print(f"  Tokens evaluated: {results['total_tokens']}")
    print(f"  Context len: {config.stage1_max_context_len}, Cont len: {config.stage1_max_cont_len}")
    print(f"{'='*60}")
    print(f"QCPC Stage 1 loss should be >= {results['loss']:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "benchmark": "stage1_upper_bound",
        "model": config.qwen3_model_path,
        "context_len": config.stage1_max_context_len,
        "continuation_len": config.stage1_max_cont_len,
        "num_samples": len(dataset),
        **results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
