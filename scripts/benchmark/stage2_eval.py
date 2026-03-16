"""Stage 2 Evaluation: Upper Bound (with context) & Lower Bound (question only).

Evaluates the LoRA-finetuned Qwen3 model on the dedicated SFT eval split:
  - Upper Bound: Input = context + question → answer (no compression)
  - Lower Bound: Input = question only → answer (no context)

QCPC Stage 2 performance should fall between these two bounds.

Uses: config.sft_eval_data_path (data/stage2/eval.json)
Format: [{"context": str, "question": str, "answer": str, "source": str}]

Metrics: Loss, PPL, F1, EM, ROUGE-L

Usage:
    python scripts/benchmark/stage2_eval.py --config config/default.yaml \
        --lora_path ./outputs/benchmark/stage2_finetuned/best_lora

    python scripts/benchmark/stage2_eval.py --config config/default.yaml \
        --lora_path ./outputs/benchmark/stage2_finetuned/best_lora --mode upper
"""

import argparse
import collections
import json
import logging
import math
import re
import string
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

try:
    from peft import PeftModel
except ImportError:
    print("Error: peft not installed. Run: pip install peft")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import QCPCConfig

logger = logging.getLogger(__name__)

PROMPT_WITH_CONTEXT = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
PROMPT_NO_CONTEXT = "Question: {question}\n\nAnswer:"


# ─── Text Metrics ───────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


# ─── Dataset ────────────────────────────────────────────────────────

class EvalLossDataset(Dataset):
    """Wraps eval records for NTP loss computation."""

    def __init__(self, records: list[dict], tokenizer, with_context: bool, max_seq_len: int):
        self.records = records
        self.tokenizer = tokenizer
        self.with_context = with_context
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        if self.with_context:
            prompt = PROMPT_WITH_CONTEXT.format(context=rec["context"], question=rec["question"])
        else:
            prompt = PROMPT_NO_CONTEXT.format(question=rec["question"])
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
        return {"input_ids": input_ids, "labels": labels, "prompt_ids": prompt_ids}


def collate_loss(batch):
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


# ─── Evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in tqdm(dataloader, desc="Computing loss"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        valid_mask = labels != -100
        n_tokens = valid_mask.sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


@torch.no_grad()
def evaluate_generation(model, records, tokenizer, device, with_context, max_seq_len, max_new_tokens):
    model.eval()
    all_f1, all_em, all_rouge = [], [], []

    for rec in tqdm(records, desc="Generating"):
        if with_context:
            prompt = PROMPT_WITH_CONTEXT.format(context=rec["context"], question=rec["question"])
        else:
            prompt = PROMPT_NO_CONTEXT.format(question=rec["question"])

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > max_seq_len:
            prompt_ids = prompt_ids[:max_seq_len]

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        pred_text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        gold = rec["answer"]

        all_f1.append(compute_f1(pred_text, gold))
        all_em.append(compute_em(pred_text, gold))
        all_rouge.append(compute_rouge_l(pred_text, gold))

    return {
        "f1": sum(all_f1) / max(len(all_f1), 1),
        "em": sum(all_em) / max(len(all_em), 1),
        "rouge_l": sum(all_rouge) / max(len(all_rouge), 1),
        "num_samples": len(all_f1),
    }


def run_one_mode(model, tokenizer, records, device, with_context, max_seq_len, batch_size, max_new_tokens):
    mode = "Upper Bound (with context)" if with_context else "Lower Bound (question only)"
    logger.info(f"\n--- {mode} ({len(records)} samples) ---")

    ds = EvalLossDataset(records, tokenizer, with_context, max_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_loss, num_workers=0)
    avg_loss, ppl = evaluate_loss(model, loader, device)

    gen = evaluate_generation(model, records, tokenizer, device, with_context, max_seq_len, max_new_tokens)
    logger.info(f"  loss={avg_loss:.4f} ppl={ppl:.2f} F1={gen['f1']:.4f} EM={gen['em']:.4f} ROUGE-L={gen['rouge_l']:.4f}")

    return {"loss": avg_loss, "ppl": ppl, **gen}


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Eval: Upper & Lower Bounds")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override qwen3_model_path from config")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--mode", choices=["both", "upper", "lower"], default="both")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--output", type=str, default="./outputs/benchmark/stage2_eval_results.json")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    config = QCPCConfig.load(args.config)
    if args.model_path:
        config.qwen3_model_path = args.model_path

    eval_path = config.sft_eval_data_path
    if not Path(eval_path).exists():
        logger.error(f"Eval data not found: {eval_path}")
        logger.error("Run: python scripts/data/data_processor.py --task sft")
        sys.exit(1)

    with open(eval_path, "r") as f:
        records = json.load(f)
    logger.info(f"Eval data: {eval_path} ({len(records)} samples)")

    # Load model + LoRA
    logger.info(f"Loading Qwen3 from {config.qwen3_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.qwen3_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.qwen3_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    logger.info(f"Loading LoRA from {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model = model.merge_and_unload()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    logger.info(f"Device: {device}")

    results = {"benchmark": "stage2_eval", "model": config.qwen3_model_path, "lora_path": args.lora_path}

    if args.mode in ("both", "upper"):
        results["upper_bound"] = run_one_mode(
            model, tokenizer, records, device, with_context=True,
            max_seq_len=args.max_seq_len, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        )
    if args.mode in ("both", "lower"):
        results["lower_bound"] = run_one_mode(
            model, tokenizer, records, device, with_context=False,
            max_seq_len=args.max_seq_len, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        )

    # Print summary
    print(f"\n{'='*65}")
    print(f"Stage 2 Evaluation Results (eval set: {len(records)} samples)")
    print(f"{'='*65}")
    print(f"{'Metric':<12} {'Upper (w/ ctx)':<20} {'Lower (q only)':<20}")
    print(f"{'-'*65}")
    if "upper_bound" in results and "lower_bound" in results:
        u, l = results["upper_bound"], results["lower_bound"]
        for m in ["loss", "ppl", "f1", "em", "rouge_l"]:
            fmt = ".2f" if m == "ppl" else ".4f"
            print(f"{m:<12} {u[m]:<20{fmt}} {l[m]:<20{fmt}}")
    elif "upper_bound" in results:
        u = results["upper_bound"]
        for m in ["loss", "ppl", "f1", "em", "rouge_l"]:
            fmt = ".2f" if m == "ppl" else ".4f"
            print(f"{m:<12} {u[m]:<20{fmt}} {'N/A':<20}")
    elif "lower_bound" in results:
        l = results["lower_bound"]
        for m in ["loss", "ppl", "f1", "em", "rouge_l"]:
            fmt = ".2f" if m == "ppl" else ".4f"
            print(f"{m:<12} {'N/A':<20} {l[m]:<20{fmt}}")
    print(f"{'='*65}")
    print(f"QCPC Stage 2 metrics should fall between these bounds.\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
