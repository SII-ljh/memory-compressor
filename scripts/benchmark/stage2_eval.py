"""Stage 2 Evaluation: Upper Bound (with context) & Lower Bound (question only).

Evaluates the LoRA-finetuned Qwen3 model in two modes:
  - Upper Bound: Input = context + question → answer
    (best possible QA performance, no compression)
  - Lower Bound: Input = question only → answer
    (no context at all, pure parametric knowledge)

QCPC Stage 2 performance should fall between these two bounds.

Metrics computed:
  - NTP Loss & PPL on answer tokens
  - Generation-based: F1, Exact Match (EM), ROUGE-L

Usage:
    # Evaluate both upper and lower bounds
    python scripts/benchmark/stage2_eval.py --config config/default.yaml \
        --lora_path ./outputs/benchmark/stage2_finetuned/best_lora

    # Evaluate only upper bound
    python scripts/benchmark/stage2_eval.py --config config/default.yaml \
        --lora_path ./outputs/benchmark/stage2_finetuned/best_lora --mode upper

    # Use custom eval data
    python scripts/benchmark/stage2_eval.py --config config/default.yaml \
        --lora_path ./outputs/benchmark/stage2_finetuned/best_lora \
        --eval_data ./data/processed/sft/eval.json
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
    """Lower text, remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Remove extra whitespace
    s = " ".join(s.split())
    return s


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
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
    """Exact match (after normalization)."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L (longest common subsequence based)."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    # LCS length
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

class EvalDataset(Dataset):
    """Evaluation dataset supporting both with-context and without-context modes."""

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        with_context: bool = True,
        max_seq_len: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.with_context = with_context
        self.max_seq_len = max_seq_len

        with open(data_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        if self.with_context:
            prompt = PROMPT_WITH_CONTEXT.format(
                context=rec["context"], question=rec["question"],
            )
        else:
            prompt = PROMPT_NO_CONTEXT.format(question=rec["question"])

        answer = " " + rec["answer"]

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            answer_ids_with_eos = answer_ids + [eos_id]
        else:
            answer_ids_with_eos = answer_ids

        # Truncate prompt if needed
        max_prompt_len = self.max_seq_len - len(answer_ids_with_eos) - 1
        if max_prompt_len < 10:
            answer_ids_with_eos = answer_ids_with_eos[:self.max_seq_len // 4]
            max_prompt_len = self.max_seq_len - len(answer_ids_with_eos) - 1
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]

        input_ids = prompt_ids + answer_ids_with_eos
        labels = [-100] * len(prompt_ids) + answer_ids_with_eos

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_ids": prompt_ids,
            "answer_text": rec["answer"],
        }


def collate_loss(batch):
    """Collate for loss computation."""
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

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ─── Evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loss(model, dataloader, device):
    """Compute average loss and PPL on answer tokens."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Computing loss"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        valid_mask = labels != -100
        n_tokens = valid_mask.sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


@torch.no_grad()
def evaluate_generation(model, dataset, tokenizer, device, max_new_tokens=256, batch_size=1):
    """Generate answers and compute F1, EM, ROUGE-L.

    Uses greedy decoding (temperature=0).
    """
    model.eval()
    all_f1 = []
    all_em = []
    all_rouge = []
    predictions = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating"):
        batch_items = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]

        # Use prompt_ids for generation (without answer tokens)
        for item in batch_items:
            prompt_ids = item["prompt_ids"]
            gold_answer = item["answer_text"]

            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode only newly generated tokens
            new_tokens = generated[0, input_ids.shape[1]:]
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            f1 = compute_f1(pred_text, gold_answer)
            em = compute_em(pred_text, gold_answer)
            rouge = compute_rouge_l(pred_text, gold_answer)

            all_f1.append(f1)
            all_em.append(em)
            all_rouge.append(rouge)
            predictions.append({
                "question": dataset.records[i + len(predictions) - len(all_f1) + len(all_f1)]["question"]
                if hasattr(dataset, "records") else "",
                "gold": gold_answer,
                "pred": pred_text,
                "f1": f1,
                "em": em,
                "rouge_l": rouge,
            })

    return {
        "f1": sum(all_f1) / max(len(all_f1), 1),
        "em": sum(all_em) / max(len(all_em), 1),
        "rouge_l": sum(all_rouge) / max(len(all_rouge), 1),
        "num_samples": len(all_f1),
        "predictions": predictions,
    }


def run_evaluation(
    model,
    tokenizer,
    eval_data_path: str,
    device: torch.device,
    with_context: bool,
    max_seq_len: int,
    batch_size: int,
    max_new_tokens: int,
    max_samples: int | None = None,
) -> dict:
    """Run full evaluation (loss + generation) for one mode."""
    mode_name = "upper_bound (with context)" if with_context else "lower_bound (question only)"
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {mode_name}")
    logger.info(f"{'='*60}")

    dataset = EvalDataset(
        eval_data_path, tokenizer, with_context=with_context, max_seq_len=max_seq_len,
    )
    if max_samples and len(dataset) > max_samples:
        dataset = torch.utils.data.Subset(dataset, range(max_samples))
        # Keep reference to original dataset for generation
        orig_records = dataset.dataset.records[:max_samples]
    else:
        orig_records = dataset.records if hasattr(dataset, "records") else None

    # Loss evaluation
    loss_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_loss, num_workers=0,
    )
    avg_loss, ppl = evaluate_loss(model, loss_loader, device)
    logger.info(f"  Loss: {avg_loss:.4f}, PPL: {ppl:.2f}")

    # Generation evaluation
    gen_dataset = EvalDataset(
        eval_data_path, tokenizer, with_context=with_context, max_seq_len=max_seq_len,
    )
    if max_samples and len(gen_dataset) > max_samples:
        gen_dataset = torch.utils.data.Subset(gen_dataset, range(max_samples))

    gen_results = evaluate_generation(
        model, gen_dataset, tokenizer, device,
        max_new_tokens=max_new_tokens, batch_size=1,
    )
    logger.info(f"  F1: {gen_results['f1']:.4f}, EM: {gen_results['em']:.4f}, "
                f"ROUGE-L: {gen_results['rouge_l']:.4f}")

    return {
        "mode": "with_context" if with_context else "question_only",
        "loss": avg_loss,
        "ppl": ppl,
        "f1": gen_results["f1"],
        "em": gen_results["em"],
        "rouge_l": gen_results["rouge_l"],
        "num_samples": gen_results["num_samples"],
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Eval: Upper & Lower Bounds")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to saved LoRA adapter (from stage2_finetune.py)")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Path to eval JSON. Defaults to data/processed/sft/eval.json")
    parser.add_argument("--mode", choices=["both", "upper", "lower"], default="both",
                        help="Which bound(s) to evaluate")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max eval samples (None = all)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--output", type=str, default="./outputs/benchmark/stage2_eval_results.json")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )

    config = QCPCConfig.load(args.config)

    eval_path = args.eval_data or "./data/processed/sft/eval.json"
    if not Path(eval_path).exists():
        logger.error(f"Eval data not found: {eval_path}")
        logger.error("Run: python scripts/benchmark/prepare_eval_data.py")
        sys.exit(1)
    logger.info(f"Eval data: {eval_path}")

    # Load base model + LoRA adapter
    logger.info(f"Loading Qwen3 from {config.qwen3_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.qwen3_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.qwen3_model_path, trust_remote_code=True, torch_dtype=torch.float32,
    )

    logger.info(f"Loading LoRA adapter from {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model = model.merge_and_unload()  # merge for faster inference

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    logger.info(f"Device: {device}")

    # Run evaluations
    results = {"benchmark": "stage2_eval", "model": config.qwen3_model_path, "lora_path": args.lora_path}

    if args.mode in ("both", "upper"):
        upper = run_evaluation(
            model, tokenizer, eval_path, device,
            with_context=True, max_seq_len=args.max_seq_len,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
        )
        results["upper_bound"] = upper

    if args.mode in ("both", "lower"):
        lower = run_evaluation(
            model, tokenizer, eval_path, device,
            with_context=False, max_seq_len=args.max_seq_len,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
        )
        results["lower_bound"] = lower

    # Print summary
    print("\n" + "=" * 70)
    print("Stage 2 Evaluation Results")
    print("=" * 70)
    print(f"{'Metric':<15} {'Upper (w/ ctx)':<20} {'Lower (q only)':<20}")
    print("-" * 70)

    if "upper_bound" in results and "lower_bound" in results:
        u, l = results["upper_bound"], results["lower_bound"]
        for metric in ["loss", "ppl", "f1", "em", "rouge_l"]:
            fmt = ".4f" if metric not in ("ppl",) else ".2f"
            print(f"{metric:<15} {u[metric]:<20{fmt}} {l[metric]:<20{fmt}}")
    elif "upper_bound" in results:
        u = results["upper_bound"]
        for metric in ["loss", "ppl", "f1", "em", "rouge_l"]:
            fmt = ".4f" if metric not in ("ppl",) else ".2f"
            print(f"{metric:<15} {u[metric]:<20{fmt}} {'N/A':<20}")
    elif "lower_bound" in results:
        l = results["lower_bound"]
        for metric in ["loss", "ppl", "f1", "em", "rouge_l"]:
            fmt = ".4f" if metric not in ("ppl",) else ".2f"
            print(f"{metric:<15} {'N/A':<20} {l[metric]:<20{fmt}}")

    print("=" * 70)
    print("\nInterpretation: QCPC Stage 2 metrics should fall between these bounds.")
    print("  Upper bound = model sees full context (no compression)")
    print("  Lower bound = model sees only the question (no context)\n")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
