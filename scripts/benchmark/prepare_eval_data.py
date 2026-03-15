"""Prepare evaluation-only dataset and SFT train/dev splits from raw JSONL data.

Creates three splits in the format expected by QADataset:
  - data/processed/sft/train.json   (for QCPC stage 2 training & Qwen baseline finetuning)
  - data/processed/sft/dev.json     (for validation during training)
  - data/processed/sft/eval.json    (held-out, evaluation only — never used in training)

Each record: {"context": str, "question": str, "answer": str}

Datasets used:
  - SQuAD v2 (train + validation)
  - HotpotQA (train + validation)

Usage:
    python scripts/benchmark/prepare_eval_data.py [--data_root ./data] [--output_root ./data/processed/sft]
    python scripts/benchmark/prepare_eval_data.py --eval_ratio 0.1 --dev_ratio 0.05
"""

import argparse
import json
import random
from pathlib import Path


def parse_squad(path: Path) -> list[dict]:
    """Parse SQuAD v2 JSONL: {context, question, answers: {text: [...]}} -> standardized format."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            answers = d.get("answers", {})
            answer_texts = answers.get("text", [])
            if not answer_texts:
                continue  # skip unanswerable
            records.append({
                "context": d["context"],
                "question": d["question"],
                "answer": answer_texts[0],
                "source": "squad_v2",
            })
    return records


def parse_hotpotqa(path: Path) -> list[dict]:
    """Parse HotpotQA JSONL: {context: {title, sentences}, question, answer} -> standardized format."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            # Flatten context: {title: [...], sentences: [[s1, s2, ...], ...]}
            ctx_obj = d["context"]
            if isinstance(ctx_obj, str):
                import ast
                ctx_obj = ast.literal_eval(ctx_obj)

            paragraphs = []
            titles = ctx_obj.get("title", [])
            sents_list = ctx_obj.get("sentences", [])
            for i, sents in enumerate(sents_list):
                title = titles[i] if i < len(titles) else ""
                text = "".join(s.strip() if s.startswith(" ") else s for s in sents)
                if title:
                    paragraphs.append(f"{title}: {text.strip()}")
                else:
                    paragraphs.append(text.strip())
            context = "\n\n".join(paragraphs)

            answer = d.get("answer", "")
            if not answer:
                continue

            records.append({
                "context": context,
                "question": d["question"],
                "answer": answer,
                "source": "hotpotqa",
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT eval data splits")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for raw data")
    parser.add_argument("--output_root", type=str, default="./data/processed/sft",
                        help="Output directory for processed splits")
    parser.add_argument("--eval_ratio", type=float, default=0.10,
                        help="Fraction of total data held out for evaluation only")
    parser.add_argument("--dev_ratio", type=float, default=0.05,
                        help="Fraction of total data for dev/validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    # --- Collect all records ---
    all_records = []

    # SQuAD v2
    for split_file in ["squad_v2_train.jsonl", "squad_v2_validation.jsonl"]:
        p = data_root / "sft" / split_file
        if p.exists():
            recs = parse_squad(p)
            print(f"  {split_file}: {len(recs)} records")
            all_records.extend(recs)

    # HotpotQA
    for split_file in ["hotpotqa_train.jsonl", "hotpotqa_validation.jsonl"]:
        p = data_root / "sft" / split_file
        if p.exists():
            recs = parse_hotpotqa(p)
            print(f"  {split_file}: {len(recs)} records")
            all_records.extend(recs)

    print(f"\nTotal records collected: {len(all_records)}")

    # --- Shuffle & split ---
    random.shuffle(all_records)

    n = len(all_records)
    n_eval = int(n * args.eval_ratio)
    n_dev = int(n * args.dev_ratio)
    n_train = n - n_eval - n_dev

    eval_set = all_records[:n_eval]
    dev_set = all_records[n_eval:n_eval + n_dev]
    train_set = all_records[n_eval + n_dev:]

    # Remove 'source' field for final output (QADataset doesn't expect it)
    def strip_source(records):
        return [{"context": r["context"], "question": r["question"], "answer": r["answer"]}
                for r in records]

    train_out = strip_source(train_set)
    dev_out = strip_source(dev_set)
    eval_out = strip_source(eval_set)

    # --- Write ---
    for name, data in [("train.json", train_out), ("dev.json", dev_out), ("eval.json", eval_out)]:
        path = output_root / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {path}: {len(data)} records")

    # --- Summary ---
    print(f"\n=== Split Summary ===")
    print(f"  Train: {len(train_out)} ({len(train_out)/n*100:.1f}%)")
    print(f"  Dev:   {len(dev_out)} ({len(dev_out)/n*100:.1f}%)")
    print(f"  Eval:  {len(eval_out)} ({len(eval_out)/n*100:.1f}%) — held out, never used in training")

    # Also save a pretrain eval split (last N samples from fineweb for stage 1 eval)
    pretrain_eval_path = output_root.parent / "pretrain_eval.jsonl"
    pretrain_src = data_root / "pretrain" / "fineweb_sampled.jsonl"
    if pretrain_src.exists():
        # Take the last 500 samples as eval set
        lines = []
        with open(pretrain_src, "r", encoding="utf-8") as f:
            for line in f:
                lines.append(line)
        eval_lines = lines[-500:]
        with open(pretrain_eval_path, "w", encoding="utf-8") as f:
            f.writelines(eval_lines)
        print(f"\nWrote {pretrain_eval_path}: {len(eval_lines)} pretrain eval samples (last 500)")


if __name__ == "__main__":
    main()
