#!/usr/bin/env python3
"""
data_processor.py — 将 data_downloader.py 下载的原始 HF 数据集转换为统一格式

输入: data/download/ (data_downloader.py 的输出)

输出结构:
  data/
    stage1/             # Stage 1 预训练数据
      train.jsonl       # 预训练训练集 (去掉末尾 eval 部分)
      eval.jsonl        # 预训练验证集 (末尾 500 条, 也用于 Stage 1 上界评估)
    stage2/             # Stage 2 QA 微调数据
      train.json        # SFT 训练集
      eval.json         # SFT 评估集 (独立, 用于 Stage 2 上下界评估)
    benchmark/          # 发论文需要跑的 benchmark
      longbench/{cfg}.json
      ruler/{cfg}.json
      halueval.json
      truthfulqa.json

用法 (从项目根目录运行):
    python scripts/data/data_processor.py --task all
    python scripts/data/data_processor.py --task pretrain
    python scripts/data/data_processor.py --task sft
    python scripts/data/data_processor.py --task eval diagnostic
    python scripts/data/data_processor.py --input_dir ./data/download --output_dir ./data
    python scripts/data/data_processor.py --force
"""

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL 文件，返回 dict 列表."""
    if not path.exists():
        print(f"  [Warn] 文件不存在: {path}")
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def write_json(data: list[dict], path: Path):
    """写入 JSON 数组文件."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _normalize_text(text: str) -> str:
    """归一化文本: 小写 + NFKC + 合并空白."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# 通用清洗 (复用 prepare_large_qa_data.py 的 clean_data 逻辑)
# ---------------------------------------------------------------------------
def clean_data(samples: list[dict], label: str = "data") -> list[dict]:
    """
    清洗 QA 样本:
    - 删除缺失 context/question/answer 的样本
    - 删除 context < 50 chars 或 question < 2 chars
    - 删除 question == answer (归一化后)
    - 保留 "No answer" 类样本
    """
    reasons = Counter()
    cleaned = []

    for sample in samples:
        ctx = sample.get("context", "").strip()
        q = sample.get("question", "").strip()
        a = sample.get("answer", "").strip()

        # 更新 strip 后的值
        sample["context"] = ctx
        sample["question"] = q
        sample["answer"] = a

        if not ctx or not q or not a:
            reasons["missing_field"] += 1
            continue
        if len(ctx) < 50:
            reasons["short_context"] += 1
            continue
        if len(q) < 2:
            reasons["short_question"] += 1
            continue

        nq = _normalize_text(q)
        na = _normalize_text(a)
        if nq == na:
            reasons["question_eq_answer"] += 1
            continue

        cleaned.append(sample)

    removed = len(samples) - len(cleaned)
    if removed > 0:
        print(f"    [{label}] 清洗: {len(samples)} → {len(cleaned)} "
              f"(移除 {removed}: {dict(reasons)})")
    else:
        print(f"    [{label}] 清洗: {len(samples)} 条, 无移除")
    return cleaned


# ===========================================================================
# 转换器: 每个数据集 raw HF → standard QA
# ===========================================================================

def convert_faitheval(records: list[dict], source: str) -> list[dict]:
    """FaithEval: 自适应探测 context 字段名."""
    results = []
    for row in records:
        # 尝试多个可能的 context 字段名
        context = (row.get("context") or row.get("passage")
                   or row.get("document") or "")
        question = row.get("question", "")
        answer = row.get("answer", "")
        if context and question:
            results.append({
                "context": str(context),
                "question": str(question),
                "answer": str(answer) if answer else "No answer",
                "source": source,
            })
    return results


def convert_conflictqa(records: list[dict]) -> list[dict]:
    """ConflictQA: counter_memory → context, ground_truths[0] → answer."""
    results = []
    for row in records:
        context = row.get("counter_memory", "")
        question = row.get("question", "")
        ground_truths = row.get("ground_truths", [])
        answer = ground_truths[0] if ground_truths else ""
        if context and question:
            results.append({
                "context": str(context),
                "question": str(question),
                "answer": str(answer) if answer else "No answer",
                "source": "conflictqa",
            })
    return results


def convert_hotpotqa(records: list[dict], max_context_chars: int = 32000) -> list[dict]:
    """HotpotQA: context.sentences → 展平拼接."""
    results = []
    for row in records:
        answer = row.get("answer", "")
        if not answer:
            continue

        # 提取 context: sentences 是嵌套列表 [[sent1, sent2], [sent3, ...], ...]
        ctx_data = row.get("context", {})
        sentences = ctx_data.get("sentences", [])
        if sentences:
            parts = []
            for sent_group in sentences:
                if isinstance(sent_group, list):
                    parts.append(" ".join(str(s) for s in sent_group))
                else:
                    parts.append(str(sent_group))
            context = " ".join(parts)
        else:
            # fallback: 使用 titles
            titles = ctx_data.get("title", [])
            context = " ".join(str(t) for t in titles) if titles else ""

        if len(context) < 100:
            continue
        context = context[:max_context_chars]

        question = row.get("question", "")
        if question:
            results.append({
                "context": context,
                "question": str(question),
                "answer": str(answer),
                "source": "hotpotqa",
            })
    return results


def convert_squad_v2(records: list[dict]) -> list[dict]:
    """SQuAD v2: answers.text[0] → answer, 不可回答 → 'No answer'."""
    results = []
    for row in records:
        context = row.get("context", "")
        question = row.get("question", "")
        answers = row.get("answers", {})
        answer_texts = answers.get("text", [])
        answer = answer_texts[0] if answer_texts else "No answer"

        if context and question:
            results.append({
                "context": str(context),
                "question": str(question),
                "answer": str(answer),
                "source": "squad_v2",
            })
    return results


def convert_natural_questions(records: list[dict], max_context_chars: int = 32000) -> list[dict]:
    """NQ: document.tokens → 过滤 HTML 拼接, annotations.short_answers 首个非空."""
    results = []
    for row in records:
        # 提取文档文本
        doc = row.get("document", {})
        tokens_data = doc.get("tokens", {})

        if isinstance(tokens_data, dict):
            token_list = tokens_data.get("token", [])
            is_html = tokens_data.get("is_html", [])
            if token_list and is_html:
                context = " ".join(
                    t for t, h in zip(token_list, is_html) if not h
                )
            else:
                context = " ".join(str(t) for t in token_list) if token_list else ""
        elif isinstance(tokens_data, list):
            context = " ".join(str(t) for t in tokens_data)
        else:
            context = doc.get("text", "")

        if len(context) < 100:
            continue
        context = context[:max_context_chars]

        # 提取问题
        question_data = row.get("question", {})
        if isinstance(question_data, dict):
            question = question_data.get("text", "")
        else:
            question = str(question_data)

        if not question:
            continue

        # 提取答案: annotations.short_answers 首个非空
        annotations = row.get("annotations", {})
        short_answers_list = annotations.get("short_answers", [])

        answer = ""
        for ann in short_answers_list:
            if isinstance(ann, dict):
                texts = ann.get("text", [])
                if texts:
                    answer = texts[0] if isinstance(texts, list) else str(texts)
                    break
            elif isinstance(ann, list):
                # 有些格式是 [[text1, text2], ...]
                for item in ann:
                    if isinstance(item, dict) and item.get("text"):
                        answer = item["text"][0] if isinstance(item["text"], list) else str(item["text"])
                        break
                    elif isinstance(item, str) and item:
                        answer = item
                        break
                if answer:
                    break

        if not answer:
            answer = "No answer"

        results.append({
            "context": context,
            "question": question,
            "answer": str(answer),
            "source": "natural_questions",
        })
    return results


def convert_gsm8k(records: list[dict]) -> list[dict]:
    """GSM8K: question → context, 固定问题, 提取 #### 后的数字."""
    results = []
    for row in records:
        context = row.get("question", "")
        answer_text = row.get("answer", "")

        # 提取 #### 后的最终答案
        match = re.search(r"####\s*(.+)", answer_text)
        answer = match.group(1).strip() if match else answer_text.strip()

        if context:
            results.append({
                "context": str(context),
                "question": "What is the answer?",
                "answer": str(answer),
                "source": "gsm8k",
            })
    return results


def convert_longbench(records: list[dict], config_name: str) -> list[dict]:
    """LongBench: context + input → question, answers 取首个."""
    results = []
    for row in records:
        context = row.get("context", "")
        question = row.get("input", "")
        answers = row.get("answers", [])
        answer = answers[0] if answers else ""

        results.append({
            "context": str(context),
            "question": str(question),
            "answer": str(answer) if answer else "",
            "source": f"longbench_{config_name}",
        })
    return results


def convert_ruler(records: list[dict], config_name: str) -> list[dict]:
    """RULER: input → context, 空 question, expected_output → answer."""
    results = []
    for row in records:
        context = row.get("input", "")
        answer = row.get("expected_output", "")

        results.append({
            "context": str(context),
            "question": "",
            "answer": str(answer) if answer else "",
            "source": f"ruler_{config_name}",
        })
    return results


def convert_halueval(records: list[dict]) -> list[dict]:
    """HaluEval: passage → context, 保留 label."""
    results = []
    for row in records:
        context = row.get("passage", row.get("context", ""))
        question = row.get("question", "")
        answer = row.get("answer", "")

        result = {
            "context": str(context) if context else "",
            "question": str(question) if question else "",
            "answer": str(answer) if answer else "",
            "source": "halueval",
        }
        # 保留 label 字段 (如果有)
        if "label" in row:
            result["label"] = row["label"]
        results.append(result)
    return results


def convert_truthfulqa(records: list[dict]) -> list[dict]:
    """TruthfulQA: 无上下文, question → question, best_answer → answer."""
    results = []
    for row in records:
        question = row.get("question", "")
        answer = row.get("best_answer", "")

        if question:
            results.append({
                "context": "",
                "question": str(question),
                "answer": str(answer) if answer else "",
                "source": "truthfulqa",
            })
    return results


# ===========================================================================
# 处理流水线
# ===========================================================================

FAITHEVAL_FILES = [
    ("faitheval_unanswerable.jsonl", "faitheval_unanswerable"),
    ("faitheval_inconsistent.jsonl", "faitheval_inconsistent"),
    ("faitheval_counterfactual.jsonl", "faitheval_counterfactual"),
]

LONGBENCH_QA_CONFIGS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader",
]

RULER_CONFIGS = [
    "cwe_4k", "cwe_8k",
    "niah_multikey_1_4k", "niah_multikey_1_8k",
    "qa_2_4k", "qa_2_8k",
    "vt_4k", "vt_8k",
]


def _estimate_tokens(text: str) -> int:
    """粗估 token 数: 词数 × 1.3 (与 data_downloader 一致)."""
    return int(len(text.split()) * 1.3)


def process_pretrain(input_dir: Path, output_dir: Path, force: bool = False,
                     eval_size: int = 500,
                     warmup_tokens: int = 100_000_000,
                     multichunk_min_chars: int = 9000):
    """处理预训练数据: 切分为 warmup / multichunk / eval 三份.

    Stage 1a (warmup):     前 ~warmup_tokens 个 token 的文档 → warmup_train.jsonl
    Stage 1b (multichunk): 剩余文档中长度 >= multichunk_min_chars 的 → multichunk_train.jsonl
    eval:                  末尾 eval_size 条 → eval.jsonl (两阶段共用)

    同时保留 train.jsonl (全量训练集) 以兼容上界评估.
    """
    stage1_out = ensure_dir(output_dir / "stage1")
    train_path = stage1_out / "train.jsonl"
    eval_path = stage1_out / "eval.jsonl"
    warmup_path = stage1_out / "warmup_train.jsonl"
    multichunk_path = stage1_out / "multichunk_train.jsonl"

    if not force and warmup_path.exists() and multichunk_path.exists() and eval_path.exists():
        print(f"  [Skip] Stage 1 数据已存在 (使用 --force 覆盖)")
        return

    src = input_dir / "pretrain" / "fineweb_sampled.jsonl"
    if not src.exists():
        print(f"  [Error] 预训练数据不存在: {src}")
        print(f"          请先运行: python scripts/data/data_downloader.py --task pretrain")
        return

    print(f"\n  [Pretrain] 切分预训练数据 (eval_size={eval_size}, "
          f"warmup_tokens={warmup_tokens:,}, multichunk_min_chars={multichunk_min_chars})")

    # 读取所有行
    lines = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line)

    total = len(lines)
    if total <= eval_size:
        print(f"  [Error] 数据量太少 ({total} 行), 无法切出 {eval_size} 条 eval")
        return

    # 末尾切出 eval
    train_lines = lines[:-eval_size]
    eval_lines = lines[-eval_size:]

    # 保留 train.jsonl (兼容上界评估脚本)
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(eval_path, "w", encoding="utf-8") as f:
        f.writelines(eval_lines)

    # 按累计 token 数切分 warmup / multichunk
    warmup_lines = []
    multichunk_lines = []
    cumulative_tokens = 0
    warmup_done = False
    short_skipped = 0

    for line in train_lines:
        try:
            record = json.loads(line.strip())
        except json.JSONDecodeError:
            continue

        text = record.get("text", "")
        est_tokens = _estimate_tokens(text)

        if not warmup_done:
            warmup_lines.append(line)
            cumulative_tokens += est_tokens
            if cumulative_tokens >= warmup_tokens:
                warmup_done = True
        else:
            # multichunk: 只保留足够长的文档 (≥ 2176 tokens ≈ multichunk_min_chars chars)
            if len(text) >= multichunk_min_chars:
                multichunk_lines.append(line)
            else:
                short_skipped += 1

    with open(warmup_path, "w", encoding="utf-8") as f:
        f.writelines(warmup_lines)
    with open(multichunk_path, "w", encoding="utf-8") as f:
        f.writelines(multichunk_lines)

    print(f"  [Stage 1 Done]")
    print(f"    train (legacy):  {len(train_lines)} samples → {train_path}")
    print(f"    warmup (1a):     {len(warmup_lines)} samples (~{cumulative_tokens:,} tokens) → {warmup_path}")
    print(f"    multichunk (1b): {len(multichunk_lines)} samples → {multichunk_path}")
    if short_skipped:
        print(f"    (跳过 {short_skipped} 条短文档 < {multichunk_min_chars} chars)")
    print(f"    eval:            {len(eval_lines)} samples → {eval_path}")


def process_sft(input_dir: Path, output_dir: Path, force: bool = False, seed: int = 42):
    """处理 SFT 数据: 合并所有 QA 数据为 train.json + eval.json."""
    stage2_out = ensure_dir(output_dir / "stage2")
    train_path = stage2_out / "train.json"
    eval_path = stage2_out / "eval.json"

    if not force and train_path.exists() and eval_path.exists():
        print(f"  [Skip] Stage 2 数据已存在 (使用 --force 覆盖)")
        return

    sft_dir = input_dir / "sft"
    if not sft_dir.exists():
        print(f"  [Error] SFT 原始数据目录不存在: {sft_dir}")
        print(f"          请先运行: python scripts/data/data_downloader.py --task sft")
        return

    all_samples = []
    rng = random.Random(seed)
    eval_size = 2000

    print(f"\n  [SFT] 处理 SFT 数据集")

    # --- 1. FaithEval ×3 ---
    for filename, source in FAITHEVAL_FILES:
        path = sft_dir / filename
        records = read_jsonl(path)
        if not records:
            continue
        converted = convert_faitheval(records, source)
        all_samples.extend(converted)
        print(f"    {source}: {len(converted)}")

    # --- 2. ConflictQA ---
    records = read_jsonl(sft_dir / "conflictqa.jsonl")
    if records:
        converted = convert_conflictqa(records)
        all_samples.extend(converted)
        print(f"    conflictqa: {len(converted)}")

    # --- 3. HotpotQA ---
    for split in ("train", "validation"):
        records = read_jsonl(sft_dir / f"hotpotqa_{split}.jsonl")
        if records:
            converted = convert_hotpotqa(records)
            all_samples.extend(converted)
            print(f"    hotpotqa_{split}: {len(records)} → {len(converted)}")

    # --- 4. SQuAD v2 ---
    for split in ("train", "validation"):
        records = read_jsonl(sft_dir / f"squad_v2_{split}.jsonl")
        if records:
            converted = convert_squad_v2(records)
            all_samples.extend(converted)
            print(f"    squad_v2_{split}: {len(records)} → {len(converted)}")

    # --- 5. NQ ---
    records = read_jsonl(sft_dir / "nq_validation.jsonl")
    if records:
        converted = convert_natural_questions(records)
        all_samples.extend(converted)
        print(f"    nq_validation: {len(records)} → {len(converted)}")

    # --- 6. GSM8K ---
    for split in ("train", "test"):
        records = read_jsonl(sft_dir / f"gsm8k_{split}.jsonl")
        if records:
            converted = convert_gsm8k(records)
            all_samples.extend(converted)
            print(f"    gsm8k_{split}: {len(records)} → {len(converted)}")

    # --- 清洗 ---
    all_samples = clean_data(all_samples, "SFT/all")
    print(f"\n  总样本数: {len(all_samples)}")

    # --- 按来源分层抽样 eval，保证 eval 中各来源比例与总体一致 ---
    by_source: dict[str, list] = {}
    for s in all_samples:
        by_source.setdefault(s["source"], []).append(s)

    eval_samples = []
    train_samples = []
    for source, items in by_source.items():
        rng.shuffle(items)
        n_eval = max(1, round(eval_size * len(items) / len(all_samples)))
        eval_samples.extend(items[:n_eval])
        train_samples.extend(items[n_eval:])

    rng.shuffle(train_samples)
    rng.shuffle(eval_samples)

    write_json(train_samples, train_path)
    write_json(eval_samples, eval_path)

    print(f"\n  [Stage 2 Done] train: {len(train_samples)} samples → {train_path}")
    print(f"                 eval:  {len(eval_samples)} samples → {eval_path}")

    train_sources = Counter(s["source"] for s in train_samples)
    eval_sources = Counter(s["source"] for s in eval_samples)
    print(f"  Train sources: {dict(train_sources)}")
    print(f"  Eval sources:  {dict(eval_sources)}")


def process_eval(input_dir: Path, output_dir: Path, force: bool = False):
    """处理 eval 数据: LongBench + RULER (不做清洗, 保持 benchmark 完整性)."""
    eval_in = input_dir / "eval"
    if not eval_in.exists():
        print(f"  [Error] Eval 原始数据目录不存在: {eval_in}")
        print(f"          请先运行: python scripts/data/data_downloader.py --task eval")
        return

    print(f"\n  [Benchmark/Eval] 处理评测数据集")

    # --- LongBench ---
    lb_in = eval_in / "longbench"
    lb_out = ensure_dir(output_dir / "benchmark" / "longbench")
    for cfg in LONGBENCH_QA_CONFIGS:
        out_path = lb_out / f"{cfg}.json"
        if not force and out_path.exists():
            continue
        records = read_jsonl(lb_in / f"{cfg}.jsonl")
        if not records:
            continue
        converted = convert_longbench(records, cfg)
        write_json(converted, out_path)
        print(f"    longbench/{cfg}: {len(converted)} samples")

    # --- RULER ---
    ruler_in = eval_in / "ruler"
    ruler_out = ensure_dir(output_dir / "benchmark" / "ruler")
    for cfg in RULER_CONFIGS:
        out_path = ruler_out / f"{cfg}.json"
        if not force and out_path.exists():
            continue
        records = read_jsonl(ruler_in / f"{cfg}.jsonl")
        if not records:
            continue
        converted = convert_ruler(records, cfg)
        write_json(converted, out_path)
        print(f"    ruler/{cfg}: {len(converted)} samples")

    print(f"  [Benchmark/Eval Done]")


def process_diagnostic(input_dir: Path, output_dir: Path, force: bool = False):
    """处理诊断数据: HaluEval + TruthfulQA (不做清洗). 输出到 benchmark/ 目录."""
    diag_in = input_dir / "diagnostic"
    if not diag_in.exists():
        print(f"  [Error] Diagnostic 原始数据目录不存在: {diag_in}")
        print(f"          请先运行: python scripts/data/data_downloader.py --task diagnostic")
        return

    bench_out = ensure_dir(output_dir / "benchmark")

    print(f"\n  [Benchmark/Diagnostic] 处理诊断数据集")

    # --- HaluEval ---
    out_path = bench_out / "halueval.json"
    if force or not out_path.exists():
        records = read_jsonl(diag_in / "halueval.jsonl")
        if records:
            converted = convert_halueval(records)
            write_json(converted, out_path)
            print(f"    halueval: {len(converted)} samples")

    # --- TruthfulQA ---
    out_path = bench_out / "truthfulqa.json"
    if force or not out_path.exists():
        records = read_jsonl(diag_in / "truthfulqa.jsonl")
        if records:
            converted = convert_truthfulqa(records)
            write_json(converted, out_path)
            print(f"    truthfulqa: {len(converted)} samples")

    print(f"  [Benchmark/Diagnostic Done]")


# ===========================================================================
# Main
# ===========================================================================
TASK_MAP = {
    "pretrain": "Pretrain (FineWeb → train/eval split)",
    "sft": "SFT (FaithEval + ConflictQA + HotpotQA + SQuAD + NQ + GSM8K)",
    "eval": "Eval (LongBench QA + RULER)",
    "diagnostic": "Diagnostic (HaluEval + TruthfulQA)",
    "all": "全部数据集",
}


def main():
    parser = argparse.ArgumentParser(
        description="数据处理器 — 将原始 HF 数据集转换为统一 QA 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 (从项目根目录运行):
  python scripts/data/data_processor.py --task all
  python scripts/data/data_processor.py --task sft
  python scripts/data/data_processor.py --task eval diagnostic
  python scripts/data/data_processor.py --input_dir ./data/download --output_dir ./data
  python scripts/data/data_processor.py --force
        """,
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=list(TASK_MAP.keys()),
        default=["all"],
        help="要处理的任务 (可多选, 默认: all)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/download",
        help="原始数据根目录 (默认: ./data/download)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="输出根目录 (默认: ./data, 会在下面生成 stage1/ stage2/ benchmark/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖已有输出",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    tasks = args.task
    if "all" in tasks:
        tasks = ["pretrain", "sft", "eval", "diagnostic"]

    print(f"\n{'#'*70}")
    print(f"# 数据处理器")
    print(f"# 输入目录: {input_dir}")
    print(f"# 输出目录: {output_dir}")
    print(f"# 任务: {', '.join(tasks)}")
    print(f"# 覆盖: {'是' if args.force else '否'}")
    print(f"{'#'*70}")

    if "pretrain" in tasks:
        process_pretrain(input_dir, output_dir, force=args.force)

    if "sft" in tasks:
        process_sft(input_dir, output_dir, force=args.force, seed=args.seed)

    if "eval" in tasks:
        process_eval(input_dir, output_dir, force=args.force)

    if "diagnostic" in tasks:
        process_diagnostic(input_dir, output_dir, force=args.force)

    print(f"\n{'#'*70}")
    print(f"# 处理完成!")
    print(f"# 输出目录: {output_dir}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
