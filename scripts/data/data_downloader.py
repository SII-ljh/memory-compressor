#!/usr/bin/env python3
"""
data_downloader.py — 科研数据集全量/采样下载器

支持断点续传、流式下载、HF镜像、CLI任务切换。

用法 (从项目根目录运行):
    # 下载全部数据集
    python scripts/data/data_downloader.py --task all

    # 仅下载预训练数据 (FineWeb 采样 ~3B tokens)
    python scripts/data/data_downloader.py --task pretrain --target_tokens 3_000_000_000

    # 仅下载 SFT 数据
    python scripts/data/data_downloader.py --task sft

    # 仅下载评测数据
    python scripts/data/data_downloader.py --task eval

    # 仅下载诊断数据
    python scripts/data/data_downloader.py --task diagnostic

    # 默认已启用 HF 镜像 (hf-mirror.com), 如需直连可禁用:
    python scripts/data/data_downloader.py --task all --no-mirror

    # Debug 模式 (从全量数据集随机采样 5%, 用于本地调试模型架构和训练)
    python scripts/data/data_downloader.py --task all --debug

    # 自定义输出目录 (默认: ./data/download)
    python scripts/data/data_downloader.py --task all --output_dir /path/to/download
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# HF 镜像配置 (需在 import datasets 之前设置)
# ---------------------------------------------------------------------------
def setup_hf_mirror(use_mirror: bool = False):
    """设置 HuggingFace 镜像站 (中国境内加速)."""
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[Mirror] 已启用 HF 镜像: https://hf-mirror.com")
    else:
        # 如果环境变量中已有镜像设置，保留用户自定义
        endpoint = os.environ.get("HF_ENDPOINT", "")
        if endpoint:
            print(f"[Mirror] 使用已有 HF_ENDPOINT: {endpoint}")


# ---------------------------------------------------------------------------
# 延迟导入 (镜像设置需在 import 前完成)
# ---------------------------------------------------------------------------
def lazy_imports():
    global datasets, load_dataset, tqdm
    import datasets as _datasets
    from datasets import load_dataset as _load_dataset
    from tqdm import tqdm as _tqdm
    datasets = _datasets
    load_dataset = _load_dataset
    tqdm = _tqdm


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(records: list[dict], path: Path, mode: str = "w"):
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(record: dict, f):
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def estimate_tokens(text: str) -> int:
    """估算文本 token 数.

    英文 web 文本: ~1.3 tokens/word, ~0.25 tokens/char (BPE tokenizer).
    中文文本: ~1.5 tokens/字, ~0.6 tokens/char.
    取 word-based 估算作为主要依据, 对中文用 char-based 补偿.
    """
    words = text.split()
    word_est = int(len(words) * 1.3)
    # 检测中文占比: 如果 word 数远少于 char 数 (中文无空格), 用 char-based
    if len(words) > 0 and len(text) / len(words) > 10:
        # 大概率是中文为主的文本
        return max(word_est, int(len(text) * 0.6))
    return max(word_est, int(len(text) * 0.25))


def count_tokens_in_jsonl(path: Path, text_key: str = "text") -> int:
    """粗略估算 JSONL 中的 token 数."""
    if not path.exists():
        return 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get(text_key, "")
                total += estimate_tokens(text)
            except json.JSONDecodeError:
                continue
    return total


def format_tokens(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def save_dataset_to_disk(ds, output_dir: Path, name: str):
    """将 HF Dataset 保存到磁盘 (Arrow 格式, 支持断点续传)."""
    save_path = output_dir / name
    if save_path.exists() and (save_path / "dataset_info.json").exists():
        print(f"  [Skip] {name} 已存在于 {save_path}")
        return
    print(f"  [Save] {name} → {save_path}")
    ds.save_to_disk(str(save_path))
    print(f"  [Done] {name} ({len(ds)} samples)")


def save_dataset_as_jsonl(ds, output_path: Path, desc: str = ""):
    """将 HF Dataset 保存为 JSONL."""
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"  [Skip] {output_path.name} 已存在 ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return
    print(f"  [Save] {desc} → {output_path}")
    records = []
    for item in tqdm(ds, desc=desc, leave=False):
        records.append(dict(item))
    write_jsonl(records, output_path)
    print(f"  [Done] {output_path.name} ({len(records)} samples, {output_path.stat().st_size / 1024 / 1024:.1f} MB)")


# ===========================================================================
# Task 1: Pre-train — FineWeb 流式采样
# ===========================================================================
def download_pretrain(output_dir: Path, target_tokens: int = 3_000_000_000, seed: int = 42, debug: bool = False):
    """
    流式采样 FineWeb sample-10BT, 目标 ~3B tokens.

    策略: 流式遍历 + 按概率采样 (sample_rate ≈ target/total).
    FineWeb sample-10BT 约 10B tokens, 采样率 ≈ 20%.
    支持断点续传: 记录已采样 token 数到 checkpoint 文件.
    """
    if debug:
        target_tokens = int(target_tokens * DEBUG_SAMPLE_RATIO)
        print(f"  [DEBUG] pretrain: target_tokens = {format_tokens(target_tokens)} ({DEBUG_SAMPLE_RATIO:.0%} of full)")
    pretrain_dir = ensure_dir(output_dir / "pretrain")
    output_path = pretrain_dir / "fineweb_sampled.jsonl"
    ckpt_path = pretrain_dir / ".fineweb_checkpoint.json"

    # 断点续传: 读取 checkpoint
    start_idx = 0
    accumulated_tokens = 0
    total_seen = 0
    if ckpt_path.exists():
        with open(ckpt_path, "r") as f:
            ckpt = json.load(f)
        start_idx = ckpt.get("total_seen", 0)
        accumulated_tokens = ckpt.get("accumulated_tokens", 0)
        total_seen = start_idx
        print(f"  [Resume] 从 checkpoint 恢复: 已处理 {total_seen:,} 条, "
              f"已采样 {format_tokens(accumulated_tokens)} tokens")
        if accumulated_tokens >= target_tokens:
            print(f"  [Skip] 已达到目标 {format_tokens(target_tokens)} tokens")
            return

    # FineWeb sample-10BT ≈ 15M docs, ~10B tokens → sample_rate ≈ target/10B
    total_pool_tokens = 10_000_000_000
    sample_rate = min(1.0, target_tokens / total_pool_tokens * 1.2)  # 1.2x 超采防不足

    print(f"\n{'='*70}")
    print(f"[Pretrain] FineWeb sample-10BT 流式采样")
    print(f"  目标: {format_tokens(target_tokens)} tokens")
    print(f"  采样率: {sample_rate:.2%}")
    print(f"  输出: {output_path}")
    print(f"{'='*70}")

    rng = random.Random(seed)
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    file_mode = "a" if start_idx > 0 else "w"
    save_interval = 10_000  # 每 1w 条保存一次 checkpoint
    sampled_count = count_lines(output_path) if start_idx > 0 else 0

    pbar = tqdm(
        desc=f"FineWeb → {format_tokens(accumulated_tokens)}/{format_tokens(target_tokens)}",
        unit=" docs",
        total=None,
    )

    try:
        with open(output_path, file_mode, encoding="utf-8") as f:
            for i, example in enumerate(ds):
                if i < start_idx:
                    # 跳过已处理的 (流式无法 seek, 只能快进)
                    if i % 500_000 == 0 and i > 0:
                        pbar.set_description(f"[Fast-forward] {i:,}/{start_idx:,}")
                    continue

                total_seen = i + 1
                pbar.update(1)

                # 按概率采样
                if rng.random() > sample_rate:
                    continue

                text = example.get("text", "")
                if not text or len(text) < 100:
                    continue

                # 估算 token 数 (英文 ~1.3 tokens/word, ~0.25 tokens/char)
                est_tokens = estimate_tokens(text)
                accumulated_tokens += est_tokens
                sampled_count += 1

                append_jsonl({"text": text, "id": example.get("id", str(i))}, f)

                # 更新进度条
                pbar.set_description(
                    f"FineWeb | sampled {sampled_count:,} | "
                    f"{format_tokens(accumulated_tokens)}/{format_tokens(target_tokens)}"
                )

                # 定期 checkpoint
                if sampled_count % save_interval == 0:
                    f.flush()
                    with open(ckpt_path, "w") as cf:
                        json.dump({
                            "total_seen": total_seen,
                            "accumulated_tokens": accumulated_tokens,
                            "sampled_count": sampled_count,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }, cf, indent=2)

                # 达到目标
                if accumulated_tokens >= target_tokens:
                    print(f"\n  [Done] 达到目标! 采样 {sampled_count:,} 条, "
                          f"估计 {format_tokens(accumulated_tokens)} tokens")
                    break
    except KeyboardInterrupt:
        print(f"\n  [Interrupt] 用户中断, 保存 checkpoint...")
    finally:
        pbar.close()
        # 保存最终 checkpoint
        with open(ckpt_path, "w") as cf:
            json.dump({
                "total_seen": total_seen,
                "accumulated_tokens": accumulated_tokens,
                "sampled_count": sampled_count,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "completed": accumulated_tokens >= target_tokens,
            }, cf, indent=2)

    file_size = output_path.stat().st_size / (1024 ** 3)
    print(f"  [Summary] {output_path}: {sampled_count:,} docs, "
          f"{format_tokens(accumulated_tokens)} tokens, {file_size:.2f} GB")


# ===========================================================================
# Task 2: SFT — 纯 QA 提取 + 反幻觉 + 推理链保真
# ===========================================================================
# 设计原则: 只训练"从长文中精准提取事实"的能力
# 绝不引入摘要/生成类数据 (BookSum, LongForm, GovReport 等会破坏 Fidelity)
#
# 数据集组成:
#   FaithEval ×3   — 反幻觉: 反事实/矛盾/不可回答  (~5k, ~16 MB)
#   ConflictQA     — 知识冲突: 参数记忆 vs 上下文事实 (~10k, ~50 MB)
#   HotpotQA       — 多跳推理: 跨段落证据链提取      (~98k, ~600 MB)
#   SQuAD v2       — 标准抽取QA + 不可回答判断        (~142k, ~100 MB)
#   NQ validation  — 真实搜索问答 (train 45GB 过大)   (~7.8k, ~200 MB)
#   GSM8K          — 数学推理链保真                   (~8.8k, ~6 MB)

FAITHEVAL_VARIANTS = [
    ("Salesforce/FaithEval-unanswerable-v1.0", "faitheval_unanswerable.jsonl"),
    ("Salesforce/FaithEval-inconsistent-v1.0", "faitheval_inconsistent.jsonl"),
    ("Salesforce/FaithEval-counterfactual-v1.0", "faitheval_counterfactual.jsonl"),
]


DEBUG_SAMPLE_RATIO = 0.05


def _debug_sample(ds, debug: bool, seed: int = 42):
    """Debug 模式下从全量数据集中随机采样 5% 的数据 (保持分布代表性)."""
    if not debug:
        return ds
    total = len(ds)
    n = max(1, int(total * DEBUG_SAMPLE_RATIO))
    sampled = ds.shuffle(seed=seed).select(range(n))
    print(f"    [DEBUG] 采样 {n}/{total} 条 ({DEBUG_SAMPLE_RATIO:.0%})")
    return sampled


def download_sft(output_dir: Path, debug: bool = False):
    sft_dir = ensure_dir(output_dir / "sft")

    print(f"\n{'='*70}")
    print(f"[SFT] 下载 QA 提取 + 反幻觉 SFT 数据集")
    print(f"  输出目录: {sft_dir}")
    print(f"{'='*70}")

    # --- 1. FaithEval ×3 (反幻觉训练: 反事实/矛盾/不可回答) ---
    print("\n  [1/6] Salesforce/FaithEval (反幻觉, 3 variants)")
    for repo_id, filename in FAITHEVAL_VARIANTS:
        output_path = sft_dir / filename
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"    [Skip] {filename} ({output_path.stat().st_size / 1024:.1f} KB)")
            continue
        try:
            ds = load_dataset(repo_id, split="test")
            ds = _debug_sample(ds, debug)
            save_dataset_as_jsonl(ds, output_path, desc=filename)
        except Exception as e:
            print(f"    [Error] {repo_id}: {e}")

    # --- 2. ConflictQA (知识冲突: 参数记忆 vs 上下文事实) ---
    print("\n  [2/6] osunlp/ConflictQA (知识冲突对抗)")
    output_path = sft_dir / "conflictqa.jsonl"
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"    [Skip] 已存在 ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        try:
            ds = load_dataset(
                "osunlp/ConflictQA",
                "ConflictQA-popQA-chatgpt",
                split="test",
            )
        except (RuntimeError, ValueError):
            # 旧版数据集使用自定义脚本, 新版 datasets 不支持
            # fallback: 从 HF auto-converted parquet 分支加载 (config=default)
            print("    [Fallback] 尝试从 parquet 分支加载...")
            ds = load_dataset(
                "osunlp/ConflictQA",
                split="test",
                revision="refs/convert/parquet",
            )
        ds = _debug_sample(ds, debug)
        save_dataset_as_jsonl(ds, output_path, desc="ConflictQA")

    # --- 3. HotpotQA distractor (多跳推理, train + validation) ---
    print("\n  [3/6] hotpotqa/hotpot_qa distractor (多跳推理)")
    for split in ["train", "validation"]:
        output_path = sft_dir / f"hotpotqa_{split}.jsonl"
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"    [Skip] hotpotqa_{split} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
            continue
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        ds = _debug_sample(ds, debug)
        save_dataset_as_jsonl(ds, output_path, desc=f"HotpotQA/{split}")

    # --- 4. SQuAD v2 (标准抽取QA + 不可回答) ---
    print("\n  [4/6] SQuAD v2 (抽取QA + unanswerable)")
    for split in ["train", "validation"]:
        output_path = sft_dir / f"squad_v2_{split}.jsonl"
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"    [Skip] squad_v2_{split} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
            continue
        # rajpurkar/squad_v2 需要 datasets>=2.20 (List feature type)
        # 旧版本 fallback 到 legacy repo "squad_v2"
        try:
            ds = load_dataset("rajpurkar/squad_v2", split=split)
        except (ValueError, TypeError):
            print(f"    [Fallback] rajpurkar/squad_v2 不兼容当前 datasets 版本, 使用 legacy repo")
            ds = load_dataset("squad_v2", split=split)
        ds = _debug_sample(ds, debug)
        save_dataset_as_jsonl(ds, output_path, desc=f"SQuAD-v2/{split}")

    # --- 5. Natural Questions (validation only, train=45GB 过大) ---
    print("\n  [5/6] Natural Questions (validation, 7.8k samples)")
    output_path = sft_dir / "nq_validation.jsonl"
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"    [Skip] 已存在 ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        ds = load_dataset(
            "google-research-datasets/natural_questions",
            split="validation",
        )
        ds = _debug_sample(ds, debug)
        save_dataset_as_jsonl(ds, output_path, desc="NQ/validation")

    # --- 6. GSM8K (数学推理链保真) ---
    print("\n  [6/6] openai/gsm8k (数学推理链)")
    for split in ["train", "test"]:
        output_path = sft_dir / f"gsm8k_{split}.jsonl"
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"    [Skip] gsm8k_{split} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
            continue
        ds = load_dataset("openai/gsm8k", "main", split=split)
        ds = _debug_sample(ds, debug)
        save_dataset_as_jsonl(ds, output_path, desc=f"GSM8K/{split}")

    print(f"\n  [SFT Done]")


# ===========================================================================
# Task 3: Eval — 长上下文压缩性能基准
# ===========================================================================
# LongBench: 只取 QA 类子集 (剔除摘要/分类/代码)
LONGBENCH_QA_CONFIGS = [
    "narrativeqa",        # 单文档叙事QA
    "qasper",             # 论文QA
    "multifieldqa_en",    # 多领域QA (英)
    "multifieldqa_zh",    # 多领域QA (中)
    "hotpotqa",           # 多文档QA
    "2wikimqa",           # 多文档QA
    "musique",            # 多跳推理QA
    "dureader",           # 中文阅读理解
]

RULER_CONFIGS = [
    "cwe_4k", "cwe_8k",
    "niah_multikey_1_4k", "niah_multikey_1_8k",
    "qa_2_4k", "qa_2_8k",
    "vt_4k", "vt_8k",
]


def download_eval(output_dir: Path, debug: bool = False):
    eval_dir = ensure_dir(output_dir / "eval")

    print(f"\n{'='*70}")
    print(f"[Eval] 下载长上下文压缩性能评测基准")
    print(f"  输出目录: {eval_dir}")
    print(f"{'='*70}")

    # --- LongBench QA 子集 ---
    print("\n  [1/2] THUDM/LongBench (QA 子集, 8 configs)")
    lb_dir = ensure_dir(eval_dir / "longbench")
    for cfg in tqdm(LONGBENCH_QA_CONFIGS, desc="LongBench QA"):
        output_path = lb_dir / f"{cfg}.jsonl"
        if output_path.exists() and output_path.stat().st_size > 0:
            continue
        try:
            ds = load_dataset("THUDM/LongBench", cfg, split="test", trust_remote_code=True)
            ds = _debug_sample(ds, debug)
            records = [dict(item) for item in ds]
            write_jsonl(records, output_path)
        except Exception as e:
            print(f"\n    [Error] LongBench/{cfg}: {e}")
    print(f"    [Done] LongBench: {sum(1 for f in lb_dir.glob('*.jsonl'))} configs")

    # --- RULER ---
    print("\n  [2/2] rbiswasfc/ruler (合成压力测试, 8 configs)")
    ruler_dir = ensure_dir(eval_dir / "ruler")
    for cfg in tqdm(RULER_CONFIGS, desc="RULER"):
        output_path = ruler_dir / f"{cfg}.jsonl"
        if output_path.exists() and output_path.stat().st_size > 0:
            continue
        try:
            ds = load_dataset("rbiswasfc/ruler", cfg, split="validation")
            ds = _debug_sample(ds, debug)
            records = [dict(item) for item in ds]
            write_jsonl(records, output_path)
        except Exception as e:
            print(f"\n    [Error] RULER/{cfg}: {e}")
    print(f"    [Done] RULER: {sum(1 for f in ruler_dir.glob('*.jsonl'))} configs")

    print(f"\n  [Eval Done]")


# ===========================================================================
# Task 4: Diagnostic — 压缩保真度探针
# ===========================================================================
# HaluEval:    10k QA 幻觉检测 (PASS/FAIL 标签, 直接量化幻觉率)
# TruthfulQA:  817 对抗性事实题 (模型是否用先验知识覆盖上下文事实)

def download_diagnostic(output_dir: Path, debug: bool = False):
    diag_dir = ensure_dir(output_dir / "diagnostic")

    print(f"\n{'='*70}")
    print(f"[Diagnostic] 下载压缩保真度诊断数据集")
    print(f"  输出目录: {diag_dir}")
    print(f"{'='*70}")

    # --- HaluEval (幻觉检测, 10k PASS/FAIL) ---
    print("\n  [1/2] flowaicom/HaluEval (幻觉检测, 10k)")
    output_path = diag_dir / "halueval.jsonl"
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"    [Skip] 已存在 ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        try:
            ds = load_dataset("flowaicom/HaluEval", split="test")
            ds = _debug_sample(ds, debug)
            save_dataset_as_jsonl(ds, output_path, desc="HaluEval")
        except Exception as e:
            print(f"    [Error] HaluEval: {e}")

    # --- TruthfulQA (对抗性事实, 817 questions) ---
    print("\n  [2/2] domenicrosati/TruthfulQA (对抗性事实题, 817)")
    output_path = diag_dir / "truthfulqa.jsonl"
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"    [Skip] 已存在 ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        try:
            ds = load_dataset("domenicrosati/TruthfulQA", split="train")
            ds = _debug_sample(ds, debug)
            save_dataset_as_jsonl(ds, output_path, desc="TruthfulQA")
        except Exception as e:
            print(f"    [Error] TruthfulQA: {e}")


# ===========================================================================
# Main
# ===========================================================================
TASK_MAP = {
    "pretrain": "Pre-train (FineWeb ~3B tokens 流式采样)",
    "sft": "SFT (FaithEval + ConflictQA + HotpotQA + SQuAD + NQ + GSM8K)",
    "eval": "Eval (LongBench QA + RULER)",
    "diagnostic": "Diagnostic (HaluEval + TruthfulQA)",
    "all": "全部数据集",
}


def main():
    parser = argparse.ArgumentParser(
        description="科研数据集下载器 — 支持流式下载、断点续传、HF镜像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 (从项目根目录运行):
  python scripts/data/data_downloader.py --task all                    # 下载全部 → data/download/
  python scripts/data/data_downloader.py --task pretrain --no-mirror   # 禁用镜像, 直连 HF
  python scripts/data/data_downloader.py --task sft eval               # 仅下载 SFT + Eval
  python scripts/data/data_downloader.py --task sft eval --debug       # Debug 模式小数据集
  python scripts/data/data_downloader.py --task pretrain --target_tokens 500_000_000  # 采样 0.5B tokens
        """,
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=list(TASK_MAP.keys()),
        default=["all"],
        help="要下载的任务 (可多选, 默认: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/download",
        help="输出根目录 (默认: ./data/download)",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        default=True,
        help="启用 HF 镜像 (hf-mirror.com, 默认开启)",
    )
    parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="禁用 HF 镜像 (直连 huggingface.co)",
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=3_000_000_000,
        help="FineWeb 采样目标 token 数 (默认: 3B)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug 模式: 从全量数据集随机采样 5%%, 用于本地调试模型架构和训练",
    )

    args = parser.parse_args()

    # 镜像配置 (必须在 import datasets 前)
    setup_hf_mirror(args.mirror)
    lazy_imports()

    output_dir = Path(args.output_dir).resolve()
    tasks = args.task
    if "all" in tasks:
        tasks = ["pretrain", "sft", "eval", "diagnostic"]

    debug = args.debug
    debug_tag = " [DEBUG]" if debug else ""

    print(f"\n{'#'*70}")
    print(f"# 科研数据集下载器{debug_tag}")
    print(f"# 输出目录: {output_dir}")
    print(f"# 任务: {', '.join(tasks)}")
    if debug:
        print(f"# 模式: DEBUG (全量随机采样 {DEBUG_SAMPLE_RATIO:.0%})")
    print(f"{'#'*70}")

    t0 = time.time()

    if "pretrain" in tasks:
        download_pretrain(output_dir, target_tokens=args.target_tokens, seed=args.seed, debug=debug)

    if "sft" in tasks:
        download_sft(output_dir, debug=debug)

    if "eval" in tasks:
        download_eval(output_dir, debug=debug)

    if "diagnostic" in tasks:
        download_diagnostic(output_dir, debug=debug)

    elapsed = time.time() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n{'#'*70}")
    print(f"# 全部完成! 耗时: {minutes}m {seconds}s")
    print(f"# 数据目录: {output_dir}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
