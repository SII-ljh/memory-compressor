#!/usr/bin/env python3
"""Pre-tokenize training data for faster startup.

Generates .npy cache files next to data files. Training auto-detects and loads them.

Usage:
    # All stages (uses default config)
    python scripts/preprocess_data.py --config config/default.yaml

    # Specific stage
    python scripts/preprocess_data.py --config config/default.yaml --stage 1b

    # With config overrides (e.g. different model/tokenizer)
    python scripts/preprocess_data.py --config config/default.yaml \
        --override qwen3_model_path=./models/Qwen3-1.7B stage1b_max_chunks=8
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer

from src.config import QCPCConfig
from src.data import (
    PretrainDataset,
    PretrainMultiChunkDataset,
    _token_cache_path,
)

logger = logging.getLogger(__name__)


def preprocess_stage1a(config: QCPCConfig, tokenizer: AutoTokenizer):
    """Pre-tokenize Stage 1a data."""
    sample_len = config.stage1a_max_context_len + config.stage1a_max_cont_len

    for label, path in [
        ("train", config.stage1a_train_data_path),
        ("eval", config.pretrain_eval_data_path),
    ]:
        cache = _token_cache_path(path, sample_len)
        if cache.exists():
            logger.info(f"[1a {label}] Cache exists: {cache.name}, skipping")
            continue
        logger.info(f"[1a {label}] Processing {path} ...")
        ds = PretrainDataset(
            path, tokenizer,
            max_context_len=config.stage1a_max_context_len,
            max_cont_len=config.stage1a_max_cont_len,
        )
        logger.info(f"[1a {label}] Done: {len(ds):,} samples")


def preprocess_stage1b(config: QCPCConfig, tokenizer: AutoTokenizer):
    """Pre-tokenize Stage 1b data."""
    sample_len = (
        config.stage1b_max_chunks * config.stage1b_chunk_len
        + config.stage1b_max_cont_len
    )

    for label, path in [
        ("train", config.stage1b_train_data_path),
        ("eval", config.pretrain_eval_data_path),
    ]:
        cache = _token_cache_path(path, sample_len)
        if cache.exists():
            logger.info(f"[1b {label}] Cache exists: {cache.name}, skipping")
            continue
        logger.info(f"[1b {label}] Processing {path} ...")
        ds = PretrainMultiChunkDataset(
            path, tokenizer,
            max_chunks=config.stage1b_max_chunks,
            min_chunks=config.stage1b_min_chunks,
            chunk_len=config.stage1b_chunk_len,
            cont_len=config.stage1b_max_cont_len,
        )
        logger.info(f"[1b {label}] Done: {len(ds):,} samples")


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize QCPC training data")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--stage", type=str, choices=["1a", "1b", "all"], default="all",
        help="Which stage data to preprocess (default: all)",
    )
    parser.add_argument("--override", nargs="*", default=[], help="Override config: key=value pairs")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )

    config = QCPCConfig.load(args.config)
    if args.override:
        from src.train import apply_overrides
        config = apply_overrides(config, args.override)

    logger.info(f"Loading tokenizer from {config.qwen3_model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.qwen3_model_path)

    if args.stage in ("all", "1a"):
        preprocess_stage1a(config, tokenizer)
    if args.stage in ("all", "1b"):
        preprocess_stage1b(config, tokenizer)

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
