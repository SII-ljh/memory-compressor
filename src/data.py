"""Data loaders for Stage 1 (pretrain) and Stage 2 (QA finetune)."""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _token_cache_path(data_path: str, sample_len: int) -> Path:
    """Cache path for pre-tokenized data: {dir}/{stem}.tokens_{sample_len}.npy"""
    p = Path(data_path)
    return p.parent / f"{p.stem}.tokens_{sample_len}.npy"


class PretrainDataset(Dataset):
    """Stage 1: Text completion pretraining dataset.

    Reads JSONL with {"text": "...", "id": "..."} format.
    Each text is tokenized and sliced into fixed-length segments (sample_len).
    Segments shorter than sample_len are discarded.
    Each segment is split into context (for compression) and continuation (target).
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_context_len: int = 512,
        max_cont_len: int = 128,
        sample_len: int = 640,
    ):
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.max_cont_len = max_cont_len
        self.sample_len = sample_len

        # Check for pre-tokenized cache
        cache = _token_cache_path(data_path, sample_len)
        if cache.exists():
            t0 = time.time()
            self.samples = np.load(str(cache), mmap_mode="r")
            logger.info(
                f"Loaded {len(self.samples):,} cached samples "
                f"from {cache.name} ({time.time() - t0:.1f}s)"
            )
            return

        logger.info(f"Tokenizing {Path(data_path).name} (sample_len={sample_len})...")
        t0 = time.time()

        # Suppress "Token indices sequence length is longer than the specified
        # maximum sequence length" warning — we segment into fixed-length
        # samples ourselves, so the tokenizer's model_max_length is irrelevant.
        _orig_max_len = tokenizer.model_max_length
        tokenizer.model_max_length = int(1e18)

        # Pre-chunk: tokenize all texts and slice into fixed-length segments
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tokens = tokenizer.encode(rec["text"], add_special_tokens=False)
                # Slice into non-overlapping segments of sample_len, drop remainder
                for start in range(0, len(tokens) - sample_len + 1, sample_len):
                    samples.append(tokens[start:start + sample_len])

        tokenizer.model_max_length = _orig_max_len
        self.samples = np.array(samples, dtype=np.int32)
        logger.info(
            f"Tokenized {len(self.samples):,} samples in {time.time() - t0:.1f}s"
        )

        # Auto-save cache (only rank 0 in distributed)
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        if rank == 0:
            np.save(str(cache), self.samples)
            logger.info(f"Saved token cache: {cache.name} ({self.samples.nbytes / 1e6:.1f} MB)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]  # exactly sample_len tokens

        # Split: context for compression, continuation as target
        split_point = min(len(tokens) - self.max_cont_len, self.max_context_len)
        context_ids = tokens[:split_point]
        cont_ids = tokens[split_point:split_point + self.max_cont_len]

        return {
            "context_ids": context_ids,
            "target_ids": cont_ids,
        }


class PretrainMultiChunkDataset(Dataset):
    """Stage 1b: Multi-chunk concatenation training dataset.

    Each sample consists of K consecutive chunks compressed independently,
    whose memory tokens are concatenated and fed to the decoder to predict
    a continuation target.

    K is determined dynamically per document:
        K = clamp(available_tokens // chunk_len, min_chunks, max_chunks)
    This avoids discarding shorter documents while still supporting long ones.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_chunks: int = 4,
        min_chunks: int = 2,
        chunk_len: int = 512,
        cont_len: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self.min_chunks = min_chunks
        self.chunk_len = chunk_len
        self.cont_len = cont_len

        sample_len = max_chunks * chunk_len + cont_len

        # Check for pre-tokenized cache
        cache = _token_cache_path(data_path, sample_len)
        if cache.exists():
            t0 = time.time()
            self.samples = np.load(str(cache), mmap_mode="r")
            logger.info(
                f"Loaded {len(self.samples):,} cached samples "
                f"from {cache.name} ({time.time() - t0:.1f}s)"
            )
            return

        logger.info(f"Tokenizing {Path(data_path).name} (sample_len={sample_len})...")
        t0 = time.time()

        # Suppress tokenizer max length warning (same as PretrainDataset)
        _orig_max_len = tokenizer.model_max_length
        tokenizer.model_max_length = int(1e18)

        # Pre-chunk: slice long texts into fixed-length samples
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tokens = tokenizer.encode(rec["text"], add_special_tokens=False)
                # Slice into non-overlapping segments of sample_len, drop remainder
                for start in range(0, len(tokens) - sample_len + 1, sample_len):
                    samples.append(tokens[start:start + sample_len])

        tokenizer.model_max_length = _orig_max_len
        self.samples = np.array(samples, dtype=np.int32)
        logger.info(
            f"Tokenized {len(self.samples):,} samples in {time.time() - t0:.1f}s"
        )

        # Auto-save cache (only rank 0 in distributed)
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        if rank == 0:
            np.save(str(cache), self.samples)
            logger.info(f"Saved token cache: {cache.name} ({self.samples.nbytes / 1e6:.1f} MB)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]  # exactly sample_len tokens
        K = self.max_chunks

        # Split: K chunks of chunk_len + cont_len target
        chunk_ids = []
        for i in range(K):
            start = i * self.chunk_len
            end = start + self.chunk_len
            chunk_ids.append(tokens[start:end])

        target_start = K * self.chunk_len
        target_ids = tokens[target_start:target_start + self.cont_len]

        return {
            "chunk_ids": chunk_ids,   # list of K lists, each chunk_len
            "target_ids": target_ids,  # list of cont_len
            "num_chunks": K,
        }


class QADataset(Dataset):
    """Stage 2: QA finetuning dataset.

    Reads JSON array with {"context": str, "question": str, "answer": str} format.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_context_len: int = 4096,
        max_prompt_len: int = 128,
        max_answer_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.max_prompt_len = max_prompt_len
        self.max_answer_len = max_answer_len

        with open(data_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        context_ids = self.tokenizer.encode(
            rec["context"], add_special_tokens=False, max_length=self.max_context_len, truncation=True
        )
        prompt_ids = self.tokenizer.encode(
            rec["question"], add_special_tokens=False, max_length=self.max_prompt_len, truncation=True
        )
        answer_ids = self.tokenizer.encode(
            rec["answer"], add_special_tokens=False, max_length=self.max_answer_len, truncation=True
        )

        return {
            "context_ids": context_ids,
            "prompt_ids": prompt_ids,
            "target_ids": answer_ids,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length sequences with padding.

    Pads each field to the max length in the batch.
    Returns tensors with attention masks.
    """
    result = {}

    for key in batch[0].keys():
        sequences = [torch.tensor(item[key], dtype=torch.long) for item in batch]
        max_len = max(s.shape[0] for s in sequences)

        padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
        masks = torch.zeros(len(sequences), max_len, dtype=torch.long)

        for i, seq in enumerate(sequences):
            padded[i, :seq.shape[0]] = seq
            masks[i, :seq.shape[0]] = 1

        result[key] = padded
        result[key.replace("_ids", "_mask")] = masks

    return result


def collate_multi_chunk_fn(batch: list[dict]) -> dict:
    """Collate for PretrainMultiChunkDataset with variable chunk counts.

    Pads the chunk dimension to the max K in the batch.
    Padded chunks get all-zero masks so the model can ignore them.
    """
    chunk_len = len(batch[0]["chunk_ids"][0])
    max_K = max(item["num_chunks"] for item in batch)
    B = len(batch)

    chunk_ids = torch.zeros(B, max_K, chunk_len, dtype=torch.long)
    chunk_mask = torch.zeros(B, max_K, chunk_len, dtype=torch.long)

    for i, item in enumerate(batch):
        K_i = item["num_chunks"]
        for k in range(K_i):
            chunk_ids[i, k] = torch.tensor(item["chunk_ids"][k], dtype=torch.long)
            chunk_mask[i, k] = 1  # all positions valid for real chunks

    # target_ids: (B, cont_len) — fixed size, no padding needed
    target_ids = torch.from_numpy(
        np.stack([item["target_ids"] for item in batch])
    ).long()
    target_mask = torch.ones_like(target_ids)

    return {
        "chunk_ids": chunk_ids,      # (B, max_K, chunk_len)
        "chunk_mask": chunk_mask,     # (B, max_K, chunk_len)
        "target_ids": target_ids,     # (B, cont_len)
        "target_mask": target_mask,   # (B, cont_len)
    }


def create_pretrain_dataloader(
    data_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_context_len: int = 4096,
    max_cont_len: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
    max_samples: int | None = None,
    drop_last: bool = True,
) -> DataLoader:
    dataset = PretrainDataset(data_path, tokenizer, max_context_len, max_cont_len)
    if max_samples is not None and len(dataset) > max_samples:
        dataset = torch.utils.data.Subset(dataset, range(len(dataset) - max_samples, len(dataset)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=drop_last,
    )


def create_multi_chunk_dataloader(
    data_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_chunks: int = 8,
    min_chunks: int = 2,
    chunk_len: int = 512,
    cont_len: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    max_samples: int | None = None,
    drop_last: bool = True,
) -> DataLoader:
    dataset = PretrainMultiChunkDataset(
        data_path, tokenizer, max_chunks, min_chunks, chunk_len, cont_len
    )
    if max_samples is not None and len(dataset) > max_samples:
        dataset = torch.utils.data.Subset(
            dataset, range(len(dataset) - max_samples, len(dataset))
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_multi_chunk_fn,
        pin_memory=True,
        drop_last=drop_last,
    )


def create_qa_dataloader(
    data_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_context_len: int = 4096,
    max_prompt_len: int = 128,
    max_answer_len: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
    max_samples: int | None = None,
    drop_last: bool = True,
) -> DataLoader:
    dataset = QADataset(data_path, tokenizer, max_context_len, max_prompt_len, max_answer_len)
    if max_samples is not None and len(dataset) > max_samples:
        dataset = torch.utils.data.Subset(dataset, range(max_samples))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=drop_last,
    )
