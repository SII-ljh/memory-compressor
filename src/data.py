"""Data loaders for Stage 1 (pretrain) and Stage 2 (QA finetune)."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class PretrainDataset(Dataset):
    """Stage 1: Text completion pretraining dataset.

    Reads JSONL with {"text": "...", "id": "..."} format.
    Splits each text into prefix (context for compression) and continuation (target).
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_context_len: int = 4096,
        max_cont_len: int = 256,
    ):
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

        # Tokenize full text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Split: first part = context (for compression), rest = continuation (target)
        total_max = self.max_context_len + self.max_cont_len
        if len(tokens) > total_max:
            # Take a window that fits
            tokens = tokens[:total_max]

        # Need at least max_cont_len tokens for a valid split
        if len(tokens) < self.max_cont_len + 10:
            # Too short, pad context minimally
            split_point = max(1, len(tokens) - self.max_cont_len)
        else:
            split_point = min(len(tokens) - self.max_cont_len, self.max_context_len)

        context_ids = tokens[:split_point]
        cont_ids = tokens[split_point:split_point + self.max_cont_len]

        return {
            "context_ids": context_ids,
            "target_ids": cont_ids,
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


def create_pretrain_dataloader(
    data_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_context_len: int = 4096,
    max_cont_len: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = PretrainDataset(data_path, tokenizer, max_context_len, max_cont_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
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
) -> DataLoader:
    dataset = QADataset(data_path, tokenizer, max_context_len, max_prompt_len, max_answer_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
