"""Automatic batch size probing and gradient accumulation calculation.

Binary-searches for the maximum per-GPU batch size that fits in memory,
then computes gradient_accumulation_steps so that
    effective_batch_size = per_gpu_bs * num_gpus * accum_steps >= target_ebs.
"""

import gc
import logging
import math
import os

# Avoid CUDA memory fragmentation from repeated OOM/free during probing
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

from .config import QCPCConfig

logger = logging.getLogger(__name__)


def _make_dummy_batch(batch_size: int, config: QCPCConfig, stage: str, device: torch.device) -> dict:
    """Create a worst-case dummy batch (max sequence lengths) for OOM probing.

    Args:
        stage: "1a" (warmup), "1b" (multi-chunk), or "2" (QA finetune).
    """
    # Use token ID 1 (not 0 which is often pad) to avoid any special-case shortcuts
    if stage == "1a":
        ctx_len = config.stage1a_max_context_len
        tgt_len = config.stage1a_max_cont_len
        batch = {
            "context_ids": torch.ones(batch_size, ctx_len, dtype=torch.long, device=device),
            "context_mask": torch.ones(batch_size, ctx_len, dtype=torch.long, device=device),
            "target_ids": torch.ones(batch_size, tgt_len, dtype=torch.long, device=device),
            "target_mask": torch.ones(batch_size, tgt_len, dtype=torch.long, device=device),
        }
    elif stage == "1b":
        K = config.stage1b_max_chunks  # worst case for OOM probing
        N = config.stage1b_chunk_len
        tgt_len = config.stage1b_max_cont_len
        batch = {
            "chunk_ids": torch.ones(batch_size, K, N, dtype=torch.long, device=device),
            "chunk_mask": torch.ones(batch_size, K, N, dtype=torch.long, device=device),
            "target_ids": torch.ones(batch_size, tgt_len, dtype=torch.long, device=device),
            "target_mask": torch.ones(batch_size, tgt_len, dtype=torch.long, device=device),
        }
    else:  # stage == "2"
        ctx_len = config.stage2_max_context_len
        prompt_len = config.stage2_max_prompt_len
        ans_len = config.stage2_max_answer_len
        batch = {
            "context_ids": torch.ones(batch_size, ctx_len, dtype=torch.long, device=device),
            "context_mask": torch.ones(batch_size, ctx_len, dtype=torch.long, device=device),
            "prompt_ids": torch.ones(batch_size, prompt_len, dtype=torch.long, device=device),
            "prompt_mask": torch.ones(batch_size, prompt_len, dtype=torch.long, device=device),
            "target_ids": torch.ones(batch_size, ans_len, dtype=torch.long, device=device),
            "target_mask": torch.ones(batch_size, ans_len, dtype=torch.long, device=device),
        }
    return batch


def _try_batch(model: nn.Module, batch: dict) -> bool:
    """Try a forward + backward pass; return True if it fits in memory."""
    result = None
    try:
        result = model(**batch)
        loss = result["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)
        return True
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        return False
    finally:
        del result
        gc.collect()
        torch.cuda.empty_cache()


def find_max_batch_size(
    model: nn.Module,
    config: QCPCConfig,
    stage: str,
    device: torch.device,
) -> int:
    """Binary search for max per-GPU batch size that fits in VRAM.

    The model must already be on `device` with the correct stage set.
    Returns the safe batch size (max_found * safety_margin, at least 1).

    In multi-GPU scenarios, the caller should run this only on rank 0
    and broadcast the result.
    """
    lo, hi = 1, config.auto_batch_upper_bound
    best = 1

    logger.info(f"Auto batch probe: searching [{lo}, {hi}] on {device}")

    model.train()
    while lo <= hi:
        mid = (lo + hi) // 2
        logger.info(f"  trying batch_size={mid} ...")
        batch = _make_dummy_batch(mid, config, stage, device)
        if _try_batch(model, batch):
            best = mid
            lo = mid + 1
            logger.info(f"    OK (best so far: {best})")
        else:
            hi = mid - 1
            logger.info(f"    OOM (shrinking upper bound to {hi})")
        del batch
        gc.collect()
        torch.cuda.empty_cache()

    safe_bs = max(1, int(best * config.auto_batch_safety_margin))
    logger.info(f"Auto batch probe done: max={best}, safe={safe_bs} "
                f"(margin={config.auto_batch_safety_margin})")
    return safe_bs


def compute_accumulation_steps(
    per_gpu_batch_size: int,
    num_gpus: int,
    target_ebs: int,
) -> tuple[int, int]:
    """Compute gradient accumulation steps to reach target effective batch size.

    Args:
        per_gpu_batch_size: batch size per GPU
        num_gpus: number of GPUs
        target_ebs: target effective batch size

    Returns:
        (accumulation_steps, actual_effective_batch_size)
    """
    if per_gpu_batch_size <= 0 or num_gpus <= 0 or target_ebs <= 0:
        raise ValueError(
            f"All arguments must be positive: per_gpu_bs={per_gpu_batch_size}, "
            f"num_gpus={num_gpus}, target_ebs={target_ebs}"
        )
    per_step_total = per_gpu_batch_size * num_gpus
    accum_steps = math.ceil(target_ebs / per_step_total)
    actual_ebs = per_step_total * accum_steps
    return accum_steps, actual_ebs
