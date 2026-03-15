"""Automatic batch size probing and gradient accumulation calculation.

Binary-searches for the maximum per-GPU batch size that fits in memory,
then computes gradient_accumulation_steps so that
    effective_batch_size = per_gpu_bs * num_gpus * accum_steps >= target_ebs.
"""

import logging
import math

import torch
import torch.nn as nn

from .config import QCPCConfig

logger = logging.getLogger(__name__)


def _make_dummy_batch(batch_size: int, config: QCPCConfig, stage: int, device: torch.device) -> dict:
    """Create a worst-case dummy batch (max sequence lengths) for OOM probing."""
    ctx_len = config.stage1_max_context_len if stage == 1 else config.stage2_max_context_len
    # Use token ID 1 (not 0 which is often pad) to avoid any special-case shortcuts
    batch = {
        "context_ids": torch.ones(batch_size, ctx_len, dtype=torch.long, device=device),
        "context_mask": torch.ones(batch_size, ctx_len, dtype=torch.long, device=device),
    }
    if stage == 1:
        tgt_len = config.stage1_max_cont_len
        batch["target_ids"] = torch.ones(batch_size, tgt_len, dtype=torch.long, device=device)
        batch["target_mask"] = torch.ones(batch_size, tgt_len, dtype=torch.long, device=device)
    else:
        prompt_len = config.stage2_max_prompt_len
        ans_len = config.stage2_max_answer_len
        batch["prompt_ids"] = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)
        batch["prompt_mask"] = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)
        batch["target_ids"] = torch.ones(batch_size, ans_len, dtype=torch.long, device=device)
        batch["target_mask"] = torch.ones(batch_size, ans_len, dtype=torch.long, device=device)
    return batch


def _try_batch(model: nn.Module, batch: dict) -> bool:
    """Try a forward + backward pass; return True if it fits in memory."""
    try:
        result = model(**batch)
        loss = result["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False


def find_max_batch_size(
    model: nn.Module,
    config: QCPCConfig,
    stage: int,
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
        # Clean up tensors
        del batch

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
