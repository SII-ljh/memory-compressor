"""QCPC model configuration."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class QCPCConfig:
    """Configuration for the Query-Conditioned Perceiver Compressor model."""

    # --- Mode switches ---
    use_decoupled_rope: bool = True
    use_prompt_bias: bool = True

    # --- Model dimensions (aligned with Qwen3-0.6B: hidden_size=1024) ---
    hidden_dim: int = 1024          # D: must match Qwen3 embedding dim
    num_heads: int = 16             # n_h: attention heads
    head_dim: int = 64              # d_h: per-head dimension (D / n_h)
    rope_dim: int = 64              # d_R: decoupled RoPE position channel dim
    num_memory_tokens: int = 128    # M: memory token count
    num_process_layers: int = 6     # L_proc: process self-attention layers
    query_mapper_mid_dim: int = 512 # D_mid: QueryMapper intermediate dim

    # --- Learnable PE (only for use_decoupled_rope=False) ---
    max_position_embeddings: int = 32768  # N_max for learnable PE

    # --- RoPE parameters (only for use_decoupled_rope=True) ---
    rope_theta: float = 1000000.0

    # --- FFN ---
    ffn_intermediate_dim: int = 2048  # SwiGLU FFN intermediate dim

    # --- Initialization ---
    init_scale: float = 0.02  # Truncated Gaussian scale

    # --- Qwen3 backbone ---
    qwen3_model_path: str = "./models/Qwen3-0.6B"
    vocab_size: int = 151936

    # --- Training: Stage 1 (pretrain) ---
    stage1_lr: float = 1e-4
    stage1_batch_size: int = 8
    stage1_max_epochs: int = 3
    stage1_max_context_len: int = 4096   # N: input context length
    stage1_max_cont_len: int = 256       # continuation length for NTP
    stage1_warmup_ratio: float = 0.05
    stage1_weight_decay: float = 0.01
    stage1_grad_clip: float = 1.0
    stage1_gradient_accumulation_steps: int = 4

    # --- Training: Stage 2 (QA finetune) ---
    stage2_lr: float = 5e-5
    stage2_batch_size: int = 4
    stage2_max_epochs: int = 5
    stage2_max_context_len: int = 4096
    stage2_max_prompt_len: int = 128     # L: prompt/question length
    stage2_max_answer_len: int = 256
    stage2_warmup_ratio: float = 0.05
    stage2_weight_decay: float = 0.01
    stage2_grad_clip: float = 1.0
    stage2_gradient_accumulation_steps: int = 4

    # --- Distributed ---
    use_fsdp: bool = True
    fsdp_sharding_strategy: str = "FULL_SHARD"

    # --- Data ---
    pretrain_data_path: str = "./data/processed/pretrain/train.jsonl"
    pretrain_eval_data_path: str = "./data/processed/pretrain/eval.jsonl"
    sft_train_data_path: str = "./data/processed/sft/train.json"
    sft_eval_data_path: str = "./data/processed/sft/eval.json"

    # --- Logging ---
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "./outputs"
    wandb_project: str = "qcpc"
    eval_samples: int = 200

    # --- Auto batch size ---
    auto_batch_size: bool = True               # Enable automatic batch size probing
    target_effective_batch_size: int = 256      # Target EBS = per_gpu_bs * num_gpus * accum_steps
    auto_batch_safety_margin: float = 0.85      # Safety factor applied to max found bs
    auto_batch_upper_bound: int = 128           # Binary search upper bound

    # --- Misc ---
    seed: int = 42
    num_workers: int = 4

    def save(self, path: str | Path) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load(cls, path: str | Path) -> "QCPCConfig":
        """Load config from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, d: dict) -> "QCPCConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
