#!/usr/bin/env bash
# Stage 2: QA finetune for Qwen3-1.7B M=128
# Auto-detects stage1 checkpoint from output_dir (searches stage1b/ → stage1a/ → stage1/)
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | xargs)}
[[ ${NUM_GPUS} -lt 1 ]] && NUM_GPUS=1

accelerate launch --num_processes "${NUM_GPUS}" --multi_gpu \
  src/train.py --config config/default.yaml --stage 2 \
  --override \
    qwen3_model_path=./models/Qwen3-1.7B \
    hidden_dim=2048 \
    num_heads=32 \
    head_dim=64 \
    ffn_intermediate_dim=4096 \
    query_mapper_mid_dim=1024 \
    num_memory_tokens=128 \
    stage2_max_epochs=1 \
    output_dir=./outputs/qwen17b_m128
