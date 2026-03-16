#!/usr/bin/env bash
# Stage 2: QA finetune for Qwen3-0.6B M=64
# Auto-detects stage1 checkpoint from output_dir/stage1/
set -euo pipefail

accelerate launch --num_processes 8 --multi_gpu \
  src/train.py --config config/default.yaml --stage 2 \
  --override \
    use_decoupled_rope=false \
    qwen3_model_path=./models/Qwen3-0.6B \
    hidden_dim=1024 \
    num_heads=16 \
    head_dim=64 \
    ffn_intermediate_dim=2048 \
    query_mapper_mid_dim=512 \
    num_memory_tokens=64 \
    stage2_max_epochs=5 \
    output_dir=./outputs/qwen06b_m64
