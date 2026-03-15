#!/usr/bin/env bash
# Ablation: Decoder=Qwen3-4B, Latent M=128 (baseline perceiver)
# Qwen3-4B: hidden_size=2560
# Perceiver: hidden=2560, heads=40, head_dim=64, ffn=5120
set -euo pipefail

accelerate launch --num_processes 8 --multi_gpu \
  src/train.py --config config/default.yaml --stage 1 \
  --override \
    use_decoupled_rope=false \
    use_prompt_bias=false \
    qwen3_model_path=./models/Qwen3-4B \
    hidden_dim=2560 \
    num_heads=40 \
    head_dim=64 \
    ffn_intermediate_dim=5120 \
    query_mapper_mid_dim=1280 \
    num_memory_tokens=128 \
    stage1_max_epochs=1 \
    output_dir=./outputs/qwen4b_m128
