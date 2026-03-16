#!/usr/bin/env bash
# Ablation: Latent length M=128, Decoder=Qwen3-0.6B (baseline perceiver)
# Perceiver: hidden=1024, heads=16, head_dim=64, ffn=2048
# This config is shared between decoder ablation and latent ablation.
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | xargs)}
[[ ${NUM_GPUS} -lt 1 ]] && NUM_GPUS=1

accelerate launch --num_processes "${NUM_GPUS}" --multi_gpu \
  src/train.py --config config/default.yaml --stage 1 \
  --override \
    use_decoupled_rope=false \
    use_prompt_bias=false \
    qwen3_model_path=./models/Qwen3-0.6B \
    hidden_dim=1024 \
    num_heads=16 \
    head_dim=64 \
    ffn_intermediate_dim=2048 \
    query_mapper_mid_dim=512 \
    num_memory_tokens=128 \
    stage1_max_epochs=3 \
    output_dir=./outputs/qwen06b_m128
