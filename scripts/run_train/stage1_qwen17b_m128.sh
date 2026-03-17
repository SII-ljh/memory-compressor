#!/usr/bin/env bash
# Ablation: Decoder=Qwen3-1.7B, Latent M=128 (baseline perceiver)
# Qwen3-1.7B: hidden_size=2048
# Perceiver: hidden=2048, heads=32, head_dim=64, ffn=4096
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | xargs)}
[[ ${NUM_GPUS} -lt 1 ]] && NUM_GPUS=1

accelerate launch --num_processes "${NUM_GPUS}" --multi_gpu \
  src/train.py --config config/default.yaml --stage 1 \
  --override \
    use_decoupled_rope=false \
    use_prompt_bias=false \
    qwen3_model_path=./models/Qwen3-1.7B \
    hidden_dim=2048 \
    num_heads=32 \
    head_dim=64 \
    ffn_intermediate_dim=4096 \
    query_mapper_mid_dim=1024 \
    num_memory_tokens=128 \
    stage1a_max_epochs=3 \
    stage1b_max_epochs=1 \
    output_dir=./outputs/qwen17b_m128
