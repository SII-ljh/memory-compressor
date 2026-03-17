#!/usr/bin/env bash
# Stage 1a only: Decoder=Qwen3-4B, Latent M=128 (baseline perceiver)
# Short window warmup: 512 ctx → Perceiver → M tokens, 128 continuation
# Qwen3-4B: hidden_size=2560
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | xargs)}
[[ ${NUM_GPUS} -lt 1 ]] && NUM_GPUS=1

if [[ ${NUM_GPUS} -eq 1 ]]; then
  python src/train.py --config config/default.yaml --stage 1a \
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
      stage1a_max_epochs=1 \
      output_dir=./outputs/qwen4b_m128
else
  accelerate launch --num_processes "${NUM_GPUS}" --multi_gpu \
    --main_process_port "${MASTER_PORT:-29500}" \
    src/train.py --config config/default.yaml --stage 1a \
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
      stage1a_max_epochs=1 \
      output_dir=./outputs/qwen4b_m128
fi
