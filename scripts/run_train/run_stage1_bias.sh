#!/usr/bin/env bash
# Stage 1 — Prompt Bias only (no decoupled RoPE)
set -euo pipefail

accelerate launch --num_processes 8 --multi_gpu \
  src/train.py --config config/default.yaml --stage 1 \
  --override \
    use_decoupled_rope=false \
    use_prompt_bias=true \
    output_dir=./outputs
