# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QCPC (Query-Conditioned Perceiver Compressor) compresses long texts into M memory tokens that serve as soft prompts for a frozen Qwen3 LLM decoder. Computation is O(M·N) instead of O(N²). Written in Python with PyTorch.

## Commands

### Testing
```bash
# Run all tests (requires Qwen3-0.6B model at ./models/Qwen3-0.6B)
pytest tests/

# Run a single test file
pytest tests/test_e2e.py

# Run a specific test
pytest tests/test_e2e.py::test_e2e_baseline_training
```

### Training
```bash
# Stage 1 (pretrain, both 1a warmup + 1b multi-chunk)
accelerate launch --num_processes 8 --multi_gpu src/train.py --config config/default.yaml --stage 1

# Stage 1a only (short window warmup)
accelerate launch --num_processes 8 --multi_gpu src/train.py --config config/default.yaml --stage 1a

# Stage 1b only (multi-chunk, loads 1a checkpoint)
accelerate launch --num_processes 8 --multi_gpu src/train.py --config config/default.yaml --stage 1b

# Stage 2 (QA finetune, loads stage 1b checkpoint)
accelerate launch --num_processes 8 --multi_gpu src/train.py --config config/default.yaml --stage 2 --resume outputs/stage1b/best.pt

# Single GPU
python src/train.py --config config/default.yaml --stage 1a

# Override config values from CLI
python src/train.py --config config/default.yaml --stage 1a --override auto_batch_size=false

# Baseline mode (no prompt bias)
python src/train.py --config config/default.yaml --stage 1a --override use_prompt_bias=false
```

### Inference
```bash
python src/inference.py --config config/default.yaml --checkpoint outputs/stage2/best.pt \
    --context "Long document text..." --question "What is...?"

# Batch inference
python src/inference.py --config config/default.yaml --checkpoint outputs/stage2/best.pt \
    --input_file data/stage2/eval.json --output_file outputs/predictions.json
```

## Architecture

### Five-Stage Pipeline
1. **Embedding** — Frozen Qwen3 embedding lookup, O(N)
2. **Latent Array** — M learnable tokens + optional prompt bias (zero-initialized α·B)
3. **Read** — Cross-attention: latents query text embeddings, O(M·N). Learnable PE injected at input level.
4. **Process** — L_proc self-attention layers among latents, O(M²)
5. **Decode** — `[<MEM>, memory, </MEM>, prompt]` injected into frozen Qwen3 LLM

### Two Operating Modes (`use_prompt_bias` switch)

| `use_prompt_bias` | Mode |
|:---:|:---|
| false | **Baseline**: learnable PE + learnable latent |
| true | **Prompt Bias**: learnable PE + query-conditioned latent bias |

### Two-Stage Training
- **Stage 1** (text completion pretrain): 1a short window warmup → 1b multi-chunk. QueryMapper and α frozen (output ≡ 0).
- **Stage 2** (QA finetune): Unfreezes QueryMapper and α. Zero-initialized prompt bias gradually activates.

### Key Source Files
- `src/model.py` — Main QCPC model, single/multi-chunk forward, stage management (`set_stage()`)
- `src/attention.py` — StandardAttention, AttentionBlock, SwiGLUFFN, RMSNorm
- `src/perceiver.py` — PerceiverIO: Read (cross-attn) + Process (self-attn) stages, learnable PE
- `src/latent.py` — LatentArray (Z_base + α·B), QueryMapper (2-layer MLP), zero-init
- `src/decoder.py` — FrozenDecoder wrapping Qwen3, special `<MEM>`/`</MEM>` tokens
- `src/embedding.py` — Frozen Qwen3 embedding layer
- `src/data.py` — Dataset classes (PretrainDataset, PretrainMultiChunkDataset, QADataset) and DataLoader factories
- `src/train.py` — Training script with auto batch size probing, multi-stage training, distributed support
- `src/config.py` — QCPCConfig dataclass, YAML loading, CLI override

## Conventions

- **No package manager**: no setup.py/pyproject.toml. Imports use `sys.path.insert(0, project_root)` then `from src.xxx import ...`.
- **Config system**: QCPCConfig dataclass → YAML file → CLI `--override key=value`. All hyperparameters live in `config/default.yaml`.
- **Frozen vs trainable**: Embedding and Decoder are always frozen. Only Perceiver components are trainable. Frozen components are excluded from FSDP sharding and optimizer.
- **Initialization**: All trainable params use truncated Gaussian (scale=0.02, clip to [-2σ, 2σ]). QueryMapper fc2 weights/bias and α are zero-initialized for smooth stage transition.
- **Mask convention**: 1=valid, 0=pad. Converted internally to True=ignore for `masked_fill`.
- **Design doc**: `plan.md` contains the full architecture specification (in Chinese).
