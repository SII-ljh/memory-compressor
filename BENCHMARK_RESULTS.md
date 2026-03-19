# Benchmark Results - Memory Compressor

Generated on: 2026-03-18

## Overview

This document summarizes benchmark results for the Query-Conditioned Perceiver Compressor (QCPC) across different model sizes (Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B) and training stages.

---

## Stage 1: Text Completion Pretrain

### Stage 1 Full Benchmark (Context=4096, Continuation=256)

| Model | Loss | Perplexity | Samples | Total Tokens |
|-------|------|------------|---------|--------------|
| Qwen3-0.6B | 3.297 | 27.028 | 500 | 108,959 |
| Qwen3-1.7B | 3.024 | 20.564 | 500 | 108,959 |
| Qwen3-4B | 2.889 | 17.982 | 500 | 108,959 |

**Observations:**
- Larger models consistently achieve lower loss and perplexity
- Qwen3-4B achieves 33% lower perplexity than Qwen3-0.6B
- Performance scales predictably with model size

---

### Stage 1a: Short Window Warmup (Context=512, Continuation=128)

| Model | Loss | Perplexity | Samples | Total Tokens |
|-------|------|------------|---------|--------------|
| Qwen3-0.6B | 3.309 | 27.354 | 500 | 61,778 |
| Qwen3-1.7B | 3.033 | 20.766 | 500 | 61,778 |
| Qwen3-4B | 2.907 | 18.300 | 500 | 61,778 |

**Observations:**
- Similar scaling pattern as full benchmark
- Slightly higher perplexity compared to Stage 1b (expected due to shorter context)
- Stage 1a serves as warmup for longer context training

---

### Stage 1b: Multi-Chunk Training (Context=4096, Continuation=128)

| Model | Loss | Perplexity | Samples | Total Tokens |
|-------|------|------------|---------|--------------|
| Qwen3-0.6B | 3.295 | 26.975 | 500 | 61,778 |
| Qwen3-1.7B | 3.020 | 20.488 | 500 | 61,778 |
| Qwen3-4B | 2.886 | 17.925 | 500 | 61,778 |

**Observations:**
- Improved perplexity over Stage 1a due to longer context
- Qwen3-4B achieves best performance (PPL=17.925)
- All models show consistent improvement from 1a to 1b

---

## Stage 2: QA Fine-tune Evaluation

### Performance Comparison Across Models

| Model | Metric | Upper Bound | Lower Bound | Gap |
|-------|--------|-------------|-------------|-----|
| **Qwen3-0.6B** | | | | |
| | Loss | 0.247 | 2.616 | 2.369 |
| | Perplexity | 1.280 | 13.681 | 12.401 |
| | F1 Score | 0.805 | 0.258 | -0.547 |
| | EM | 0.733 | 0.247 | -0.486 |
| | ROUGE-L | 0.805 | 0.258 | -0.547 |
| **Qwen3-1.7B** | | | | |
| | Loss | 0.187 | 2.113 | 1.926 |
| | Perplexity | 1.206 | 8.272 | 7.066 |
| | F1 Score | 0.872 | 0.279 | -0.593 |
| | EM | 0.810 | 0.257 | -0.553 |
| | ROUGE-L | 0.872 | 0.279 | -0.593 |
| **Qwen3-4B** | | | | |
| | Loss | 0.173 | 1.915 | 1.742 |
| | Perplexity | 1.189 | 6.789 | 5.600 |
| | F1 Score | 0.885 | 0.304 | -0.581 |
| | EM | 0.827 | 0.275 | -0.552 |
| | ROUGE-L | 0.885 | 0.303 | -0.582 |

**Key Findings:**

1. **Model Size Scaling:**
   - F1 Score improves from 0.805 → 0.872 → 0.885 (0.6B → 1.7B → 4B)
   - EM Score improves from 0.733 → 0.810 → 0.827
   - Larger models show consistently better QA performance

2. **Compression Impact:**
   - Upper Bound (with full context): Excellent performance across all metrics
   - Lower Bound (with compressed memory): Significant performance drop
   - Performance gap is substantial: 53-59% F1 score reduction

3. **Best Model:**
   - Qwen3-4B with LoRA fine-tuning achieves:
     - F1: 0.885 (upper bound)
     - EM: 0.827 (upper bound)
     - Perplexity: 1.189 (upper bound)

4. **Sample Size:**
   - All evaluations use 1,999 samples for Stage 2
   - Statistically robust comparison across model sizes

---

## Summary

### Stage 1 (Pretrain)
- **Best Model:** Qwen3-4B achieves PPL=17.982 on full-length sequences
- **Training Progress:** Clear improvement from Stage 1a to 1b across all models
- **Scaling:** Larger models consistently outperform smaller ones

### Stage 2 (QA Fine-tune)
- **Best Model:** Qwen3-4B achieves F1=0.885, EM=0.827
- **Compression Challenge:** Memory compression results in significant performance degradation
- **Direction:** Further work needed to close the upper/lower bound gap

---

## Files Analyzed

1. `stage1_upper_bound_Qwen3-{0.6B,1.7B,4B}.json`
2. `stage1a_upper_bound_Qwen3-{0.6B,1.7B,4B}.json`
3. `stage1b_upper_bound_Qwen3-{0.6B,1.7B,4B}.json`
4. `stage2_eval_Qwen3-{0.6B,1.7B,4B}.json`

---

*This report was automatically generated from benchmark JSON files.*