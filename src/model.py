"""QCPC: Query-Conditioned Perceiver Compressor — full model."""

import torch
import torch.nn as nn

from .config import QCPCConfig
from .embedding import FrozenEmbedding
from .perceiver import PerceiverIO
from .decoder import FrozenDecoder


class QCPC(nn.Module):
    """Query-Conditioned Perceiver Compressor.

    Full pipeline:
    1. Frozen Embedding: token IDs → embeddings O(N)
    2. Perceiver IO: compress N embeddings → M memory tokens O(M*N)
    3. Frozen Decoder: [<MEM>, memory, </MEM>, prompt] → LLM generation

    Two boolean switches control 4 operating modes:
    - use_decoupled_rope: standard learnable PE vs decoupled RoPE
    - use_prompt_bias: whether to inject query-conditioned latent bias
    """

    def __init__(self, config: QCPCConfig):
        super().__init__()
        self.config = config

        # Frozen embedding (shared for text and prompt)
        self.embedding = FrozenEmbedding(config)

        # Perceiver IO compressor (trainable)
        self.perceiver = PerceiverIO(config)

        # Frozen LLM decoder
        self.decoder = FrozenDecoder(config)

    def forward(
        self,
        context_ids: torch.LongTensor,
        context_mask: torch.Tensor | None = None,
        prompt_ids: torch.LongTensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        target_ids: torch.LongTensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> dict:
        """Full forward pass.

        Args:
            context_ids: (B, N) long context token IDs
            context_mask: (B, N) attention mask (1=valid, 0=pad)
            prompt_ids: (B, L) prompt/question token IDs (optional)
            prompt_mask: (B, L) prompt mask
            target_ids: (B, T) target token IDs for loss (optional, training only)
            target_mask: (B, T) target mask

        Returns:
            dict with 'loss' (if training), 'logits', 'memory_tokens'
        """
        # 1. Embed context
        context_embeds = self.embedding(context_ids)  # (B, N, D)

        # 2. Embed prompt (for bias and decoder)
        prompt_embeds = None
        if prompt_ids is not None:
            prompt_embeds = self.embedding(prompt_ids)  # (B, L, D)

        # 3. Perceiver IO compression
        memory_tokens = self.perceiver(
            text_embeds=context_embeds,
            text_mask=context_mask,
            prompt_embeds=prompt_embeds if self.config.use_prompt_bias else None,
            prompt_mask=prompt_mask if self.config.use_prompt_bias else None,
        )  # (B, M, D)

        # 4. Decode: [<MEM>, memory, </MEM>, prompt] → LLM
        decoder_result = self.decoder(
            memory_tokens=memory_tokens,
            prompt_ids=prompt_ids,
            prompt_embeds=None,  # use IDs for decoder prompt
            target_ids=target_ids,
            prompt_mask=prompt_mask,
            target_mask=target_mask,
        )

        decoder_result["memory_tokens"] = memory_tokens
        return decoder_result

    def get_trainable_params(self, stage: int = 1) -> list[nn.Parameter]:
        """Get trainable parameters for a given training stage.

        Stage 1 (pretrain): Perceiver IO + Z_base (no prompt bias)
        Stage 2 (QA finetune): Perceiver IO + Z_base + QueryMapper + alpha
        """
        params = []

        # Perceiver IO parameters (always trainable)
        for name, param in self.perceiver.named_parameters():
            if stage == 1:
                # Stage 1: skip QueryMapper and alpha
                if "query_mapper" in name or "alpha" in name:
                    param.requires_grad = False
                    continue
            param.requires_grad = True
            params.append(param)

        return params

    def set_stage(self, stage: int) -> None:
        """Configure model for training stage.

        Stage 1: Freeze prompt bias pathway, train perceiver only.
        Stage 2: Unfreeze prompt bias pathway.
        """
        # First freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # Then selectively unfreeze
        for name, param in self.perceiver.named_parameters():
            if stage == 1:
                if "query_mapper" in name or "alpha" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif stage == 2:
                param.requires_grad = True

    def count_params(self) -> dict:
        """Count parameters by component."""
        def _count(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        emb_t, emb_tr = _count(self.embedding)
        perc_t, perc_tr = _count(self.perceiver)
        dec_t, dec_tr = _count(self.decoder)

        return {
            "embedding": {"total": emb_t, "trainable": emb_tr},
            "perceiver": {"total": perc_t, "trainable": perc_tr},
            "decoder": {"total": dec_t, "trainable": dec_tr},
            "total": emb_t + perc_t + dec_t,
            "total_trainable": emb_tr + perc_tr + dec_tr,
        }
