"""Perceiver IO main body: Read (cross-attention) + Process (self-attention) stages."""

import torch
import torch.nn as nn

from .config import QCPCConfig
from .attention import AttentionBlock, RMSNorm
from .latent import LatentArray


class PerceiverIO(nn.Module):
    """Perceiver IO compressor: Read + Process stages.

    Read stage: Cross-attention where latent Z (M tokens) attends to
                long text embeddings E_X (N tokens). O(M*N).
    Process stage: L_proc layers of self-attention among M latent tokens. O(M^2).

    Supports two position encoding modes:
    - use_decoupled_rope=False: Learnable PE injected once at input level.
    - use_decoupled_rope=True: Decoupled RoPE applied at every layer.
    """

    def __init__(self, config: QCPCConfig):
        super().__init__()
        self.config = config
        D = config.hidden_dim

        # Latent array (Z_base + optional prompt bias)
        self.latent = LatentArray(config)

        # Learnable positional embeddings (Mode A only)
        self.use_decoupled_rope = config.use_decoupled_rope
        if not config.use_decoupled_rope:
            self.pe_text = nn.Parameter(
                torch.zeros(config.max_position_embeddings, D)
            )
            self.pe_latent = nn.Parameter(
                torch.zeros(config.num_memory_tokens, D)
            )
            nn.init.trunc_normal_(self.pe_text, std=config.init_scale, a=-2*config.init_scale, b=2*config.init_scale)
            nn.init.trunc_normal_(self.pe_latent, std=config.init_scale, a=-2*config.init_scale, b=2*config.init_scale)

        # Read stage: single cross-attention block
        self.read_block = AttentionBlock(config)

        # Process stage: L_proc self-attention blocks
        self.process_blocks = nn.ModuleList([
            AttentionBlock(config)
            for _ in range(config.num_process_layers)
        ])

        # Final norm
        self.final_norm = RMSNorm(D)

    def forward(
        self,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress long text into M memory tokens.

        Args:
            text_embeds: (B, N, D) long text embeddings from frozen Qwen3 embedding
            text_mask: (B, N) attention mask for text (1=valid, 0=pad)
            prompt_embeds: (B, L, D) prompt embeddings (for prompt bias, optional)
            prompt_mask: (B, L) attention mask for prompt (1=valid, 0=pad)

        Returns:
            memory_tokens: (B, M, D) compressed memory representations
        """
        B, N, D = text_embeds.shape

        # Initialize latent array
        Z = self.latent(
            batch_size=B,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
        )  # (B, M, D)

        # Add learnable PE (Mode A only)
        if not self.use_decoupled_rope:
            text_embeds = text_embeds + self.pe_text[:N]  # (B, N, D)
            Z = Z + self.pe_latent  # (B, M, D)

        # Convert mask for attention: 1=valid → True=ignore for masked positions
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = ~text_mask.bool()  # True=ignore

        # Read stage: cross-attention (latent queries attend to text)
        Z = self.read_block(Z, text_embeds, key_padding_mask=key_padding_mask)

        # Process stage: self-attention among latent tokens
        for block in self.process_blocks:
            Z = block(Z, Z)  # self-attention: no mask needed for latent

        # Final normalization
        Z = self.final_norm(Z)

        return Z
