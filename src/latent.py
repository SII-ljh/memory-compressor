"""Latent Array initialization with optional Prompt Bias."""

import torch
import torch.nn as nn

from .config import QCPCConfig


def truncated_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02):
    """Initialize tensor with truncated normal distribution [-2*std, 2*std]."""
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(mean - 2 * std, mean + 2 * std)
    return tensor


class QueryMapper(nn.Module):
    """Two-layer MLP that maps prompt embedding to latent bias.

    D -> D_mid -> GELU -> D_mid -> M*D
    Second layer is zero-initialized for smooth training start.
    """

    def __init__(self, config: QCPCConfig):
        super().__init__()
        D = config.hidden_dim
        D_mid = config.query_mapper_mid_dim
        M = config.num_memory_tokens

        self.fc1 = nn.Linear(D, D_mid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(D_mid, M * D)

        self.M = M
        self.D = D

        # Initialize fc1 normally
        truncated_normal_(self.fc1.weight, std=config.init_scale)
        nn.init.zeros_(self.fc1.bias)

        # Zero-initialize fc2 (output layer) for smooth start
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, prompt_embed: torch.Tensor) -> torch.Tensor:
        """Map mean-pooled prompt embedding to latent bias.

        Args:
            prompt_embed: (batch_size, D) mean-pooled prompt vector

        Returns:
            bias: (batch_size, M, D) latent bias matrix
        """
        x = self.fc1(prompt_embed)   # (B, D_mid)
        x = self.act(x)              # (B, D_mid)
        x = self.fc2(x)              # (B, M*D)
        return x.view(-1, self.M, self.D)  # (B, M, D)


class LatentArray(nn.Module):
    """Latent Array with optional Prompt Bias.

    Z = Z_base                         (if use_prompt_bias=False)
    Z = Z_base + alpha * B             (if use_prompt_bias=True)

    where B = QueryMapper(mean_pool(E_P)) and alpha is a learnable
    scalar initialized to 0.
    """

    def __init__(self, config: QCPCConfig):
        super().__init__()
        self.config = config
        M = config.num_memory_tokens
        D = config.hidden_dim

        # Z_base: learnable latent array with truncated Gaussian init
        self.z_base = nn.Parameter(torch.empty(M, D))
        truncated_normal_(self.z_base, std=config.init_scale)

        # Prompt Bias pathway (only created if config says so)
        self.use_prompt_bias = config.use_prompt_bias
        if self.use_prompt_bias:
            self.query_mapper = QueryMapper(config)
            # Gate alpha: scalar initialized to 0
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        batch_size: int,
        prompt_embeds: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute latent array Z for a batch.

        Args:
            batch_size: number of samples in the batch
            prompt_embeds: (batch_size, L, D) prompt embeddings (required if use_prompt_bias=True)
            prompt_mask: (batch_size, L) attention mask for prompt (1=valid, 0=pad)

        Returns:
            Z: (batch_size, M, D) latent array
        """
        # Expand Z_base for the batch
        Z = self.z_base.unsqueeze(0).expand(batch_size, -1, -1)  # (B, M, D)

        if self.use_prompt_bias and prompt_embeds is not None:
            # Mean pooling over prompt sequence (respecting mask)
            if prompt_mask is not None:
                mask_expanded = prompt_mask.unsqueeze(-1).float()  # (B, L, 1)
                pooled = (prompt_embeds * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = prompt_embeds.mean(dim=1)  # (B, D)

            B = self.query_mapper(pooled)  # (B, M, D)
            Z = Z + self.alpha * B

        return Z
