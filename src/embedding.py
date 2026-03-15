"""Frozen Qwen3 Embedding Lookup layer."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .config import QCPCConfig


class FrozenEmbedding(nn.Module):
    """Frozen embedding layer that reuses Qwen3's embedding table.

    Loads only the embedding weights from a pretrained Qwen3 model
    and freezes all parameters. Complexity: O(N).
    """

    def __init__(self, config: QCPCConfig):
        super().__init__()
        self.config = config

        # Load only the Qwen3 config to get vocab size and embedding dim
        qwen_config = AutoConfig.from_pretrained(
            config.qwen3_model_path, trust_remote_code=True
        )

        # Create embedding table matching Qwen3
        self.embed_tokens = nn.Embedding(
            qwen_config.vocab_size, qwen_config.hidden_size
        )

        # Load pretrained weights
        qwen_model = AutoModel.from_pretrained(
            config.qwen3_model_path, trust_remote_code=True
        )
        self.embed_tokens.load_state_dict(
            qwen_model.embed_tokens.state_dict()
        )
        del qwen_model  # free memory

        # Freeze all parameters
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Embed input token IDs.

        Args:
            input_ids: (batch_size, seq_len) token IDs

        Returns:
            embeddings: (batch_size, seq_len, hidden_dim)
        """
        return self.embed_tokens(input_ids)
