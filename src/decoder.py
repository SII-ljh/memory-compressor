"""Decode stage: Memory Token + Prompt → frozen LLM Decoder."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import QCPCConfig


class FrozenDecoder(nn.Module):
    """Frozen Qwen3 LLM Decoder with memory token injection.

    Concatenates [<MEM>, H_1..H_M, </MEM>, prompt_embeds] and feeds
    to the frozen Qwen3 decoder for generation/loss computation.

    Special tokens <MEM> and </MEM> are added to mark memory boundaries.
    """

    def __init__(self, config: QCPCConfig):
        super().__init__()
        self.config = config

        # Load tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.qwen3_model_path, trust_remote_code=True
        )
        special_tokens = {"additional_special_tokens": ["<MEM>", "</MEM>"]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)

        # Load frozen LLM decoder
        self.lm = AutoModelForCausalLM.from_pretrained(
            config.qwen3_model_path, trust_remote_code=True
        )
        # Resize embeddings for new special tokens
        if num_added > 0:
            self.lm.resize_token_embeddings(len(self.tokenizer))

        # Freeze all LLM parameters
        for param in self.lm.parameters():
            param.requires_grad = False

        # Cache special token IDs
        self.mem_start_id = self.tokenizer.convert_tokens_to_ids("<MEM>")
        self.mem_end_id = self.tokenizer.convert_tokens_to_ids("</MEM>")

        # Get the frozen embedding layer for encoding special tokens
        self.embed_tokens = self.lm.get_input_embeddings()

    def _make_special_embed(self, token_id: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create embedding for a single special token, expanded for batch."""
        ids = torch.tensor([[token_id]], device=device)
        with torch.no_grad():
            emb = self.embed_tokens(ids)  # (1, 1, D)
        return emb.expand(batch_size, -1, -1)  # (B, 1, D)

    def forward(
        self,
        memory_tokens: torch.Tensor,
        prompt_ids: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        target_ids: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass through frozen decoder.

        Input sequence: [<MEM>, H_1..H_M, </MEM>, prompt_tokens, target_tokens]

        For training: provide target_ids, computes cross-entropy loss on target only.
        For inference: provide only prompt_ids/embeds, returns logits.

        Args:
            memory_tokens: (B, M, D) compressed memory from Perceiver IO
            prompt_ids: (B, L) tokenized prompt (question) IDs
            prompt_embeds: (B, L, D) prompt embeddings (alternative to prompt_ids)
            target_ids: (B, T) target token IDs for loss computation
            prompt_mask: (B, L) attention mask for prompt
            target_mask: (B, T) attention mask for target
            memory_mask: (B, M) attention mask for memory (0 = padded chunk memory)

        Returns:
            dict with 'loss' (if target_ids given), 'logits', 'input_length'
        """
        B = memory_tokens.shape[0]
        device = memory_tokens.device

        # Build embedding sequence piece by piece
        parts = []
        mask_parts = []

        # 1. <MEM> token
        mem_start = self._make_special_embed(self.mem_start_id, B, device)
        parts.append(mem_start)
        mask_parts.append(torch.ones(B, 1, device=device))

        # 2. Memory tokens H_1..H_M
        parts.append(memory_tokens)
        if memory_mask is not None:
            mask_parts.append(memory_mask.float())
        else:
            mask_parts.append(torch.ones(B, memory_tokens.shape[1], device=device))

        # 3. </MEM> token
        mem_end = self._make_special_embed(self.mem_end_id, B, device)
        parts.append(mem_end)
        mask_parts.append(torch.ones(B, 1, device=device))

        # 4. Prompt embeddings
        if prompt_embeds is not None:
            parts.append(prompt_embeds)
            if prompt_mask is not None:
                mask_parts.append(prompt_mask.float())
            else:
                mask_parts.append(torch.ones(B, prompt_embeds.shape[1], device=device))
        elif prompt_ids is not None:
            with torch.no_grad():
                p_emb = self.embed_tokens(prompt_ids)
            parts.append(p_emb)
            if prompt_mask is not None:
                mask_parts.append(prompt_mask.float())
            else:
                mask_parts.append(torch.ones(B, prompt_ids.shape[1], device=device))

        # Track where prompt ends (everything before target is input)
        input_length = sum(p.shape[1] for p in parts)

        # 5. Target tokens (for training)
        if target_ids is not None:
            with torch.no_grad():
                t_emb = self.embed_tokens(target_ids)
            parts.append(t_emb)
            if target_mask is not None:
                mask_parts.append(target_mask.float())
            else:
                mask_parts.append(torch.ones(B, target_ids.shape[1], device=device))

        # Concatenate all parts
        inputs_embeds = torch.cat(parts, dim=1)  # (B, total_len, D)
        attention_mask = torch.cat(mask_parts, dim=1)  # (B, total_len)

        # Forward through frozen LLM (using inputs_embeds, not input_ids)
        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits  # (B, total_len, vocab_size)

        result = {"logits": logits, "input_length": input_length}

        # Compute loss only on target tokens
        if target_ids is not None:
            # Shift: predict next token at each position
            # We want loss only on target positions
            # logits at positions [input_length-1 : input_length-1+T] predict target tokens
            target_logits = logits[:, input_length - 1 : input_length - 1 + target_ids.shape[1]]
            # target_ids are the ground truth tokens
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                target_logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
            )
            # Apply target mask if provided
            if target_mask is not None:
                loss = loss.view(B, -1) * target_mask.float()
                loss = loss.sum() / target_mask.float().sum().clamp(min=1)
            else:
                loss = loss.mean()
            result["loss"] = loss

        return result
