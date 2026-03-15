"""Inference script for QCPC: compress long text and generate answers.

Supports multi-GPU via Accelerate.

Usage:
    # Single GPU inference
    python src/inference.py --config config/default.yaml --checkpoint outputs/stage2/best.pt \
        --context "Long document text..." --question "What is...?"

    # Multi-GPU inference
    accelerate launch src/inference.py --config config/default.yaml --checkpoint outputs/stage2/best.pt \
        --input_file data/stage2/eval.json --output_file outputs/predictions.json

    # Batch inference from file
    python src/inference.py --config config/default.yaml --checkpoint outputs/stage2/best.pt \
        --input_file data/stage2/eval.json --output_file outputs/predictions.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QCPCConfig
from src.model import QCPC

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="QCPC Inference")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--context", type=str, default=None, help="Context text (for single inference)")
    parser.add_argument("--question", type=str, default=None, help="Question (for single inference)")
    parser.add_argument("--input_file", type=str, default=None, help="JSON file with list of {context, question}")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


class QCPCInference:
    """QCPC inference wrapper."""

    def __init__(
        self,
        config: QCPCConfig,
        checkpoint_path: str,
        device: torch.device | None = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = QCPC(config)

        # Load perceiver checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.perceiver.load_state_dict(ckpt["model"], strict=False)

        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = self.model.decoder.tokenizer

    @torch.no_grad()
    def compress(
        self,
        context: str,
        question: str | None = None,
        max_context_len: int | None = None,
        max_prompt_len: int | None = None,
    ) -> torch.Tensor:
        """Compress context into memory tokens.

        Args:
            context: Long text to compress
            question: Optional question for query-conditioned compression
            max_context_len: Override max context length
            max_prompt_len: Override max prompt length

        Returns:
            memory_tokens: (1, M, D) compressed representation
        """
        max_ctx = max_context_len or self.config.stage2_max_context_len
        max_pmt = max_prompt_len or self.config.stage2_max_prompt_len

        # Tokenize context
        ctx_ids = self.tokenizer.encode(
            context, add_special_tokens=False, max_length=max_ctx, truncation=True
        )
        ctx_tensor = torch.tensor([ctx_ids], device=self.device)

        # Tokenize question (optional)
        prompt_ids = None
        if question and self.config.use_prompt_bias:
            pmt_ids = self.tokenizer.encode(
                question, add_special_tokens=False, max_length=max_pmt, truncation=True
            )
            prompt_ids = torch.tensor([pmt_ids], device=self.device)

        # Embed
        ctx_embeds = self.model.embedding(ctx_tensor)
        prompt_embeds = None
        if prompt_ids is not None:
            prompt_embeds = self.model.embedding(prompt_ids)

        # Compress
        memory_tokens = self.model.perceiver(
            text_embeds=ctx_embeds,
            prompt_embeds=prompt_embeds if self.config.use_prompt_bias else None,
        )

        return memory_tokens

    @torch.no_grad()
    def generate(
        self,
        context: str,
        question: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> str:
        """Compress context and generate answer.

        Args:
            context: Long text to compress
            question: Optional question
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold

        Returns:
            Generated text string
        """
        # Compress context
        memory_tokens = self.compress(context, question)

        # Prepare prompt IDs for decoder
        prompt_ids = None
        if question:
            pmt_ids = self.tokenizer.encode(
                question, add_special_tokens=False,
                max_length=self.config.stage2_max_prompt_len, truncation=True
            )
            prompt_ids = torch.tensor([pmt_ids], device=self.device)

        # Build initial input: [<MEM>, memory, </MEM>, prompt]
        B = 1
        parts = []

        # <MEM>
        mem_start = self.model.decoder._make_special_embed(
            self.model.decoder.mem_start_id, B, self.device
        )
        parts.append(mem_start)

        # Memory tokens
        parts.append(memory_tokens)

        # </MEM>
        mem_end = self.model.decoder._make_special_embed(
            self.model.decoder.mem_end_id, B, self.device
        )
        parts.append(mem_end)

        # Prompt
        if prompt_ids is not None:
            p_emb = self.model.decoder.embed_tokens(prompt_ids)
            parts.append(p_emb)

        inputs_embeds = torch.cat(parts, dim=1)  # (1, seq_len, D)

        # Autoregressive generation
        generated_ids = []
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is None:
                outputs = self.model.decoder.lm(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                )
            else:
                # Only feed the last generated token
                last_embed = self.model.decoder.embed_tokens(
                    torch.tensor([[generated_ids[-1]]], device=self.device)
                )
                outputs = self.model.decoder.lm(
                    inputs_embeds=last_embed,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # (1, vocab)

            # Sampling
            if temperature <= 0:
                # Greedy
                next_token = logits.argmax(dim=-1).item()
            else:
                logits = logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[mask] = float("-inf")
                    logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            generated_ids.append(next_token)

            # Stop on EOS
            if next_token == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(
        self,
        contexts: list[str],
        questions: list[str | None],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> list[str]:
        """Generate answers for a batch of context-question pairs."""
        results = []
        for ctx, q in zip(contexts, questions):
            result = self.generate(ctx, q, max_new_tokens, temperature, top_p)
            results.append(result)
        return results


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = QCPCConfig.load(args.config)

    # Initialize inference
    inferencer = QCPCInference(config, args.checkpoint)
    logger.info("Model loaded successfully")

    if args.context:
        # Single inference
        answer = inferencer.generate(
            context=args.context,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\nAnswer: {answer}")

    elif args.input_file:
        # Batch inference from file
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []
        for i, item in enumerate(data):
            context = item.get("context", "")
            question = item.get("question", None)

            answer = inferencer.generate(
                context=context,
                question=question,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            results.append({
                "context": context[:200] + "..." if len(context) > 200 else context,
                "question": question,
                "prediction": answer,
                "reference": item.get("answer", ""),
            })

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(data)} samples")

        # Save results
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_path}")
        else:
            for r in results[:5]:
                print(f"\nQ: {r['question']}")
                print(f"Pred: {r['prediction']}")
                print(f"Ref:  {r['reference']}")
    else:
        print("Please provide --context or --input_file")
        sys.exit(1)


if __name__ == "__main__":
    main()
