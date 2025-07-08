from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import GenerationConfig


@dataclass
class FusedLogitsWarper:
    """
    A class that performs top-k then top-p filtering, optionally applying a temperature.

    Top-k filtering only keeps the `k` tokens with the best scores.

    Top-p filtering only keeps the top tokens whose cumulated probability is above `p`.

    The filtered tokens are returned as a list of indices, along with the corresponding subset of
    the original logits.

    If only top-k filtering is active, the filtered tokens are sorted by descending order.

    If top-p filtering is active, the filtered tokens are sorted by ascending order.

    Args:
        temperature (`float`):
            Strictly positive float value used to modulate the logits distribution. A value smaller than `1` decreases
            randomness (and vice versa), with `0` being equivalent to shifting all probability mass to the most likely
            token.
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    @classmethod
    def from_config(cls, generation_config: GenerationConfig) -> "FusedLogitsWarper":
        """Instantiate a fused warper from a generation configuration.

        Args:
            generation_config (`~transformers.generation.GenerationConfig`):
                The generation configuration to be used as base parametrization for the fused warper.

        Returns:
            a `FusedLogitsWarper` or None if neither top-k nor top-p are configured.
        """
        if generation_config.do_sample and generation_config.top_k == 0 and generation_config.top_p == 1.0:
            raise ValueError("Multinomial sampling requires at least top-k or top-p to be specified.")
        return cls(generation_config.temperature, generation_config.top_k, generation_config.top_p)

    def __call__(self, logits: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.LongTensor]:
        if self.temperature != 1.0:
            logits = logits / self.temperature

        do_top_k = self.top_k > 0 and self.top_k < logits.shape[-1]
        do_top_p = self.top_p < 1.0 and self.top_p > 0.0

        if not do_top_k and not do_top_p:
            return logits, None

        if do_top_k:
            sorted_logits, sorted_indices = torch.topk(logits, self.top_k)
        else:
            # Warning: not applying top-k filtering leads to this very slow sort operation
            sorted_logits, sorted_indices = torch.sort(logits)

        if do_top_p:
            if do_top_k:
                # logits have been sorted in descending order, so we need to flip them
                sorted_logits = torch.flip(sorted_logits, [-1])
                sorted_indices = torch.flip(sorted_indices, [-1])
            # We always keep the best logits and those whose cumulative probability is strictly higher than top_p
            cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            keep_mask = cum_probs > (1 - self.top_p)
            keep_mask[:, -1] = True
            # Set rejected logits to -inf so that they are ignored in downstream comparisons
            sorted_logits[~keep_mask] = float("-Inf")
            # Clip the [batch_size, vocab_size] logits tensor to speed-up downstream ops
            keep_by_batch = torch.sum(keep_mask, dim=-1)
            keep = torch.amax(keep_by_batch)
            sorted_logits = sorted_logits[:, -keep:]
            sorted_indices = sorted_indices[:, -keep:]

        return sorted_logits, sorted_indices
