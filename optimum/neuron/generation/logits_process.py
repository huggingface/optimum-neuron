import torch
from transformers.generation import LogitsWarper


class FastTopKLogitsWarper(LogitsWarper):
    r"""Returns [batch_size, top_k] scores and indices instead of [batch_size, vocab_size] scores."""

    def __init__(self, top_k: int):
        self.top_k = top_k

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        return torch.topk(scores, top_k)
