import logging
from typing import Optional

import torch
from transformers.generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.generation.utils import GenerationMode

from .logits_process import FusedLogitsWarper


logger = logging.getLogger(__name__)


class TokenSelector:
    """Implements the token selection logic corresponding to a generation configuration.

    This class combines and uses the logits processors and stopping criterias implemented in
    the transformers library.

    The algorithm to select these objects is heavily inspired by the transformers `GenerationMixin.generate()`
    method, but the actual token selection methods are specific.

    The reason why this class does not inherit from `GenerationMixin` is because it does not
    include the code to produce the tokens logits.
    Separating the production of the tokens logits from the tokens selection allows this class
    to be used with different generation paradigms, either synchronously using a single `TokenSelector` in
    `GenerationMixin.generate()` or asynchronously using multiple `TokenSelector` inside an inference endpoint.

    The constructor of this class should not be called directly: instances should be obtained by
    calling `TokenSelector.create()`.
    """

    def __init__(
        self,
        mode: GenerationMode,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        eos_token_id: int,
        pad_token_id: int,
        logits_warper: Optional[LogitsProcessorList] = None,
    ):
        self.mode = mode
        self.logits_processor = logits_processor
        self.stopping_criteria = stopping_criteria
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.logits_warper = logits_warper

    @classmethod
    def create(
        cls,
        input_ids: torch.Tensor,
        generation_config: GenerationConfig,
        model: GenerationMixin,
        max_seq_length: int,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
    ) -> "TokenSelector":
        r"""Creates the `TokenSelector` for a specific generation configuration.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            generation_config (`~transformers.generation.GenerationConfig`, *optional*):
                The generation configuration to parametrize the token selection.
            model (`~transformers.generation.GenerationMixin`):
                The model provides the internal helpers allowing to select the logits processors and stopping criterias.
            max_seq_length (`int`):
                The maximum number of input + generated tokens for this model. It depends on the model compilation parameters.
            stopping_criteria (`Optional[transformers.generation.StoppingCriteriaList], defaults to `None`):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config.
        Return:
            `torch.LongTensor`: A `torch.LongTensor` containing the selected tokens.
        """
        generation_config.validate()

        unsupported_generation_flags = [
            "output_attentions",
            "output_hidden_states",
            "output_scores",
            "return_dict_in_generate",
        ]
        for flag in unsupported_generation_flags:
            if getattr(generation_config, flag, False):
                raise ValueError("{flag} is not supported for generation.")

        if generation_config.max_new_tokens is not None:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
            generation_config.max_length = generation_config.max_new_tokens + input_ids.shape[-1]

        min_length = generation_config.min_length
        if min_length > max_seq_length:
            raise ValueError(
                f"The minimum generation length ({min_length}) exceeds the model maximum sequence length ({max_seq_length})"
            )
        max_length = generation_config.max_length
        if max_length > max_seq_length:
            logger.warning(
                f"Adjusting the maximum generation length ({max_length}) to the model maximum sequence length ({max_seq_length})"
            )
            generation_config.max_length = max_seq_length

        # Instantiate transformers library processors and criterias
        logits_processor = model._get_logits_processor(
            generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
        )
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
        stopping_criteria = model._get_stopping_criteria(generation_config, stopping_criteria=stopping_criteria)

        # The generation requires special tokens
        eos_token_id = generation_config.eos_token_id
        # This is not supposed to happen for any of the models we support
        assert eos_token_id is not None and not isinstance(eos_token_id, list)
        if generation_config.pad_token_id is None:
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        generation_mode = model._get_generation_mode(generation_config, None)
        if generation_mode not in [GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE]:
            raise ValueError("Unsupported generation mode")

        logits_warper = None
        if generation_mode == GenerationMode.SAMPLE:
            logits_warper = FusedLogitsWarper.from_config(generation_config)

        return cls(
            mode=generation_mode,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            logits_warper=logits_warper,
            eos_token_id=eos_token_id,
            pad_token_id=generation_config.pad_token_id,
        )

    def select(self, input_ids: torch.LongTensor, logits: torch.Tensor) -> torch.LongTensor:
        """Select the next tokens from the candidate logits.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation (not used in all generation modes).
            logits (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The logits corresponding to the generated tokens.

        Return:
            `torch.LongTensor`: A `torch.LongTensor` containing the selected tokens.
        """
        scores = self.logits_processor(input_ids, logits)
        if self.mode == GenerationMode.SAMPLE:
            return self._sample(scores)
        else:
            return torch.argmax(scores, dim=-1)

    def _sample(self, scores: torch.Tensor) -> torch.LongTensor:
        # Get [batch_size, kept] scores and indices instead of [batch_size, vocab_size] scores
        scores, next_token_indices = self.logits_warper(scores)

        # sample
        probs = torch.nn.functional.softmax(scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        # Convert the filtered tokens to actual vocabulary tokens
        next_tokens = torch.gather(next_token_indices, 1, next_tokens)
        return next_tokens.squeeze(1)
