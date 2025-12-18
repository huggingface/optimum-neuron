# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from abc import ABC, abstractmethod

import torch
from transformers import GenerationConfig
from transformers.generation import GenerationMixin, SampleDecoderOnlyOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

from .sampling import (
    Sampler,
    prepare_sampling_params,
)


logger = logging.getLogger("Neuron")


def increase_attention_mask(attention_mask: torch.LongTensor, num_new_tokens: int) -> torch.LongTensor:
    # Prepend num_new_tokens to the attention mask (inputs are right padded)
    return torch.cat(
        [attention_mask.new_ones((attention_mask.shape[0], num_new_tokens)), attention_mask],
        dim=-1,
    )


def position_ids_from_attention_mask(attention_mask: torch.LongTensor) -> torch.LongTensor:
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def increase_position_ids(position_ids: torch.LongTensor, num_new_tokens: int) -> torch.LongTensor:
    if position_ids.shape[1] > 1:
        position_ids = torch.amax(position_ids, 1, keepdim=True)
    return position_ids + num_new_tokens


class NxDGenerationMixin(GenerationMixin, ABC):
    """A generation Mixin that can be used to extend NxDPreTrainedModel based classes"""

    # These are expected to be set by the GenerationMixin code
    main_input_name = "input_ids"
    _is_stateful = False
    _supports_cache_class = False

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        assert hasattr(self, "neuron_config")  # Must be set by the super class
        # Initialize default generation config
        self.generation_config = GenerationConfig.from_model_config(config)
        self.sampler = None

    def can_generate(self):
        # Still required in transformers <= 4.50
        return True

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        generation_config: "GenerationConfig | None" = None,
        **kwargs,
    ):
        # Sanity check
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.neuron_config.sequence_length:
            raise ValueError(
                f"The input sequence length ({sequence_length}) exceeds the model static sequence length ({self.neuron_config.sequence_length})"
            )
        if batch_size > self.neuron_config.batch_size:
            raise ValueError(
                f"The specified batch_size ({batch_size}) exceeds the model static batch size ({self.neuron_config.batch_size})"
            )
        # Keep generation stateless.
        self.reset()
        return super().generate(
            input_ids, attention_mask=attention_mask, generation_config=generation_config, **kwargs
        )

    # TODO: Remove _sample and define separate flow for on-device sampling that doesn't use HF.
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        attention_mask: torch.LongTensor,
        seq_ids: torch.Tensor = None,
        **kwargs,
    ) -> SampleDecoderOnlyOutput | torch.LongTensor:
        explicit_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if explicit_kwargs:
            logger.warning(f"The following kwargs are not supported for neuron model: {list(explicit_kwargs.keys())}")
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # Prepare input tensors
        position_ids = position_ids_from_attention_mask(attention_mask)
        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])
        batch_size = attention_mask.shape[0]
        top_k = generation_config.top_k if do_sample else 1
        top_p = generation_config.top_p if do_sample else 1.0
        temperature = generation_config.temperature if do_sample else 1.0
        sampling_params = prepare_sampling_params(
            batch_size=batch_size,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        # init scores / logits tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False
        is_for_token_generation = False
        # auto-regressive generation
        while not this_peer_finished:
            # forward pass to get next token
            outputs = self.forward(
                input_ids[:, -1:] if is_for_token_generation else input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                return_dict=True,
            )

            if self.neuron_config.on_device_sampling:
                next_tokens = outputs.tokens
            else:
                next_token_logits = outputs.logits[:, -1, :].clone()

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)

                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)

                if self.sampler is None:
                    self.sampler = Sampler(self.neuron_config, do_sample=True, on_cpu=True)

                next_tokens = self.sampler(next_token_scores, sampling_params)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0

            # update attention_mask, position_ids for next step
            is_for_token_generation = True
            attention_mask = increase_attention_mask(attention_mask, 1)
            position_ids = increase_position_ids(position_ids, 1)

        if return_dict_in_generate:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
            )
        else:
            return input_ids

    def prepare_inputs_for_generation(
        self,
        input_ids,
        is_decode,
        attention_mask=None,
        position_ids=None,
        seq_ids=None,
        sampling_params=None,
    ):
        if is_decode:
            input_ids = input_ids[:, -1:]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if is_decode:
                position_ids = torch.amax(position_ids, 1, keepdim=True)

        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "sampling_params": sampling_params,
            "seq_ids": seq_ids,
        }

        return model_inputs

    def prepare_inputs_for_prefill(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        seq_ids=None,
        sampling_params=None,
    ):
        return self.prepare_inputs_for_generation(
            input_ids,
            is_decode=False,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
        )

    def prepare_inputs_for_decode(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        seq_ids=None,
        sampling_params=None,
    ):
        return self.prepare_inputs_for_generation(
            input_ids,
            is_decode=True,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
        )

    def _assisted_decoding(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        assistant_model: "NxDGenerationMixin",
        attention_mask: torch.LongTensor,
        seq_ids: torch.Tensor = None,
        **kwargs,
    ):
        explicit_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if explicit_kwargs:
            logger.warning(f"The following kwargs are not supported for neuron model: {list(explicit_kwargs.keys())}")

        pad_token_id = generation_config.pad_token_id

        if assistant_model.neuron_config.on_device_sampling:
            raise ValueError("Assistant model must not use on-device sampling")
        if self.neuron_config.batch_size > 1:
            raise ValueError("Assisted decoding is only supported for batch size 1")

        # Other auxiliary variables
        max_len = stopping_criteria[0].max_length
        spec_len = self.neuron_config.speculation_length

        # Prepare input tensors
        position_ids = position_ids_from_attention_mask(attention_mask)
        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])
        batch_size = attention_mask.shape[0]
        do_sample = generation_config.do_sample
        assert do_sample is False, "Assisted decoding is only supported for greedy decoding."
        sampling_params = prepare_sampling_params(batch_size=batch_size)

        # Prefill the target model KV cache and get the first accepted token
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            return_dict=True,
        )
        next_tokens = outputs.logits[:, 0, :].argmax(dim=-1, keepdim=True)

        # Run the assistant model once to fill its kv cache
        assistant_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            return_dict=True,
        )

        # Increment input_ids, attention mask and position ids for the accepted token
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
        attention_mask = increase_attention_mask(attention_mask, 1)
        position_ids = increase_position_ids(position_ids, 1)

        while input_ids.shape[1] < max_len:
            # At this stage:
            # - the target and assistant models must have seen the same tokens
            # - the attention mask and position ids have been incremented for processing the last accepted token
            next_tokens = input_ids[:, -1:]
            assistant_tokens = input_ids.clone()
            assistant_attention_mask = attention_mask.clone()
            assistant_position_ids = position_ids.clone()
            candidate_tokens = torch.full((batch_size, spec_len), pad_token_id, device=input_ids.device)
            for i in range(spec_len):
                next_tokens = (
                    assistant_model.forward(
                        input_ids=next_tokens,
                        attention_mask=assistant_attention_mask,
                        position_ids=assistant_position_ids,
                        seq_ids=seq_ids,
                        sampling_params=sampling_params,
                        return_dict=True,
                    )
                    .logits[:, 0, :]
                    .argmax(dim=-1, keepdim=True)
                )
                candidate_tokens[:, i] = next_tokens.squeeze(-1)
                assistant_tokens = torch.cat((assistant_tokens, next_tokens), dim=-1)
                assistant_attention_mask = increase_attention_mask(assistant_attention_mask, 1)
                assistant_position_ids = increase_position_ids(assistant_position_ids, 1)
                # Stop assistant generation on EOS
                if stopping_criteria(assistant_tokens, None)[0]:
                    break
            # The next step is to validate the speculated tokens with the original model
            # The input ids are the concatenation of the last accepted token and the candidate tokens minus one.
            # The last accepted token must be added because it has only been predicted but never processed by
            # the original model yet.
            # This last candidate token on the other hand is not passed as it is only a prediction, and its logits
            # have not been computed yet by the assitant model (and it is not in the assistant model KV cache).
            validation_tokens = torch.cat((input_ids[:, -1:], candidate_tokens[:, :-1]), dim=-1)
            # The attention mask and seq_ids are the same as before
            # The position ids are not incremented, only expanded to cover the new tokens
            validation_position_ids = torch.arange(position_ids.amax(), position_ids.amax() + spec_len).expand(
                1, spec_len
            )
            outputs = self.forward(
                input_ids=validation_tokens,
                attention_mask=attention_mask,
                position_ids=validation_position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                return_dict=True,
            )
            selected_tokens = outputs.logits.argmax(dim=-1)
            # The returned logits are the probabilities that the next token is each of the vocabulary tokens.
            # We can therefore compare the argmax of these logits with the candidate tokens
            n_matches = ((~(candidate_tokens == selected_tokens)).cumsum(dim=-1) < 1).sum()
            logger.info(f"Assisted decoding: accepted {n_matches} tokens from assistant model")
            # Since all accepted tokens are correct, the next token predicted by the target model (if any) is also correct
            # We must however not slip beyond the candidate tokens length nor generate beyond max_len
            n_selected_tokens = min(n_matches + 1, selected_tokens.shape[1], max_len - input_ids.shape[1])
            accepted_tokens = selected_tokens[:, :n_selected_tokens]
            # If needed, decode the accepted tokens using the assistant model to fill its kv cache
            # Fast forward the assistant model's KV cache for accepted tokens
            # we also increment the attention mask and position ids accordingly
            # Note that we MUST NOT pass the last accepted token as it has not been seen yet by the
            # original model and will be processed only in the next iteration
            for i in range(accepted_tokens.shape[1] - 1):
                attention_mask = increase_attention_mask(attention_mask, 1)
                position_ids = increase_position_ids(position_ids, 1)
                if i >= n_matches:
                    # This token was not accepted, decode with the correct token to update the kv cache
                    assistant_model.forward(
                        input_ids=accepted_tokens[:, i : i + 1],
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        seq_ids=seq_ids,
                        sampling_params=sampling_params,
                        return_dict=True,
                    )
            # Prepare the inputs for the next iteration
            input_ids = torch.cat((input_ids, accepted_tokens), dim=-1)
            if stopping_criteria(input_ids, None)[0]:
                break
            # We only need to increment attention mask and position ids by one since they were already
            # incremented while fast-forwarding the assistant model's KV cache for the accepted tokens
            attention_mask = increase_attention_mask(attention_mask, 1)
            position_ids = increase_position_ids(position_ids, 1)

        return input_ids

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # We dont want HF to move parameters to device
        return torch.device("cpu")

    @abstractmethod
    def reset(self):
        raise SystemError(f"The reset method must be implemented by {self.__class__.__name__}")
