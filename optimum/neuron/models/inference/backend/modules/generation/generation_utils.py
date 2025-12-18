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
import copy
from abc import ABC, abstractmethod
from typing import Any

import torch
from transformers import GenerationConfig, PreTrainedModel
from transformers.generation import GenerationMixin, SampleDecoderOnlyOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.modeling_outputs import ModelOutput

from .sampling import (
    Sampler,
    prepare_sampling_params,
)


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
        **model_kwargs,
    ) -> SampleDecoderOnlyOutput | torch.LongTensor:
        r"""
        We override the GenerationMixin sample function (_sample for transformers>=4.39.0) to add support for right side padding.
        """

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        batch_size = model_kwargs["attention_mask"].shape[0]
        top_k = generation_config.top_k if do_sample else 1
        top_p = generation_config.top_p if do_sample else 1.0
        temperature = generation_config.temperature if do_sample else 1.0
        sampling_params = prepare_sampling_params(
            batch_size=batch_size,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        model_kwargs["sampling_params"] = sampling_params

        # init scores / logits tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False
        is_for_token_generation = False
        # auto-regressive generation
        while not this_peer_finished:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, is_decode=is_for_token_generation, **model_kwargs
            )
            model_kwargs["attention_mask"] = model_inputs.get("attention_mask")

            # forward pass to get next token
            outputs = self.forward(**model_inputs, return_dict=True)

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

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            is_for_token_generation = True
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_for_token_generation=is_for_token_generation
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0

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
        sampling_params=None,
        seq_ids=None,
        **kwargs,
    ):
        if is_decode:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
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

        # WARNING: This is needed for propagating additional kwargs to the neuron model
        additional_kwargs = self.get_required_kwargs()
        for arg in additional_kwargs:
            model_inputs.update({arg: kwargs.get(arg, None)})

        return model_inputs

    def prepare_inputs_for_prefill(
        self,
        input_ids,
        attention_mask=None,
        sampling_params=None,
        **kwargs,
    ):
        return self.prepare_inputs_for_generation(
            input_ids,
            is_decode=False,
            attention_mask=attention_mask,
            sampling_params=sampling_params,
            **kwargs,
        )

    def prepare_inputs_for_decode(
        self,
        input_ids,
        attention_mask=None,
        sampling_params=None,
        **kwargs,
    ):
        return self.prepare_inputs_for_generation(
            input_ids,
            is_decode=True,
            attention_mask=attention_mask,
            sampling_params=sampling_params,
            **kwargs,
        )

    # We override this function because we want to change the way attention_mask
    # is updated each iteration.
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_for_token_generation: bool,
    ) -> dict[str, Any]:
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if is_for_token_generation:
                # Prepend 1 to the attention mask (inputs are right padded)
                attention_mask = torch.cat(
                    [attention_mask.new_ones((attention_mask.shape[0], 1)), attention_mask],
                    dim=-1,
                )
            model_kwargs["attention_mask"] = attention_mask
        return model_kwargs

    def _assisted_decoding(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        assistant_model: "PreTrainedModel | None" = None,
        **model_kwargs,
    ):
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id

        if assistant_model.neuron_config.on_device_sampling:
            raise ValueError("Assistant model must not use on-device sampling")

        # Init values
        if eos_token_id is not None and pad_token_id is None:
            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

        # Prepare assistant model's keys of inputs
        assistant_kwargs = copy.deepcopy(model_kwargs)

        # Other auxiliary variables
        max_len = stopping_criteria[0].max_length
        cur_len = input_ids.shape[-1]
        spec_len = self.neuron_config.speculation_length

        # Run the target model once and get the first generated token
        model_inputs = self.prepare_inputs_for_prefill(input_ids, **model_kwargs)
        outputs = self.forward(**model_inputs)

        curr_pos = model_inputs["position_ids"][0].argmax(dim=-1)
        new_token = outputs.logits[:, 0].argmax(dim=-1, keepdim=True)

        # Prepare the input ids and attention mask for the draft model
        candidate_input_ids = input_ids

        # This is the finally return outputs; append the first generated token
        returned_ids = torch.cat((input_ids[:, : curr_pos + 1], new_token), dim=1)

        # Speculation loop
        while True:
            # 1 Token generation using draft model
            is_for_token_generation = assistant_model.kv_cache_populated
            for _ in range(spec_len):
                # 1.1 Prepare assistant model inputs
                assistant_inputs = assistant_model.prepare_inputs_for_generation(
                    candidate_input_ids,
                    is_decode=is_for_token_generation,
                    **assistant_kwargs,
                )

                # 1.2 Use the assistant model to obtain the next candidate logits
                assistant_model_outputs = assistant_model.forward(**assistant_inputs)
                assistant_new_token = assistant_model_outputs.logits[:, 0, :].argmax(dim=-1)

                # 1.3 Update inputs and args for next iteration
                candidate_input_ids = torch.cat((candidate_input_ids, assistant_new_token[:, None]), dim=-1)
                assistant_kwargs = assistant_model._update_model_kwargs_for_generation(
                    assistant_model_outputs,
                    assistant_kwargs,
                    is_for_token_generation,
                )

                # 1.4 Stop assistant generation on EOS
                if eos_token_id_tensor is not None:
                    last_assistant_token_is_eos = assistant_new_token.tile(eos_token_id_tensor.shape[0], 1)
                    last_assistant_token_is_eos = (
                        ~last_assistant_token_is_eos.ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0).bool()
                    )
                    if last_assistant_token_is_eos:
                        break
                else:
                    last_assistant_token_is_eos = False

            # 2 Validation of draft model output using the original model
            #   The length could be shorter if the draft loop ends earlier
            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]

            # 2.1 Prepare the input arguments
            input_ids = torch.cat((new_token, candidate_input_ids[:, -candidate_length:-1]), dim=-1)
            attention_mask = model_inputs["attention_mask"]
            pos = curr_pos + 1
            position_ids = torch.arange(pos, pos + spec_len).expand(1, spec_len)
            # Pad the input_ids if needed
            if input_ids.shape[-1] < spec_len:
                input_ids = torch.cat(
                    (input_ids, torch.full((1, spec_len - input_ids.shape[-1]), pad_token_id)),
                    dim=-1,
                )

            # 2.2. Run a forward pass on the candidate sequence
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            # 2.3. Process the new logits
            new_tokens = outputs.logits.argmax(dim=-1)
            selected_tokens = outputs.logits[:, : candidate_length - 1].argmax(dim=-1)

            # 3. Compare the argmax from the original model logits with the assistant forecasted tokens. We can keep
            # the assistant forecasted tokens until the first mismatch, or until the max length is reached.
            candidate_new_tokens = candidate_input_ids[:, -candidate_length:-1]
            n_matches = ((~(candidate_new_tokens == selected_tokens)).cumsum(dim=-1) < 1).sum()

            # 4. Ensure we don't generate beyond max_len or an EOS token
            if last_assistant_token_is_eos and n_matches == candidate_length:
                n_matches -= 1
            n_matches = min(n_matches, max_len - cur_len - 1)
            # n_matches = 4

            # 5. Get the valid continuation, after the matching tokens. We also consider the extra token
            # generated by the original model. Update the return ids accordingly
            valid_tokens = new_tokens[:, : n_matches + 1]
            returned_ids = torch.cat((returned_ids, valid_tokens), dim=1)
            # if last_assistant_token_is_eos and n_matches == candidate_length-1:
            #    break;

            # 6. Update the args for the next iteration.
            #    Feed the last correct token to the next loop
            new_token = valid_tokens[:, -1:]
            if new_token[0] in torch.tensor(eos_token_id):
                break
            input_ids = valid_tokens[:, -1:]
            candidate_input_ids = valid_tokens[:, -1:]
            model_inputs_attn_mask = model_inputs["attention_mask"]
            n_matches_concat_tensor = torch.zeros(1, n_matches + 1, dtype=model_inputs_attn_mask.dtype)
            model_inputs_attn_mask = torch.cat([model_inputs_attn_mask, n_matches_concat_tensor], dim=-1)
            model_inputs["attention_mask"] = model_inputs_attn_mask.index_fill(
                1, torch.arange(curr_pos + 1, curr_pos + 1 + n_matches + 1), 1
            )

            curr_pos = curr_pos + n_matches + 1
            assistant_kwargs["attention_mask"] = copy.deepcopy(model_inputs["attention_mask"])

            # 7. Update with the generated token length and check for stopping condition.
            cur_len = cur_len + n_matches + 1
            if cur_len >= max_len:
                break
            # 8. If the rest length is smaller than speculation length, we directly run the target model to finish
            if max_len - cur_len < spec_len:
                # @yihsian: TODO: complete with using target tokengen model
                break

        return returned_ids

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
