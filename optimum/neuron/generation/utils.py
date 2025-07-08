# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Utilities for generation with Neuron."""

import copy
import inspect
import warnings
from functools import wraps
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers import parallel_state
from transformers import GenerationMixin
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import (
    BeamSearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchOutput,
    GenerateOutput,
    GreedySearchDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput,
    GreedySearchOutput,
)
from transformers.utils import ModelOutput, logging

from ..utils.misc import args_and_kwargs_to_kwargs_only


logger = logging.get_logger(__name__)


def _move_dict_args_to_device(kwargs: dict[str, Any], device: str = "cpu") -> dict[str, Any]:
    """
    Takes keyword arguments which will be passed to a model's forward function
    and moves its values to `device` if
    they are of type `torch.Tensor`. If the key is a dictionary it does the same to the
    respective values.
    Args:
        kwargs: (`dict[str, Any]`):
            The kwargs to be passed to the models forward function.
        device: (`str`, defaults to `cpu`):
            The target device to which tensors should be moved.
    Returns:
        `dict[str, Any]`: The kwargs dict with its tensors moved to `device`.
    """

    def needs_move(src_device, tgt_device):
        return src_device != tgt_device

    for k, v in kwargs.items():
        # Handle nested dicts
        if isinstance(v, dict):
            for k_, v_ in v.items():
                if isinstance(v_, torch.Tensor):
                    if needs_move(v_.device, device):
                        v[k_] = v_.to(device=device)

        # Handle tensor types
        elif isinstance(v, torch.Tensor):
            if needs_move(v.device, device):
                kwargs[k] = v.to(device=device)

        # Handle past_key_value tuples
        elif k == "past_key_values":
            if v is not None:
                new_past_key_values = ()
                for layer_past in v:
                    new_layer_past = ()
                    for past_state in layer_past:
                        if needs_move(past_state.device, device):
                            new_layer_past += (past_state.to(device=device),)
                        else:
                            new_layer_past += (past_state,)
                    new_past_key_values += (new_layer_past,)
                kwargs[k] = new_past_key_values

    return kwargs


def _pad_input_ids_for_general_sampling(
    input_ids: torch.Tensor, num_padding_values: int, pad_token_id: int
) -> torch.Tensor:
    """
    Pads `input_ids` with `num_padding_values` padding tokens along the second dimension.
    Args:
        input_ids (`torch.Tensor`):
            Input ids to be padded.
        num_padding_values (`int`):
            Number of padding values to add.
        pad_token_id (`int`):
            Token ID of padding token.
    Returns:
        `torch.Tensor`: Padded `input_ids`.
    """
    bsz = input_ids.size(0)
    input_ids = torch.cat(
        [input_ids, torch.ones((bsz, num_padding_values), device=input_ids.device, dtype=torch.long) * pad_token_id], 1
    )
    return input_ids


def _get_fwd_for_general_sampling(
    current_fwd: Callable,
    generation_config: GenerationConfig,
    is_encoder_decoder: bool,
    vocab_size: int,
    main_device: str,
    to_device: str = "cpu",
    output_dtype: torch.dtype = torch.float32,
) -> Callable:
    """
    Wraps the passed forward function and extends it such that before each forward call
    the `decoder_input_ids` are padded and all tensors are moved to `main_device` (e.g. XLA).
    Then the original forward passed is called followed by a `xm.mark_step`. Subsequently,
    an "unpadding" of the logits is performed. This way, all functions that process the logits
    can be called without making any changes.
    Args:
        current_fwd (`Callable`):
            The current forward function of the model.
        generation_config (`GenerationConfig`):
            The GenerationConfig of the model.
        is_encoder_decoder (`bool`):
            Defines if this is a encoder-decoder model.
        vocab_size (`int`):
            The total number of vocabs of the current model.
        main_device (`str`):
            The device on which the forward pass should be executed.
        to_device (`str`, defaults to `cpu`):
            The device on which all other processing should be executed.
        output_dtype (`torch.dtype`, defaults to `torch.float32`):
            The expected data type of the output logits.
    Returns:
        `Callable`: The extended forward function.
    """

    @wraps(current_fwd)
    def new_fwd(*args, **kwargs):
        # Pad input to max length
        cur_len = None
        input_ids_string = "decoder_input_ids" if is_encoder_decoder else "input_ids"
        if input_ids_string in kwargs:
            current_input_ids = kwargs[input_ids_string]
            batch_size, cur_len = current_input_ids.shape
            num_padding_values = generation_config.max_length - cur_len
            kwargs[input_ids_string] = _pad_input_ids_for_general_sampling(
                current_input_ids, num_padding_values, generation_config.pad_token_id
            )

            # For decoder only models, pad decoder attention mask in addition to prompts
            if "attention_mask" in kwargs and not is_encoder_decoder and num_padding_values > 0:
                kwargs["attention_mask"] = torch.cat(
                    [
                        kwargs["attention_mask"],
                        torch.zeros((batch_size, (generation_config.max_length - cur_len)))
                        .long()
                        .to(kwargs["attention_mask"].device),
                    ],
                    1,
                )
                # create position_ids on the fly for batch generation
                if "position_ids" in set(inspect.signature(current_fwd).parameters.keys()):
                    position_ids = kwargs["attention_mask"].long().cumsum(-1) - 1
                    position_ids.masked_fill_(kwargs["attention_mask"] == 0, 1)
                    kwargs["position_ids"] = position_ids

        # Move inputs to device
        _move_dict_args_to_device(kwargs, main_device)

        # Forward
        kwargs = args_and_kwargs_to_kwargs_only(current_fwd, args, kwargs)
        outputs = current_fwd(**kwargs)
        # Gather outputs if NxD tensor parallelism is applied and the output logits have not been gathered.
        if (
            parallel_state.model_parallel_is_initialized()
            and parallel_state.get_tensor_model_parallel_size() > 1
            and outputs["logits"].shape[-1] != vocab_size
        ):
            outputs["logits"] = xm.all_gather(
                outputs["logits"],
                dim=-1,
                groups=parallel_state.get_tensor_model_parallel_group(as_list=True),
            )
        xm.mark_step()

        # Move to CPU
        _move_dict_args_to_device(outputs, to_device)

        # Post-process output as a function of cur_len
        outputs["logits"] = outputs["logits"][:, :cur_len, ...].to(output_dtype)

        return outputs

    return new_fwd


class GeneralNeuronGenerationMixin(GenerationMixin):
    """
    A class containing all functions for auto-regressive text generation on Trn1, to be used as a mixin in [`PreTrainedModel`].
    The generation will be handled on both CPU and TRN1 in the following way:
      1. Model forward pass will be executed on TRN1
      2. All other logics including padding, searching, and sampling will be handled by general device (CPU).
    This implementation allows us to support general searching and sampling methods with minimal code changes.
    """

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        **kwargs,
    ):
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # two conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same).
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. set generation parameters if not already defined
        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs and move to CPU
        general_device = "cpu"
        if "input_ids" in kwargs and kwargs["input_ids"] is not None:
            kwargs["input_ids"] = kwargs["input_ids"].to(general_device)
        if inputs is not None:
            inputs = inputs.to(general_device)
        input_ids, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )

        # 4. set Neuron specific generation configurations
        original_forward = copy.deepcopy(self.forward)
        try:
            general_forward = _get_fwd_for_general_sampling(
                self.forward,
                generation_config,
                self.config.is_encoder_decoder,
                self.config.vocab_size,
                self.device,
            )
            self.forward = general_forward
            if generation_config.use_cache:
                warnings.warn(
                    "use_cache is not supported for generation on Neuron devices, switching to use_cache=False."
                )
                # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
                # generating the first new token or not, and we only want to use the embeddings for the first new token)
                if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
                    raise ValueError("Decoder-only models with inputs_embeds forwarding must use `use_cache=True`")
            generation_config.use_cache = False
            if generation_config.max_new_tokens is not None:
                generation_config.max_length = generation_config.max_new_tokens + input_ids.shape[-1]

            # 5. Run HuggingFace generate function
            return super().generate(inputs, generation_config, **kwargs)
        finally:
            self.forward = original_forward

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: str | None = None
    ) -> dict[str, Any]:
        """Move the input tensor to XLA device and move the output tensors back to CPU."""
        output = super()._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor.to(self.device), model_kwargs, model_input_name
        )
        _move_dict_args_to_device(output, "cpu")
        return output


class NeuronGenerationMixin(GenerationMixin):
    """
    A class containing all functions for auto-regressive text generation on Trn1, to be used as a mixin in [`PreTrainedModel`].

    The class exposes [`~generation.GenerationMixin.generate`], which can be used for:
        - *greedy decoding* by calling [`~generation.GenerationMixin.greedy_search`] if `num_beams=1` and
          `do_sample=False`
        - *contrastive search* by calling [`~generation.GenerationMixin.contrastive_search`] if `penalty_alpha>0` and
          `top_k>1`
        - *multinomial sampling* by calling [`~generation.GenerationMixin.sample`] if `num_beams=1` and
          `do_sample=True`
        - *beam-search decoding* by calling [`~generation.GenerationMixin.beam_search`] if `num_beams>1` and
          `do_sample=False`
        - *beam-search multinomial sampling* by calling [`~generation.GenerationMixin.beam_sample`] if `num_beams>1`
          and `do_sample=True`
        - *diverse beam-search decoding* by calling [`~generation.GenerationMixin.group_beam_search`], if `num_beams>1`
          and `num_beam_groups>1`
        - *constrained beam-search decoding* by calling [`~generation.GenerationMixin.constrained_beam_search`], if
          `constraints!=None` or `force_words_ids!=None`

    You do not need to call any of the above methods directly. Pass custom parameter values to 'generate' instead. To
    learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

    @staticmethod
    def _initialize_attention(
        model_kwargs,
        num_padding_values,
        batch_size,
        device,
        is_encoder_decoder,
    ):
        """Initializes the appropriate attention mask -- encoder-decoder models use `decoder_attention_mask`"""
        if is_encoder_decoder:
            # One 1 for decoder_start_token_id, 0s for the currently-unfilled locations in the past_key_values tensor,
            # 1s for the actual input_ids
            decoder_attention_mask = torch.cat(
                [
                    torch.zeros((batch_size, num_padding_values), dtype=torch.int32),
                    torch.ones((batch_size, 2), dtype=torch.int32),
                ],
                axis=1,
            ).to(device)
            mask = {"decoder_attention_mask": decoder_attention_mask}
        else:
            attention_mask = model_kwargs.pop("attention_mask")
            # 0s for the currently-unfilled locations in the past_key_values tensor, 1s for the actual input_ids
            attention_mask = torch.cat(
                [
                    torch.zeros(
                        (batch_size, num_padding_values), dtype=attention_mask.dtype, device=attention_mask.device
                    ),
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device),
                ],
                axis=1,
            )
            mask = {"attention_mask": attention_mask}

        return mask

    @staticmethod
    def _update_attention(model_kwargs, batch_size, is_encoder_decoder):
        """Updates the appropriate attention mask -- encoder-decoder models use `decoder_attention_mask`"""

        attention_mask_name = "decoder_attention_mask" if is_encoder_decoder else "attention_mask"
        attention_mask = model_kwargs.pop(attention_mask_name)
        attention_mask_update_slice = torch.ones(
            (batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat([attention_mask[:, 1:], attention_mask_update_slice], dim=-1)
        mask = {attention_mask_name: attention_mask}
        return mask

    @staticmethod
    def _initialize_past(past_key_values, num_padding_values):
        """Initialize past_key_values with zeros -- the structure depends on `batch_axis`"""

        new_past = ()
        for past_layer in past_key_values:
            new_past_layer = list(past_layer)
            for i in range(len(new_past_layer[:2])):
                b, n_heads, _, head_dim = past_layer[i].shape
                new_past_layer[i] = torch.cat(
                    [
                        torch.zeros(
                            (b, n_heads, num_padding_values, head_dim),
                            dtype=past_layer[i].dtype,
                            device=past_layer[i].device,
                        ),
                        past_layer[i],
                    ],
                    dim=2,
                )
            new_past += (tuple(new_past_layer),)

        return new_past

    @staticmethod
    def _update_past(past_key_values):
        new_past = ()
        for past_layer in past_key_values:
            new_past_layer = list(past_layer)
            for i, _ in enumerate(new_past_layer[:2]):
                new_past_layer[i] = past_layer[i][:, :, 1:]
            new_past += (tuple(new_past_layer),)

        return new_past

    def _update_model_kwargs_for_xla_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        batch_size: int,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        max_length: int | None = None,
        seq_length: int | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        if use_cache:
            past_key_values = self._extract_past_from_model_output(outputs)
            if past_key_values is None:
                raise ValueError(
                    "No known `past_key_values variable` found in model outputs (model outputs keys:"
                    f" {list(outputs.keys())})"
                )
            is_past_initialized = model_kwargs.pop("past_key_values", None) is not None

            if not is_past_initialized:
                # The padded version of `past_key_values` has a length of `max_length - 1`, as `past_key_values` holds information relative to
                # previous autoregressive generation steps (step 0 has no past_key_values, step 1 has 1 past_key_values value, ..., the last step
                # has `max_length - 1` past_key_values values).
                num_padding_values = max_length - seq_length
                mask = self._initialize_attention(
                    model_kwargs, num_padding_values, batch_size, outputs.logits.device, is_encoder_decoder
                )
                new_past = self._initialize_past(past_key_values, num_padding_values)
            else:
                mask = self._update_attention(model_kwargs, batch_size, is_encoder_decoder)
                new_past = self._update_past(past_key_values)

            # sets the updated variables (mask and past_key_values)
            model_kwargs.update(mask)
            model_kwargs["past_key_values"] = tuple(new_past)
        else:
            model_kwargs["past_key_values"] = None
            if "token_type_ids" in model_kwargs:
                token_type_ids = model_kwargs["token_type_ids"]
                model_kwargs["token_type_ids"] = torch.cat(
                    [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
                )

            if not is_encoder_decoder:
                # update attention mask
                if "attention_mask" in model_kwargs:
                    batch_size = model_kwargs["attention_mask"].shape[0]
                    update_indices = torch.stack(
                        [torch.arange(batch_size), torch.tensor(seq_length).repeat(batch_size)], dim=-1
                    )
                    model_kwargs["attention_mask"][update_indices[:, 0], update_indices[:, 1]] = model_kwargs[
                        "attention_mask"
                    ].new_ones((batch_size, 1))

            else:
                # update decoder attention mask
                if "decoder_attention_mask" in model_kwargs:
                    batch_size = model_kwargs["decoder_attention_mask"].shape[0]
                    update_indices = torch.stack(
                        [torch.arange(batch_size), torch.tensor(seq_length).repeat(batch_size)], dim=-1
                    )
                    model_kwargs["decoder_attention_mask"][update_indices[:, 0], update_indices[:, 1]] = model_kwargs[
                        "decoder_attention_mask"
                    ].new_ones((batch_size, 1))

        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    if len(dict_to_expand[key].shape) == 2:
                        dict_to_expand[key] = (
                            dict_to_expand[key].repeat(1, expand_size).view(-1, dict_to_expand[key].shape[1])
                        )
                    elif len(dict_to_expand[key].shape) <= 1:
                        dict_to_expand[key] = dict_to_expand[key].repeat(expand_size)
                    else:
                        dict_to_expand[key] = torch.concat(
                            [tensor.unsqueeze(0).repeat(expand_size, 1, 1) for tensor in dict_to_expand[key]]
                        )
            return dict_to_expand

        if input_ids is not None:
            # Manual repeat interleave
            input_ids = input_ids.repeat(1, expand_size).view(-1, input_ids.shape[1])

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor | None, list[int]]] = None,
        synced_gpus: bool | None = None,
        is_traced_inference: bool = False,
        **kwargs,
    ) -> GenerateOutput | torch.LongTensor:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor | None`, defaults to `None`):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`GenerationConfig | None`, defaults to `None`):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList | None`, defaults to `None`):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList | None`, defaults to `None`):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], list[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool | None`, defaults to `None`):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            is_traced_inference (`bool`, defaults to `False`):
                Whether the decoder is traced or using XLA lazy tensor. If the decoder is traced, next tokens and the beam scores
                are computed inside the decoder.
            kwargs:
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchDecoderOnlyOutput`],
                    - [`~generation.SampleDecoderOnlyOutput`],
                    - [`~generation.BeamSearchDecoderOnlyOutput`],
                    - [`~generation.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchEncoderDecoderOutput`],
                    - [`~generation.SampleEncoderDecoderOutput`],
                    - [`~generation.BeamSearchEncoderDecoderOutput`],
                    - [`~generation.BeamSampleEncoderDecoderOutput`]
        """

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        if generation_config.use_cache and not is_traced_inference:
            warnings.warn("use_cache is not supported for generation on Neuron devices, switching to use_cache=False.")
            model_kwargs["use_cache"] = False
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs and not is_traced_inference

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            if (
                generation_config.pad_token_id is not None
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs and not is_traced_inference:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # Pad to max_length
        input_ids = torch.cat(
            [
                input_ids,
                (
                    torch.ones(
                        (batch_size, (generation_config.max_length - input_ids_seq_length)),
                    )
                    .long()
                    .to(input_ids.device)
                )
                * generation_config.pad_token_id,
            ],
            1,
        )
        # For decoder only models, pad decoder attention mask in addition to prompts
        if (
            "attention_mask" in model_kwargs
            and model_kwargs.get("use_cache", False) is False
            and not self.config.is_encoder_decoder
        ):
            model_kwargs["attention_mask"] = torch.cat(
                [
                    model_kwargs["attention_mask"],
                    torch.zeros((batch_size, (generation_config.max_length - input_ids_seq_length)))
                    .long()
                    .to(model_kwargs["attention_mask"].device),
                ],
                1,
            )

        # 7. determine generation mode
        is_constraint_gen_mode = (
            generation_config.constraints is not None or generation_config.force_words_ids is not None
        )

        is_contrastive_search_gen_mode = (
            (generation_config.num_beams == 1)
            and generation_config.top_k is not None
            and generation_config.top_k > 1
            and generation_config.do_sample is False
            and generation_config.penalty_alpha is not None
            and generation_config.penalty_alpha > 0
        )

        is_greedy_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )

        if generation_config.num_beam_groups > generation_config.num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")

        if hasattr(self, "device") and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing greedy search, "
                    f"but is {generation_config.num_return_sequences}."
                )

            # 11. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                seq_length=input_ids_seq_length,
                is_traced_inference=is_traced_inference,
                **model_kwargs,
            )
        elif is_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device="cpu",
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                seq_length=input_ids_seq_length,
                is_traced_inference=is_traced_inference,
                **model_kwargs,
            )

        else:
            raise ValueError("Only greedy search and beam search are supported on Neuron.")

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        max_length: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int | None] = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_scores: bool | None = None,
        return_dict_in_generate: bool | None = None,
        synced_gpus: bool = False,
        seq_length: int | None = None,
        is_traced_inference: bool = False,
        **model_kwargs,
    ) -> GreedySearchOutput | torch.LongTensor:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. list of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. list of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int | list[int]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            seq_length (`int | None`, defaults to `False`):
                Length of current input_ids sequence
            is_traced_inference (`bool`, defaults to `False`):
                Whether the decoder is traced or using XLA lazy tensor. If the decoder is traced, next tokens and the beam scores
                are computed inside the decoder.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from optimum.neuron import NeuronModelForSeq2SeqLM

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> input_shapes = {"batch_size": 1, "sequence_length": 128, "num_beams": 1}
        >>> model = NeuronModelForSeq2SeqLM.from_pretrained("t5-small", export=True, dynamic_batch_size=False, **input_shapes)

        >>> input_prompt = "translate English to German: Lets eat good food."
        >>> inputs = tokenizer(input_prompt, return_tensors="pt")

        >>> outputs = model.greedy_search(input_ids)

        >>> results = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
        ```
        """
        # init values
        if logits_processor is not None and is_traced_inference:
            logger.warning(
                "`logits_processor` will not be neglected because in `optimum-neuron`, `next_tokens` is computed inside the compiled decoder. If you want us to support custom logits_processor during the compilation, please file an issue to https://github.com/huggingface/optimum-neuron."
            )
        elif logits_processor is None:
            logits_processor = LogitsProcessorList()
        use_cache = model_kwargs.pop("use_cache", False)
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = None
        if return_dict_in_generate and output_scores:
            if is_traced_inference:
                logger.warning(
                    "`output_scores` will be neglected because currently we do not trace `next_token_scores` for greedy search (we do only in beam search). If you want us to support the option during the compilation, please file an issue to https://github.com/huggingface/optimum-neuron."
                )
            else:
                scores = ()
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            if use_cache:
                # From max_length-sized input_ids, select first
                # seq_length - 1 values.

                if model_kwargs.get("past_key_values") is None:
                    input_ids_ = input_ids[:, :seq_length]
                else:
                    update_indices = torch.stack(
                        [torch.arange(input_ids.size(0)), torch.tensor(seq_length - 1).repeat(input_ids.size(0))],
                        dim=-1,
                    )
                    input_ids_ = input_ids[update_indices[:, 0], update_indices[:, 1], None]

                model_inputs = self.prepare_inputs_for_generation(input_ids_, **model_kwargs)
            else:
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            if not is_traced_inference:
                if not use_cache:
                    one_hot = (
                        torch.cat(
                            [
                                torch.tensor([0]).repeat(1, seq_length - 1),
                                torch.tensor([1]).repeat(1, 1),
                                torch.tensor([0]).repeat(1, input_ids.size(1) - seq_length),
                            ],
                            dim=1,
                        )
                        .to(device=outputs.logits.device)
                        .float()
                    )
                    next_token_logits = torch.matmul(one_hot, outputs.logits)
                    next_token_logits = next_token_logits.squeeze(1)
                else:
                    next_token_logits = outputs.logits[:, -1, :]

                # pre-process distribution
                # Move to cpu to handle arbitrary logits_processor
                next_tokens_scores = logits_processor(input_ids.to("cpu")[:, :seq_length], next_token_logits.to("cpu"))
                next_tokens_scores = next_tokens_scores.to(input_ids.device)

                # argmax
                next_tokens = torch.argmax(next_tokens_scores, dim=-1)

                if return_dict_in_generate and output_scores:
                    scores += (next_tokens_scores,)
            else:
                next_tokens = outputs[0]

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            batch_size, _ = input_ids.shape
            update_indices = torch.stack(
                [torch.arange(batch_size), torch.tensor(seq_length).repeat(batch_size)], dim=-1
            )
            input_ids[update_indices[:, 0], update_indices[:, 1]] = next_tokens[:]
            model_kwargs = self._update_model_kwargs_for_xla_generation(
                outputs=outputs,
                model_kwargs=model_kwargs,
                batch_size=batch_size,
                is_encoder_decoder=self.config.is_encoder_decoder,
                max_length=stopping_criteria.max_length,
                seq_length=seq_length,
                use_cache=use_cache,
            )

            seq_length += 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            if not is_traced_inference:
                xm.mark_step()

            # stop when each sentence is finished, or if we exceed the maximum length
            stop_criterion_1 = unfinished_sequences.max() == 0

            if isinstance(stopping_criteria, list):
                if len(stopping_criteria) == 1:
                    stopping_criteria = stopping_criteria[0]

            # Cases that can be handled in XLA without requiring
            # non-padded input_ids
            if isinstance(stopping_criteria, MaxLengthCriteria):
                stop_criterion_2 = seq_length >= stopping_criteria.max_length
            elif isinstance(stopping_criteria, MaxTimeCriteria):
                stop_criterion_2 = stopping_criteria(input_ids, scores)
            else:
                # Other cases will be handled on CPU
                batch_size, _ = input_ids.shape
                mask = torch.cat(
                    [torch.ones(batch_size, seq_length), torch.zeros(batch_size, input_ids.shape[1] - seq_length)],
                    dim=1,
                ).bool()
                input_ids_cpu = torch.masked_select(input_ids, mask).reshape((batch_size, seq_length)).to("cpu")
                scores_cpu = scores.to("cpu") if torch.is_tensor(scores) else scores
                stop_criterion_2 = stopping_criteria(input_ids_cpu, scores_cpu)

            if stop_criterion_1 or stop_criterion_2:
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        max_length: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int | None] = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_scores: bool | None = None,
        return_dict_in_generate: bool | None = None,
        synced_gpus: bool | None = False,
        seq_length: int | None = None,
        is_traced_inference: bool = False,
        **model_kwargs,
    ) -> BeamSearchOutput | torch.LongTensor:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. list of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. list of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int | list[int]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            seq_length (`int | None`, defaults to `False`):
                Length of current input_ids sequence
            is_traced_inference (`bool`, defaults to `False`):
                Whether the decoder is traced or using XLA lazy tensor. If the decoder is traced, next tokens and the beam scores
                are computed inside the decoder.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from optimum.neuron import NeuronModelForSeq2SeqLM

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> input_shapes = {"batch_size": 1, "sequence_length": 128, "num_beams": 4}
        >>> model = NeuronModelForSeq2SeqLM.from_pretrained("t5-small", export=True, dynamic_batch_size=False, **input_shapes)

        >>> input_prompt = "translate English to German: Lets eat good food."
        >>> inputs = tokenizer(input_prompt, return_tensors="pt")

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }
        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ```
        """
        # init values
        if logits_processor is not None and is_traced_inference:
            logger.warning(
                "`logits_processor` will be neglected because in `optimum-neuron`, `next_tokens` is computed inside the compiled decoder. If you want us to support custom logits_processor during the compilation, please file an issue to https://github.com/huggingface/optimum-neuron."
            )
        elif logits_processor is None:
            logits_processor = LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        # Overwrite cur_len
        cur_len = seq_length

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores_device = "cpu"
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=beam_scores_device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            if model_kwargs["use_cache"]:
                # From max_length-sized input_ids, select first
                # cur_len - 1 values.
                update_indices = torch.stack(
                    [torch.arange(input_ids.size(0)), torch.tensor(cur_len - 1).repeat(input_ids.size(0))], dim=-1
                )
                input_ids_ = input_ids[update_indices[:, 0], update_indices[:, 1], None]
                model_inputs = self.prepare_inputs_for_generation(input_ids_, **model_kwargs)
            else:
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if is_traced_inference:
                outputs = self(
                    **model_inputs,
                    beam_scores=beam_scores,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                next_token_scores = outputs.next_token_scores
                next_tokens = outputs.next_tokens
                next_indices = outputs.next_indices

                if return_dict_in_generate and output_scores:
                    scores += (next_token_scores,)
            else:
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                if synced_gpus and this_peer_finished:
                    cur_len = cur_len + 1
                    continue  # don't waste resources running the code we don't need

                if not model_kwargs["use_cache"]:
                    one_hot = (
                        torch.cat(
                            [
                                torch.tensor([0]).repeat(1, cur_len - 1),
                                torch.tensor([1]).repeat(1, 1),
                                torch.tensor([0]).repeat(1, input_ids.size(1) - cur_len),
                            ],
                            dim=1,
                        )
                        .to(device=outputs.logits.device)
                        .float()
                    )
                    next_token_logits = torch.matmul(one_hot, outputs.logits)
                    next_token_logits = next_token_logits.squeeze(1)
                else:
                    next_token_logits = outputs.logits[:, -1, :]

                # Manually compute log softmax
                # log_softmax(vi) = vi - max(vi) - log(sum(exp(vi - max(vi))))
                logit_max, _ = torch.max(next_token_logits, dim=-1, keepdim=True)
                logsumexp = torch.log(torch.exp(next_token_logits - logit_max).sum(dim=-1, keepdim=True))
                next_token_scores = next_token_logits - logit_max - logsumexp
                # (batch_size * num_beams, vocab_size)

                xm.mark_step()

                # We don't want to change every single logit processor, so
                # we perform this processing on CPU.
                input_ids_ = input_ids.to("cpu")[:, :cur_len]
                next_token_scores_ = next_token_scores.to("cpu")
                next_token_scores_processed = logits_processor(input_ids_, next_token_scores_)

                next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

                # reshape for beam search
                vocab_size = next_token_scores.shape[-1]
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
                next_token_scores = next_token_scores * 1

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                if return_dict_in_generate and output_scores:
                    scores += (next_token_scores_processed,)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )
            # stateless
            beam_outputs = beam_scorer.process(
                input_ids.to("cpu")[:, :cur_len],
                next_token_scores.to("cpu"),
                next_tokens.to("cpu"),
                next_indices.to("cpu"),
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            update_indices = torch.stack(
                [torch.arange(batch_beam_size), torch.tensor(cur_len - 1).repeat(batch_beam_size)], dim=-1
            )
            update_indices_2 = torch.stack(
                [torch.arange(batch_beam_size), torch.tensor(cur_len).repeat(batch_beam_size)], dim=-1
            )
            # First select beam_indices
            device = input_ids.device
            beam_idx_device = beam_idx.to(device=input_ids.device)
            input_ids[:, :] = input_ids[beam_idx_device.long(), :]

            # Then append new tokens
            if is_traced_inference:
                # int64 is not natively supported by inf2 and has been cast down to int32
                input_ids[update_indices_2[:, 0], update_indices_2[:, 1], None] = (
                    beam_next_tokens.unsqueeze(-1).to(device).to(torch.long)
                )
            else:
                input_ids[update_indices_2[:, 0], update_indices_2[:, 1], None] = beam_next_tokens.unsqueeze(-1).to(
                    device
                )
            input_ids = input_ids * 1  # Hack to materialize tensor

            # update generated ids, model inputs, and length for next step
            model_kwargs = self._update_model_kwargs_for_xla_generation(
                outputs=outputs,
                model_kwargs=model_kwargs,
                batch_size=batch_beam_size,
                is_encoder_decoder=self.config.is_encoder_decoder,
                max_length=stopping_criteria.max_length,
                seq_length=cur_len,
                use_cache=model_kwargs["use_cache"],
            )
            if is_traced_inference:
                self._reorder_cache(beam_idx.to(torch.int64))
            elif model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            # stop when each sentence is finished, or if we exceed the maximum length
            stop_criterion_1 = beam_scorer.is_done
            if isinstance(stopping_criteria, list):
                if len(stopping_criteria) == 1:
                    stopping_criteria = stopping_criteria[0]

            # Cases that can be handled in XLA without requiring
            # non-padded input_ids
            if isinstance(stopping_criteria, MaxLengthCriteria):
                stop_criterion_2 = cur_len >= stopping_criteria.max_length
            elif isinstance(stopping_criteria, MaxTimeCriteria):
                stop_criterion_2 = stopping_criteria(input_ids, scores)
            else:
                # Other cases will be handled on CPU
                batch_size, _ = input_ids.shape
                input_ids_cpu = input_ids.to("cpu")
                mask = torch.cat(
                    [torch.ones(batch_size, cur_len), torch.zeros(batch_size, input_ids.shape[1] - cur_len)], dim=1
                ).bool()
                input_ids_cpu = torch.masked_select(input_ids_cpu, mask).reshape((batch_size, cur_len))
                scores_cpu = scores.to("cpu") if torch.is_tensor(scores) else scores
                stop_criterion_2 = stopping_criteria(input_ids_cpu, scores_cpu)

            # TODO: validate with @JingyaHuang
            if stop_criterion_1 or torch.all(stop_criterion_2):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids.to("cpu"),
            beam_scores.to("cpu"),
            next_tokens.to("cpu"),
            next_indices.to("cpu"),
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        for k, v in sequence_outputs.items():
            if type(v) is torch.Tensor:
                sequence_outputs[k] = sequence_outputs[k].to(input_ids.device)

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]
