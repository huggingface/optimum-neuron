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
"""NeuroModelForXXX classes for seq2seq models' inference on Neuron devices."""

import copy
import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForSeq2SeqLM, GenerationConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.utils import ModelOutput

from ..exporters.neuron import (
    NeuronDefaultConfig,
    main_export,
)
from ..exporters.tasks import TasksManager
from ..utils.save_utils import maybe_load_preprocessors
from .generation import NeuronGenerationMixin
from .modeling_traced import NeuronTracedModel
from .utils import (
    DECODER_NAME,
    ENCODER_NAME,
    NEURON_FILE_NAME,
    is_neuronx_available,
    is_neuronx_distributed_available,
)


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

if is_neuronx_available():
    import torch_neuronx

if is_neuronx_distributed_available():
    import neuronx_distributed

logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "AutoTokenizer"

NEURON_SEQ2SEQ_MODEL_START_DOCSTRING = r"""
    This model inherits from [`~neuron.modeling.NeuronTracedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)

    Args:
        encoder (`torch.jit._script.ScriptModule`): [torch.jit._script.ScriptModule](https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html) is the TorchScript module of the encoder with embedded NEFF(Neuron Executable File Format) compiled by neuron(x) compiler.
        decoder (`torch.jit._script.ScriptModule`): [torch.jit._script.ScriptModule](https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html) is the TorchScript module of the decoder with embedded NEFF(Neuron Executable File Format) compiled by neuron(x) compiler.
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig) is the Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`optimum.neuron.modeling.NeuronTracedModel.from_pretrained`] method to load the model weights.
"""

NEURON_SEQ2SEQ_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`](https://huggingface.co/docs/transformers/autoclass_tutorial#autotokenizer).
            See [`PreTrainedTokenizer.encode`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.encode) and
            [`PreTrainedTokenizer.__call__`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__) for details.
            [What are input IDs?](https://huggingface.co/docs/transformers/glossary#input-ids)
        attention_mask (`Union[torch.Tensor, None]` of shape `({0})`, defaults to `None`):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](https://huggingface.co/docs/transformers/glossary#attention-mask)
"""


class NeuronModelForConditionalGeneration(NeuronTracedModel, ABC):
    base_model_prefix = "neuron_model"
    config_name = "config.json"

    def __init__(
        self,
        encoder: torch.jit._script.ScriptModule,
        decoder: torch.jit._script.ScriptModule,
        config: "PretrainedConfig",
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        encoder_file_name: Optional[str] = NEURON_FILE_NAME,
        decoder_file_name: Optional[str] = NEURON_FILE_NAME,
        preprocessors: Optional[List] = None,
        neuron_configs: Optional[Dict[str, "NeuronDefaultConfig"]] = None,
        configs: Optional[Dict[str, "PretrainedConfig"]] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        self.config = config
        self.configs = configs
        self.neuron_configs = neuron_configs
        self.input_static_shapes = NeuronModelForConditionalGeneration.get_input_static_shapes(
            self.neuron_configs[ENCODER_NAME]
        )  # only for the encoder
        self._attributes_init(model_save_dir, preprocessors, **kwargs)
        self.encoder = NeuronEncoder(
            encoder,
            self,
            self.configs[ENCODER_NAME],
            self.neuron_configs[ENCODER_NAME],
        )
        self.decoder = NeuronDecoder(
            decoder,
            self,
            self.configs[DECODER_NAME],
            self.neuron_configs[DECODER_NAME],
        )
        self.dynamic_batch_size = all(
            neuron_config._config.neuron["dynamic_batch_size"] for neuron_config in self.neuron_configs.values()
        )
        self.encoder_file_name = encoder_file_name
        self.decoder_file_name = decoder_file_name

        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(self.configs[DECODER_NAME])
        self.generation_config = generation_config
        self.tensor_parallel_size = self.neuron_configs[DECODER_NAME].tensor_parallel_size

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~optimum.neuron.modeling_traced.NeuronTracedModel.from_pretrained`] class method.

        Args:
            save_directory (`Union[str, Path]`):
                Directory where to save the model file.
        """
        shutil.copytree(self.model_save_dir, save_directory, dirs_exist_ok=True)
        self.generation_config.save_pretrained(save_directory)

    @staticmethod
    def load_model(
        encoder_path: Union[str, Path],
        decoder_path: Union[str, Path],
        tensor_parallel_size: int,
    ):
        if tensor_parallel_size == 1:
            # Initialize Neuron Runtime before loading models
            runtime = torch.classes.neuron.Runtime()
            runtime.initialize()
            runtime.set_default_neuron_cores(0, 1)
            encoder = NeuronTracedModel.load_model(encoder_path)
            decoder = NeuronTracedModel.load_model(decoder_path)
            torch_neuronx.move_trace_to_device(decoder, 0)
        else:
            encoder = neuronx_distributed.trace.parallel_model_load(encoder_path)
            decoder = neuronx_distributed.trace.parallel_model_load(decoder_path)
        return encoder, decoder

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        encoder_file_name: Optional[str] = NEURON_FILE_NAME,
        decoder_file_name: Optional[str] = NEURON_FILE_NAME,
        subfolder: str = "",
        local_files_only: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        model_id = str(model_id)

        if not os.path.isdir(model_id):
            # Downloads all repo's files matching the allowed patterns
            model_id = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                force_download=force_download,
                ignore_patterns=["*.msgpack", "*.safetensors", "*.bin"],  # only download *.neuron artifacts
            )

        preprocessors = maybe_load_preprocessors(model_id, subfolder=subfolder)

        new_model_save_dir = Path(model_id)

        model_and_config_save_paths = {
            "encoder": (
                new_model_save_dir / ENCODER_NAME / encoder_file_name,
                new_model_save_dir / ENCODER_NAME / cls.config_name,
            ),
            "decoder": (
                new_model_save_dir / DECODER_NAME / decoder_file_name,
                new_model_save_dir / DECODER_NAME / cls.config_name,
            ),
        }

        # Re-build pretrained configs and neuron configs
        configs, neuron_configs = {}, {}
        for name, file_paths in model_and_config_save_paths.items():
            if file_paths[1].is_file():
                model_config = AutoConfig.from_pretrained(file_paths[1])
                configs[name] = model_config
                neuron_configs[name] = cls._neuron_config_init(model_config)

        encoder, decoder = cls.load_model(
            encoder_path=model_and_config_save_paths[ENCODER_NAME][0],
            decoder_path=model_and_config_save_paths[DECODER_NAME][0],
            tensor_parallel_size=configs["decoder"].neuron.get("tensor_parallel_size", 1),
        )

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
        except OSError:
            logger.info("Generation config file not found, using a generation config created from the model config.")

        return cls(
            encoder=encoder,
            decoder=decoder,
            config=config,
            model_save_dir=model_save_dir,
            encoder_file_name=encoder_file_name,
            decoder_file_name=decoder_file_name,
            preprocessors=preprocessors,
            neuron_configs=neuron_configs,
            configs=configs,
            generation_config=generation_config,
        )

    @classmethod
    def _from_transformers(cls, *args, **kwargs):
        # Deprecate it when optimum uses `_export` as from_pretrained_method in a stable release.
        return cls._export(*args, **kwargs)

    @classmethod
    def _export(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        compiler_workdir: Optional[str] = None,
        tensor_parallel_size: Optional[int] = 1,
        inline_weights_to_neff: bool = True,
        optlevel: str = "2",
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        task: Optional[str] = None,
        auto_cast: Optional[str] = "matmul",
        auto_cast_type: Optional[str] = "bf16",
        disable_fast_relayout: Optional[bool] = False,
        disable_fallback: bool = False,
        dynamic_batch_size: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs_shapes,
    ) -> "NeuronModelForConditionalGeneration":
        if dynamic_batch_size is True:
            logger.warning(
                "Sequence-to-sequence models don't support dynamic batch size yet, `dynamic_batch_size` will be set to False."
            )

        if task is None:
            task = TasksManager.infer_task_from_model(cls.auto_model_class)

        # Get compilation arguments
        auto_cast_type = None if auto_cast is None else auto_cast_type
        compiler_kwargs = {
            "auto_cast": auto_cast,
            "auto_cast_type": auto_cast_type,
            "disable_fast_relayout": disable_fast_relayout,
            "disable_fallback": disable_fallback,
        }

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            compiler_kwargs=compiler_kwargs,
            tensor_parallel_size=tensor_parallel_size,
            task=task,
            dynamic_batch_size=dynamic_batch_size,
            cache_dir=cache_dir,
            compiler_workdir=compiler_workdir,
            inline_weights_to_neff=inline_weights_to_neff,
            optlevel=optlevel,
            trust_remote_code=trust_remote_code,
            subfolder=subfolder,
            revision=revision,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            do_validation=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs_shapes,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            model_save_dir=save_dir,
        )

    def _save_config(self, save_directory):
        save_directory = Path(save_directory)
        self.configs[ENCODER_NAME].save_pretrained(save_directory / ENCODER_NAME)
        self.configs[DECODER_NAME].save_pretrained(save_directory / DECODER_NAME)
        combined_config = self._combine_encoder_decoder_config(
            encoder_config=self.configs[ENCODER_NAME],
            decoder_config=self.configs[DECODER_NAME],
        )
        combined_config.save_pretrained(save_directory)

    def _combine_encoder_decoder_config(self, encoder_config: "PretrainedConfig", decoder_config: "PretrainedConfig"):
        encoder_neuron_config = encoder_config.neuron
        decoder_neuron_config = decoder_config.neuron
        combined_config = copy.deepcopy(encoder_config)

        encoder_neuron_config["encoder_input_names"] = encoder_neuron_config.pop("input_names")
        encoder_neuron_config["encoder_output_names"] = encoder_neuron_config.pop("output_names")
        decoder_neuron_config["decoder_input_names"] = decoder_neuron_config.pop("input_names")
        decoder_neuron_config["decoder_output_names"] = decoder_neuron_config.pop("output_names")

        encoder_neuron_config.update(decoder_neuron_config)
        encoder_neuron_config.pop("model_type")
        combined_config.__setattr__("neuron", encoder_neuron_config)

        return combined_config


TRANSLATION_EXAMPLE = r"""
    *(Following models are compiled with neuronx compiler and can only be run on INF2.)*
    Example of text-to-text generation with small T5 model:

    ```python
    from transformers import {processor_class}
    from optimum.neuron import {model_class}

    neuron_model = {model_class}.from_pretrained({checkpoint_regular}, export=True, dynamic_batch_size=False, batch_size=1, sequence_length=64, num_beams=4)
    neuron_model.save_pretrained("t5_small_neuronx")
    del neuron_model

    neuron_model = {model_class}.from_pretrained("t5_small_neuronx")
    tokenizer = {processor_class}.from_pretrained("t5_small_neuronx")
    inputs = tokenizer("translate English to German: Lets eat good food.", return_tensors="pt")

    output = neuron_model.generate(
        **inputs,
        num_return_sequences=1,
    )
    results = [tokenizer.decode(t, skip_special_tokens=True) for t in output]
    ```
    
    *(For large models, in order to fit into Neuron cores, we need to apply tensor parallelism. Here below is an example ran on `inf2.24xlarge`.)*
    Example of text-to-text generation with tensor parallelism:
    
    ```python
    from transformers import {processor_class}
    from optimum.neuron import {model_class}
    # 1. compile
    if __name__ == "__main__":  # compulsory for parallel tracing since the API will spawn multiple processes.
        neuron_model = {model_class}.from_pretrained(
            {checkpoint_tp}, export=True, tensor_parallel_size=8, dynamic_batch_size=False, batch_size=1, sequence_length=128, num_beams=4,
        )
        neuron_model.save_pretrained("flan_t5_xl_neuronx_tp8/")
        del neuron_model

    # 2. inference
    neuron_model = {model_class}.from_pretrained("flan_t5_xl_neuronx_tp8")
    tokenizer = {processor_class}.from_pretrained("flan_t5_xl_neuronx_tp8")
    inputs = tokenizer("translate English to German: Lets eat good food.", return_tensors="pt")

    output = neuron_model.generate(
        **inputs,
        num_return_sequences=1,
    )
    results = [tokenizer.decode(t, skip_special_tokens=True) for t in output]
    ```
"""  # noqa: W293


@add_start_docstrings(
    """
    Neuron Sequence-to-sequence model with a language modeling head for text2text-generation tasks.
    """,
    NEURON_SEQ2SEQ_MODEL_START_DOCSTRING,
)
class NeuronModelForSeq2SeqLM(NeuronModelForConditionalGeneration, NeuronGenerationMixin):
    auto_model_class = AutoModelForSeq2SeqLM
    main_input_name = "input_ids"

    @add_start_docstrings_to_model_forward(
        NEURON_SEQ2SEQ_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + TRANSLATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForSeq2SeqLM",
            checkpoint_regular="google-t5/t5-small",
            checkpoint_tp="google/flan-t5-xl",
        )
    )
    def forward(
        self,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        beam_scores: Optional[torch.FloatTensor] = None,
        return_dict: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Union[Tuple[torch.FloatTensor], ModelOutput]:
        hidden_states = encoder_outputs["last_hidden_state"]

        if not hasattr(self, "beam_idx"):
            # Infering the number of beams from the attention mask
            num_beams = attention_mask.shape[0]
            self.beam_idx = torch.arange(0, num_beams, dtype=torch.int64)

        outputs = self.decoder(
            decoder_input_ids, decoder_attention_mask, hidden_states, attention_mask, self.beam_idx, beam_scores
        )

        # Fetch optional outputs
        cur_idx = 0
        cross_attentions = None
        decoder_attentions = None
        decoder_hidden_states = None

        # Skip pkv which can't be copied from memory to buffer
        if output_attentions and self.configs["decoder"].neuron.get("output_attentions"):
            if self.config.is_encoder_decoder:
                cross_attentions = outputs[-self.config.num_decoder_layers :]
                cur_idx += self.config.num_decoder_layers
            decoder_attentions = outputs[-(self.config.num_decoder_layers + cur_idx) : -cur_idx]
            cur_idx += self.config.num_decoder_layers

        if output_hidden_states and self.configs["decoder"].neuron.get("output_hidden_states"):
            decoder_hidden_states = outputs[-(self.config.num_decoder_layers + 1 + cur_idx) : -cur_idx]

        decoder_outputs = ModelOutput(
            next_token_scores=outputs[0],
            next_tokens=outputs[1],
            next_indices=outputs[2],
            cross_attentions=cross_attentions,
            decoder_attentions=decoder_attentions,
            decoder_hidden_states=decoder_hidden_states,
        )

        if return_dict:
            return decoder_outputs
        else:
            return decoder_outputs.to_tuple()

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        num_return_sequences: int = 1,
        **kwargs,
    ):
        max_length = self.neuron_configs[ENCODER_NAME].sequence_length
        num_beams = self.neuron_configs[ENCODER_NAME].num_beams
        batch_size = self.neuron_configs[ENCODER_NAME].batch_size

        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        inputs = self._pad_to_compiled_shape(inputs)

        past_key_values = self.encoder(**inputs)

        decoder_attention_mask = torch.cat(
            [
                torch.zeros((batch_size, max_length - 1), dtype=torch.int64),
                torch.ones((batch_size, 1), dtype=torch.int64),
            ],
            axis=1,
        )

        if self.tensor_parallel_size == 1:
            # copy the new cache state to the decoder
            for state, tensor in zip(self.decoder.model.parameters(), past_key_values):
                state.copy_(tensor)
        else:
            # Here we iterate sharded encoders and decoders since the encoder on each rank will return cache as device tensors,
            # we want to assign them to the cache of the sharded decoder on the same rank to avoid the copy. The KV cache always
            # use pre-allocated memory, no host-device communication overhead.
            for decoder_tp, encoder_tp in zip(self.decoder.model.models, self.encoder.model.models):
                decoder_tp.load_state_dict(encoder_tp.state_dict(), strict=False)

        output = super().generate(
            **inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            assistant_model=assistant_model,
            num_return_sequences=num_return_sequences,
            max_length=kwargs.pop("max_length", None) or max_length,
            max_new_tokens=kwargs.pop("max_new_tokens", None),
            output_attentions=kwargs.pop("output_attentions", False),
            output_hidden_states=kwargs.pop("output_hidden_states", False),
            output_scores=kwargs.pop("output_scores", False),
            return_dict_in_generate=kwargs.pop("return_dict_in_generate", False),
            num_beams=num_beams,
            do_sample=kwargs.pop("do_sample", False),
            use_cache=True,  # pkv is cached by default in
            decoder_attention_mask=decoder_attention_mask,
            # Pass fake encoder_outputs so the transfomers code will not invoke the encoder
            encoder_outputs={"last_hidden_state": torch.ones((batch_size, max_length, 1))},
            is_traced_inference=True,
        )
        return output

    def _reorder_cache(self, beam_idx):
        """
        The cache was reordered during the tracing of the decoder so we can skip it here. This is needed for beam search and not greedy sampling.
        """
        self.beam_idx = beam_idx

    def get_encoder(self) -> "NeuronEncoder":
        return self.encoder

    def _update_model_kwargs_for_xla_generation(
        self,
        model_kwargs: Dict[str, Any],
        batch_size: int,
        is_encoder_decoder: bool = False,
        # Leave following kwargs for compatibility, will not have any effect.
        outputs: "ModelOutput" = None,
        standardize_cache_format: bool = False,
        max_length: Optional[int] = None,
        seq_length: Optional[int] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        mask = self._update_attention(model_kwargs, batch_size, is_encoder_decoder)
        # sets the updated variables (mask and past_key_values)
        model_kwargs.update(mask)

        return model_kwargs

    # Override to cut the input_ids to just last token
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids as past is cached
        input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
        }

    def _validate_static_shape(self, input_shapes: List[int], target_shapes: List[int]) -> bool:
        """
        Checks if a input needs to be padded.
        """
        return input_shapes == target_shapes

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True


class _NeuronSeq2SeqModelPart:
    """
    For Seq2Seq architecture, we usually compile it to multiple neuron models. Each represents a part of the model.
    """

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronTracedModel,
        config: Optional["PretrainedConfig"] = None,
        neuron_config: Optional["NeuronDefaultConfig"] = None,
        model_type: str = "encoder",
        device: Optional[int] = None,
    ):
        self.model = model
        self.parent_model = parent_model
        self.config = config
        self.neuron_config = neuron_config
        self.model_type = model_type
        self.device = device

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class NeuronEncoder(_NeuronSeq2SeqModelPart):
    """
    Encoder part of the encoder-decoder model for Neuron inference. (Actually it's a monolith of encoder + decoder without past_key_values to workaround the control flow in the decoder).
    """

    main_input_name = "input_ids"

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronTracedModel,
        config: Optional["PretrainedConfig"] = None,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, config, neuron_config, "encoder")

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor):
        inputs = (
            input_ids,
            attention_mask,
        )
        outputs = self.model(*inputs)
        return outputs


class NeuronDecoder(_NeuronSeq2SeqModelPart):
    """
    Decoder part of the encoder-decoder model for Neuron inference. (Actually it's decoder with past_key_values).
    """

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronTracedModel,
        config: Optional["PretrainedConfig"] = None,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, config, neuron_config, "decoder")

    def forward(
        self,
        input_ids: torch.LongTensor,
        decoder_attention_mask: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor,
        beam_idx: torch.LongTensor,
        beam_scores: torch.FloatTensor,
    ):
        inputs = (
            input_ids,
            decoder_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            beam_idx,
            beam_scores,
        )
        outputs = self.model(*inputs)
        return outputs
