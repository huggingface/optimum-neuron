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
"""NeuroModelForXXX classes for seq2seq models' inference on neuron devices."""
import logging
import os
import shutil
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForSeq2SeqLM, GenerationConfig, PreTrainedTokenizerBase
from transformers.generation.beam_search import BeamScorer
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
)
from transformers.generation.utils import (
    BeamSearchOutput,
    GreedySearchOutput,
)
from transformers.modeling_outputs import ModelOutput, Seq2SeqLMOutput

from ..exporters.neuron import (
    NeuronConfig,
    main_export,
)
from ..exporters.neuron.model_configs import *  # noqa: F403
from ..exporters.tasks import TasksManager
from ..utils.save_utils import maybe_load_preprocessors
from .generation import NeuronGenerationMixin
from .modeling_base import NeuronBaseModel
from .utils import (
    DECODER_NAME,
    ENCODER_NAME,
    NEURON_FILE_NAME,
    is_neuronx_available,
)


if TYPE_CHECKING:
    from transformers import PretrainedConfig

if is_neuronx_available():
    import torch_neuronx

logger = logging.getLogger(__name__)


class NeuronModelForConditionalGeneration(NeuronBaseModel):
    base_model_prefix = "neuron_model"
    config_name = "config.json"

    def __init__(
        self,
        encoder: torch.jit._script.ScriptModule,
        decoder: torch.jit._script.ScriptModule,
        configs: Optional[Dict[str, "PretrainedConfig"]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        encoder_file_name: Optional[str] = NEURON_FILE_NAME,
        decoder_file_name: Optional[str] = NEURON_FILE_NAME,
        preprocessors: Optional[List] = None,
        neuron_configs: Optional[Dict[str, "NeuronConfig"]] = None,
        generation_config: Optional[GenerationConfig] = None,
        model_and_config_save_paths: Optional[Dict[str, Tuple[str, Path]]] = None,
        **kwargs,
    ):
        self.configs = configs
        self.neuron_configs = neuron_configs
        self._attributes_init(model_save_dir, preprocessors, **kwargs)
        self.model_and_config_save_paths = model_and_config_save_paths if model_and_config_save_paths else None
        self.encoder = NeuronEncoder(
            encoder,
            self,
            self.configs[ENCODER_NAME],
            self.neuron_configs[ENCODER_NAME],
        )
        self.decoder = NeuronEncoder(
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

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        encoder_file_name: str = NEURON_FILE_NAME,
        decoder_file_name: str = NEURON_FILE_NAME,
    ):
        """
        Saves the model encoder and decoder as well as their configuration files to a
        directory, so that it can be re-loaded using the
        [`~optimum.neuron.modeling_seq2seq.NeuronModelForSeq2SeqLM.from_pretrained`] class method.

        Args:
            save_directory (`Union[str, Path`]):
                The directory where to save the model files.
        """
        if self.model_and_config_save_paths is None:
            logger.warning(
                "`model_save_paths` is None which means that no path of Neuron model is defined. Nothing will be saved."
            )
            return

        save_directory = Path(save_directory)
        if not self.model_and_config_save_paths.get(ENCODER_NAME)[0].is_file():
            self.model_and_config_save_paths.pop(ENCODER_NAME)

        if not self.model_and_config_save_paths.get(DECODER_NAME)[0].is_file():
            self.model_and_config_save_paths.pop(DECODER_NAME)

        dst_paths = [
            save_directory / ENCODER_NAME / encoder_file_name,
            save_directory / DECODER_NAME / decoder_file_name,
        ]
        src_paths = [
            Path(self.model_and_config_save_paths[model_name][0])
            for model_name in set(self.model_and_config_save_paths.keys()).intersection([ENCODER_NAME, DECODER_NAME])
        ]

        for src_path, dst_path in zip(src_paths, dst_paths):
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if src_path.is_file():
                shutil.copyfile(src_path, dst_path)

        self.generation_config.save_pretrained(save_directory)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
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
                use_auth_token=use_auth_token,
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

        encoder = cls.load_model(model_and_config_save_paths["encoder"][0])
        decoder = cls.load_model(model_and_config_save_paths["decoder"][0])
        torch_neuronx.move_trace_to_device(decoder, 0)

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=os.path.join(subfolder, DECODER_NAME),
            )
        except OSError:
            logger.info("Generation config file not found, using a generation config created from the model config.")

        return cls(
            encoder=encoder,
            decoder=decoder,
            configs=configs,
            model_save_dir=model_save_dir,
            encoder_file_name=encoder_file_name,
            decoder_file_name=decoder_file_name,
            preprocessors=preprocessors,
            neuron_configs=neuron_configs,
            generation_config=generation_config,
            model_and_config_save_paths=model_and_config_save_paths,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        task: Optional[str] = None,
        auto_cast: Optional[str] = "matmul",
        auto_cast_type: Optional[str] = "bf16",
        disable_fast_relayout: Optional[bool] = False,
        disable_fallback: bool = False,
        dynamic_batch_size: bool = False,
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
            task=task,
            dynamic_batch_size=dynamic_batch_size,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            subfolder=subfolder,
            revision=revision,
            force_download=force_download,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            do_validation=False,
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

        encoder_neuron_config["encoder_input_names"] = encoder_neuron_config.pop("input_names")
        encoder_neuron_config["encoder_output_names"] = encoder_neuron_config.pop("output_names")
        decoder_neuron_config["decoder_input_names"] = decoder_neuron_config.pop("input_names")
        decoder_neuron_config["decoder_output_names"] = decoder_neuron_config.pop("output_names")

        neuron_config = encoder_neuron_config.update(decoder_neuron_config)
        encoder_config.__setattr__("neuron", neuron_config)

        return encoder_config


class NeuronModelForSeq2SeqLM(NeuronModelForConditionalGeneration, NeuronGenerationMixin):
    auto_model_class = AutoModelForSeq2SeqLM
    main_input_name = "input_ids"

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        encoder = self.get_encoder()
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(inputs_tensor, model_kwargs["attention_mask"])
        return model_kwargs

    def _update_model_kwargs_for_xla_generation(
        self,
        model_kwargs: Dict[str, Any],
        batch_size: int,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        max_length: Optional[int] = None,
        seq_length: Optional[int] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        def _update_attention(model_kwargs, is_encoder_decoder):
            """Updates the appropriate attention mask -- encoder-decoder models use `decoder_attention_mask`"""

            attention_mask_name = "decoder_attention_mask" if is_encoder_decoder else "attention_mask"
            attention_mask = model_kwargs.pop(attention_mask_name)
            attention_mask_update_slice = torch.ones(
                (batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask[:, 1:], attention_mask_update_slice], dim=-1)
            mask = {attention_mask_name: attention_mask}
            return mask

        mask = _update_attention(model_kwargs, is_encoder_decoder)
        # sets the updated variables (mask and past_key_values)
        model_kwargs.update(mask)

        # Set a mock cache tensor
        model_kwargs["past_key_values"] = torch.tensor([])

        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        This is needed for beam search and not greedy sampling
        We reorder the cache within the trace so we can skip it in modelling_t5.py. So we override the _reorder_cache
        """
        self.beam_idx = beam_idx
        return past_key_values

    def generate(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        prompt: str,
        max_length: int,
        num_beams: int,
        num_return_sequences: int,
        device: str,
    ):
        batch_encoding = tokenizer(
            prompt, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt"
        )

        past_key_values = self.encoder(batch_encoding["input_ids"], batch_encoding["attention_mask"])

        decoder_attention_mask = torch.cat(
            [torch.zeros((1, max_length - 1), dtype=torch.int32), torch.ones((1, 1), dtype=torch.int32)], axis=1
        )

        # copy the new cache state to the decoder
        if device == "xla":
            for state, tensor in zip(self.decoder.parameters(), past_key_values):
                state.copy_(tensor)
        else:
            # First half of the cache is self attention and the rest is cross attention
            self.decoder.past_key_values_sa = past_key_values[: len(past_key_values) // 2]
            self.decoder.past_key_values_ca = past_key_values[len(past_key_values) // 2 :]

        output = super().generate(
            **batch_encoding,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=False,
            use_cache=True,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs={"last_hidden_state": torch.ones((1, 128, 1))},
        )  # Pass fake encoder_outputs so the transfomers code will not invoke the encoder
        return output

    def forward(
        self,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        beam_scores=None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        hidden_states = encoder_outputs["last_hidden_state"]

        if not hasattr(self, "beam_idx"):
            # Infering the number of beams from the attention mask
            num_beams = attention_mask.shape[0]
            self.beam_idx = torch.arange(0, num_beams, dtype=torch.int64)

        decoder_outputs = self.decoder(
            decoder_input_ids, decoder_attention_mask, hidden_states, attention_mask, self.beam_idx, beam_scores
        )

        # lm_logits = decoder_outputs[0]
        next_token_scores = decoder_outputs[0]
        next_tokens = decoder_outputs[1]
        next_indices = decoder_outputs[2]

        return next_token_scores, next_tokens, next_indices

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: "BeamScorer",
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        seq_length: Optional[int] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
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

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        # beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores_device = "cpu"
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=beam_scores_device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
            # prepare model inputs
            # From max_length-sized input_ids, select first
            # cur_len - 1 values.
            update_indices = torch.stack(
                [torch.arange(input_ids.size(0)), torch.tensor(cur_len - 1).repeat(input_ids.size(0))], dim=-1
            )
            input_ids_ = input_ids[update_indices[:, 0], update_indices[:, 1], None]
            model_inputs = self.prepare_inputs_for_generation(input_ids_, **model_kwargs)

            next_token_scores, next_tokens, next_indices = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                beam_scores=beam_scores,
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
            input_ids[update_indices_2[:, 0], update_indices_2[:, 1], None] = (
                beam_next_tokens.unsqueeze(-1).to(device).to(torch.long)
            )
            input_ids = input_ids * 1  # Hack to materialize tensor

            # update generated ids, model inputs, and length for next step
            model_kwargs = self._update_model_kwargs_for_xla_generation(
                model_kwargs,
                batch_size=batch_beam_size,
                is_encoder_decoder=self.config.is_encoder_decoder,
                max_length=stopping_criteria.max_length,
                seq_length=cur_len,
                use_cache=model_kwargs["use_cache"],
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(
                    model_kwargs["past_key_values"], beam_idx.to(torch.int64)
                )

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

            if stop_criterion_1 or stop_criterion_2:
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
            if type(v) == torch.Tensor:
                sequence_outputs[k] = sequence_outputs[k].to(input_ids.device)

        return sequence_outputs["sequences"]

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional["LogitsProcessorList"] = None,
        stopping_criteria: Optional["StoppingCriteriaList"] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        seq_length: Optional[int] = int,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        """
        Overriding greedy sampling to use next tokens returned from neuron device instead of logits.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        use_cache = model_kwargs["use_cache"] if "use_cache" in model_kwargs else False
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
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

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            # prepare model inputs
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

            # forward pass to get next token
            output = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_tokens = output[0]

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
                model_kwargs,
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

            if this_peer_finished:
                break

        if streamer is not None:
            streamer.end()

        return input_ids


class _NeuronSeq2SeqModelPart:
    """
    For Seq2Seq architecture, we usually compile it to multiple neuron models. Each represents a part of the model.
    """

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        config: Optional["PretrainedConfig"] = None,
        neuron_config: Optional["NeuronConfig"] = None,
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

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
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
        parent_model: NeuronBaseModel,
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
