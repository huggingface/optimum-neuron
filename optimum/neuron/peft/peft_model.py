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
import json
import os
import warnings
from pathlib import Path
from typing import Any

import neuronx_distributed
import torch
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
)
from transformers import PreTrainedModel

from ..models.training import (
    adapt_peft_config_for_model,
    adapt_state_dict,
    create_parameter_metadata,
    get_pipeline_parameters_for_current_stage,
    specialize_transformation_specs_for_model,
    to_original_peft_config_for_model,
)
from ..utils.import_utils import is_peft_available
from ..utils.patching import Patcher
from ..utils.training_utils import _get_model_param_count
from .utils.save_and_load import get_peft_model_state_dict


if is_peft_available():
    from peft import PeftConfig, PeftModel
    from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_PREFIX_MAPPING
    from peft.tuners import XLoraModel
    from peft.utils import (
        load_peft_weights,
        set_peft_model_state_dict,
    )

else:

    class PeftModel:
        pass

    class PeftConfig:
        pass

    class XLoraModel:
        pass

    def load_peft_weights(*args, **kwargs):
        pass

    def set_peft_model_state_dict(*args, **kwargs):
        pass

    PEFT_TYPE_TO_CONFIG_MAPPING = {}
    PEFT_TYPE_TO_PREFIX_MAPPING = {}

ADAPTER_MODEL_PARALLEL_SHARDS_DIR_NAME = "adapter_shards"


class NeuronPeftModel(PeftModel):
    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
        **kwargs: Any,
    ) -> None:
        from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING

        # We adapt the PEFT config for the model using the transformation specs.
        peft_config = adapt_peft_config_for_model(model, peft_config, inplace=False)

        patcher = Patcher(
            [
                ("peft.peft_model.PEFT_TYPE_TO_TUNER_MAPPING", PEFT_TYPE_TO_TUNER_MAPPING),
                ("peft.auto.MODEL_TYPE_TO_PEFT_MODEL_MAPPING", MODEL_TYPE_TO_PEFT_MODEL_MAPPING),
            ]
        )
        with patcher:
            super().__init__(
                model,
                peft_config,
                adapter_name=adapter_name,
                autocast_adapter_dtype=autocast_adapter_dtype,
                low_cpu_mem_usage=False,
            )
        # We specialize the transformation specs for the PeFT model.
        specialize_transformation_specs_for_model(self)

        # We need to update the names of the parameters for the current stage after initialization because we have
        # added the `default` adapter.
        self.recompute_parameters_for_current_stage()

    def recompute_parameters_for_current_stage(self):
        self._parameter_names_for_current_pp_rank = get_pipeline_parameters_for_current_stage(self)

    @property
    def parameters_for_current_stage(self) -> set[str]:
        if not hasattr(self, "_parameter_names_for_current_pp_rank"):
            self.recompute_pipeline_parameters_for_current_pp_rank()
        return self._parameter_names_for_current_pp_rank

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = False,
        selected_adapters: list[str] | None = None,
        save_embedding_layers: str | bool = "auto",
        is_main_process: bool = True,
        path_initial_model_for_weight_conversion: str | None = None,
        **kwargs: Any,
    ) -> None:
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )

        def save_mutated_as_lora(peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs):
            if peft_config.use_rslora and (peft_config.rank_pattern or peft_config.alpha_pattern):
                msg = (
                    "Passing `path_initial_model_for_weight_conversion` to `save_pretrained` is not supported when "
                    "using `rank_pattern` or `alpha_pattern` at the same time as `use_rslora=True`."
                )
                raise ValueError(msg)

            if not any(
                str(peft_config.init_lora_weights).lower().startswith(prefix)
                for prefix in ["pissa", "corda", "olora", "true"]
            ):
                warnings.warn(
                    "`path_initial_model_for_weight_conversion` only works for converting a PiSSA/CorDA/OLoRA adapter to "
                    "a LoRA adapter"
                )
            initial_adapter_name = os.path.basename(path_initial_model_for_weight_conversion)
            try:
                self.load_adapter(
                    os.path.dirname(path_initial_model_for_weight_conversion),
                    subfolder=initial_adapter_name,
                    adapter_name=initial_adapter_name,
                )
                is_pissa = str(self.peft_config[initial_adapter_name].init_lora_weights).lower().startswith("pissa")
                is_corda = str(self.peft_config[initial_adapter_name].init_lora_weights).lower() == "corda"
                is_olora = str(self.peft_config[initial_adapter_name].init_lora_weights).lower() == "olora"
                if is_pissa or is_corda or is_olora:
                    raise ValueError(
                        "The `init_lora_weights` parameter of the initial adapter should be set to `True`. "
                        "Otherwise, `self.load_adapter` will subtract the decomposed values again based on the "
                        "residual model."
                    )
                output_state_dict = self.base_model.subtract_mutated_init(
                    output_state_dict, initial_adapter_name, kwargs
                )
            finally:
                self.delete_adapter(initial_adapter_name)
            return output_state_dict

        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            self.create_or_update_model_card(save_directory)

        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
                save_embedding_layers=save_embedding_layers,
            )
            output_dir = os.path.join(save_directory, f"adapter_{adapter_name}")

            os.makedirs(output_dir, exist_ok=True)

            # Save the metadata required to consolidate the checkpoints properly.
            # Note: we do not need to do it for each adapter, as the metadata is the same for all adapters but we do it
            # for simplicity.
            if get_data_parallel_rank() == 0 and get_tensor_model_parallel_rank() == 0:
                metadata = create_parameter_metadata(self)
                pp_rank = get_pipeline_model_parallel_rank()
                metadata_path = os.path.join(
                    output_dir, ADAPTER_MODEL_PARALLEL_SHARDS_DIR_NAME, f"mp_metadata_pp_rank_{pp_rank}.json"
                )
                Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, "w") as f:
                    f.write(json.dumps(metadata, indent=4))

            if is_main_process:
                if path_initial_model_for_weight_conversion is not None:
                    peft_config = copy.deepcopy(peft_config)
                    peft_config.init_lora_weights = True
                    peft_config.save_pretrained(path_initial_model_for_weight_conversion)
                    output_state_dict = save_mutated_as_lora(
                        peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs
                    )

            # Save the adapter weights.
            neuronx_distributed.trainer.save_checkpoint(
                output_dir,
                tag=ADAPTER_MODEL_PARALLEL_SHARDS_DIR_NAME,
                model=output_state_dict,
                use_xser=self.trn_config.use_xser,
                async_save=self.trn_config.async_save,
                num_workers=self.trn_config.num_local_ranks_per_step,
            )

            # Save the config and change the inference mode to `True`

            # We first transform the PEFT config to the original one for the original model implementation.
            peft_config = to_original_peft_config_for_model(self.base_model, peft_config, inplace=False)

            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if peft_config.is_prompt_learning
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            if peft_config.task_type is None:
                # deal with auto mapping
                base_model_class = self._get_base_model_class(
                    is_prompt_tuning=peft_config.is_prompt_learning,
                )
                parent_library = base_model_class.__module__

                auto_mapping_dict = {
                    "base_model_class": base_model_class.__name__,
                    "parent_library": parent_library,
                }
            else:
                auto_mapping_dict = None

            if is_main_process:
                if path_initial_model_for_weight_conversion is not None:
                    peft_config.init_lora_weights = True
                    peft_config.r *= 2
                    if not peft_config.use_rslora:
                        peft_config.lora_alpha *= 2
                    else:
                        # with rslora, we have scaling = alpha / sqrt(r), we thus adjust alpha to keep the same scaling
                        peft_config.lora_alpha *= 2**0.5

                    if peft_config.rank_pattern:
                        peft_config.rank_pattern = {key: 2 * val for key, val in peft_config.rank_pattern.items()}
                    if peft_config.alpha_pattern:
                        peft_config.alpha_pattern = {key: 2 * val for key, val in peft_config.alpha_pattern.items()}

                peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: str | os.PathLike,
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: PeftConfig | None = None,
        autocast_adapter_dtype: bool = True,
        **kwargs: Any,
    ) -> PeftModel:
        from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING

        # load the config
        if config is None:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                    use_auth_token=kwargs.get("use_auth_token", None),
                    token=kwargs.get("token", None),
                )
            ].from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        # ** Difference from original from_pretrained **
        # No runtime configuration here.

        # ** Difference from original from_pretrained **
        # No hf_device_map here.

        if config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable
        if isinstance(getattr(model, "base_model", None), XLoraModel):
            raise NotImplementedError(
                "XLoraModel is not supported in Optimum Neuron. Please use open an issue or a PR if needed."
            )

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(
                model,
                config,
                adapter_name,
                autocast_adapter_dtype=autocast_adapter_dtype,
            )
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](
                model,
                config,
                adapter_name,
                autocast_adapter_dtype=autocast_adapter_dtype,
            )

        load_result = model.load_adapter(
            model_id,
            adapter_name,
            is_trainable=is_trainable,
            autocast_adapter_dtype=autocast_adapter_dtype,
            **kwargs,
        )

        # 1. Remove VB-LoRA vector bank, since it's a shared parameter set via the VBLoRAModel
        # 2. Remove the prompt encoder, as it does not need to be part of the checkpoint
        missing_keys = [
            k for k in load_result.missing_keys if "vblora_vector_bank" not in k and "prompt_encoder" not in k
        ]
        if missing_keys:
            # Let's warn here since (in contrast to load_adapter) we don't return the load result, so it could be quite
            # difficult for users to even notice that something might have gone wrong here. As we filter out non PEFT
            # keys from the missing keys, this gives no false positives.
            warnings.warn(f"Found missing adapter keys while loading the checkpoint: {missing_keys}")

        return model

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False) -> None:
        peft_config = adapt_peft_config_for_model(self, peft_config, inplace=False)
        super().add_adapter(adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage)
        # We need to update the names of the parameters for the current stage after adding a new adapter.
        self.recompute_parameters_for_current_stage()

    def load_adapter(
        self,
        model_id: str | os.PathLike,
        adapter_name: str,
        is_trainable: bool = False,
        torch_device: str | None = None,
        autocast_adapter_dtype: bool = True,
        **kwargs: Any,
    ):
        low_cpu_mem_usage = False

        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        if torch_device is None:
            torch_device = self.base_model.device if hasattr(self.base_model, "device") else "cpu"

        if adapter_name not in self.peft_config:
            # load the config
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    **hf_hub_download_kwargs,
                )
            ].from_pretrained(
                model_id,
                **hf_hub_download_kwargs,
            )
            self._check_new_adapter_config(peft_config, is_trainable=is_trainable)
            peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage)

        adapters_weights = load_peft_weights(model_id, **hf_hub_download_kwargs)

        # ** Difference from original load_adapter **
        # We need to adapt the adapters_weights to the model.
        upstanding_sharded_params = {}
        adapters_weights = adapt_state_dict(
            self, adapters_weights, upstanding_sharded_params, inplace=True, adapter_name=adapter_name
        )

        # load the weights into the model
        ignore_mismatched_sizes = kwargs.get("ignore_mismatched_sizes", False)
        load_result = set_peft_model_state_dict(
            self,
            adapters_weights,
            adapter_name=adapter_name,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        tuner = self.peft_config[adapter_name].peft_type
        tuner_prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(tuner, "")
        adapter_missing_keys = []

        # Filter missing keys specific to the current adapter and tuner prefix.
        for key in load_result.missing_keys:
            if tuner_prefix in key and adapter_name in key:
                adapter_missing_keys.append(key)

        load_result.missing_keys.clear()
        load_result.missing_keys.extend(adapter_missing_keys)

        # ** Difference from original load_adapter **
        # No hf_device_map here.

        if hasattr(self.base_model, "_cast_adapter_dtype"):
            self.base_model._cast_adapter_dtype(
                adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
            )

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.eval()
        return load_result

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        return _get_model_param_count(self)


class NeuronPeftModelForCausalLM(NeuronPeftModel):
    pass
