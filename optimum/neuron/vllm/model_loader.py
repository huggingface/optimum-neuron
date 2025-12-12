# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""An optimum-neuron vLLM model loader."""

import logging

import torch
import torch.nn as nn
from vllm.config import LoadConfig, ModelConfig, ParallelConfig, SchedulerConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from ..cache.hub_cache import select_hub_cached_entries
from ..configuration_utils import NeuronConfig
from ..models.inference.modeling_utils import NeuronModelForCausalLM
from ..utils.system import get_available_cores
from ..utils.version_utils import get_neuronxcc_version


logger = logging.getLogger("Neuron")

neuronxcc_version = get_neuronxcc_version()


class OptimumNeuronModelForCausalLM(nn.Module):
    def __init__(self, model: NeuronModelForCausalLM) -> None:
        super().__init__()
        self.model = model
        self.logits_processor = LogitsProcessor(self.model.config.vocab_size, logits_as_input=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        seq_ids: torch.Tensor,
        sampling_params: torch.Tensor,
    ) -> torch.Tensor:
        # sort block ids sequentially for perf/neuron support reasons
        sorted_seq_ids, sorted_indices = torch.sort(seq_ids)
        input_ids = torch.index_select(input_ids, 0, sorted_indices)
        position_ids = torch.index_select(position_ids, 0, sorted_indices)
        sampling_params = torch.index_select(sampling_params, 0, sorted_indices)

        output = self.model(
            input_ids,
            position_ids=position_ids,
            seq_ids=sorted_seq_ids,
            sampling_params=sampling_params,
        )
        # on-device sampling
        if self.model.neuron_config.on_device_sampling:
            output = output
        else:
            output = output[:, -1, :]

        restored_indices = torch.argsort(sorted_indices)
        if seq_ids.shape[0] != 1:
            output = torch.index_select(output, 0, restored_indices)

        return output


def get_optimum_neuron_model(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    scheduler_config: SchedulerConfig,
    load_config: LoadConfig,
) -> OptimumNeuronModelForCausalLM:
    """Initializes a neuron-optimized model for inference."""
    if parallel_config.pipeline_parallel_size > 1:
        raise ValueError(
            "optimum-neuron does not support pipeline parallelism. "
            "Please set pipeline_parallel_size to 1 in the parallel config."
        )
    if parallel_config.data_parallel_size > 1:
        raise ValueError(
            "optimum-neuron does not support data parallelism. "
            "Please set data_parallel_size to 1 in the parallel config."
        )
    tp_degree = parallel_config.tensor_parallel_size
    available_cores = get_available_cores()
    if tp_degree > available_cores:
        raise ValueError(
            f"The specified tensor parallelism degree ({tp_degree}) is higher"
            f" than the number of available Neuron cores ({available_cores})."
            " Please set tensor_parallel_size to a value less than or equal "
            "to the number of available Neuron cores."
        )
    model_id = model_config.served_model_name
    model_name_or_path = model_config.model
    revision = model_config.revision or "main"
    token = model_config.hf_token
    try:
        # Look for a NeuronConfig in the model directory
        neuron_config = NeuronConfig.from_pretrained(model_name_or_path, revision=revision, token=token)
    except EnvironmentError:
        neuron_config = None
    if neuron_config is not None:
        neuron_model = NeuronModelForCausalLM.from_pretrained(
            model_name_or_path,
            revision=revision,
            token=token,
        )
    else:
        # Model needs to be exported: look for compatible hub cached configs
        batch_size = scheduler_config.max_num_seqs
        sequence_length = scheduler_config.max_model_len
        torch_dtype = None if model_config.dtype is None else model_config.dtype

        cached_entries = select_hub_cached_entries(
            model_name_or_path,
            task="text-generation",
            batch_size=batch_size,
            sequence_length=sequence_length,
            tensor_parallel_size=tp_degree,
            torch_dtype=torch_dtype,
        )
        cached_only = load_config.model_loader_extra_config != "allow_non_cached_model"
        if len(cached_entries) == 0:
            if cached_only:
                hub_cache_url = "https://huggingface.co/aws-neuron/optimum-neuron-cache"  # noqa: E501
                neuron_export_url = "https://huggingface.co/docs/optimum-neuron/main/en/guides/export_model"  # noqa: E501
                error_msg = f"No cached version found for {model_id}"
                if batch_size is not None:
                    error_msg += f", batch size = {batch_size}"
                if sequence_length is not None:
                    error_msg += f", sequence length = {sequence_length},"
                if tp_degree is not None:
                    error_msg += f", tp = {tp_degree}"
                if torch_dtype is not None:
                    error_msg += f", dtype = {torch_dtype}"
                error_msg += (
                    f".You can start a discussion to request it on {hub_cache_url} "
                    "Alternatively, you can export your own neuron model "
                    f"as explained in {neuron_export_url}"
                )
                raise ValueError(error_msg)
            else:
                logger.warning("No cached version found for %s", model_id)
                if batch_size is not None:
                    logger.warning("  batch size = %d", batch_size)
                if sequence_length is not None:
                    logger.warning("  sequence length = %d", sequence_length)
                if tp_degree is not None:
                    logger.warning("  tp = %d", tp_degree)
                if torch_dtype is not None:
                    logger.warning("  dtype = %s", torch_dtype)
                logger.warning("The model will be exported on the fly, which may take some time.")
                logger.warning(
                    "To avoid this, you can set export parameters "
                    "in the model config corresponding to a cached configuration."
                )
        else:
            logger.warning("%s is not a neuron model: it will be exported using cached artifacts.", model_id)

        neuron_config = NeuronModelForCausalLM.get_neuron_config(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tensor_parallel_size=tp_degree,
        )
        neuron_model = NeuronModelForCausalLM.export(
            model_name_or_path,
            neuron_config=neuron_config,
            token=token,
            revision=revision,
            load_weights=True,
        )
    return OptimumNeuronModelForCausalLM(neuron_model)
