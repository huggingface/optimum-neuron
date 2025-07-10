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
from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import CompletionSequenceGroupOutput, Logprob, SequenceOutput

from ..cache import get_hub_cached_entries
from ..configuration_utils import NeuronConfig
from ..modeling_decoder import NeuronModelForCausalLM
from ..utils import map_torch_dtype
from ..utils.system import get_available_cores
from ..utils.version_utils import get_neuronxcc_version


logger = logging.getLogger("Neuron")

available_cores = get_available_cores()
neuronxcc_version = get_neuronxcc_version()


class OptimumNeuronModelForCausalLM(nn.Module):
    def __init__(self, model: NeuronModelForCausalLM) -> None:
        super().__init__()
        self.model = model
        self.logits_processor = LogitsProcessor(self.model.config.vocab_size, logits_as_input=True)
        self.sampler = Sampler()

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
            attention_mask=None,
            position_ids=position_ids,
            seq_ids=sorted_seq_ids,
            sampling_params=sampling_params,
        )
        # on-device sampling
        if self.model.neuron_config.on_device_sampling:
            output = output.hidden_states
        else:
            output = output.logits[:, -1, :]

        restored_indices = torch.argsort(sorted_indices)
        if seq_ids.shape[0] != 1:
            output = torch.index_select(output, 0, restored_indices)

        return output

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        # on-device sampling
        if self.model.neuron_config.on_device_sampling:
            batch_size = logits.shape
            seq_ids = [seq_id for sg in sampling_metadata.seq_groups for seq_id in sg.seq_ids]
            assert len(seq_ids) == list(batch_size)[0], "batch size mismatch"
            # Organize input tensors by step instead of by sequence.
            accepted_token_ids_by_step = logits.flatten()
            accepted_token_ids_by_step = accepted_token_ids_by_step.tolist()

            step_output_token_ids = []
            for i, seq_id in enumerate(seq_ids):
                token_id = accepted_token_ids_by_step[i]
                step_output_token_ids.append(
                    CompletionSequenceGroupOutput(
                        samples=[
                            SequenceOutput(
                                parent_seq_id=seq_id, output_token=token_id, logprobs={token_id: Logprob(token_id)}
                            )
                        ],
                        prompt_logprobs=None,
                    )
                )
            return SamplerOutput(outputs=step_output_token_ids)
        else:
            return self.sampler(logits, sampling_metadata)


def check_neuron_config_compatibility(
    neuron_config_dict: dict,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    tensor_parallel_size: int | None = None,
    torch_dtype: str | torch.dtype | None = None,
) -> bool:
    """Check if the cached entry is compatible with the current environment."""
    logger.debug(
        "Checking the provided neuron config %s is compatible with the local setup and provided environment",
        neuron_config_dict,
    )

    # Local setup compat checks
    if neuron_config_dict["tp_degree"] > available_cores:
        logger.debug("Not enough neuron cores available to use tp_degree %d", neuron_config_dict["tp_degree"])
        return False

    if neuron_config_dict["neuronxcc_version"] != neuronxcc_version:
        logger.debug(
            "Compiler version conflict, the local one (%s) differs from the one used to compile the model (%s)",
            neuronxcc_version,
            neuron_config_dict["neuronxcc_version"],
        )
        return False
    if neuron_config_dict["batch_size"] > 1 and not neuron_config_dict["continuous_batching"]:
        logger.debug("Continuous batching is not enabled")
        return False
    if batch_size is not None and neuron_config_dict["batch_size"] != batch_size:
        logger.debug(
            "The target batch size %d is different from the neuron config batch size %d",
            batch_size,
            neuron_config_dict["batch_size"],
        )
        return False
    if sequence_length is not None and neuron_config_dict["sequence_length"] != sequence_length:
        logger.debug(
            "The target sequence length %d is different from the neuron  config sequence length %d",
            sequence_length,
            neuron_config_dict["sequence_length"],
        )
        return False
    if tensor_parallel_size is not None and neuron_config_dict["tp_degree"] != tensor_parallel_size:
        logger.debug(
            "The target parallel size %d is different from the neuron config tp degree %d",
            tensor_parallel_size,
            neuron_config_dict["tp_degree"],
        )
        return False
    if torch_dtype is not None:
        neuron_config_value = map_torch_dtype(str(neuron_config_dict["torch_dtype"]))
        target_value = map_torch_dtype(torch_dtype)
        if target_value != neuron_config_value:
            logger.debug(
                "The target dtype %s is different from the neuron config dtype %s", target_value, neuron_config_value
            )
            return False

    return True


def _is_cached(
    model_id: str,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    tensor_parallel_size: int | None = None,
    torch_dtype: str | torch.dtype | None = None,
) -> bool:
    # Look for cached entries for the specified model
    in_cache = False
    entries = get_hub_cached_entries(model_id)
    # Look for compatible entries
    for entry in entries:
        if check_neuron_config_compatibility(
            entry,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tensor_parallel_size=tensor_parallel_size,
            torch_dtype=torch_dtype,
        ):
            in_cache = True
            break
    return in_cache


def get_optimum_neuron_model(
    model_config: ModelConfig, parallel_config: ParallelConfig, scheduler_config: SchedulerConfig
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
    if tp_degree > available_cores:
        raise ValueError(
            f"The specified tensor parallelism degree ({tp_degree}) is higher"
            f" than the number of available Neuron cores ({available_cores})."
            " Please set tensor_parallel_size to a value less than or equal "
            "to the number of available Neuron cores."
        )
    model_id = model_config.model
    revision = model_config.revision or "main"
    token = model_config.hf_token
    try:
        # Look for a NeuronConfig in the model directory
        neuron_config = NeuronConfig.from_pretrained(model_id, revision=revision, token=token)
    except Exception:
        neuron_config = None
    if neuron_config is not None:
        neuron_model = NeuronModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            token=token,
            export=False,
        )
    else:
        # Model needs to be exported: look for compatible hub cached configs
        batch_size = scheduler_config.max_num_seqs
        sequence_length = scheduler_config.max_model_len
        tensor_parallel_size = parallel_config.tensor_parallel_size
        torch_dtype = None if model_config.dtype is None else model_config.dtype
        if not _is_cached(
            model_id,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tensor_parallel_size=tensor_parallel_size,
            torch_dtype=torch_dtype,
        ):
            hub_cache_url = "https://huggingface.co/aws-neuron/optimum-neuron-cache"  # noqa: E501
            neuron_export_url = "https://huggingface.co/docs/optimum-neuron/main/en/guides/export_model#exporting-neuron-models-using-neuronx-tgi"  # noqa: E501
            error_msg = (
                f"No cached version found for {model_id} with "
                f"batch size = {batch_size}, seq len = {sequence_length},"
                f" tp = {tensor_parallel_size}, dtype = {torch_dtype}."
                f"You can start a discussion to request it on {hub_cache_url}"
                f"Alternatively, you can export your own neuron model "
                f"as explained in {neuron_export_url}"
            )
            raise ValueError(error_msg)
        logger.warning("%s is not a neuron model: it will be exported using cached artifacts.", model_id)
        neuron_model = NeuronModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            token=token,
            export=True,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_cores=tensor_parallel_size,
            auto_cast_type=torch_dtype,
        )
    return OptimumNeuronModelForCausalLM(neuron_model)
