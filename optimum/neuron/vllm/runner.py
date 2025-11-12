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
"""An optimum-neuron vLLM runner class."""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import DeviceConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from .model_loader import OptimumNeuronModelForCausalLM, get_optimum_neuron_model


logger = logging.getLogger("Neuron")


@dataclass
class OptimumNeuronCachedRequest:
    """Holds cached requests for optimum-neuron runner."""

    req_id: str
    sampling_params: SamplingParams
    num_prompt_tokens: int
    output_token_ids: list[int] | None = None

    def __post_init__(self):
        if self.sampling_params.temperature == 0.0:
            # For vLLM zero temperature means greedy decoding, but Neuron uses top_k=1
            self.sampling_params.top_k = 1
            self.sampling_params.top_p = 1.0
            self.sampling_params.temperature = 1.0
        self.output_token_ids = []

    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)


class OptimumNeuronCachedBatch:
    """Holds batch state for optimum-neuron runner."""

    def __init__(self, vllm_config: VllmConfig):
        # Initialize cached request states (one for each sequence slot in the batch)
        self.cached_requests: list[OptimumNeuronCachedRequest | None] = [
            None,
        ] * vllm_config.scheduler_config.max_num_seqs

    def add_request(self, new_request_data: NewRequestData) -> int:
        seq_id = -1
        for i, cached_request in enumerate(self.cached_requests):
            assert new_request_data.prompt_token_ids is not None
            if cached_request is None:
                self.cached_requests[i] = OptimumNeuronCachedRequest(
                    req_id=new_request_data.req_id,
                    sampling_params=new_request_data.sampling_params,
                    num_prompt_tokens=len(new_request_data.prompt_token_ids),
                )
                seq_id = i
                break
        return seq_id

    def remove_requests(self, req_ids: set[str]) -> None:
        for i, cached_request in enumerate(self.cached_requests):
            if cached_request is not None and cached_request.req_id in req_ids:
                logger.info(f"Removing request {cached_request.req_id} at index {i} from the cached batch.")
                self.cached_requests[i] = None
                break

    def request(self, req_id: str) -> tuple[OptimumNeuronCachedRequest, int]:
        for seq_id, cached_request in enumerate(self.cached_requests):
            if cached_request is not None and cached_request.req_id == req_id:
                return cached_request, seq_id
        raise KeyError(f"Request id {req_id} not found in the cached requests.")

    @property
    def req_ids(self) -> list[str]:
        req_ids: list[str] = []
        for cached_request in self.cached_requests:
            if cached_request is not None:
                req_ids.append(cached_request.req_id)
        return req_ids


class OptimumNeuronModelRunner:
    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.model: OptimumNeuronModelForCausalLM | None = None
        device_config = self.device_config if self.device_config is not None else DeviceConfig()
        self.device = device_config.device
        self.pin_memory = is_pin_memory_available()
        self.batch: OptimumNeuronCachedBatch = OptimumNeuronCachedBatch(vllm_config)

    def load_model(self) -> None:
        self.model = get_optimum_neuron_model(
            self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            load_config=self.load_config,
        )

    def tensor_for_sampling_params(self, sampling_params: list[SamplingParams]) -> torch.Tensor:
        if self.model.model.neuron_config.on_device_sampling:
            max_topk = self.model.model.neuron_config.max_topk
        else:
            max_topk = self.model.model.config.vocab_size

        sampling_params_list: list[list[Any]] = []
        for params in sampling_params:
            top_k = params.top_k if params.top_k > 0 else max_topk
            top_k = min(top_k, max_topk)
            sampling_params_list.append([top_k, params.top_p, params.temperature])
        return torch.tensor(sampling_params_list, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        # Remove finished requests from the cached states.
        self.batch.remove_requests(scheduler_output.finished_req_ids)

        n_prompt_reqs = len(scheduler_output.scheduled_new_reqs)
        n_decode_reqs = len(scheduler_output.scheduled_cached_reqs.req_ids)

        if n_prompt_reqs == 0 and n_decode_reqs == 0:
            logger.info("No requests to schedule.")
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )

        # Note: We cannot process prompt and decode requests together,
        # because they use different graphs.

        def get_next_tokens(
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            seq_ids: torch.LongTensor,
            sampling_params: SamplingParams,
        ):
            assert self.model is not None

            sampling_params = self.tensor_for_sampling_params(sampling_params)

            logits_or_tokens = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
            )

            if self.model.model.neuron_config.on_device_sampling:
                return logits_or_tokens
            else:
                # Sample tokens from logits
                return self.model.sample(
                    logits=logits_or_tokens,
                    sampling_params=sampling_params,
                )

        # We start by decoding the next tokens for the requests already in the batch.
        req_ids = []
        tokens = None
        if n_decode_reqs > 0:
            req_ids = scheduler_output.scheduled_cached_reqs.req_ids
            (input_ids, position_ids, seq_ids, sampling_params) = self._prepare_decode(
                scheduler_output.scheduled_cached_reqs
            )
            tokens = get_next_tokens(input_ids, position_ids, seq_ids, sampling_params)
            logger.info(f"Generated {n_decode_reqs} new tokens from cached requests: {tokens}")

        # Then process new prompt requests.
        if n_prompt_reqs > 0:
            (prompt_req_ids, input_ids, position_ids, seq_ids, sampling_params) = self._prepare_prompt(
                scheduler_output
            )
            req_ids += prompt_req_ids
            prompt_tokens = get_next_tokens(input_ids, position_ids, seq_ids, sampling_params)
            logger.info(f"Generated {n_prompt_reqs} tokens from new requests: {prompt_tokens}")
            tokens = prompt_tokens if tokens is None else torch.cat([tokens, prompt_tokens], dim=0)

        sampled_token_ids: list[list[int]] = []
        for i, req_id in enumerate(req_ids):
            token = tokens[i].item()
            # Cache the sampled tokens in the model runner, because the scheduler
            # won't send them back.
            cached_request, _ = self.batch.request(req_id)
            cached_request.output_token_ids.append(token)
            sampled_token_ids.append([token])

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(req_ids),
        )

    def _prepare_prompt(
        self,
        scheduler_output: SchedulerOutput,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, list[SamplingParams]]:
        assert len(scheduler_output.scheduled_new_reqs) > 0
        req_ids: list[str] = []
        input_tokens: list[list[int]] = []
        input_positions: list[list[int]] = []
        input_seq_ids: list[int] = []
        sampling_params: list[SamplingParams] = []

        seq_lens: list[int] = []
        for new_request_data in scheduler_output.scheduled_new_reqs:
            seq_id = self.batch.add_request(new_request_data)
            if seq_id == -1:
                raise RuntimeError("No available sequence slot for the new request.")
            req_ids.append(new_request_data.req_id)
            input_seq_ids.append(seq_id)
            assert new_request_data.prompt_token_ids is not None
            seq_len = len(new_request_data.prompt_token_ids)
            seq_lens.append(seq_len)
            input_tokens.append(new_request_data.prompt_token_ids)
            input_positions.append(list(range(seq_len)))
            assert new_request_data.sampling_params is not None
            sampling_params.append(new_request_data.sampling_params)

        max_seq_len = max(seq_lens)
        assert max_seq_len > 0
        input_ids = make_tensor_with_pad(
            input_tokens, pad=0, max_len=max_seq_len, dtype=torch.long, device=self.device
        )
        position_ids = make_tensor_with_pad(
            input_positions, pad=0, max_len=max_seq_len, dtype=torch.long, device=self.device
        )
        seq_ids = torch.tensor(input_seq_ids, dtype=torch.long, device=self.device)

        return (req_ids, input_ids, position_ids, seq_ids, sampling_params)

    def _prepare_decode(
        self,
        cached_request_data: CachedRequestData,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[SamplingParams]]:
        # Sanity check
        assert len(cached_request_data.req_ids) > 0

        if sorted(self.batch.req_ids) != sorted(cached_request_data.req_ids):
            # FIXME: This should not happen, log a warning for now.
            # The reason is probably that we don't remove finished requests from the batch
            # when we generated an EOS token for them.
            logger.warning("Mismatch between batch cached requests and scheduler cached requests.")
            logger.warning(f"Batch req_ids: {self.batch.req_ids}")
            logger.warning(f"Cached req_ids: {cached_request_data.req_ids}")

        input_tokens: list[list[int]] = []
        input_positions: list[list[int]] = []
        input_seq_ids: list[int] = []
        sampling_params: list[SamplingParams] = []
        for req_id in cached_request_data.req_ids:
            cached_request, seq_id = self.batch.request(req_id)
            input_tokens.append(cached_request.output_token_ids[-1:])
            input_positions.append([cached_request.num_tokens() - 1])
            input_seq_ids.append(seq_id)
            sampling_params.append(cached_request.sampling_params)

        input_ids = make_tensor_with_pad(input_tokens, pad=0, max_len=1, dtype=torch.long, device=self.device)
        position_ids = make_tensor_with_pad(input_positions, pad=0, max_len=1, dtype=torch.long, device=self.device)
        seq_ids = torch.tensor(input_seq_ids, dtype=torch.long, device=self.device)

        return input_ids, position_ids, seq_ids, sampling_params
