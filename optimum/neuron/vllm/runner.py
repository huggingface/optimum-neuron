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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import DeviceConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

from .model_loader import OptimumNeuronModel, OptimumNeuronModelForCausalLM, OptimumNeuronModelForEmbedding
from .sampler import NeuronSampler


logger = logging.getLogger("Neuron")


@dataclass
class OptimumNeuronCachedRequest:
    """Holds cached requests for optimum-neuron runner."""

    req_id: str
    seq_id: int
    sampling_params: SamplingParams
    prompt_token_ids: list[int]
    output_token_ids: list[int] | None = None

    def __post_init__(self):
        if self.sampling_params.temperature == 0.0:
            # For vLLM zero temperature means greedy decoding, but Neuron uses top_k=1
            self.sampling_params.top_k = 1
            self.sampling_params.top_p = 1.0
            self.sampling_params.temperature = 1.0
        self.output_token_ids = []

    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)


class OptimumNeuronCachedBatch:
    """Holds batch state for optimum-neuron runner."""

    def __init__(self, vllm_config: VllmConfig):
        # Initialize cached request states (one for each sequence slot in the batch)
        self.cached_requests: list[OptimumNeuronCachedRequest | None] = [
            None,
        ] * vllm_config.scheduler_config.max_num_seqs

    def add_request(self, new_request_data: NewRequestData) -> OptimumNeuronCachedRequest | None:
        request = None
        for i, cached_request in enumerate(self.cached_requests):
            assert new_request_data.prompt_token_ids is not None
            if cached_request is None:
                request = OptimumNeuronCachedRequest(
                    req_id=new_request_data.req_id,
                    seq_id=i,
                    prompt_token_ids=new_request_data.prompt_token_ids,
                    sampling_params=new_request_data.sampling_params,
                )
                logger.info(f"Added new request {request.req_id} at index {i} to the cached batch.")
                self.cached_requests[i] = request
                break
        return request

    def remove_requests(self, req_ids: set[str]) -> None:
        for i, cached_request in enumerate(self.cached_requests):
            if cached_request is not None and cached_request.req_id in req_ids:
                logger.info(f"Removed request {cached_request.req_id} at index {i} from the cached batch.")
                self.cached_requests[i] = None

    def request(self, req_id: str) -> OptimumNeuronCachedRequest:
        for cached_request in self.cached_requests:
            if cached_request is not None and cached_request.req_id == req_id:
                return cached_request
        raise KeyError(f"Request id {req_id} not found in the cached requests.")

    @property
    def req_ids(self) -> set[str]:
        req_ids: set[str] = set()
        for cached_request in self.cached_requests:
            if cached_request is not None:
                req_ids.add(cached_request.req_id)
        return req_ids

    @property
    def capacity(self) -> int:
        return len(self.cached_requests)

    @property
    def num_cached_requests(self) -> int:
        return sum(1 for r in self.cached_requests if r is not None)


def create_sampling_metadata(requests: list[OptimumNeuronCachedRequest], vocab_size: int) -> SamplingMetadata:
    all_greedy = True
    all_random = True
    temperature = []
    top_p = []
    top_k = []
    max_num_logprobs = None
    no_penalties = True
    frequency_penalties = []
    presence_penalties = []
    repetition_penalties = []
    prompt_token_ids_list = []
    max_prompt_len = 0
    output_token_ids_list = []
    max_output_len = 0
    allowed_token_ids_mask: torch.Tensor | None = None
    bad_word_tokens_ids = {}
    for request in requests:
        if request.sampling_params.top_k == 1:
            all_random = False
        else:
            all_greedy = False
        temperature.append(request.sampling_params.temperature)
        top_p.append(request.sampling_params.top_p)
        top_k.append(request.sampling_params.top_k if request.sampling_params.top_k > 0 else vocab_size)
        if request.sampling_params.logprobs is not None:
            logprobs = request.sampling_params.logprobs
            if max_num_logprobs is None:
                max_num_logprobs = logprobs
            else:
                max_num_logprobs = max(logprobs, max_num_logprobs)
        if (
            request.sampling_params.frequency_penalty != 0.0
            or request.sampling_params.presence_penalty != 0.0
            or request.sampling_params.repetition_penalty != 0.0
        ):
            no_penalties = False
        max_prompt_len = max(max_prompt_len, len(request.prompt_token_ids))
        prompt_token_ids_list.append(request.prompt_token_ids)
        assert request.output_token_ids is not None
        output_token_ids_list.append(request.output_token_ids)
        max_output_len = max(max_output_len, len(request.output_token_ids))
        frequency_penalties.append(request.sampling_params.frequency_penalty)
        presence_penalties.append(request.sampling_params.presence_penalty)
        repetition_penalties.append(request.sampling_params.repetition_penalty)
        if request.sampling_params.allowed_token_ids is not None:
            if allowed_token_ids_mask is None:
                # Even if only one request uses allowed mask, we need to create the
                # Tensor for the whole batch
                allowed_token_ids_mask = torch.full((len(requests), vocab_size), True)
            allowed_token_ids_mask[request.seq_id] = False
            for token_id in request.sampling_params.allowed_token_ids:
                allowed_token_ids_mask[request.seq_id, token_id] = True
        if request.sampling_params.bad_words_token_ids is not None:
            bad_word_tokens_ids[request.seq_id] = request.sampling_params.bad_words_token_ids

    for i, prompt_token_ids in enumerate(prompt_token_ids_list):
        prompt_token_ids_list[i] = prompt_token_ids + [vocab_size] * (max_prompt_len - len(prompt_token_ids))

    # Note that we pass an empty logits processors for now, as even though we added the builtin processors,
    # we don't support updating them when the batch changes.
    # This means that the following features provided by the builtin logits processors are not supported yet:
    # - min_p
    # - logits_bias
    # - min_length
    return SamplingMetadata(
        temperature=torch.tensor(temperature),
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=torch.tensor(top_p),
        top_k=torch.tensor(top_k),
        generators={},
        max_num_logprobs=max_num_logprobs,
        no_penalties=no_penalties,
        prompt_token_ids=torch.tensor(prompt_token_ids_list),
        frequency_penalties=torch.tensor(frequency_penalties),
        presence_penalties=torch.tensor(presence_penalties),
        repetition_penalties=torch.tensor(repetition_penalties),
        output_token_ids=output_token_ids_list,
        allowed_token_ids_mask=allowed_token_ids_mask,
        bad_words_token_ids=bad_word_tokens_ids,
        logitsprocs=LogitsProcessors(),  # Empty logits processors for now
    )


class OptimumNeuronModelRunner(ABC):
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
        device_config = self.device_config if self.device_config is not None else DeviceConfig()
        self.device = device_config.device
        self.pin_memory = is_pin_memory_available()

    @staticmethod
    def create(vllm_config: VllmConfig) -> "OptimumNeuronModelRunner":
        task = vllm_config.model_config.task or "generate"
        if task == "generate":
            return OptimumNeuronModelRunnerForCausalLM(vllm_config)
        elif task == "embed":
            return OptimumNeuronModelRunnerForEmbedding(vllm_config)
        else:
            raise ValueError(f"Task {task} is not supported for Neuron.")

    @abstractmethod
    def get_supported_tasks(self) -> tuple[str, ...]:
        raise NotImplementedError()


class OptimumNeuronModelRunnerForCausalLM(OptimumNeuronModelRunner):
    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        super().__init__(vllm_config)
        self.model: OptimumNeuronModel | None = None
        self.batch: OptimumNeuronCachedBatch = OptimumNeuronCachedBatch(vllm_config)
        self.logitsproc = LogitsProcessors()

    def load_model(self) -> None:
        self.model = OptimumNeuronModelForCausalLM.create(
            self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            load_config=self.load_config,
        )
        if not self.model.model.neuron_config.on_device_sampling:
            self.sampler = NeuronSampler(self.model.model.neuron_config)

    def get_supported_tasks(self) -> tuple[str, ...]:
        return ("generate",)

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
        if len(scheduler_output.finished_req_ids) > 0:
            self.batch.remove_requests(scheduler_output.finished_req_ids)
            logger.info(
                f"Number of cached requests in the batch: {self.batch.num_cached_requests}/{self.batch.capacity}"
            )

        n_prompt_reqs = len(scheduler_output.scheduled_new_reqs)
        n_decode_reqs = len(scheduler_output.scheduled_cached_reqs.req_ids)

        if n_prompt_reqs == 0 and n_decode_reqs == 0:
            logger.debug("No requests to schedule.")
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
            requests: list[OptimumNeuronCachedRequest],
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            seq_ids: torch.Tensor,
            sampling_params: list[SamplingParams],
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            assert self.model is not None

            sampling_params_tensor = self.tensor_for_sampling_params(sampling_params)

            if self.model.model.neuron_config.on_device_sampling:
                return self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    seq_ids=seq_ids,
                    sampling_params=sampling_params_tensor,
                ), None
            else:
                logits = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    seq_ids=seq_ids,
                    sampling_params=sampling_params_tensor,
                )
                sampling_metadata = create_sampling_metadata(requests, vocab_size=self.model.model.config.vocab_size)
                sampler_outputs = self.sampler(logits, sampling_metadata)
                return sampler_outputs.sampled_token_ids, logits

        # We start by decoding the next tokens for the requests already in the batch.
        req_ids = []
        tokens = None
        logprobs = None
        if n_decode_reqs > 0:
            req_ids = scheduler_output.scheduled_cached_reqs.req_ids
            (requests, input_ids, position_ids, seq_ids, sampling_params) = self._prepare_decode(
                scheduler_output.scheduled_cached_reqs
            )
            tokens, logprobs = get_next_tokens(requests, input_ids, position_ids, seq_ids, sampling_params)
            logger.debug(f"Generated {n_decode_reqs} new tokens from cached requests: {tokens}")

        # Then process new prompt requests.
        if n_prompt_reqs > 0:
            (requests, input_ids, position_ids, seq_ids, sampling_params) = self._prepare_prompt(scheduler_output)
            req_ids += [request.req_id for request in requests]
            prompt_tokens, prompt_logprobs = get_next_tokens(
                requests, input_ids, position_ids, seq_ids, sampling_params
            )
            logger.info(
                f"Number of cached requests in the batch: {self.batch.num_cached_requests}/{self.batch.capacity}"
            )
            logger.debug(f"Generated {n_prompt_reqs} tokens from new requests: {prompt_tokens}")
            tokens = prompt_tokens if tokens is None else torch.cat([tokens, prompt_tokens], dim=0)
            logprobs = prompt_logprobs if logprobs is None else torch.cat([logprobs, prompt_logprobs], dim=0)

        # TODO: return logprobs when they are available (for now only if on_device_sampling is disabled)

        sampled_token_ids: list[list[int]] = []
        for i, req_id in enumerate(req_ids):
            token = tokens[i].item()
            # Cache the sampled tokens in the model runner, because the scheduler
            # won't send them back.
            cached_request = self.batch.request(req_id)
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
    ) -> tuple[list[OptimumNeuronCachedRequest], torch.Tensor, torch.Tensor, torch.Tensor, list[SamplingParams]]:
        assert len(scheduler_output.scheduled_new_reqs) > 0
        requests: list[OptimumNeuronCachedRequest] = []
        input_tokens: list[list[int]] = []
        input_positions: list[list[int]] = []
        input_seq_ids: list[int] = []
        sampling_params: list[SamplingParams] = []

        seq_lens: list[int] = []
        for new_request_data in scheduler_output.scheduled_new_reqs:
            request = self.batch.add_request(new_request_data)
            if request is None:
                raise RuntimeError("No available sequence slot for the new request.")
            requests.append(request)
            input_seq_ids.append(request.seq_id)
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

        return (requests, input_ids, position_ids, seq_ids, sampling_params)

    def _prepare_decode(
        self,
        cached_request_data: CachedRequestData,
    ) -> tuple[list[OptimumNeuronCachedRequest], torch.Tensor, torch.Tensor, torch.Tensor, list[SamplingParams]]:
        scheduled_req_ids = set(cached_request_data.req_ids)
        # Sanity check
        assert len(scheduled_req_ids) > 0

        if self.batch.req_ids != scheduled_req_ids:
            if not scheduled_req_ids.issubset(self.batch.req_ids):
                logger.error(
                    "The scheduled cached requests contain request ids not present in the batch."
                    f" Scheduled: {scheduled_req_ids}, Batch: {self.batch.req_ids}"
                )
                raise RuntimeError("Inconsistent scheduled cached requests and batch state.")
            # Unscheduled requests are in the batch but not scheduled for decoding.
            # Theoretically they could be rescheduled in the next iteration.
            # For now we log a warning since it is unexpected.
            unscheduled_req_ids = self.batch.req_ids - scheduled_req_ids
            logger.warning(f"Request ids {unscheduled_req_ids} are in the batch but not scheduled for decoding.")

        requests: list[OptimumNeuronCachedRequest] = []
        input_tokens: list[list[int]] = []
        input_positions: list[list[int]] = []
        input_seq_ids: list[int] = []
        sampling_params: list[SamplingParams] = []
        for req_id in cached_request_data.req_ids:
            cached_request = self.batch.request(req_id)
            requests.append(cached_request)
            assert cached_request.output_token_ids is not None
            input_tokens.append(cached_request.output_token_ids[-1:])
            input_positions.append([cached_request.num_tokens() - 1])
            input_seq_ids.append(cached_request.seq_id)
            sampling_params.append(cached_request.sampling_params)

        input_ids = make_tensor_with_pad(input_tokens, pad=0, max_len=1, dtype=torch.long, device=self.device)
        position_ids = make_tensor_with_pad(input_positions, pad=0, max_len=1, dtype=torch.long, device=self.device)
        seq_ids = torch.tensor(input_seq_ids, dtype=torch.long, device=self.device)

        return requests, input_ids, position_ids, seq_ids, sampling_params


class OptimumNeuronModelRunnerForEmbedding(OptimumNeuronModelRunner):
    def load_model(self) -> None:
        self.model = OptimumNeuronModelForEmbedding.create(
            self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            load_config=self.load_config,
        )

    def get_supported_tasks(self) -> tuple[str, ...]:
        return ("embed",)

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        n_reqs = len(scheduler_output.scheduled_new_reqs)

        if n_reqs == 0:
            logger.debug("No requests to schedule.")
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )

        input_ids, attention_mask = self._prepare_inputs(scheduler_output)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # For embedding models, the outputs should be the pooled embeddings
        # Each row corresponds to one request's embedding
        # Convert the (batch_size, 1, hidden_size) tensor to a list of 1D vectors, one per request
        pooler_output = [outputs[i, 0, :].to("cpu", non_blocking=False) for i in range(outputs.shape[0])]

        req_ids = [new_request_data.req_id for new_request_data in scheduler_output.scheduled_new_reqs]
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
        )

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(scheduler_output.scheduled_new_reqs) > 0
        input_tokens: list[list[int]] = []
        input_mask: list[list[int]] = []

        seq_lens: list[int] = []
        for new_request_data in scheduler_output.scheduled_new_reqs:
            assert new_request_data.prompt_token_ids is not None
            seq_len = len(new_request_data.prompt_token_ids)
            seq_lens.append(seq_len)
            input_tokens.append(new_request_data.prompt_token_ids)
            input_mask.append([1] * seq_len)

        max_seq_len = max(seq_lens)
        assert max_seq_len > 0
        input_ids = make_tensor_with_pad(
            input_tokens, pad=0, max_len=max_seq_len, dtype=torch.long, device=self.device
        )
        attention_mask = make_tensor_with_pad(
            input_mask, pad=0, max_len=max_seq_len, dtype=torch.long, device=self.device
        )

        return (input_ids, attention_mask)
