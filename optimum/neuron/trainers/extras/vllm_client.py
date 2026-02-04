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

import atexit
import random
import time
from collections import namedtuple
from typing import Union

import requests
import torch
import torch_xla
from optimum.utils import logging
from trl.extras.vllm_client import VLLMClient as TRLVLLMClient
from trl.import_utils import is_vllm_available


if is_vllm_available():
    from vllm.distributed.utils import StatelessProcessGroup
else:

    class StatelessProcessGroup:
        pass


logger = logging.get_logger()

# Set up the communication group for weight broadcasting using CPU communicator
Group = namedtuple("Group", "barrier")


class CPUCommunicator:
    def __init__(self, store, rank):
        self.rank = rank
        self.store = store
        self.group = Group(barrier=self.barrier)

    def broadcast(self, tensor, src):
        # Move tensor to CPU to ensure compatibility with vLLM server
        if tensor.device.type == "xla":
            tensor = tensor.cpu()
            torch_xla.sync()
        self.store.broadcast_obj(tensor, src=self.rank)

    def barrier(self):
        self.store.barrier()

    def __del__(self):
        del self.store


class VLLMClient(TRLVLLMClient):
    """VLLMClient with CPU-based communication for Neuron environments."""

    def __init__(
        self,
        base_url: str | None = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        super().__init__(
            base_url=base_url,
            host=host,
            server_port=server_port,
            group_port=group_port,
            connection_timeout=connection_timeout,
        )

    def init_communicator(self, device: Union[torch.device, str, int] = 0):
        # Get the world size from the server
        url = f"{self.base_url}/get_world_size/"
        response = requests.get(url)
        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = vllm_world_size + 1  # add the client to the world
        self.rank = vllm_world_size  # the client's rank is the last process

        # Initialize weight update group
        url = f"{self.base_url}/init_communicator/"

        # Use dummy UUID for CPU/Neuron environments
        client_device_uuid = "42"

        # In the server side, the host is set to 0.0.0.0
        response = self.session.post(
            url,
            json={
                "host": "0.0.0.0",
                "port": self.group_port,
                "world_size": world_size,
                "client_device_uuid": client_device_uuid,
            },
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Brief delay to allow server initialization. While not strictly required (client socket will retry on
        # connection failure), this prevents log warnings like:
        # [W416 23:24:57.460001114 socket.cpp:204] [c10d] The hostname of the client socket cannot be retrieved. err=-3
        time.sleep(0.1)

        pg = StatelessProcessGroup.create(host=self.host, port=self.group_port, rank=self.rank, world_size=world_size)
        self.communicator = CPUCommunicator(pg, self.rank)

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator)


class MockVLLMClient(VLLMClient):
    """
    Mock VLLMClient that generates completions without a vLLM server.

    Used for neuron_parallel_compile and testing. Generates completions by cycling
    through prompt tokens (echo mode), producing deterministic, non-garbage output.
    """

    def __init__(self, tokenizer, max_completion_length=256, min_completion_length=10, seed=None):
        self.tokenizer = tokenizer
        self.max_completion_length = max_completion_length
        self.min_completion_length = min(min_completion_length, max_completion_length)
        self.random = random.Random(seed)

        logger.warning(
            "Using MockVLLMClient for neuron_parallel_compile or testing. "
            "This generates echo completions and should only be used for compilation/testing."
        )

    def generate(
        self,
        prompts: list[str],
        images=None,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 256,
        repetition_penalty: float = 1.0,
        truncate_prompt_tokens=None,
        guided_decoding_regex=None,
        generation_kwargs=None,
    ):
        prompt_ids = []
        completion_ids = []
        logprobs = []

        # Fallback tokens if prompt is empty
        vocab_size = self.tokenizer.vocab_size
        fallback_token_id = min(100, vocab_size - 1)

        for prompt in prompts:
            # Tokenize prompt
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

            # Truncate if needed
            if truncate_prompt_tokens is not None and len(prompt_tokens) > truncate_prompt_tokens:
                prompt_tokens = prompt_tokens[-truncate_prompt_tokens:]

            prompt_ids.append(prompt_tokens)

            # Generate n completions per prompt
            for _ in range(n):
                # Random completion length within bounds
                max_len = min(max_tokens, self.max_completion_length)
                completion_length = self.random.randint(self.min_completion_length, max_len)

                # Echo mode: cycle through prompt tokens
                if len(prompt_tokens) > 0:
                    completion = [prompt_tokens[i % len(prompt_tokens)] for i in range(completion_length)]
                else:
                    # Fallback if prompt is empty
                    completion = [fallback_token_id] * completion_length

                completion_ids.append(completion)

                # Logprobs: simulate higher confidence for echoed tokens
                completion_logprobs = [-self.random.uniform(0.5, 2.0) for _ in range(completion_length)]
                logprobs.append(completion_logprobs)

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
        }

    def init_communicator(self, device):
        pass

    def update_named_param(self, name, weights):
        pass

    def reset_prefix_cache(self):
        pass

    def close_communicator(self):
        pass
