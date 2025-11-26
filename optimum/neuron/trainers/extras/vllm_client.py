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
import requests
import time
from typing import Union
from collections import namedtuple

import torch
import torch_xla
from trl.extras.vllm_client import VLLMClient as TRLVLLMClient

from trl.import_utils import is_vllm_available


if is_vllm_available():
    from vllm.distributed.utils import StatelessProcessGroup
else:
    class StatelessProcessGroup:
        pass

# Set up the communication group for weight broadcasting using CPU communicator
Group = namedtuple('Group', 'barrier')

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
    """
    VLLMClient for Neuron environments.

    This class inherits all functionality from trl.extras.vllm_client.VLLMClient and only overrides methods
    to enable CPU-based communication suitable for Neuron setups and development/testing scenarios.
    """

    def init_communicator(self, device: Union[torch.device, str, int] = 0):
        """
        Initializes the weight update group in a distributed setup for model synchronization.

        This method uses CPU-based communication via object broadcasting, suitable for Neuron
        environments and development/testing scenarios.

        Args:
            device (`torch.device`, `str`, or `int`, *optional*, defaults to `0`):
                Device parameter for compatibility. Communication is handled via CPU.
        """
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

        pg = StatelessProcessGroup.create(
            host=self.host, port=self.group_port, rank=self.rank, world_size=world_size
        )
        self.communicator = CPUCommunicator(pg, self.rank)

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator)
