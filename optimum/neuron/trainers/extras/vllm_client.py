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
import time
from typing import Union

import torch
from trl.extras.vllm_client import VLLMClient as TRLVLLMClient

from trl.import_utils import is_vllm_available


if is_vllm_available():
    from vllm.distributed.utils import StatelessProcessGroup
else:
    class StatelessProcessGroup:
        pass




class VLLMClient(TRLVLLMClient):
    """
    Extension of TRL's VLLMClient that adds CPU support for development and testing.

    This class inherits all functionality from trl.extras.vllm_client.VLLMClient and only
    overrides the init_communicator method to add CPU fallback support when neither CUDA
    nor XPU devices are available.
    """

    def init_communicator(self, device: Union[torch.device, str, int] = 0):
        """
        Initializes the weight update group in a distributed setup for model synchronization.

        This method extends the parent implementation to support CPU-only environments by adding
        a CPU fallback communicator when neither XPU nor CUDA devices are available.

        Args:
            device (`torch.device`, `str`, or `int`, *optional*, defaults to `0`):
                Device of trainer main process. It's the device that will be used for the weights synchronization. Can
                be a `torch.device` object, a string like `'cuda:0'`, or an integer device index.
        """
        # Get the world size from the server
        import requests
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
        # Will simplify it after torch xpu 2.9 support get uuid.
        if is_torch_xpu_available():
            if hasattr(torch.xpu.get_device_properties(device), "uuid"):
                client_device_uuid = str(torch.xpu.get_device_properties(device).uuid)
            else:
                client_device_uuid = "42"
        elif torch.cuda.is_available():
            client_device_uuid = str(torch.cuda.get_device_properties(device).uuid)
        else:
            # CPU fallback - use dummy UUID
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

        # Set up the communication group for weight broadcasting
        if is_torch_xpu_available():
            store = torch.distributed.TCPStore(
                host_name=self.host, port=self.group_port, world_size=world_size, is_master=(self.rank == 0)
            )
            prefixed_store = c10d.PrefixStore("client2server", store)
            pg = c10d.ProcessGroupXCCL(
                store=prefixed_store,
                rank=self.rank,
                size=world_size,
            )
            self.communicator = pg
        elif torch.cuda.is_available():
            pg = StatelessProcessGroup.create(
                host=self.host, port=self.group_port, rank=self.rank, world_size=world_size
            )
            self.communicator = PyNcclCommunicator(pg, device=device)
        else:
            # CPU fallback - create a custom communicator that uses object broadcasting
            from collections import namedtuple
            Group = namedtuple('Group', 'barrier')

            class CPUCommunicator:
                def __init__(self, store, rank):
                    self.rank = rank
                    self.store = store
                    self.group = Group(barrier=self.barrier)

                def broadcast(self, tensor, src):
                    # Move tensor to CPU to avoid issues on the server side when running vLLM+CPU
                    tensor = tensor.cpu()
                    self.store.broadcast_obj(tensor, src=self.rank)

                def barrier(self):
                    self.store.barrier()

                def __del__(self):
                    del self.store

            pg = StatelessProcessGroup.create(
                host=self.host, port=self.group_port, rank=self.rank, world_size=world_size
            )
            self.communicator = CPUCommunicator(pg, self.rank)

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator)
