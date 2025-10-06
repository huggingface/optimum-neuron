import asyncio
import contextlib
import logging
import os
import random
import subprocess
import sys
import time
from typing import List

import huggingface_hub
import pytest
import torch

from optimum.neuron.utils.import_utils import is_package_available


if is_package_available("openai"):
    from openai import APIConnectionError, AsyncOpenAI
else:

    class AsyncOpenAI:
        pass


OPTIMUM_CACHE_REPO_ID = "optimum-internal-testing/neuron-testing-cache"
HF_TOKEN = huggingface_hub.get_token()
DEFAULT_LLM_SERVICE = "optimum-neuron-vllm"


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


class TestClient(AsyncOpenAI):
    def __init__(self, service_name: str, model_name: str, port: int):
        super().__init__(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        self.model_name: str = model_name
        self.service_name: str = service_name

    async def sample(
        self,
        prompt: str,
        max_output_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: List[str] | None = None,
    ):
        response = await self.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        generated_tokens = response.usage.completion_tokens
        generated_text = response.choices[0].message.content
        return generated_tokens, generated_text

    async def greedy(self, prompt: str, max_output_tokens: int, stop: List[str] | None = None):
        return await self.sample(prompt, max_output_tokens=max_output_tokens, temperature=0, stop=stop)


class LauncherHandle:
    def __init__(self, service_name: str, model_name: str, port: int):
        self.client = TestClient(service_name, model_name, port)

    def _inner_health(self):
        raise NotImplementedError

    async def health(self, timeout: int = 60):
        assert timeout > 0
        for i in range(timeout):
            if not self._inner_health():
                raise RuntimeError(f"Service crashed after {i} seconds.")

            try:
                models = await self.client.models.list()
                model_name = models.data[0].id
                if self.client.model_name != model_name:
                    raise ValueError(f"The service exposes {model_name} but {self.client.service_name} was expected.")
                logger.info(f"Service started after {i} seconds")
                return
            except APIConnectionError:
                time.sleep(1)
            except Exception as e:
                raise RuntimeError(f"Querying container model failed with: {e}")
        raise RuntimeError(f"Service failed to start after {i} seconds.")


class SubprocessLauncherHandle(LauncherHandle):
    def __init__(self, service_name, model_name, port: int, process: subprocess.Popen):
        super().__init__(service_name, model_name, port)
        self.process = process

    def _inner_health(self) -> bool:
        return self.process.poll() is None


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def vllm_launcher(event_loop):
    """Utility fixture to expose a vLLM inference service.

    Args:
        service_name (`str`):
            Used to identify test configurations and adjust test expectations,
        model_name_or_path (`str`):
            The model to use (can be a hub model or a path)
        trust_remote_code (`bool`):
            Must be set to True for gated models.

    Returns:
        A `LauncherHandle` containing both a vLLM server and OpenAI client.
    """

    @contextlib.contextmanager
    def launcher(
        service_name: str,
        model_name_or_path: str,
        batch_size: int | None = None,
        sequence_length: int | None = None,
        tensor_parallel_size: int | None = None,
        dtype: str | None = None,
    ):
        port = random.randint(8000, 10_000)

        cache_repo = os.environ.get("CUSTOM_CACHE_REPO", None)
        os.environ["CUSTOM_CACHE_REPO"] = OPTIMUM_CACHE_REPO_ID

        command = [
            "optimum-cli",
            "neuron",
            "serve",
            "--model",
            model_name_or_path,
            "--port",
            str(port),
        ]
        if batch_size is not None:
            command += ["--batch_size", str(batch_size)]
        if sequence_length is not None:
            command += ["--sequence_length", str(sequence_length)]
        if tensor_parallel_size is not None:
            command += ["--tensor_parallel_size", str(tensor_parallel_size)]
        if dtype is not None:
            if isinstance(dtype, torch.dtype):
                # vLLM does not accept torch dtype, convert to string
                dtype = str(dtype).split(".")[-1]
            command += ["--dtype", dtype]

        p = subprocess.Popen(
            command,
            shell=False,
        )

        if cache_repo is not None:
            os.environ["CUSTOM_CACHE_REPO"] = cache_repo

        logger.info(f"Starting {service_name} with model {model_name_or_path}")
        yield SubprocessLauncherHandle(service_name, model_name_or_path, port, p)
        logger.info(f"Stopping {service_name} with model {model_name_or_path}")
        p.terminate()
        p.wait()
        logger.info(f"Stopped {service_name} with model {model_name_or_path}")

    return launcher
