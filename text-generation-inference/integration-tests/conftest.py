import asyncio
import contextlib
import os
import random
import shlex
import subprocess
import sys
import time
from tempfile import TemporaryDirectory
from typing import List

import docker
import pytest
from aiohttp import ClientConnectorError, ClientOSError, ServerDisconnectedError
from docker.errors import NotFound
from text_generation import AsyncClient
from text_generation.types import Response


DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "neuronx-tgi:latest")
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", None)
DOCKER_VOLUME = os.getenv("DOCKER_VOLUME", "/data")


class LauncherHandle:
    def __init__(self, port: int):
        self.client = AsyncClient(f"http://localhost:{port}")

    def _inner_health(self):
        raise NotImplementedError

    async def health(self, timeout: int = 60):
        assert timeout > 0
        for _ in range(timeout):
            if not self._inner_health():
                raise RuntimeError("Launcher crashed")

            try:
                await self.client.generate("test")
                return
            except (ClientConnectorError, ClientOSError, ServerDisconnectedError):
                time.sleep(1)
        raise RuntimeError("Health check failed")


class ContainerLauncherHandle(LauncherHandle):
    def __init__(self, docker_client, container_name, port: int):
        super(ContainerLauncherHandle, self).__init__(port)
        self.docker_client = docker_client
        self.container_name = container_name

    def _inner_health(self) -> bool:
        container = self.docker_client.containers.get(self.container_name)
        return container.status in ["running", "created"]


class ProcessLauncherHandle(LauncherHandle):
    def __init__(self, process, port: int):
        super(ProcessLauncherHandle, self).__init__(port)
        self.process = process

    def _inner_health(self) -> bool:
        return self.process.poll() is None


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def data_volume():
    tmpdir = TemporaryDirectory()
    yield tmpdir.name
    # Cleanup the temporary directory using sudo as it contains root files created by the container
    subprocess.run(shlex.split(f"sudo rm -rf {tmpdir.name}"))


@pytest.fixture(scope="module")
def launcher(event_loop, data_volume):
    @contextlib.contextmanager
    def docker_launcher(
        model_id: str,
        trust_remote_code: bool = False,
    ):
        port = random.randint(8000, 10_000)

        args = ["--model-id", model_id, "--env"]

        if trust_remote_code:
            args.append("--trust-remote-code")

        client = docker.from_env()

        container_name = f"tgi-tests-{model_id.split('/')[-1]}"

        try:
            container = client.containers.get(container_name)
            container.stop()
            container.wait()
        except NotFound:
            pass

        env = {"LOG_LEVEL": "info,text_generation_router=debug"}

        if HUGGING_FACE_HUB_TOKEN is not None:
            env["HUGGING_FACE_HUB_TOKEN"] = HUGGING_FACE_HUB_TOKEN

        for var in ["HF_BATCH_SIZE", "HF_SEQUENCE_LENGTH", "HF_AUTOCAST_TYPE", "HF_NUM_CORES"]:
            if var in os.environ:
                env[var] = os.environ[var]

        volumes = [f"{data_volume}:/data"]

        container = client.containers.run(
            DOCKER_IMAGE,
            command=args,
            name=container_name,
            environment=env,
            auto_remove=False,
            detach=True,
            devices=["/dev/neuron0"],
            volumes=volumes,
            ports={"80/tcp": port},
            shm_size="1G",
        )

        yield ContainerLauncherHandle(client, container.name, port)

        try:
            container.stop()
            container.wait()
        except NotFound:
            pass

        container_output = container.logs().decode("utf-8")
        print(container_output, file=sys.stderr)

        container.remove()

    return docker_launcher


@pytest.fixture(scope="module")
def generate_load():
    async def generate_load_inner(client: AsyncClient, prompt: str, max_new_tokens: int, n: int) -> List[Response]:
        futures = [
            client.generate(prompt, max_new_tokens=max_new_tokens, decoder_input_details=True) for _ in range(n)
        ]

        return await asyncio.gather(*futures)

    return generate_load_inner
