import asyncio
import contextlib
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from typing import List

import huggingface_hub
import pytest
from docker.errors import NotFound
from openai import APIConnectionError, AsyncOpenAI

import docker


OPTIMUM_CACHE_REPO_ID = "optimum-internal-testing/neuron-testing-cache"
HF_TOKEN = huggingface_hub.get_token()
DEFAULT_LLM_SERVICE = "optimum-neuron-vllm"


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


def get_docker_image():
    docker_image = os.getenv("DOCKER_IMAGE", None)
    if docker_image is None:
        client = docker.from_env()
        logger.info("No image specified, trying to identify an image locally")
        images = client.images.list(filters={"reference": DEFAULT_LLM_SERVICE})
        if not images:
            raise ValueError(f"No {DEFAULT_LLM_SERVICE} image found on this host to run tests.")
        docker_image = images[0].tags[0]
    return docker_image


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


class ContainerLauncherHandle(LauncherHandle):
    def __init__(self, service_name, model_name, docker_client, container_name, port: int):
        super(ContainerLauncherHandle, self).__init__(service_name, model_name, port)
        self.docker_client = docker_client
        self.container_name = container_name
        self._log_since = time.time()

    def _inner_health(self) -> bool:
        container = self.docker_client.containers.get(self.container_name)
        container_output = container.logs(since=self._log_since).decode("utf-8")
        self._log_since = time.time()
        if container_output != "":
            print(container_output, end="")
        return container.status in ["running", "created"]


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def neuron_launcher(event_loop):
    """Utility fixture to expose an inference service.

    The fixture uses a single event loop for each module, but it can create multiple
    docker services with different parameters using the parametrized inner context.

    Args:
        service_name (`str`):
            Used to identify test configurations and adjust test expectations,
        model_name_or_path (`str`):
            The model to use (can be a hub model or a path)
        trust_remote_code (`bool`):
            Must be set to True for gated models.

    Returns:
        A `ContainerLauncherHandle` containing both a TGI server and client.
    """

    @contextlib.contextmanager
    def docker_launcher(
        service_name: str,
        model_name_or_path: str,
        trust_remote_code: bool = False,
    ):
        port = random.randint(8000, 10_000)

        client = docker.from_env()

        container_name = f"optimum-neuron-tests-{service_name}-{port}"

        try:
            container = client.containers.get(container_name)
            container.stop()
            container.wait()
        except NotFound:
            pass

        env = {
            "LOGLEVEL": "DEBUG",
            "CUSTOM_CACHE_REPO": OPTIMUM_CACHE_REPO_ID,
        }

        if HF_TOKEN is not None:
            env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
            env["HF_TOKEN"] = HF_TOKEN

        for var in [
            "SM_VLLM_MAX_NUM_SEQS",
            "SM_VLLM_TENSOR_PARALLEL_SIZE",
            "SM_VLLM_MAX_MODEL_LEN",
        ]:
            if var in os.environ:
                env[var] = os.environ[var]

        base_image = get_docker_image()
        if os.path.isdir(model_name_or_path):
            # Create a sub-image containing the model to workaround docker dind issues preventing
            # to share a volume from the container running tests

            test_image = f"{container_name}-img"
            logger.info(
                "Building image on the flight derivated from %s, tagged with %s",
                base_image,
                test_image,
            )
            with tempfile.TemporaryDirectory() as context_dir:
                # Copy model directory to build context
                model_path = os.path.join(context_dir, "model")
                shutil.copytree(model_name_or_path, model_path)
                # Create Dockerfile
                container_model_id = f"/data/{model_name_or_path}"
                docker_content = f"""
                FROM {base_image}
                COPY model {container_model_id}
                """
                with open(os.path.join(context_dir, "Dockerfile"), "wb") as f:
                    f.write(docker_content.encode("utf-8"))
                    f.flush()
                image, logs = client.images.build(path=context_dir, dockerfile=f.name, tag=test_image)
                env["SM_VLLM_MODEL"] = container_model_id
            logger.info("Successfully built image %s", image.id)
            logger.debug("Build logs %s", logs)
        else:
            test_image = base_image
            image = None
            container_model_id = model_name_or_path

        args = ["--env"]

        if trust_remote_code:
            args.append("--trust-remote-code")

        container = client.containers.run(
            test_image,
            command=args,
            name=container_name,
            environment=env,
            auto_remove=False,
            detach=True,
            devices=["/dev/neuron0"],
            ports={"8080/tcp": port},
            shm_size="1G",
        )

        logger.info(f"Starting {container_name} container")
        yield ContainerLauncherHandle(service_name, container_model_id, client, container.name, port)

        try:
            container.stop(timeout=60)
            container.wait(timeout=60)
        except Exception as e:
            logger.exception(f"Ignoring exception while stopping container: {e}.")
            pass
        finally:
            logger.info("Removing container %s", container_name)
            try:
                container.remove(force=True)
            except Exception as e:
                logger.error("Error while removing container %s, skipping", container_name)
                logger.exception(e)

            # Cleanup the build image
            if image:
                logger.info("Cleaning image %s", image.id)
                try:
                    image.remove(force=True)
                except NotFound:
                    pass
                except Exception as e:
                    logger.error("Error while removing image %s, skipping", image.id)
                    logger.exception(e)

    return docker_launcher
