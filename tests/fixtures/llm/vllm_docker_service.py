import contextlib
import logging
import os
import random
import shutil
import sys
import tempfile
import time

import huggingface_hub
import pytest

from optimum.neuron.utils.import_utils import is_package_available
from optimum.neuron.version import __version__


if is_package_available("docker"):
    from docker.errors import NotFound

    import docker

from .vllm_service import LauncherHandle


OPTIMUM_CACHE_REPO_ID = "optimum-internal-testing/neuron-testing-cache"
HF_TOKEN = huggingface_hub.get_token()
DEFAULT_LLM_SERVICE = f"optimum-neuron-vllm:{__version__}"


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


def get_docker_image():
    docker_image = os.getenv("DOCKER_IMAGE", None)
    if docker_image is None:
        logger.info(f"No base test image specified in environment, trying {DEFAULT_LLM_SERVICE}")
        docker_image = DEFAULT_LLM_SERVICE
    # Check the image is actually available
    client = docker.from_env()
    images = client.images.list(filters={"reference": docker_image})
    if not images:
        raise ValueError(
            f"No {docker_image} image found on this host to run tests."
            "You can generate the latest image using 'make optimum-neuron-vllm'."
        )
    logger.info(f"Using {docker_image} as base test image")
    docker_image = images[0].tags[0]
    return docker_image


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
def vllm_docker_launcher(event_loop):
    """Utility fixture to expose a vLLM inference service.

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
        A `ContainerLauncherHandle` containing both a vLLM server and OpenAI client.
    """

    @contextlib.contextmanager
    def docker_launcher(
        service_name: str,
        model_name_or_path: str,
        served_model_name: str | None = None,
        batch_size: int | None = None,
        sequence_length: int | None = None,
        tensor_parallel_size: int | None = None,
        params_as_env: bool = True,
        propagate_hf_token: bool = True,
    ):
        port = random.randint(8000, 10_000)

        client = docker.from_env()

        container_name = f"optimum-neuron-tests-{service_name}-{port}"

        command = []

        def add_param(key, value):
            value = str(value)
            if params_as_env:
                upper_key = key.upper()
                env[f"SM_ON_{upper_key}"] = value
            else:
                command.append(f"--{key}")
                command.append(value)

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

        if HF_TOKEN is not None and propagate_hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
            env["HF_TOKEN"] = HF_TOKEN

        if served_model_name is not None:
            add_param("served_model_name", served_model_name)
        if batch_size is not None:
            add_param("batch_size", batch_size)
        if sequence_length is not None:
            add_param("sequence_length", sequence_length)
        if tensor_parallel_size is not None:
            add_param("tensor_parallel_size", tensor_parallel_size)

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
                container_model_name_or_path = f"/data/{model_name_or_path}"
                docker_content = f"""
                FROM {base_image}
                COPY model {container_model_name_or_path}
                """
                with open(os.path.join(context_dir, "Dockerfile"), "wb") as f:
                    f.write(docker_content.encode("utf-8"))
                    f.flush()
                image, logs = client.images.build(path=context_dir, dockerfile=f.name, tag=test_image)
            logger.info("Successfully built image %s", image.id)
            logger.debug("Build logs %s", logs)
        else:
            test_image = base_image
            image = None
            container_model_name_or_path = model_name_or_path

        add_param("model", container_model_name_or_path)
        container = client.containers.run(
            test_image,
            name=container_name,
            environment=env,
            auto_remove=False,
            detach=True,
            devices=["/dev/neuron0"],
            ports={"8080/tcp": port},
            shm_size="1G",
            command=command,
        )

        logger.info(f"Starting {container_name} container")
        model_name = served_model_name if served_model_name is not None else container_model_name_or_path
        yield ContainerLauncherHandle(
            service_name,
            model_name,
            client,
            container.name,
            port,
        )

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
