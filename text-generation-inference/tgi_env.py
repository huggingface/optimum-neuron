#!/usr/bin/env python

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from huggingface_hub import constants
from transformers import AutoConfig

from optimum.neuron.modeling_decoder import get_available_cores
from optimum.neuron.utils import get_hub_cached_entries
from optimum.neuron.utils.version_utils import get_neuronxcc_version


logger = logging.getLogger(__name__)

tgi_router_env_vars = ["MAX_BATCH_SIZE", "MAX_TOTAL_TOKENS", "MAX_INPUT_LENGTH"]
tgi_server_env_vars = ["HF_BATCH_SIZE", "HF_SEQUENCE_LENGTH", "HF_NUM_CORES", "HF_AUTO_CAST_TYPE"]

env_config_peering = [
    ("MAX_BATCH_SIZE", "batch_size"),
    ("MAX_TOTAL_TOKENS", "sequence_length"),
    ("HF_BATCH_SIZE", "batch_size"),
    ("HF_AUTO_CAST_TYPE", "auto_cast_type"),
    ("HF_SEQUENCE_LENGTH", "sequence_length"),
    ("HF_NUM_CORES", "num_cores"),
]

# By the end of this script all env var should be specified properly
env_vars = tgi_server_env_vars + tgi_router_env_vars

available_cores = get_available_cores()
neuronxcc_version = get_neuronxcc_version()


def parse_cmdline_and_set_env(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    if not argv:
        argv = sys.argv
    # All these are params passed to tgi and intercepted here
    parser.add_argument("--max-input-length", type=int, default=os.getenv("MAX_INPUT_LENGTH", 0))
    parser.add_argument(
        "--max-total-tokens", type=int, default=os.getenv("MAX_TOTAL_TOKENS", os.getenv("HF_SEQUENCE_LENGTH", 0))
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=os.getenv("MAX_BATCH_SIZE", os.getenv("HF_BATCH_SIZE", 0))
    )
    parser.add_argument("--model-id", type=str, default=os.getenv("MODEL_ID"))
    parser.add_argument("--revision", type=str, default=os.getenv("REVISION"))

    args = parser.parse_known_args(argv)[0]

    if not args.model_id:
        raise Exception("No model id provided ! Either specify it using --model-id cmdline " "or MODEL_ID env var")

    # Override env with cmdline params
    os.environ["MODEL_ID"] = args.model_id

    # Set all tgi router and tgi server values to consistent values as early as possible
    # from the order of the parser defaults, the tgi router value can override the tgi server ones
    if args.max_total_tokens > 0:
        os.environ["MAX_TOTAL_TOKENS"] = str(args.max_total_tokens)
        os.environ["HF_SEQUENCE_LENGTH"] = str(args.max_total_tokens)

    if args.max_input_length > 0:
        os.environ["MAX_INPUT_LENGTH"] = str(args.max_input_length)

    if args.max_batch_size > 0:
        os.environ["MAX_BATCH_SIZE"] = str(args.max_batch_size)
        os.environ["HF_BATCH_SIZE"] = str(args.max_batch_size)

    if args.revision:
        os.environ["REVISION"] = str(args.revision)

    return args


def neuron_config_to_env(neuron_config):
    with open(os.environ["ENV_FILEPATH"], "w") as f:
        for env_var, config_key in env_config_peering:
            f.write("export {}={}\n".format(env_var, neuron_config[config_key]))
        max_input_length = os.getenv("MAX_INPUT_LENGTH")
        if not max_input_length:
            max_input_length = int(neuron_config["sequence_length"]) // 2
            if max_input_length == 0:
                raise Exception("Model sequence length should be greater than 1")
        f.write("export MAX_INPUT_LENGTH={}\n".format(max_input_length))

        # max_prefill_tokens = os.getenv("MAX_BATCH_PREFILL_TOKENS")
        # if not max_prefill_tokens:
        #     max_prefill_tokens = max_input_length
        # f.write("export MAX_BATCH_PREFILL_TOKENS={}\n".format(max_prefill_tokens))
        #
        # max_batch_total_tokens = os.getenv("MAX_BATCH_TOTAL_TOKENS")
        # if not max_batch_total_tokens:
        #     max_batch_total_tokens = str(neuron_config["sequence_length"] * neuron_config["batch_size"])
        # f.write("export MAX_BATCH_TOTAL_TOKENS={}\n".format(max_batch_total_tokens))


def lookup_compatible_cached_model(model_id: str, revision: Optional[str]) -> Optional[Dict[str, Any]]:
    # Reuse the same mechanic as the one in use to configure the tgi server part
    # The only difference here is that we stay as flexible as possible on the compatibility part
    entries = get_hub_cached_entries(model_id, "inference")

    logger.debug("Found %d cached entries for model %s, revision %s", len(entries), model_id, revision)

    for entry in entries:
        if check_env_and_neuron_config_compatibility(entry):
            break
    else:
        entry = None

    if not entry:
        logger.debug(
            "No compatible cached entry found for model %s, env %s, available cores %s, " "neuronxcc version %s",
            model_id,
            get_env_dict(),
            available_cores,
            neuronxcc_version,
        )
    else:
        logger.info("Compatible neuron cached model found %s", entry)
    return entry


def check_env_and_neuron_config_compatibility(neuron_config: Dict[str, Any]) -> bool:

    logger.debug(
        "Checking the provided neuron config %s is compatible with the local setup and provided environment",
        neuron_config,
    )

    # Local setup compat checks
    if neuron_config["num_cores"] > available_cores:
        logger.debug("Not enough neuron cores available to run the provided neuron config")
        return False

    if neuron_config["compiler_version"] != neuronxcc_version:
        logger.debug(
            "Compiler version conflict, the local one " "(%s) differs from the one used to compile the model (%s)",
            neuronxcc_version,
            neuron_config["compiler_version"],
        )
        return False

    for env_var, config_key in env_config_peering:
        neuron_config_value = str(neuron_config[config_key])
        env_value = os.getenv(env_var, str(neuron_config_value))
        if env_value != neuron_config_value:
            logger.debug(
                "The provided env var '%s' and the neuron config '%s' param differ (%s != %s)",
                env_var,
                config_key,
                env_value,
                neuron_config_value,
            )
            return False

    if os.getenv("MAX_INPUT_LENGTH"):
        max_input_length = int(os.environ["MAX_INPUT_LENGTH"])
        sequence_length = neuron_config["sequence_length"]
        if max_input_length >= sequence_length:
            logger.debug(
                "Specified max input length is not compatible with config sequence length " "( %s >= %s)",
                max_input_length,
                sequence_length,
            )
            return False

    return True


def get_env_dict() -> Dict[str, str]:
    d = {}
    for k in env_vars:
        d[k] = os.getenv(k)
    return d


def main():
    """
    This script determines proper default TGI env variables for the neuron precompiled models to
    work properly
    :return:
    """
    logging.basicConfig(level=logging.DEBUG, force=True)

    args = parse_cmdline_and_set_env()

    for env_var in env_vars:
        if not os.getenv(env_var):
            break
    else:
        logger.info("All env vars %s already set, skipping, user know what they are doing", env_vars)
        sys.exit(0)

    cache_dir = constants.HF_HUB_CACHE

    logger.info("Cache dir %s", cache_dir)

    config = AutoConfig.from_pretrained(args.model_id, revision=args.revision)
    neuron_config = getattr(config, "neuron", None)
    if neuron_config is not None:
        compatible = check_env_and_neuron_config_compatibility(neuron_config)
        if not compatible:
            env_dict = get_env_dict()
            msg = (
                "Invalid neuron config and env. Config {}, env {}, available cores {}, " "neuronxcc version {}"
            ).format(neuron_config, env_dict, available_cores, neuronxcc_version)
            logger.error(msg)
            raise Exception(msg)
    else:
        neuron_config = lookup_compatible_cached_model(args.model_id, args.revision)

    if not neuron_config:
        msg = (
            "No compatible neuron config found. Provided env {}, " "available cores {}, neuronxcc version {}"
        ).format(get_env_dict(), available_cores, neuronxcc_version)
        logger.error(msg)
        raise Exception(msg)

    neuron_config_to_env(neuron_config)


if __name__ == "__main__":
    main()
