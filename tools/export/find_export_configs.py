#!/usr/bin/env python3
"""
Generic script to test export configurations for any Hugging Face model.

This script accepts a model_id, identifies its task using TasksManager,
and automatically deduces valid configuration parameters from the system
and model structure as specified in AGENTS.md.

Usage:
    python test_export_configs.py <model_id> [--output_file <path>]

Example:
    python test_export_configs.py Qwen/Qwen3-Embedding-8B
    python test_export_configs.py meta-llama/Llama-3.1-8B --output_file configs.json
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from optimum.exporters.tasks import TasksManager
from transformers import AutoConfig

from optimum.neuron.utils.instance import current_instance_type
from optimum.neuron.utils.system import cores_per_device, get_available_cores


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fixed candidate batch sizes (as per AGENTS.md)
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128]

# Task-specific model class mappings
TASK_TO_MODEL_CLASS = {
    "text-generation": "NeuronModelForCausalLM",
    "feature-extraction": "NeuronModelForEmbedding",
}


def get_num_attention_heads(config: AutoConfig) -> int:
    """Get the number of attention heads from model config."""
    # Different model architectures use different attribute names
    if hasattr(config, "num_attention_heads"):
        return config.num_attention_heads
    elif hasattr(config, "n_head"):
        return config.n_head
    elif hasattr(config, "num_heads"):
        return config.num_heads
    else:
        raise ValueError(
            f"Could not determine number of attention heads from config. "
            f"Available attributes: {list(config.to_dict().keys())}"
        )


def get_max_position_embeddings(config: AutoConfig) -> int:
    """Get the maximum position embeddings from model config."""
    # Different model architectures use different attribute names
    if hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    elif hasattr(config, "n_positions"):
        return config.n_positions
    elif hasattr(config, "max_seq_len"):
        return config.max_seq_len
    elif hasattr(config, "seq_length"):
        return config.seq_length
    else:
        logger.warning("Could not determine max positions from config, defaulting to 32768")
        return 32768


def compute_valid_tp_sizes(num_attention_heads: int, num_cores: int, min_tp: int) -> List[int]:
    """
    Compute valid tensor parallel sizes based on AGENTS.md rules:
    - Must be at least 2
    - Must be a multiple of the number of cores per device
    - Must be less than the total number of neuron cores available
    - Must divide the model number of attention heads
    """
    valid_tp_sizes = []
    # Start from cores_per_device (minimum TP that uses full device)
    tp = min_tp
    while tp <= num_cores:
        if num_attention_heads % tp == 0:
            valid_tp_sizes.append(tp)
        tp += min_tp
    return valid_tp_sizes


def compute_valid_sequence_lengths(max_positions: int) -> List[int]:
    """
    Compute valid sequence lengths based on AGENTS.md rules:
    - Must not be less than 1024
    - Must be a power-of-two
    - Must be lower than the model maximum number of positions
    """
    valid_seq_lengths = []
    sl = 1024
    while sl <= max_positions:
        valid_seq_lengths.append(sl)
        sl *= 2
    return valid_seq_lengths


def load_existing_configs(output_file: Path, model_id: str) -> Dict[str, Any]:
    """Load existing configurations from file."""
    if output_file.exists():
        with open(output_file, "r") as f:
            return json.load(f)
    return {model_id: []}


def config_exists(configs: Dict[str, Any], model_id: str, tp: int, bs: int, sl: int) -> bool:
    """Check if a configuration has already been tested."""
    for config in configs.get(model_id, []):
        if config["tensor_parallel_size"] == tp and config["batch_size"] == bs and config["sequence_length"] == sl:
            return True
    return False


def has_valid_config_for_tp_bs(configs: Dict[str, Any], model_id: str, tp: int, bs: int) -> bool:
    """
    Check if any valid configuration exists for a given (tp, bs) pair.

    When resuming, if a valid config exists for (tp, bs), we can skip all
    sequence lengths for that pair since:
    - Larger sequence lengths were already tested and failed
    - Smaller sequence lengths will also work
    """
    for config in configs.get(model_id, []):
        if config["tensor_parallel_size"] == tp and config["batch_size"] == bs:
            return True
    return False


def get_max_valid_batch_size_for_tp(configs: Dict[str, Any], model_id: str, tp: int) -> int | None:
    """
    Get the maximum batch size that has a valid configuration for a given TP.

    When resuming, since batch sizes are tested in decreasing order, if a valid
    config exists for a given TP, configurations with higher batch sizes have
    already been tested and failed. We should resume from this batch size.

    Returns None if no valid configuration exists for this TP.
    """
    max_bs = None
    for config in configs.get(model_id, []):
        if config["tensor_parallel_size"] == tp:
            if max_bs is None or config["batch_size"] > max_bs:
                max_bs = config["batch_size"]
    return max_bs


def add_config(
    configs: Dict[str, Any],
    model_id: str,
    task: str,
    instance_type: str,
    tp: int,
    bs: int,
    sl: int,
) -> None:
    """Add a valid configuration (only valid configs are stored)."""
    if model_id not in configs:
        configs[model_id] = []
    # Check if already exists
    for config in configs[model_id]:
        if config["tensor_parallel_size"] == tp and config["batch_size"] == bs and config["sequence_length"] == sl:
            return  # Already exists
    # Add new config
    configs[model_id].append(
        {
            "task": task,
            "instance_type": instance_type,
            "batch_size": bs,
            "sequence_length": sl,
            "tensor_parallel_size": tp,
        }
    )


def save_configs(configs: Dict[str, Any], output_file: Path) -> None:
    """Save configurations to file."""
    with open(output_file, "w") as f:
        json.dump(configs, f, indent=2)
    logger.info(f"Saved configurations to {output_file}")


def export_model(model_id: str, task: str, tp: int, bs: int, sl: int, output_dir: str) -> bool:
    """Export model with given configuration."""
    cmd = [
        "optimum-cli",
        "export",
        "neuron",
        "-m",
        model_id,
        "--batch_size",
        str(bs),
        "--sequence_length",
        str(sl),
        "--tensor_parallel_size",
        str(tp),
        "--task",
        task,
        output_dir,
    ]

    logger.info(f"Exporting: TP={tp}, BS={bs}, SL={sl}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            logger.info("✓ Export successful")
            return True
        else:
            logger.error("✗ Export failed")
            if result.stderr:
                logger.error(f"Error output: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("✗ Export timeout (exceeded 3600 seconds)")
        return False
    except Exception as e:
        logger.error(f"✗ Export error: {e}")
        return False


def load_model(model_id: str, task: str, output_dir: str) -> bool:
    """Load exported model to verify it works on Neuron devices."""
    model_class = TASK_TO_MODEL_CLASS.get(task, "NeuronModel")
    cmd = (
        f'python -c "from optimum.neuron import {model_class}; '
        f"model = {model_class}.from_pretrained('{output_dir}')\""
    )

    logger.info(f"Loading model from {output_dir}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            logger.info("✓ Load successful")
            return True
        else:
            logger.error("✗ Load failed")
            if result.stderr:
                logger.error(f"Error output: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("✗ Load timeout (exceeded 600 seconds)")
        return False
    except Exception as e:
        logger.error(f"✗ Load error: {e}")
        return False


def print_summary(configs: Dict[str, Any], model_id: str) -> None:
    """Print a summary of test results."""
    valid_configs = configs.get(model_id, [])

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model: {model_id}")
    print(f"Valid configurations: {len(valid_configs)}")

    if valid_configs:
        print("\nValid configurations:")
        for config in sorted(
            valid_configs, key=lambda c: (c["tensor_parallel_size"], -c["batch_size"], -c["sequence_length"])
        ):
            print(f"  TP={config['tensor_parallel_size']}, BS={config['batch_size']}, SL={config['sequence_length']}")


def test_configurations(
    model_id: str,
    task: str,
    instance_type: str,
    output_file: Path,
    tp_sizes: List[int],
    batch_sizes: List[int],
    sequence_lengths: List[int],
) -> None:
    """Test all configurations following the workflow."""
    configs = load_existing_configs(output_file, model_id)

    logger.info(f"Testing configurations for {model_id} (task: {task})")
    logger.info(f"Candidate TP sizes: {tp_sizes}")
    logger.info(f"Candidate batch sizes: {batch_sizes}")
    logger.info(f"Candidate sequence lengths: {sequence_lengths}")
    logger.info(f"Output file: {output_file}")

    # Iterate: increasing TP, decreasing BS, decreasing SL
    for tp in tp_sizes:
        # Optimization: when resuming, skip batch sizes larger than the first valid one
        # for this TP, since they have already been tested and failed
        max_valid_bs = get_max_valid_batch_size_for_tp(configs, model_id, tp)
        for bs in sorted(batch_sizes, reverse=True):
            if max_valid_bs is not None and bs > max_valid_bs:
                logger.info(f"Skipping (tp={tp}, bs={bs}): higher batch sizes already tested and failed")
                continue

            # Optimization: if any valid config exists for this (tp, bs) pair,
            # skip all sequence lengths since larger ones already failed and
            # smaller ones will also work
            if has_valid_config_for_tp_bs(configs, model_id, tp, bs):
                logger.info(f"Skipping (tp={tp}, bs={bs}): valid config already exists")
                continue

            for sl in sorted(sequence_lengths, reverse=True):
                if config_exists(configs, model_id, tp, bs, sl):
                    logger.info(f"Skipping (already valid): TP={tp}, BS={bs}, SL={sl}")
                    continue

                # Test this configuration
                output_dir = f"./data/{Path(model_id).name}_tp{tp}_bs{bs}_sl{sl}"

                # Export
                if not export_model(model_id, task, tp, bs, sl, output_dir):
                    logger.info(f"✗ Configuration invalid: TP={tp}, BS={bs}, SL={sl}")
                    continue

                # Load
                if not load_model(model_id, task, output_dir):
                    logger.info(f"✗ Configuration invalid: TP={tp}, BS={bs}, SL={sl}")
                    continue

                # Valid configuration found
                logger.info(f"✓ Configuration valid: TP={tp}, BS={bs}, SL={sl}")
                add_config(configs, model_id, task, instance_type, tp, bs, sl)
                save_configs(configs, output_file)
                # Skip testing smaller sequence lengths for this (tp, bs) pair
                # since they will also work if a larger sequence length succeeded
                break

    print_summary(configs, model_id)


def main():
    parser = argparse.ArgumentParser(
        description="Test export configurations for a Hugging Face model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_export_configs.py Qwen/Qwen3-Embedding-8B
  python test_export_configs.py meta-llama/Llama-3.1-8B --output_file configs.json

Configuration parameters (TP sizes, sequence lengths) are automatically
deduced from the system and model structure as per AGENTS.md guidelines.
        """,
    )
    parser.add_argument("model_id", type=str, help="The Hugging Face model ID (e.g., 'meta-llama/Llama-3.1-8B')")
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Output JSON file for configurations (default: <model_name>_configs.json)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--dry-run", action="store_true", help="List configurations to test without actually running export/load tests"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Infer task from model_id
    try:
        task = TasksManager.infer_task_from_model(args.model_id)
        logger.info(f"Inferred task from model: {task}")
    except Exception as e:
        logger.error(f"Failed to infer task from model {args.model_id}: {e}")
        sys.exit(1)

    # Load model config to deduce parameters
    try:
        config = AutoConfig.from_pretrained(args.model_id)
        logger.info(f"Loaded model config for {args.model_id}")
    except Exception as e:
        logger.error(f"Failed to load model config for {args.model_id}: {e}")
        sys.exit(1)

    # Get system information
    try:
        instance_type = current_instance_type()
        logger.info(f"Instance type: {instance_type}")
        num_cores = get_available_cores()
        min_tp = cores_per_device()
        logger.info(f"Cores per device: {min_tp}")
        logger.info(f"Available Neuron cores: {num_cores}")
    except Exception as e:
        logger.error(f"Failed to get available Neuron cores: {e}")
        logger.error("Make sure you are running on a Neuron-enabled instance.")
        sys.exit(1)

    # Get model parameters
    try:
        num_attention_heads = get_num_attention_heads(config)
        max_positions = get_max_position_embeddings(config)
        logger.info(f"Model attention heads: {num_attention_heads}")
        logger.info(f"Model max positions: {max_positions}")
    except Exception as e:
        logger.error(f"Failed to extract model parameters: {e}")
        sys.exit(1)

    # Compute valid configuration parameters
    tp_sizes = compute_valid_tp_sizes(num_attention_heads, num_cores, min_tp)
    if not tp_sizes:
        logger.error(
            f"No valid tensor parallel sizes found. "
            f"Attention heads ({num_attention_heads}) must be divisible by a multiple of "
            f"cores_per_device ({min_tp}) and <= available cores ({num_cores})."
        )
        sys.exit(1)
    logger.info(f"Valid TP sizes: {tp_sizes}")

    sequence_lengths = compute_valid_sequence_lengths(max_positions)
    if not sequence_lengths:
        logger.error(f"No valid sequence lengths found. Max positions ({max_positions}) must be >= 1024.")
        sys.exit(1)
    logger.info(f"Valid sequence lengths: {sequence_lengths}")

    # Batch sizes are fixed as per AGENTS.md
    batch_sizes = BATCH_SIZES
    logger.info(f"Batch sizes: {batch_sizes}")

    # Set output file if not specified
    if args.output_file is None:
        model_name = Path(args.model_id).name.lower()
        args.output_file = Path(f"{model_name}_configs.json")

    # Dry run: just print the configurations
    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("DRY RUN - Configuration Summary")
        print(f"{'=' * 60}")
        print(f"Model: {args.model_id}")
        print(f"Task: {task}")
        print(f"Output file: {args.output_file}")
        print("\nSystem:")
        print(f"  Instance type: {instance_type}")
        print(f"  Cores per device: {min_tp}")
        print(f"  Available Neuron cores: {num_cores}")
        print("\nModel parameters:")
        print(f"  Attention heads: {num_attention_heads}")
        print(f"  Max positions: {max_positions}")
        print("\nConfigurations to test:")
        print(f"  Tensor parallel sizes: {tp_sizes}")
        print(f"  Batch sizes: {batch_sizes}")
        print(f"  Sequence lengths: {sequence_lengths}")
        total_configs = len(tp_sizes) * len(batch_sizes) * len(sequence_lengths)
        print(f"\nTotal configurations: {total_configs}")
        sys.exit(0)

    # Run tests
    try:
        test_configurations(
            args.model_id,
            task,
            instance_type,
            args.output_file,
            tp_sizes=tp_sizes,
            batch_sizes=batch_sizes,
            sequence_lengths=sequence_lengths,
        )
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Testing failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
