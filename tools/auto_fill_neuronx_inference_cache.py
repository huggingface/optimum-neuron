import argparse
import subprocess
import time
from tempfile import TemporaryDirectory

from optimum.neuron.utils import synchronize_hub_cache


MODEL_CONFIGURATIONS = {
    "openai-community/gpt2": [
        {"batch_size": 1, "sequence_length": 1024, "num_cores": 1, "auto_cast_type": "fp16"},
        {"batch_size": 16, "sequence_length": 1024, "num_cores": 1, "auto_cast_type": "fp16"},
    ],
    "meta-llama/Llama-2-7b-chat-hf": [
        {"batch_size": 1, "sequence_length": 4096, "num_cores": 2, "auto_cast_type": "fp16"},
        {"batch_size": 1, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "fp16"},
        {"batch_size": 1, "sequence_length": 4096, "num_cores": 24, "auto_cast_type": "fp16"},
        {"batch_size": 4, "sequence_length": 4096, "num_cores": 2, "auto_cast_type": "fp16"},
        {"batch_size": 4, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "fp16"},
        {"batch_size": 4, "sequence_length": 4096, "num_cores": 24, "auto_cast_type": "fp16"},
        {"batch_size": 8, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "fp16"},
        {"batch_size": 8, "sequence_length": 4096, "num_cores": 24, "auto_cast_type": "fp16"},
        {"batch_size": 16, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "fp16"},
        {"batch_size": 16, "sequence_length": 4096, "num_cores": 24, "auto_cast_type": "fp16"},
    ],
    "meta-llama/Llama-2-13b-chat-hf": [
        {"batch_size": 1, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "fp16"},
        {"batch_size": 1, "sequence_length": 4096, "num_cores": 24, "auto_cast_type": "fp16"},
        {"batch_size": 4, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "fp16"},
        {"batch_size": 4, "sequence_length": 4096, "num_cores": 24, "auto_cast_type": "fp16"},
        {"batch_size": 8, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "fp16"},
        {"batch_size": 8, "sequence_length": 4096, "num_cores": 24, "auto_cast_type": "fp16"},
    ],
    "meta-llama/Llama-2-70b-chat-hf": [
        {"batch_size": 1, "sequence_length": 4096, "num_cores": 24, "auto_cast_type": "fp16"},
    ],
    "mistralai/Mistral-7B-Instruct-v0.1": [
        {"batch_size": 1, "sequence_length": 4096, "num_cores": 2, "auto_cast_type": "bf16"},
        {"batch_size": 1, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "bf16"},
        {"batch_size": 4, "sequence_length": 4096, "num_cores": 2, "auto_cast_type": "bf16"},
        {"batch_size": 4, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "bf16"},
        {"batch_size": 8, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "bf16"},
        {"batch_size": 16, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "bf16"},
    ],
    "HuggingFaceH4/zephyr-7b-beta": [
        {"batch_size": 1, "sequence_length": 4096, "num_cores": 2, "auto_cast_type": "bf16"},
        {"batch_size": 1, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "bf16"},
        {"batch_size": 4, "sequence_length": 4096, "num_cores": 2, "auto_cast_type": "bf16"},
        {"batch_size": 4, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "bf16"},
        {"batch_size": 8, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "bf16"},
        {"batch_size": 16, "sequence_length": 4096, "num_cores": 8, "auto_cast_type": "bf16"},
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--cache-repo-id", type=str, default=None)
    args = parser.parse_args()
    model_ids = [args.model_id] if args.model_id else MODEL_CONFIGURATIONS.keys()
    for model_id in model_ids:
        for export_kwargs in MODEL_CONFIGURATIONS[model_id]:
            print(f"Exporting {model_id} with parameters {export_kwargs}")
            start = time.time()
            # Export in a separate process to reset the number of used cores
            with TemporaryDirectory() as tmpdir:
                command = f"optimum-cli export neuron -m {model_id}"
                for kwarg, value in export_kwargs.items():
                    command += f" --{kwarg} {value}"
                command += f" {tmpdir}"
                print(command)
                p = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                p.communicate()
                assert p.returncode == 0
            end = time.time()
            print(f"Model successfully exported in {end -start:.2f} s.")
            synchronize_hub_cache(args.cache_repo_id)


if __name__ == "__main__":
    main()
