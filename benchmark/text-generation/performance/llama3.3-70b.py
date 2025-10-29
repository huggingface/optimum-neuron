import torch
from transformers import AutoTokenizer

from benchmark import run
from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.configuration_utils import NeuronConfig
from optimum.neuron.utils.system import get_available_cores


def main():
    NUM_CORES = 24
    num_cores = get_available_cores()
    if num_cores < NUM_CORES:
        raise ValueError(f"This benchmark can only run on an instance with at least {NUM_CORES} cores.")

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    model_configurations = {
        "Llama-3.3-70B-BS1": [model_id, 1, 4096],
        "Llama-3.3-70B-BS4": [model_id, 4, 4096],
        "Llama-3.3-70B-BS8": [model_id, 8, 4096],
    }

    for model_name, model_configuration in model_configurations.items():
        model_id, batch_size, seq_length = model_configuration
        try:
            neuron_config = NeuronConfig.from_pretrained(model_id)
            assert neuron_config.batch_size == batch_size, (
                f"Model {model_name} is not configured for batch size {batch_size}."
            )
            assert neuron_config.tp_degree == NUM_CORES, f"Model {model_name} is not configured for {NUM_CORES} cores."
            assert neuron_config.sequence_length == seq_length, (
                f"Model {model_name} is not configured for sequence length {seq_length}."
            )
            assert neuron_config.dtype == torch.bfloat16, f"Model {model_name} is not configured for bf16."
            model = NeuronModelForCausalLM.from_pretrained(model_id)
        except Exception:
            model = NeuronModelForCausalLM.from_pretrained(
                model_id,
                export=True,
                batch_size=batch_size,
                sequence_length=seq_length,
                auto_cast_type="bf16",
                num_cores=NUM_CORES,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        json_path = f"{model_name}.json"
        run(model, tokenizer, 256, 2048, json_path=json_path)


if __name__ == "__main__":
    main()
