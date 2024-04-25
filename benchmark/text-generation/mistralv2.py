from tempfile import TemporaryDirectory

from transformers import AutoTokenizer

from benchmark import run
from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.modeling_decoder import get_available_cores


def main():
    NUM_CORES = 8
    num_cores = get_available_cores()
    if num_cores < NUM_CORES:
        raise ValueError(f"This benchmark can only run on an instance with at least {NUM_CORES} cores.")

    model_configurations = {
        "Mistral-7B-v2-BS1": ["mistralai/Mistral-7B-Instruct-v0.2", 1, 4096],
        "Mistral-7B-v2-BS4": ["mistralai/Mistral-7B-Instruct-v0.2", 4, 4096],
        "Mistral-7B-v2-BS8": ["mistralai/Mistral-7B-Instruct-v0.2", 8, 4096],
        "Mistral-7B-v2-BS16": ["mistralai/Mistral-7B-Instruct-v0.2", 16, 4096],
        "Mistral-7B-v2-BS32": ["mistralai/Mistral-7B-Instruct-v0.2", 32, 4096],
    }

    for model_name, model_configuration in model_configurations.items():
        model_id, batch_size, seq_length = model_configuration
        model = NeuronModelForCausalLM.from_pretrained(
            model_id,
            export=True,
            batch_size=batch_size,
            sequence_length=seq_length,
            auto_cast_type="bf16",
            num_cores=NUM_CORES,
        )
        with TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(tmpdir)
            json_path = f"{model_name}.json"
            run(tmpdir, 256, 2048, json_path=json_path)


if __name__ == "__main__":
    main()
