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
        "Llama-3.1-8B-BS1": ["meta-llama/Meta-Llama-3.1-8B", 1, 4096],
        "Llama-3.1-8B-BS4": ["meta-llama/Meta-Llama-3.1-8B", 4, 4096],
        "Llama-3.1-8B-BS8": ["meta-llama/Meta-Llama-3.1-8B", 8, 4096],
        "Llama-3.1-8B-BS16": ["meta-llama/Meta-Llama-3.1-8B", 16, 4096],
        "Llama-3.1-8B-BS32": ["meta-llama/Meta-Llama-3.1-8B", 32, 4096],
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
