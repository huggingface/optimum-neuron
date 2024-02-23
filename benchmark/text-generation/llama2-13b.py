from tempfile import TemporaryDirectory

from transformers import AutoTokenizer

from benchmark import run
from optimum.neuron import NeuronModelForCausalLM


model_configurations = {
    "Llama-2-13B-BS1": ["meta-llama/Llama-2-13b-chat-hf", 1, 4096],
    "Llama-2-13B-BS4": ["meta-llama/Llama-2-13b-chat-hf", 4, 4096],
    "Llama-2-13B-BS8": ["meta-llama/Llama-2-13b-chat-hf", 8, 4096],
    "Llama-2-13B-BS16": ["meta-llama/Llama-2-13b-chat-hf", 16, 4096],
}


for model_name, model_configuration in model_configurations.items():
    model_id, batch_size, seq_length = model_configuration
    model = NeuronModelForCausalLM.from_pretrained(
        model_id, export=True, batch_size=batch_size, sequence_length=seq_length, auto_cast_type="fp16"
    )
    with TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(tmpdir)
        json_path = f"{model_name}.json"
        run(tmpdir, 256, 1024, json_path=json_path)
