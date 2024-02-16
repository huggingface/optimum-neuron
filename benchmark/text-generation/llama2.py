import os
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer

from benchmark import run
from optimum.neuron import NeuronModelForCausalLM


model_configurations = {
    "Llama-2-7BL": ["meta-llama/Llama-2-7b-chat-hf", 1, 2048],
    "Llama-2-7BT": ["meta-llama/Llama-2-7b-chat-hf", 4, 2048],
}

num_cores = len(os.listdir("/sys/class/neuron_device/")) * 2
if num_cores >= 4:
    extra_model_configurations = {
        "Llama-2-13BL": ["meta-llama/Llama-2-13b-chat-hf", 1, 2048],
        "Llama-2-13BT": ["meta-llama/Llama-2-13b-chat-hf", 4, 2048],
    }
    model_configurations = {**model_configurations, **extra_model_configurations}

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
