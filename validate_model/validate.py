import torch
import typer
from model_runner import MixtralRunner

from transformers import GenerationConfig

# model_id = "/home/ubuntu/model_hf/Mixtral-8x7B-v0.1/"
# traced_model_path = "/home/ubuntu/traced_model/Mixtral-8x7B-v0.1/"

torch.manual_seed(0)


def model_sample(model_id: str = "Maykeye/TinyLLama-v0", traced_model_path: str = None):
    """
    Validate a model with a specific configuration.

    Args:
        model_id: The path to the model.
        traced_model_path: The path to the traced model. If None, it will be set from the model_id.
    """
    if traced_model_path is None:
        traced_model_path = "traced_" + model_id.replace("/", "_")

    # Compile the model for a specific configuration
    generation_config = GenerationConfig.from_pretrained(model_id)
    generation_config.top_k = 1
    generation_config.do_sample = False

    runner = MixtralRunner(model_id, tokenizer_path=model_id, generation_config=generation_config)

    batch_size = 2
    max_prompt_length = 1024
    sequence_length = 2048

    runner.trace(
        traced_model_path=traced_model_path,
        tp_degree=32,
        batch_size=batch_size,
        max_prompt_length=max_prompt_length,
        sequence_length=sequence_length,
    )
    exit()
    # Load model weights into Neuron devise
    # We will use the returned model to run accuracy and perf tests
    print("\nLoading model to Neuron device ..")
    neuron_model = runner.load_neuron_model(traced_model_path)

    # Confirm the traced model matches the huggingface model run on cpu
    print("\nChecking accuracy ..")
    runner.check_accuracy(neuron_model, batch_size, sequence_length)

    # Perform inference
    prompts = ["I believe the meaning of life is", "The color of the sky is"]
    print("\nGenerating ..")
    _, outputs = runner.generate_on_neuron(prompts, neuron_model)
    print("Generated outputs:")
    for idx, output in enumerate(outputs):
        print(f"output {idx}: {output}")



if __name__ == "__main__":
    # setup typer app
    app = typer.Typer()
    app.command()(model_sample)
    app()
    # model_id = "/home/ubuntu/model_hf/Mixtral-8x7B-v0.1/"
    # traced_model_path = "/home/ubuntu/traced_model/Mixtral-8x7B-v0.1/"
    # model_sample(model_id, traced_model_path)