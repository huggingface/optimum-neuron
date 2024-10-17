from llama2.llama2_runner import LlamaRunner
from transformers import AutoTokenizer


model_path = "/home/ubuntu/optimum-neuron/examples/nxd/Llama-2-7b-chat-hf"
traced_model_path = "/home/ubuntu/optimum-neuron/examples/nxd/llama-2-7b-chat-hf-trace"


def main():
    max_context_length = 128
    max_new_tokens = 384
    batch_size = 2
    tp_degree = 24

    runner = LlamaRunner(model_path=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_size = "right"

    runner.trace(
        traced_model_path=traced_model_path,
        tp_degree=tp_degree,
        batch_size=batch_size,
        context_lengths=max_context_length,
        new_token_counts=max_new_tokens,
        on_device_sampling=True,
    )

    neuron_model = runner.load_neuron_model(traced_model_path)

    prompt = ["I believe the meaning of life is", "The color of the sky is"]

    generate_ids, outputs = runner.generate(neuron_model, tokenizer, prompt)

    for idx, output in enumerate(outputs):
        print(f"output {idx}: {output}")


if __name__ == "__main__":
    # freeze_support()
    main()
