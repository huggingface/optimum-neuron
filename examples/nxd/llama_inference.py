model_path = "/home/ubuntu/neuronx-distributed/examples/inference/Llama-2-7b-chat-hf"
traced_model_path = "/home/ubuntu/neuronx-distributed/examples/inference/llama-2-7b-chat-hf-trace"

from llama2.llama2_runner import LlamaRunner


def main():
    max_context_length = 128
    max_new_tokens = 384
    batch_size = 2
    tp_degree = 24

    runner = LlamaRunner(model_path=model_path,
                        tokenizer_path=model_path)

    runner.trace(traced_model_path=traced_model_path,
                tp_degree=tp_degree,
                batch_size=batch_size,
                context_lengths=max_context_length,
                new_token_counts=max_new_tokens,
                on_device_sampling=True)

    neuron_model = runner.load_neuron_model(traced_model_path)

    prompt = ["I believe the meaning of life is", "The color of the sky is"]

    generate_ids, outputs = runner.generate_on_neuron(prompt, neuron_model)

    for idx, output in enumerate(outputs):
        print(f"output {idx}: {output}")


if __name__ == '__main__':
    #freeze_support()
    main()
