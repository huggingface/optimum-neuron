import argparse
import json
import os
import time
from typing import Union

import torch
from modeling_decoder import NeuronModelForCausalLM
from modules.config import NeuronInferenceConfig
from transformers import AutoTokenizer, GenerationConfig, set_seed
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput


SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]


def generate(model, tokenizer, prompts, max_new_tokens):
    # Sanity checks
    if len(prompts) != model.config.max_batch_size:
        raise ValueError(f"Number of prompts should match batch size {model.config.max_batch_size}")
    set_seed(0)  # to avoid randomness in sampling if any
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")

    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.max_new_tokens = max_new_tokens
    # This is hard-coded because of on-device generation sampling
    generation_config.do_sample = True
    generation_config.top_k = 1

    start = time.time()
    outputs = model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
    )
    end = time.time()
    model.reset()

    if isinstance(outputs, SampleOutput.__args__):
        # Get token ids from output when return_dict_in_generate=True
        output_ids = outputs.sequences
    else:
        output_ids = outputs
    output_tokens = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_tokens, (end - start)


def timed_generate(model, input_ids, output_length):
    start = time.time()
    with torch.inference_mode():
        output_tokens = model.generate(input_ids, do_sample=False, min_length=output_length, max_length=output_length)
    end = time.time()
    model.reset()
    return output_tokens, (end - start)


def benchmark(model, tokenizer, inc_length, max_length, new_tokens, json_path=None):
    # Encode the reference prompt
    local_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(local_path, "wiki.txt")) as f:
        prompt = f.read()
    tokens = tokenizer([prompt], return_tensors="pt")
    # Evaluate the batch size
    batch_size = model.config.batch_size

    def get_input_ids(tokens, batch_size, input_length):
        return tokens.input_ids[0, :input_length].repeat((batch_size, 1))

    benchmark = {"results": []}
    for input_length in range(inc_length, max_length + 1, inc_length):
        # Generate a single input, just to evaluate the context encoding time
        input_ids = get_input_ids(tokens, batch_size, input_length + 1)
        _, encoding_time = timed_generate(model, input_ids, 1)
        output_ids, duration = timed_generate(model, input_ids, input_length + new_tokens)
        latency = (duration - encoding_time) / new_tokens * 1000
        throughput = new_tokens * batch_size / duration
        result = {
            "input_length": input_length,
            "batch_size": batch_size,
            "encoding_time": encoding_time,
            "new_tokens": new_tokens,
            "latency": latency,
            "throughput": throughput,
        }
        benchmark["results"].append(result)
    if json_path is not None:
        with open(json_path, "w") as fp:
            json.dump(benchmark, fp, indent=4)
    return benchmark


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Action to perform", dest="action")
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("model", type=str, help="The HF Hub model id or a local directory.")
    export_parser = subparsers.add_parser("export", parents=[parent_parser], help="Convert model to Neuron.")
    export_parser.add_argument("--save_dir", type=str, required=True, help="The save directory.")
    export_parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size.",
    )
    export_parser.add_argument("--sequence_length", type=int, help="The maximum sequence length.")
    export_parser.add_argument("--tp_degree", type=int, default=2, help="The level of tensor parallelism.")
    run_parser = subparsers.add_parser(
        "run", parents=[parent_parser], help="Generate tokens using the specified model."
    )
    run_parser.add_argument(
        "--prompts",
        type=str,
        default="The color of the sky is",
        help="The prompts to use for generation, using | as separator.",
    )
    run_parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="The number of new tokens in the generated sequences."
    )
    bench_parser = subparsers.add_parser("bench", parents=[parent_parser], help="Benchmark the specified model.")
    bench_parser.add_argument(
        "--inc-length", type=int, default=512, help="The number of input tokens in each increment."
    )
    bench_parser.add_argument("--max-length", type=int, default=4096, help="The maximum number of input tokens.")
    bench_parser.add_argument("--new-tokens", type=int, default=128, help="The number of generated tokens.")

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_size = "right"

    if args.action == "export":
        neuron_config = NeuronInferenceConfig.from_model_config(args.model)
        neuron_config.enable_bucketing = True
        neuron_config.tp_degree = args.tp_degree
        neuron_config.batch_size = args.batch_size
        max_length = args.sequence_length
        if max_length is None:
            max_length = neuron_config.max_position_embeddings
        neuron_config.max_context_length = int(max_length * 0.75)
        neuron_config.max_new_tokens = max_length
        neuron_config.max_length = max_length
        neuron_config.n_positions = max_length
        neuron_config.max_batch_size = args.batch_size
        neuron_config.padding_side = "right"
        neuron_config.pad_token_id = neuron_config.eos_token_id
        # WARNING: We use on-device sampling, meaning that the generation parameters
        # cannot be changed at inference
        neuron_config.on_device_sampling = True
        neuron_config.do_sample = True
        neuron_config.top_k = 1

        NeuronModelForCausalLM.export(
            args.model,
            neuron_config,
            serialize_base_path=args.save_dir,
        )
        # Also save tokenizer to be able to use it when reloading the model
        tokenizer.save_pretrained(args.save_dir)
    elif args.action == "run":
        start = time.time()
        neuron_model = NeuronModelForCausalLM.load(args.model)
        end = time.time()
        print(f"Neuron model loaded in {end - start:.2f} s")
        batch_size = neuron_model.config.batch_size
        prompts = args.prompts.split("|")
        if len(prompts) < batch_size:
            prompts = prompts + [prompts[-1]] * (batch_size - len(prompts))
        outputs, latency = generate(neuron_model, tokenizer, prompts, args.max_new_tokens)

        for idx, output in enumerate(outputs):
            print(f"output {idx}: {output}")
        print(f"{len(outputs)} outputs generated using Neuron model in {latency:.4f} s")
    elif args.action == "bench":
        neuron_model = NeuronModelForCausalLM.load(args.model)
        model_name = os.path.basename(os.path.normpath(args.model))
        benchmark(
            neuron_model, tokenizer, args.inc_length, args.max_length, args.new_tokens, json_path=f"{model_name}.json"
        )


if __name__ == "__main__":
    main()
