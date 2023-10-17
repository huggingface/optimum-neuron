import argparse
import json
import os
import time

import torch
from transformers import AutoConfig, AutoTokenizer, set_seed

from optimum.neuron import NeuronModelForCausalLM


def generate(model, input_ids, length):
    start = time.time()
    with torch.inference_mode():
        output_tokens = model.generate(input_ids, do_sample=False, min_length=length, max_length=length)
    end = time.time()
    return output_tokens, (end - start)


def run(model_id, inc_length, max_length, json_path=None):
    prompts = ["One of my fondest memory"]
    config = AutoConfig.from_pretrained(model_id)
    batch_size = config.neuron["batch_size"]
    if len(prompts) < batch_size:
        prompts = prompts + [prompts[-1]] * (batch_size - len(prompts))
    model = NeuronModelForCausalLM.from_pretrained(model_id, export=False, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Specify padding options for decoder-only architecture
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    # Encode the first input tokens
    tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    bootstrap_input_ids = tokens.input_ids
    # Generate the first set of inputs
    input_ids, latency = generate(model, bootstrap_input_ids, inc_length)
    input_length = input_ids.size()[-1]
    neuron_config = getattr(model.config, "neuron")
    benchmark = {"neuron_config": neuron_config, "results": []}
    while input_length < max_length:
        # Generate a single input, just to evaluate the context encoding time
        _, encoding_time = generate(model, input_ids, input_length + 1)
        result = {
            "input_length": input_length,
            "batch_size": batch_size,
            "encoding_time": encoding_time,
            "generations": [],
        }
        for sequence_length in range(input_length + inc_length, max_length + 1, inc_length):
            output_ids, latency = generate(model, input_ids, sequence_length)
            throughput = batch_size * sequence_length / latency
            result["generations"].append(
                {
                    "sequence_length": sequence_length,
                    "new_tokens": sequence_length - input_length,
                    "latency": latency,
                    "generation_time": latency - encoding_time,
                    "throughput": throughput,
                }
            )
        # Reuse the first generated tokens for the next step
        input_length += inc_length
        input_ids = output_ids[:, :input_length]
        benchmark["results"].append(result)
    if json_path is not None:
        with open(json_path, "w") as fp:
            json.dump(benchmark, fp, indent=4)
    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="A neuron model in a local directory.")
    parser.add_argument("--inc-length", type=int, default=128, help="The number of tokens in each increment.")
    parser.add_argument("--max-length", type=int, default=2048, help="The maximum number of generated tokens.")
    parser.add_argument("--seed", type=int, default=None, help="Pass a seed for reproducibility.")
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    model_name = os.path.basename(os.path.normpath(args.model))
    benchmark = run(args.model, args.inc_length, args.max_length, json_path=f"{model_name}.json")
    # Dump encoding times
    results = benchmark["results"]
    print(f"{benchmark['neuron_config']}")
    print("Encoding times")
    print([result["input_length"] for result in results])
    print([f"{result['encoding_time']:.2f}" for result in results])
    # Just look at the first set of generations
    generations = results[0]["generations"]
    print(f"Latency and throughput for {args.inc_length} input tokens")
    print([generation["new_tokens"] for generation in generations])
    print([f"{generation['latency']:.2f}" for generation in generations])
    print([f"{generation['throughput']:.2f}" for generation in generations])
