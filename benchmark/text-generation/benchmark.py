import argparse
import json
import os
import time

import torch
from transformers import AutoConfig, AutoTokenizer, set_seed

from optimum.neuron import NeuronModelForCausalLM


def generate(model, input_ids, output_length):
    start = time.time()
    with torch.inference_mode():
        output_tokens = model.generate(input_ids, do_sample=False, min_length=output_length, max_length=output_length)
    end = time.time()
    return output_tokens, (end - start)


def run(model_id, inc_length, max_length, json_path=None):
    # Encode the reference prompt
    local_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(local_path, "wiki.txt")) as f:
        prompt = f.read()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokens = tokenizer([prompt], return_tensors="pt")
    # Evaluate the batch size
    config = AutoConfig.from_pretrained(model_id)
    batch_size = config.neuron["batch_size"]
    model = NeuronModelForCausalLM.from_pretrained(model_id, export=False, low_cpu_mem_usage=True)

    def get_input_ids(tokens, batch_size, input_length):
        return tokens.input_ids[0, :input_length].repeat((batch_size, 1))

    neuron_config = getattr(model.config, "neuron")
    benchmark = {"neuron_config": neuron_config, "results": []}
    for input_length in range(inc_length, max_length - inc_length + 1, inc_length):
        # Generate a single input, just to evaluate the context encoding time
        input_ids = get_input_ids(tokens, batch_size, input_length + 1)
        _, encoding_time = generate(model, input_ids, 1)
        new_tokens = inc_length
        output_ids, duration = generate(model, input_ids, input_length + new_tokens)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="A neuron model in a local directory.")
    parser.add_argument("--inc-length", type=int, default=512, help="The number of tokens in each increment.")
    parser.add_argument("--max-length", type=int, default=4096, help="The maximum number of generated tokens.")
    parser.add_argument("--seed", type=int, default=None, help="Pass a seed for reproducibility.")
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    model_name = os.path.basename(os.path.normpath(args.model))
    benchmark = run(args.model, args.inc_length, args.max_length, json_path=f"{model_name}.json")
    # Dump encoding times
    print(f"{benchmark['neuron_config']}")
    results = benchmark["results"]
    print(results)
