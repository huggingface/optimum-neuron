import argparse
import time

import torch
from transformers import AutoTokenizer, set_seed

from optimum.neuron import NeuronModelForCausalLM


def generate(model, tokenizer, prompts, max_new_tokens, temperature):
    # Specifiy padding options for decoder-only architecture
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Encode tokens and generate using temperature
    tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    start = time.time()
    with torch.inference_mode():
        sample_output = model.generate(
            **tokens, do_sample=True, max_new_tokens=max_new_tokens, temperature=temperature, top_k=50, top_p=0.9
        )
    end = time.time()
    outputs = [tokenizer.decode(tok) for tok in sample_output]
    return outputs, (end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="The HF Hub model id or a local directory.")
    parser.add_argument(
        "--prompts",
        type=str,
        default="One of my fondest memory is",
        help="The prompts to use for generation, using | as separator.",
    )
    parser.add_argument("--length", type=int, default=128, help="The number of tokens in the generated sequences.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature to generate. 1.0 has no effect, lower tend toward greedy sampling.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Pass a seed for reproducibility.")
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    start = time.time()
    model = NeuronModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True)
    end = time.time()
    print(f"Neuron model loaded in {end - start:.2f} s.")
    batch_size = model.neuron_config.batch_size
    prompts = args.prompts.split("|")
    if len(prompts) < batch_size:
        prompts = prompts + [prompts[-1]] * (batch_size - len(prompts))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    outputs, latency = generate(model, tokenizer, prompts, args.length, args.temperature)
    print(outputs)
    print(f"{len(outputs)} outputs generated using Neuron model in {latency:.4f} s")
