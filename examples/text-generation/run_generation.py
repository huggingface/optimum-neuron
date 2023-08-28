import argparse
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

from optimum.neuron import NeuronModelForCausalLM


def load_llm_optimum(model_id_or_path, batch_size, num_cores, auto_cast_type):
    config = AutoConfig.from_pretrained(model_id_or_path)
    export = getattr(config, "neuron", None) is None

    # Load and convert the Hub model to Neuron format
    return NeuronModelForCausalLM.from_pretrained(
        model_id_or_path,
        export=export,
        low_cpu_mem_usage=True,
        # These are parameters required for the conversion
        batch_size=batch_size,
        num_cores=num_cores,
        auto_cast_type=auto_cast_type,
    )


def generate(model, tokenizer, prompts, length, temperature):
    # Specifiy padding options for decoder-only architecture
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Encode tokens and generate using temperature
    tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    start = time.time()
    with torch.inference_mode():
        sample_output = model.generate(
            **tokens,
            do_sample=True,
            max_length=length,
            temperature=temperature,
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
        "--num_cores", type=int, default=2, help="The number of cores on which the model should be split."
    )
    parser.add_argument("--auto_cast_type", type=str, default="f32", help="One of f32, f16, bf16.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature to generate. 1.0 has no effect, lower tend toward greedy sampling.",
    )
    parser.add_argument(
        "--save_dir", type=str, help="The save directory. Allows to avoid recompiling the model every time."
    )
    parser.add_argument("--compare", action="store_true", help="Compare with the genuine transformers model on CPU.")
    parser.add_argument("--seed", type=int, default=None, help="Pass a seed for reproducibility.")
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    prompts = args.prompts.split("|")
    batch_size = len(prompts)
    model = load_llm_optimum(args.model, batch_size, args.num_cores, args.auto_cast_type)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    outputs, latency = generate(model, tokenizer, prompts, args.length, args.temperature)
    print(outputs)
    print(f"Outputs generated using Neuron model in {latency:.4f} s")
    if args.compare:
        cpu_model = AutoModelForCausalLM.from_pretrained("gpt2")
        outputs, latency = generate(cpu_model, tokenizer, prompts, args.length, args.temperature)
        print(outputs)
        print(f"Outputs generated using pytorch model in {latency:.4f} s")

    if args.save_dir:
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
