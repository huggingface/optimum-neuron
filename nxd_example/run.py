import copy
import json
from functools import partial
from typing import List

import fire
import torch
from config import Llama1B
from model import Transformer, load_llama_checkpoint
from neuronx_distributed.trace import ModelBuilder
from neuronx_distributed.trace.model_builder import BaseModelInstance
from tokenizer import Tokenizer


def generate(model: torch.nn.Module,
             max_len: int,
             prompt_tokens: List[List[int]],
             stop_tokens: List[int]):

    # Track max pos per batch
    last_pos = torch.tensor([len(prompt)-1 for prompt in prompt_tokens], dtype=torch.int32)

    # Pad all batch lines to the same sequence length
    padded_tokens = [prompt + [0] * (max_len - len(prompt)) for prompt in prompt_tokens]
    tokens = torch.tensor(padded_tokens, dtype=torch.int32)

    input_tokens = tokens
    input_bs, input_len = input_tokens.shape

    # A tensor to keep track of generation completion per batch line
    is_gen_complete = torch.full((input_bs, 1), False)

    while True:

        logits = model.forward(input_tokens, last_pos)
        last_pos = last_pos + 1

        # assuming we are doing greedy sampling
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

        input_tokens = next_token.to(torch.int32)

        # Add the new token to prompt
        for idx, prompt  in enumerate(prompt_tokens):
            if not is_gen_complete[idx][0].item():
                prompt.append(next_token[idx].item())

        for stop_token in stop_tokens:
            is_gen_complete = is_gen_complete.logical_or(next_token == stop_token)

        # Stop generation when all batch lines are complete
        if is_gen_complete.all():
            break

        if torch.max(last_pos).item() >= max_len:
            break

    return prompt_tokens

@torch.inference_mode()
def generate_cpu(batch_size=2,
                 seq_len=128,
                 model_path="/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth",
                 tokenizer_path="/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model",
                 prompts=["I will just count till 20 - 1,2,3,4",
                          "I will just count till 20 - 1,2,3,4,5,6"]):

    checkpoint = load_llama_checkpoint(Llama1B, model_path)

    model: Transformer = Transformer(Llama1B, batch_size, seq_len)
    model.load_state_dict(checkpoint, strict=False)

    tokenizer = Tokenizer(model_path=tokenizer_path)
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    generate(model, seq_len, prompt_tokens, stop_tokens=tokenizer.stop_tokens)

    for prompt in prompt_tokens:
        print(tokenizer.decode(prompt))

@torch.inference_mode()
def generate_nxd(traced_model_path="/home/ubuntu/workspace-trn1/src/Aazhiko-workplace/scripts/modeling_example/traced_model/",
                 tokenizer_path="/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model",
                 prompts=["I will just count till 20 - 1,2,3,4",
                          "I will just count till 20 - 1,2,3,4,5,6"]):
    import os

    from safetensors.torch import load_file

    with open(traced_model_path + "config.json", 'r') as file:
        cfg = json.load(file)

    bs, seq_len, tp_degree = cfg["batch_size"], cfg["seq_len"], cfg["tp_degree"]

    if len(prompts) != bs:
        raise ValueError(f"Prompts size does not match batch size {cfg['batch_size']}")

    weights = []
    for rank in range(tp_degree):
        ckpt = load_file(os.path.join(traced_model_path, f"weights/tp{rank}_sharded_checkpoint.safetensors"))
        weights.append(ckpt)

    model = torch.jit.load(traced_model_path + "nxd_model.pt")
    start_rank_tensor = torch.tensor([0], dtype=torch.int32, device="cpu")
    model.nxd_model.initialize(weights, start_rank_tensor)
    print("loaded", flush=True)

    tokenizer = Tokenizer(model_path=tokenizer_path)
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    generate(model, seq_len, prompt_tokens, stop_tokens=tokenizer.stop_tokens)

    for prompt in prompt_tokens:
        print(tokenizer.decode(prompt))


def compile(batch_size=2,
            seq_len=128,
            tp_degree=32,
            model_path="/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth",
            output_path="traced_model/"):

    # ModelBuilder takes in a object of type BaseModelInstance. This object
    # should have two functions `load_module` and `get`. The object declares
    # how the object can be initialized so the tracing function can create its own
    # instance.
    #
    # Why cannot ModelBuilder just take model instance as input?
    # At the time of writing ModelBuilder, we needed ModelBuilder to run within
    # a different process for each trace. We have since fixed this limiation and
    # can relax the APIs. As a part of planned ModelBuilder improvement, we will
    # remove the need for `BaseModelInstance`.

    # What is an alias? and why do I need it?
    # If you have any state that you want to use across model invocations
    # you would want to use an alias. Aliasing, tells the compiler that
    # the output can be written to the same input buffer. This avoids
    # the creation of duplicate memory allocations for the output.
    #
    # On NxD, all output tensors are copied back from device to CPU
    # after a model invocation. But the aliased tensors are not returned
    # and are retained on device.
    #
    # So if you have a buffer that is expensive to repeatedly copy
    # to and from the device, you should use an alias. KV Cache is a good
    # candidate for aliasing.
    #
    # How do I define an alias?
    # Alias is a map. It maps buffer -> output index.
    #
    # Say we have Module defined as,
    #
    #  Module(torch.nn.Module):
    #
    #    def __init__(self):
    #      self.register_buffer("cache", torch.zeros(...))
    #
    #    def forward(input_A, input_B):
    #       ...
    #       return output, output_A
    #
    # And we want to alias input_A and output_A. The alias would say,
    #
    #  module = Module()
    #  alias = { module.cache : 1 }
    #
    #  This means `cache` is aliased to the output number 1 which is `output_A`

    # CURRENT MODEL BUILDER FLOW. LOOK AFTER THIS SECTION FOR HOW THE NEW API
    # WILL LOOK LIKE.
    class Instance(BaseModelInstance):

        def __init__(self):
            self.module = None

        def load_module(self):
            self.module = Transformer(Llama1B, batch_size, seq_len)

        def get(self, bucket_rank, **kwargs):

            # The Transformer model return logits as index 0. We want to start
            # aliasing from output index 1
            #
            # Transformer() -> (logits,
            #                   k_cache_lay1, .., k_cache_layN,
            #                   v_cache_lay1, ... v_cache_layN)

            aliases = {}
            output_index = 1
            for i, layer in enumerate(self.module.layers):
                aliases[layer.attention.cache_k] = output_index
                output_index = output_index + 1
            for i, layer in enumerate(self.module.layers):
                aliases[layer.attention.cache_v] = output_index
                output_index = output_index + 1

            return self.module, aliases

    builder = ModelBuilder(router=None,
                           tp_degree=tp_degree,
                           checkpoint_loader=partial(load_llama_checkpoint, Llama1B, model_path, tp_degree),
                           debug=True)
    builder.add(key="prefil",
                model_instance=Instance(),
                example_inputs=[(torch.ones((batch_size, seq_len),dtype=torch.int32), # input tokens
                                 torch.tensor([0] * batch_size, dtype=torch.int32),)],   # last_pos
                compiler_args="--auto-cast=none")
    builder.add(key="decode",
                model_instance=Instance(),
                example_inputs=[(torch.ones((batch_size, 1),dtype=torch.int32), # input tokens
                                 torch.tensor([0] * batch_size, dtype=torch.int32),)],   # last_pos
                compiler_args="--auto-cast=none")

    traced_model = builder.trace(initialize_model_weights=False)

    builder.shard_checkpoint(serialize_path=output_path + "weights/")
    torch.jit.save(traced_model, output_path + "nxd_model.pt")

    # Lets store the config along with the saved model
    data = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tp_degree": tp_degree
    }
    with open(output_path + "config.json", 'w') as f:
        json.dump(data, f, indent=4)

    '''
    Sneek peek at how the new Model Builder API will look like :

    --- Simple Case ---

    torch.dist.init_process_group(backend="xla")
    parallel_state.initialize_model_parallel(tp_degree=32)

    model = Model()

    model = ModelBuilder(model, world_size=32)
    .spmd_trace(torch.randn(1024))
    .compile()

    model.save("model.nxdpt")

    model = NxDModel.load("model.nxdpt").to_neuron()
    # HERE you could call model.set_weights(different_sharded_weights) to replace the weights

    output = model(torch.randn(1024))

    --- You can load weights after compilation too ---

    model = NxDModel.load("model.nxdpt")
    sharded_checkpoint = shard(model, checkpoint: TensorDict, device="neuron")
    model.set_weights(sharded_checkpoint)
    model.to_neuron()

    '''


def test_attention(model_path="/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth"):

    from config import Llama1B
    from model import Attention, precompute_rope

    # Lets test with float32
    cfg = copy.deepcopy(Llama1B)
    cfg.dtype = torch.float32

    bs, seqlen, hidden_size = 1, 64, 2048 # batch, seq len, hidden size
    hidden = torch.randn((bs, seqlen, hidden_size), dtype=torch.float32)
    start_pos = torch.tensor([0])
    mask = torch.full((64, 64), torch.finfo(hidden.dtype).min)
    mask = torch.triu(mask, diagonal=1)
    rope_cache = precompute_rope("cpu", 500000.0, 64, 64)

    import re
    state_dict = load_llama_checkpoint(model_path=model_path, cfg=cfg)
    attn_dict = {re.sub('layers.*.attention.','',k):v.to(torch.float32) for (k,v) in state_dict.items() if "attention" in k}

    class Instance(BaseModelInstance):

        def __init__(self):
            self.module = None

        def load_module(self):
            self.module = Attention(cfg, bs, seqlen)

        def get(self, bucket_rank, **kwargs):
            return self.module, {self.module.cache_k:1, self.module.cache_v:2}

    builder = ModelBuilder(router=None,
                           tp_degree=1,
                           checkpoint_loader=lambda: attn_dict,
                           debug=True)
    builder.add(key="prefil",
                model_instance=Instance(),
                example_inputs=[(hidden, start_pos, mask, rope_cache)],
                compiler_args="--auto-cast=none")
    neuron_attn = builder.trace()

    start_rank_tensor = torch.tensor([0], dtype=torch.int32, device="cpu")
    neuron_attn.nxd_model.initialize([attn_dict], start_rank_tensor)

    attn = Attention(cfg, bs, seqlen)
    attn.load_state_dict(attn_dict, strict=False)

    hidden = torch.randn((bs,seqlen,hidden_size), dtype=torch.float32)
    start_pos = torch.tensor([0])
    mask = torch.full((64, 64), torch.finfo(hidden.dtype).min)
    mask = torch.triu(mask, diagonal=1)
    rope_cache = precompute_rope("cpu", 500000.0, 64, 64)

    cpu_o, cache_k, cache_v = attn(hidden, start_pos, mask, rope_cache)

    # Note: If you alias the tensors, the compiled model will not return it.
    # They are kept on device. We cannot return a signle tensor is because
    # each core has its own copy of the tensor. As we are SPMD the output
    # is assumed to be the same, so return the output of the first core.
    neuron_o = neuron_attn(hidden, start_pos, mask, rope_cache)

    # But we can access them this way per rank. This is how you do it.
    rank=0
    cache_k1 = neuron_attn.nxd_model.state[rank]['cache_k'].to("cpu")
    cache_v1 = neuron_attn.nxd_model.state[rank]['cache_v'].to("cpu")

    torch.testing.assert_close(cpu_o, neuron_o)
    torch.testing.assert_close(cache_k,cache_k1)
    torch.testing.assert_close(cache_v,cache_v1)


if __name__ == '__main__':
    fire.Fire({
        'generate_cpu': generate_cpu,
        'generate_nxd': generate_nxd,
        'compile': compile,
        'test_attention': test_attention,
    })
