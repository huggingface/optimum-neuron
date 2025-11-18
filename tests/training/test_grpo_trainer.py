import os
import torch

# Ensure a single-process torch.distributed group is initialized before importing
# trainer code that requires `torch.distributed.is_initialized()` during construction.
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
if torch.distributed.is_available() and not torch.distributed.is_initialized():
    try:
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    except Exception:
        # If initialization fails, tests will surface the relevant error.
        pass

from transformers import AutoTokenizer
from optimum.neuron.trainers.grpo_trainer import NeuronGRPOTrainer

model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

trainer = NeuronGRPOTrainer(
    model=model_name,
    tokenizer=tokenizer,
    args=None,
    reward_funcs=[lambda prompts, completions: [1.0]*len(completions)],
)

print("Trainer instantiated successfully!")
