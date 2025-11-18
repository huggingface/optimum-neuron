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

# Also initialize neuronx model-parallel groups so imports that rely on them succeed
try:
    from neuronx_distributed.parallel_layers.parallel_state import initialize_model_parallel

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        skip_collective_init=True,
        lnc_size=1,
        mesh_only=False,
    )
except Exception:
    # If neuronx isn't available or initialization fails, let the test run and surface the error.
    pass

from transformers import AutoTokenizer

# Ensure the convenience `warning` function exists on `optimum.utils.logging`.
try:
    import optimum.utils.logging as opt_logging

    if not hasattr(opt_logging, "warning"):
        def _warning(msg, *args, **kwargs):
            opt_logging.get_logger().warning(msg, *args, **kwargs)

        opt_logging.warning = _warning
except Exception:
    # If importing the module fails, we'll surface that during test import.
    pass

from torch import nn

class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # return a simple object with `logits` attribute to mimic HF models
        x = self.lin(torch.zeros(1, 4))
        return type("Out", (), {"logits": x})()


from optimum.neuron.trainers.grpo_trainer import NeuronGRPOTrainer

model = DummyModel()
tokenizer = DummyTokenizer()

trainer = NeuronGRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=None,
    reward_funcs=[lambda prompts, completions: [1.0] * len(completions)],
)

print("Trainer instantiated successfully!")
