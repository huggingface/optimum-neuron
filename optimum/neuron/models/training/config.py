from dataclasses import dataclass
from pathlib import Path

import torch
from neuronx_distributed.parallel_layers import parallel_state

from ...configuration_utils import NeuronConfig, register_neuron_config
from ...utils.torch_xla_and_neuronx_initialization import init_process_group


@dataclass
@register_neuron_config
class TrainingNeuronConfig(NeuronConfig):
    tensor_parallel_size: int = 1
    sequence_parallel_enabled: bool = False
    kv_size_multiplier: int | None = None
    pipeline_parallel_size: int = 1
    virtual_pipeline_parallel_size: int = 1
    pipeline_parallel_num_microbatches: int = 1
    pipeline_parallel_use_zero1_optimizer: bool = False
    gradient_checkpointing: bool = False
    checkpoint_dir: str | Path | None = None
    num_local_ranks_per_step: int = 8
    use_xser: bool = True
    async_save: bool = False
    fuse_qkv: bool = False
    recompute_causal_mask: bool = True
    transpose_nki_inputs: bool = True

    def __post_init__(self):
        if self.tensor_parallel_size < 1:
            raise ValueError(f"The tensor parallel size must be >= 1, but {self.tensor_parallel_size} was given here.")
        if self.pipeline_parallel_size < 1:
            raise ValueError(
                f"The pipeline parallel size must be >= 1, but {self.pipeline_parallel_size} was given here."
            )
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)

        if not torch.distributed.is_initialized():
            init_process_group()
            if not torch.distributed.is_initialized():
                raise ValueError(
                    "Neuron training requires torch distributed to be initialized. "
                    "You can initialize it by running `torchrun`."
                )

        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=self.tensor_parallel_size,
                pipeline_model_parallel_size=self.pipeline_parallel_size,
            )

    def auto_kv_size_multiplier(self, num_key_value_heads: int) -> int:
        kv_size_multiplier = max(1, self.tensor_parallel_size // num_key_value_heads)
        if self.kv_size_multiplier is not None and self.kv_size_multiplier != kv_size_multiplier:
            raise ValueError(
                "A kv size multiplier was already specified and is different from the inferred one: "
                f"{self.kv_size_multiplier}"
            )
        return kv_size_multiplier

    @property
    def should_parallelize(self):
        return self.tensor_parallel_size > 1 or self.pipeline_parallel_size > 1
