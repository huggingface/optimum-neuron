from typing import Any

import torch
import torch_xla.core.xla_model as xm
from optimum.utils import logging
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..utils import is_trl_available
from .grpo_config import NeuronGRPOConfig
from .transformers import NeuronTrainer
from .trl_utils import TRL_VERSION
from neuronx_distributed.pipeline import NxDPPModel


logger = logging.get_logger()


def identity(x):
    # Identity function for data collator as no collation needed for GRPO
    return x


if is_trl_available():
    # Import TRL classes only when available to avoid hard dependency at import time.
    from trl import GRPOConfig, GRPOTrainer  # type: ignore
else:

    class GRPOTrainer:
        """Placeholder used when `trl` is not installed."""


    class GRPOConfig:
        """Placeholder config used when `trl` is not installed."""


# Create a new class that inherits from NeuronTrainer and uses the source methods from GRPOTrainer.
_GRPOTrainer = type(
    "_GRPOTrainer",
    (NeuronTrainer,),
    GRPOTrainer.__dict__.copy()
)


class NeuronGRPOTrainer(_GRPOTrainer):
    """
    GRPOTrainer adapted for Neuron. Only Neuron-specific wiring is kept; GRPO logic is reused from TRL.
    Note: vLLM/offline optimizations should be ported separately in the future.
    """

    def __init__(
        self,
        model: PreTrainedModel | torch.nn.Module | str,
        args: GRPOConfig | None = None,
        data_collator: Any | None = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, Any] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ):
        if not is_trl_available(required_version=TRL_VERSION): 
            raise RuntimeError(f"Using NeuronGRPOTrainer requires trl=={TRL_VERSION}.")

        args_is_none = args is None
        if args is None:
            args = NeuronGRPOConfig(output_dir="tmp_trainer")
        elif args is not None and args.__class__.__name__ == "NeuronTrainingArguments":
            args_as_dict = args.to_dict()
            args_as_dict.update({k: getattr(args, k) for k in args_as_dict.keys() if k.endswith("_token")})
            args = NeuronGRPOConfig(**args_as_dict)

        if args_is_none:
            log_level = args.get_process_log_level() 
            logging.set_verbosity(log_level)
            logging.warning(f"No `GRPOConfig` passed, using `output_dir={args.output_dir}`.")

        if data_collator is None:
            data_collator = identity

        NeuronTrainer.__init__(
            self,
            model,
            args, 
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class if tokenizer is None else tokenizer,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        )

        if not hasattr(self, "_train_batch_size"):
            # Fallback to per-device train batch size when the base trainer doesn't set this field
            try:
                self._train_batch_size = self.args.per_device_train_batch_size
            except Exception:
                self._train_batch_size = 1

        # TRL 0.24 uses self.num_generations in sampler
        if not hasattr(self, "num_generations"):
            try:
                self.num_generations = int(getattr(self.args, "num_generations", getattr(self.args, "steps_per_generation", 1)))
            except Exception:
                self.num_generations = 1

        try:
            gen_bs = getattr(self.args, "generation_batch_size", None)
        except Exception:
            gen_bs = None
        if gen_bs is None:
            try:
                self.args.generation_batch_size = int(self._train_batch_size) * int(self.num_generations)
            except Exception:
                self.args.generation_batch_size = int(self.num_generations)

        # TRL's dataloader/sampler references `self.num_iterations`; default to 1
        if not hasattr(self, "num_iterations"):
            self.num_iterations = 1

        # TRL's dataloader checks `self.shuffle_dataset`; default to True
        if not hasattr(self, "shuffle_dataset"):
            try:
                self.shuffle_dataset = bool(getattr(self.args, "shuffle_dataset", True))
            except Exception:
                self.shuffle_dataset = True

        # capture reward functions if provided for TRL's GRPO loss
        if "reward_funcs" in kwargs:
            self.reward_funcs = kwargs["reward_funcs"]

    def train(self, resume_from_checkpoint: str | bool | None = None):
        return NeuronTrainer.train(self, resume_from_checkpoint=resume_from_checkpoint)

    # must override train_step with grpo loss computation
    def train_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        manager = self.autocast_smart_context_manager()

        if isinstance(model, NxDPPModel):
            with manager:
                loss = model.run_train(**inputs)
            if self.pp_rank != self.pp_size - 1:
                dtype = torch.bfloat16 if self.args.bf16 else torch.float32
                loss = torch.tensor(0, dtype=dtype).to(xm.xla_device())
            return loss

        with manager:
            # Delegate GRPO loss computation to TRL's inherited logic
            loss = self.compute_loss(model, inputs, return_outputs=False) 

        if isinstance(num_items_in_batch, torch.Tensor):
            num_items = num_items_in_batch.item()
        else:
            num_items = num_items_in_batch

        if num_items is not None:
            loss = loss / num_items
        else:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)
        return loss
