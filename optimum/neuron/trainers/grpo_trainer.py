from typing import Any, Iterator
from collections import defaultdict
import inspect
import os
import torch
import torch_xla.core.xla_model as xm
from optimum.utils import logging
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from neuronx_distributed.pipeline import NxDPPModel
from neuronx_distributed.parallel_layers.parallel_state import get_data_parallel_replica_groups

from ..utils import is_trl_available
from .grpo_config import NeuronGRPOConfig
from .transformers import NeuronTrainer
from .trl_utils import TRL_VERSION


logger = logging.get_logger()
logger.setLevel(logging.INFO)


if is_trl_available():
    import trl.trainer.utils
    
    def safe_nanmin(tensor: torch.Tensor) -> torch.Tensor:
        """ XLA-compatible nanmin that handles empty tensors """
        logger.info("[safe_nanmin] Called")
        if tensor.numel() == 0:
            return torch.tensor(float('nan'), device=tensor.device)
        return torch.min(tensor)
    
    def safe_nanmax(tensor: torch.Tensor) -> torch.Tensor:
        """ XLA-compatible nanmax that handles empty tensors """
        logger.info("[safe_nanmax] Called")
        if tensor.numel() == 0:
            return torch.tensor(float('nan'), device=tensor.device)
        return torch.max(tensor)
    
    trl.trainer.utils.nanmin = safe_nanmin
    trl.trainer.utils.nanmax = safe_nanmax
    
    from trl import GRPOConfig, GRPOTrainer
    from trl.data_utils import is_conversational
    from trl.trainer.utils import (
        split_tensor_dict,
        shuffle_sequence_dict,
        selective_log_softmax,
        entropy_from_logits,
        pad,
    )
    _GRPO = GRPOTrainer
else:
    class GRPOTrainer:
        pass
    class GRPOConfig:
        pass
    _GRPO = None


def identity(x):
    return x


def pad_to_neuron_sequence_length(tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
    """Pad tensor to be a multiple of 2048 for Neuron flash attention."""
    NEURON_SEQ_LENGTH_MULTIPLE = 2048
    seq_len = tensor.size(1)
    
    if seq_len % NEURON_SEQ_LENGTH_MULTIPLE == 0:
        return tensor
    
    pad_length = NEURON_SEQ_LENGTH_MULTIPLE - (seq_len % NEURON_SEQ_LENGTH_MULTIPLE)
    if tensor.dim() == 2:
        padded = torch.nn.functional.pad(tensor, (0, pad_length), value=pad_value)
    elif tensor.dim() == 3:
        padded = torch.nn.functional.pad(tensor, (0, 0, 0, pad_length), value=pad_value)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
    
    return padded


class NeuronGRPOTrainer(NeuronTrainer):
    """
    GRPO Trainer for Neuron/Trainium devices.
    
    Key adaptations for Neuron:
    1. Generator model loads before distributed training
    2. Tensors kept on CPU for tokenizer operations
    3. XLA-compatible gathering for logging
    4. Sequence padding for Neuron flash attention
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
        reward_funcs = None,
        **kwargs,
    ):
        if not is_trl_available(required_version=TRL_VERSION):
            raise RuntimeError(f"Using NeuronGRPOTrainer requires trl=={TRL_VERSION}.")

        # Extract model name
        if isinstance(model, str):
            self.model_name_or_path = model
        else:
            self.model_name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
        
        if not self.model_name_or_path:
            raise ValueError(
                "NeuronGRPOTrainer requires a model ID string. "
                "Pass the model as: model='Qwen/Qwen2-0.5B-Instruct'"
            )

        # Setup args
        args_is_none = args is None
        if args is None:
            args = NeuronGRPOConfig(output_dir="tmp_trainer")
        elif args.__class__.__name__ == "NeuronTrainingArguments":
            args_as_dict = args.to_dict()
            args_as_dict.update({k: getattr(args, k) for k in args_as_dict.keys() if k.endswith("_token")})
            args = NeuronGRPOConfig(**args_as_dict)

        # Set GRPO params early
        self.max_prompt_length = getattr(args, "max_prompt_length", 512)
        self.max_completion_length = getattr(args, "max_completion_length", 128)
        self.num_generations = getattr(args, "num_generations", 1)
        self.temperature = getattr(args, "temperature", 1.0)
        self.top_p = getattr(args, "top_p", 1.0)
        self.top_k = getattr(args, "top_k", None)
        
        if isinstance(model, PreTrainedModel):
            self.model_kwarg_keys = inspect.signature(model.forward).parameters.keys()
        else:
            self.model_kwarg_keys = set()
        
        # Load generator model before distributed setup
        logger.info("[PRE-INIT] Loading generator model before distributed setup...")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self._generator_tokenizer = processing_class or tokenizer or AutoTokenizer.from_pretrained(
            self.model_name_or_path
        )
        
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Keep generator on CPU for simplicity and XLA compatibility
        generator_device = getattr(args, "generator_device", "cpu")
        if generator_device == "cuda" and torch.cuda.is_available():
            self.generator_model = self.generator_model.to("cuda")
            logger.info("[PRE-INIT] Generator on CUDA")
        else:
            self.generator_model = self.generator_model.to("cpu")
            logger.info("[PRE-INIT] Generator on CPU")
        
        self.generator_model.eval()
        
        generation_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "pad_token_id": self._generator_tokenizer.pad_token_id,
            "eos_token_id": self._generator_tokenizer.eos_token_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            generation_kwargs["top_k"] = self.top_k
        self.generation_config = GenerationConfig(**generation_kwargs)
        
        logger.info("[PRE-INIT] Generator model ready")
        
        if args_is_none:
            log_level = args.get_process_log_level()
            logging.set_verbosity(log_level)
            logging.warning(f"No `GRPOConfig` passed, using `output_dir={args.output_dir}`.")

        if data_collator is None:
            data_collator = identity

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class if tokenizer is None else tokenizer,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        )
        
        # Set model_kwarg_keys after model is prepared
        if hasattr(self, 'model') and self.model is not None:
            model_to_check = self.model
            if isinstance(model_to_check, NxDPPModel):
                model_to_check = model_to_check.module if hasattr(model_to_check, 'module') else model_to_check
            elif hasattr(model_to_check, 'get_base_model'):
                model_to_check = model_to_check.get_base_model()
            
            self.model_kwarg_keys = inspect.signature(model_to_check.forward).parameters.keys()

        # GRPO-specific attributes from config (extract all potential attributes from args)
        self.repetition_penalty = getattr(self.args, "repetition_penalty", None)
        self.min_p = getattr(self.args, "min_p", None)
        self.chat_template_kwargs = getattr(self.args, "chat_template_kwargs", {}) or {}
        self.loss_type = getattr(self.args, "loss_type", "grpo")
        self.mask_truncated_completions = getattr(self.args, "mask_truncated_completions", True)
        self.top_entropy_quantile = getattr(self.args, "top_entropy_quantile", 1.0)
        self.beta = getattr(self.args, "beta", 0.0)
        
        # Epsilon parameters
        self.epsilon_low = getattr(self.args, "epsilon", 0.2)
        _eps_high = getattr(self.args, "epsilon_high", None)
        self.epsilon_high = _eps_high if _eps_high is not None else self.epsilon_low
        
        # Additional GRPO config attributes that TRL might expect (always set with defaults)
        common_grpo_attrs = {
            "alpha": 0.0,
            "gamma": 1.0,
            "cliprange": 0.2,
            "cliprange_value": 0.2,
            "vf_coef": 0.1,
            "ref_free": False,
            "importance_sampling_level": "token",
        }
        for attr, default_value in common_grpo_attrs.items():
            setattr(self, attr, getattr(self.args, attr, default_value))
        
        # Reference model setup (before super().__init__ to avoid distribution issues)
        if self.beta != 0.0 and not self.ref_free:
            logger.info("[PRE-INIT] Loading reference model before distributed setup...")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            self.ref_model = self.ref_model.to("cpu")
            self.ref_model.eval()
            logger.info("[PRE-INIT] Reference model ready")
        else:
            self.ref_model = None
        
        self.mode = "train"
        
        # TRL compatibility attributes
        self.processing_class = self._generator_tokenizer
        self.pad_token_id = self._generator_tokenizer.pad_token_id
        self.eos_token_id = self._generator_tokenizer.eos_token_id
        self.model_wrapped = self.model
        
        # Metrics tracking
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        
        # Generation buffer for batching
        self._buffered_inputs = None
        self._buffer_index = 0
        self._generation_step_counter = 0
        
        # Store reward functions
        self.reward_funcs = reward_funcs or []
        
        # vLLM flags (not used in Neuron but kept for TRL compatibility)
        self.use_vllm = False
        self.vllm_importance_sampling_correction = False
        self.use_transformers_paged = False
        
        # TRL's GRPO trainer uses this for generation frequency
        self.num_iterations = self.args.gradient_accumulation_steps
        
    def _generate_single_turn(self, prompts, images=None):
        """
        Override to use CPU generator model for XLA compatibility.
        Returns prompt_ids, completion_ids, logprobs, extra_fields as expected by TRL.
        """
        generator_device = next(self.generator_model.parameters()).device
        
        processor_kwargs = {
            "return_tensors": "pt",
            "padding": "max_length",
            "padding_side": "left",
            "max_length": self.max_prompt_length,
            "truncation": True,
            "add_special_tokens": False,
        }
        
        if is_conversational({"prompt": prompts[0]}):
            generate_inputs = self._generator_tokenizer.apply_chat_template(
                conversation=prompts,
                **processor_kwargs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **self.chat_template_kwargs,
            )
        else:
            generate_inputs = self._generator_tokenizer(text=prompts, **processor_kwargs)
        
        generate_inputs = {
            k: v.to(generator_device) if isinstance(v, torch.Tensor) else v 
            for k, v in generate_inputs.items()
        }
        
        with torch.no_grad():
            prompt_completion_ids = self.generator_model.generate(
                **generate_inputs,
                max_new_tokens=self.max_completion_length,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                do_sample=self.generation_config.do_sample,
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                top_k=self.generation_config.top_k,
            )
        
        # Move to CPU for processing
        prompt_completion_ids = prompt_completion_ids.cpu()
        prompt_ids = generate_inputs["input_ids"].cpu()
        prompt_mask = generate_inputs["attention_mask"].cpu()
        
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # Create completion mask
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), completion_ids.size(1), dtype=torch.long)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        sequence_indices = torch.arange(completion_ids.size(1)).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Convert to Python lists
        prompt_ids_list = []
        completion_ids_list = []
        
        for p, pm, c, cm in zip(prompt_ids, prompt_mask, completion_ids, completion_mask):
            valid_prompt_tokens = p[pm.bool()]
            valid_completion_tokens = c[cm.bool()]
            prompt_ids_list.append(valid_prompt_tokens.tolist())
            completion_ids_list.append(valid_completion_tokens.tolist())
        
        return prompt_ids_list, completion_ids_list, None, {}

    # Inherit TRL's _generate method (calls our _generate_single_turn)
    _generate = _GRPO._generate
    
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """
        Override to create tensors on CPU for XLA compatibility.
        This avoids the device property issue and ensures tokenizer compatibility.
        """
        # Always use CPU for tensor creation to avoid XLA issues with tokenizer
        device = torch.device("cpu")
        mode = "train" if self.model.training else "eval"
        
        prompts = [x["prompt"] for x in inputs]
        
        # Generate completions (returns lists)
        prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields = (
            self._generate(prompts, None)
        )
        
        # Convert lists to tensors on CPU (not XLA device)
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None
        
        # Mask truncated completions if needed
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
        
        # Concatenate for attention mask
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        
        # Compute old_per_token_logps if needed
        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                # Policy model logps will be on XLA; moved to XLA in get_batch_samples before loss
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,  # CPU tensors, moved to XLA in _get_per_token_logps_and_entropies
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                old_per_token_logps = None
        
        # Compute ref_per_token_logps if reference model exists
        ref_per_token_logps = None
        if self.beta != 0.0 and self.ref_model is not None:
            with torch.no_grad():
                # Ref model logps stay on CPU; moved to XLA in get_batch_samples before loss
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
        
        # Calculate rewards on CPU
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        prompts_decoded = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        rewards = self._calculate_rewards(inputs, prompts_decoded, completions, completion_ids_list)
        
        # Compute advantages (group-relative normalization)
        # Group rewards by num_generations and normalize
        rewards_reshaped = rewards.view(-1, self.num_generations)  # (num_prompts, num_generations)
        mean_grouped_rewards = rewards_reshaped.mean(dim=1)  # (num_prompts,)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)  # (total_samples,)
        advantages = rewards - mean_grouped_rewards
        
        # Build output dict (policy model logps on XLA, ref model logps on CPU; both moved to XLA in get_batch_samples)
        output = {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "prompt_mask": prompt_mask,
            "completion_mask": completion_mask,
            "prompt_completion_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "logits_to_keep": logits_to_keep,
            "rewards": rewards,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "sampling_per_token_logps": sampling_per_token_logps,
            "num_items_in_batch": num_items_in_batch,
            **extra_fields,
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        
        # Log output structure
        logger.info(f"[_generate_and_score_completions] Output keys: {list(output.keys())}")
        if "prompt_ids" in output:
            logger.info(f"[_generate_and_score_completions] Input tensors on device: {output['prompt_ids'].device}")
        if "old_per_token_logps" in output and output["old_per_token_logps"] is not None:
            logger.info(f"[_generate_and_score_completions] old_per_token_logps on device: {output['old_per_token_logps'].device}")
        if "ref_per_token_logps" in output and output["ref_per_token_logps"] is not None:
            logger.info(f"[_generate_and_score_completions] ref_per_token_logps on device: {output['ref_per_token_logps'].device}")
        
        return output
    
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """Calculate rewards using reward functions."""
        if not self.reward_funcs:
            # Default to zero rewards if no reward functions provided
            return torch.zeros(len(completions), device="cpu")
        
        rewards = []
        for reward_func in self.reward_funcs:
            func_rewards = reward_func(prompts, completions)
            rewards.append(torch.tensor(func_rewards, device="cpu"))
        
        # Average rewards from multiple functions
        if len(rewards) > 1:
            rewards = torch.stack(rewards).mean(dim=0)
        else:
            rewards = rewards[0]

        logger.info(f"[_calculate_rewards] Prompts shape: {len(prompts)}")
        logger.info(f"[_calculate_rewards] Completions shape: {len(completions)}")
        logger.info(f"[_calculate_rewards] Rewards shape: {len(rewards)}")
        return rewards

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        token_type_ids=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Override to handle CPU->XLA tensor movement and padding for Neuron.
        Handles both ref_model (CPU) and policy_model (XLA) cases.
        """
        # Filter out image-related parameters
        image_keys = ["pixel_values", "image_grid_thw", "num_images", "pixel_attention_mask", "image_sizes"]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in image_keys}
        
        # Detect model device to determine tensor placement
        model_device = next(model.parameters()).device
        
        # If model is on CPU (ref_model), keep everything on CPU
        if model_device.type == "cpu":
            # Use tensors as-is on CPU
            input_ids_cpu = input_ids
            attention_mask_cpu = attention_mask
            cpu_kwargs = {}
            if token_type_ids is not None:
                cpu_kwargs["token_type_ids"] = token_type_ids
            
            cpu_kwargs.update(filtered_kwargs)
            
            # Call TRL's implementation with CPU tensors
            logps, entropies = _GRPO._get_per_token_logps_and_entropies(
                self,
                model,
                input_ids_cpu,
                attention_mask_cpu,
                logits_to_keep,
                batch_size=batch_size,
                compute_entropy=compute_entropy,
                **cpu_kwargs,
            )
            
            # Return CPU tensors
            return logps, entropies
        
        # Otherwise, model is on XLA (policy_model), move tensors to XLA
        xla_device = xm.xla_device()
        
        if input_ids.device.type != "xla":
            input_ids = input_ids.to(xla_device)
        if attention_mask.device.type != "xla":
            attention_mask = attention_mask.to(xla_device)
        
        # Pad to Neuron's flash attention requirement
        input_ids_padded = pad_to_neuron_sequence_length(input_ids, self.pad_token_id)
        attention_mask_padded = pad_to_neuron_sequence_length(attention_mask, 0)
        
        # Move other tensors to XLA if present
        xla_kwargs = {}
        if token_type_ids is not None:
            xla_kwargs["token_type_ids"] = pad_to_neuron_sequence_length(
                token_type_ids.to(xla_device) if token_type_ids.device.type != "xla" else token_type_ids, 0
            )
        
        xla_kwargs.update(filtered_kwargs)
        
        # Call TRL's implementation with XLA tensors
        logps, entropies = _GRPO._get_per_token_logps_and_entropies(
            self,
            model,
            input_ids_padded,
            attention_mask_padded,
            logits_to_keep,
            batch_size=batch_size,
            compute_entropy=compute_entropy,
            **xla_kwargs,
        )
        
        # Return only the non-padded part, keep on XLA for computation graph
        return logps[:, :logits_to_keep], entropies[:, :logits_to_keep] if entropies is not None else None

    _compute_loss = _GRPO._compute_loss

    def _set_signature_columns_if_needed(self):
        """Override to set GRPO-specific signature columns."""
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def get_batch_samples(
        self,
        epoch_iterator: Iterator,
        num_batches: int,
        device: torch.device | None = None,
        prefetch_size: int | None = None,
    ) -> tuple[list[dict[str, Any]], int | torch.Tensor | None]:
        """Override to handle GRPO generation and buffering logic."""
        generate_every = self.args.steps_per_generation * self.num_iterations
        if self._generation_step_counter % generate_every == 0 or self._buffered_inputs is None:
            logger.info(f"[get_batch_samples] Generating at step {self._generation_step_counter}")
            
            total_batches_needed = self.args.steps_per_generation
            raw_batches = []
            
            for _ in range(total_batches_needed):
                try:
                    batch = next(epoch_iterator)
                    raw_batches.append(batch)
                except StopIteration:
                    break
            
            if not raw_batches:
                return [], None
            
            raw_samples = []
            for batch in raw_batches:
                if isinstance(batch, list):
                    raw_samples.extend(batch)
                else:
                    raw_samples.append(batch)
            
            # Generate and score (returns CPU tensors)
            generation_batch = self._generate_and_score_completions(raw_samples)
            
            logger.info(f"[get_batch_samples] Generated batch with tensors on: {generation_batch.get('prompt_ids', torch.tensor(0)).device}")
            
            # Separate sequence-like values (tensors, lists) from scalar values (int, float, None)
            sequence_dict = {}
            scalar_dict = {}
            for key, val in generation_batch.items():
                if val is None:
                    scalar_dict[key] = val
                elif isinstance(val, torch.Tensor) and val.ndim >= 1:
                    sequence_dict[key] = val
                elif isinstance(val, (list, tuple)):
                    sequence_dict[key] = val
                else:
                    scalar_dict[key] = val
            
            # Shuffle only sequence-like values
            shuffled_sequences = shuffle_sequence_dict(sequence_dict)
            
            # Split sequence-like values
            generation_batches = split_tensor_dict(shuffled_sequences, self.args.steps_per_generation)
            
            # Add scalar values to each batch (scalars are shared across all batches)
            for batch in generation_batches:
                batch.update(scalar_dict)
            self._buffered_inputs = generation_batches
            self._buffer_index = 0
        
        current_batch = self._buffered_inputs[self._buffer_index]
        self._buffer_index += 1
        self._generation_step_counter += 1
        
        if self._buffer_index >= len(self._buffered_inputs):
            self._buffer_index = 0
        
        # Move to XLA device now for training
        if device is not None and device.type == "xla":
            logger.info(f"[get_batch_samples] Moving batch to XLA device")
            current_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in current_batch.items()
            }
        
        return [current_batch], None

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | torch.Tensor | None = None,
    ):
        """Compute GRPO loss."""
        if return_outputs:
            raise ValueError("return_outputs=True is not supported for GRPO")
        
        loss = self._compute_loss(model, inputs) 
        return loss

    def train_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training step that works with GRPO's prepared inputs."""
        manager = self.autocast_smart_context_manager()

        # Always compute GRPO loss using TRL's compute_loss
        with manager:
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        
        # Handle backward pass based on model type
        if isinstance(model, NxDPPModel):
            # For pipeline parallelism, handle backward pass through pipeline
            # The loss is already computed, so we can use accelerator.backward
            self.accelerator.backward(loss)
            
            # Zero out loss on non-last pipeline stages (as before)
            if self.pp_rank != self.pp_size - 1:
                dtype = torch.bfloat16 if self.args.bf16 else torch.float32
                loss = torch.tensor(0, dtype=dtype).to(xm.xla_device())
        else:
            # Standard backward pass for non-pipeline models
            self.accelerator.backward(loss)
        
        return loss

    def train(self, resume_from_checkpoint: str | bool | None = None):
        """Use NeuronTrainer's training loop."""
        return super().train(resume_from_checkpoint=resume_from_checkpoint)
