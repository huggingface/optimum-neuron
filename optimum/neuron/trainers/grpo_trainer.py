from typing import Any
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


if is_trl_available():
    from trl import GRPOConfig, GRPOTrainer
    from trl.data_utils import is_conversational
    from trl.trainer.utils import (
        split_tensor_dict,
        shuffle_sequence_dict,
        selective_log_softmax,
        entropy_from_logits,
    )
    _GRPO = GRPOTrainer
else:
    class GRPOTrainer:
        pass
    class GRPOConfig:
        pass
    _GRPO = None


def identity(x):
    # Identity collator for GRPO
    return x


def pad_to_neuron_sequence_length(tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
    # Pad tensor to be a multiple of 2048 for Neuron flash attention
    NEURON_SEQ_LENGTH_MULTIPLE = 2048
    seq_len = tensor.size(1)
    
    if seq_len % NEURON_SEQ_LENGTH_MULTIPLE == 0:
        return tensor
    
    pad_length = NEURON_SEQ_LENGTH_MULTIPLE - (seq_len % NEURON_SEQ_LENGTH_MULTIPLE)
    # Pad along sequence dimension (dim 1)
    if tensor.dim() == 2:  # (batch, seq_len)
        padded = torch.nn.functional.pad(tensor, (0, pad_length), value=pad_value)
    elif tensor.dim() == 3:  # (batch, seq_len, features)
        padded = torch.nn.functional.pad(tensor, (0, 0, 0, pad_length), value=pad_value)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
    
    return padded


class NeuronGRPOTrainer(NeuronTrainer):
    """
    GRPO Trainer for Neuron/Trainium devices.
    
    Critical constraints for Neuron:
    1. Generator model must load before distributed training fork
    2. All tensors must have fixed shapes
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

        # set up critical section before parent init to avoid fork issues

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

        # Set GRPO params early (needed for generator init)
        self.max_prompt_length = getattr(args, "max_prompt_length", 512)
        self.max_completion_length = getattr(args, "max_completion_length", 128)
        self.num_generations = getattr(args, "num_generations", 1)
        self.temperature = getattr(args, "temperature", 1.0)
        self.top_p = getattr(args, "top_p", 1.0)
        self.top_k = getattr(args, "top_k", None)
        
        # Some models don't support `logits_to_keep` argument - check before wrapping
        if isinstance(model, PreTrainedModel):
            self.model_kwarg_keys = inspect.signature(model.forward).parameters.keys()
        else:
            self.model_kwarg_keys = set()  # Will be set after model is prepared
        
        # init generator before parent init
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
        
        generator_device = getattr(args, "generator_device", "cpu")
        if generator_device == "cuda" and torch.cuda.is_available():
            self.generator_model = self.generator_model.to("cuda")
            logger.info("[PRE-INIT] Generator on CUDA")
        else:
            self.generator_model = self.generator_model.to("cpu")
            logger.info("[PRE-INIT] Generator on CPU")
        
        self.generator_model.eval()
        
        # Setup generation config
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
        
        # critical section: parent init (will fork for distributed training)
        # CRITICAL SECTION 2: Parent init ()
        
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
        
        # post-init setup
        
        # Set model_kwarg_keys after model is prepared
        if hasattr(self, 'model') and self.model is not None:
            # For distributed models, check the base model
            model_to_check = self.model
            if isinstance(model_to_check, NxDPPModel):
                # For NxDPPModel, we need to get the actual wrapped model
                model_to_check = model_to_check.module if hasattr(model_to_check, 'module') else model_to_check
            elif hasattr(model_to_check, 'get_base_model'):
                model_to_check = model_to_check.get_base_model()
            
            self.model_kwarg_keys = inspect.signature(model_to_check.forward).parameters.keys()

        # GRPO-specific attributes
        self.repetition_penalty = getattr(self.args, "repetition_penalty", None)
        self.min_p = getattr(self.args, "min_p", None)
        self.chat_template_kwargs = getattr(self.args, "chat_template_kwargs", {}) or {}
        
        self.loss_type = getattr(self.args, "loss_type", "grpo")
        self.scale_rewards = getattr(self.args, "scale_rewards", "group")
        self.importance_sampling_level = getattr(self.args, "importance_sampling_level", "token")
        self.mask_truncated_completions = getattr(self.args, "mask_truncated_completions", False)
        self.beta = getattr(self.args, "beta", 0.0)
        self.epsilon_low = getattr(self.args, "epsilon", 0.1)
        self.epsilon_high = getattr(self.args, "epsilon_high", self.epsilon_low)
        self.use_vllm = getattr(self.args, "use_vllm", False)
        self.use_liger_loss = getattr(self.args, "use_liger_loss", False)
        self.top_entropy_quantile = getattr(self.args, "top_entropy_quantile", 1.0)
        
        self._step = 0
        self._buffered_inputs = None
        self._generation_step_counter = 0
        self._buffer_index = 0
        
        self.num_iterations = getattr(self.args, "num_iterations", 1)
        self.shuffle_dataset = getattr(self.args, "shuffle_dataset", True)
        
        # Calculate how many steps we generate for at once
        if not hasattr(self.args, "steps_per_generation"):
            self.args.steps_per_generation = self.args.gradient_accumulation_steps
        
        if not hasattr(self.args, "generation_batch_size") or self.args.generation_batch_size is None:
            self.args.generation_batch_size = self.args.per_device_train_batch_size * self.args.steps_per_generation

        # Reward functions
        if reward_funcs is not None:
            self.reward_funcs = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
            self.reward_func_names = []
            for rf in self.reward_funcs:
                if hasattr(rf, '__name__'):
                    self.reward_func_names.append(rf.__name__)
                else:
                    self.reward_func_names.append(str(rf))
        else:
            raise ValueError("reward_funcs must be provided for GRPO training")
        
        # Reward weights
        if hasattr(self.args, 'reward_weights') and self.args.reward_weights is not None:
            self.reward_weights = torch.tensor(self.args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(self.reward_funcs), dtype=torch.float32)
        
        self.reward_processing_classes = [None] * len(self.reward_funcs)
        
        # Initialize metrics tracking
        from collections import defaultdict, deque
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._logs = {
            "images": deque(maxlen=args.per_device_train_batch_size),
            "prompt": deque(maxlen=args.per_device_train_batch_size),
            "completion": deque(maxlen=args.per_device_train_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.per_device_train_batch_size)),
            "advantages": deque(maxlen=args.per_device_train_batch_size),
        }
        
        self.pad_token = self.processing_class.pad_token
        self.pad_token_id = self.processing_class.pad_token_id
        self.eos_token_id = self.processing_class.eos_token_id

    def _generate_single_turn(self, prompts: list, images=None):
        # generate completions with fixed tensor shapes
        # must return list of same length token lists (across all batches) for neuron compilation
        generator_device = next(self.generator_model.parameters()).device
        
        # generation inputs
        processor_kwargs = {
            "return_tensors": "pt",
            "padding": "max_length",  # pad to fixed length
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
        
        # Move to generator device
        generate_inputs = {
            k: v.to(generator_device) if isinstance(v, torch.Tensor) else v 
            for k, v in generate_inputs.items()
        }
        
        # Generate with fixed output length
        with torch.no_grad():
            prompt_completion_ids = self.generator_model.generate(
                **generate_inputs,
                max_length=self.max_prompt_length + self.max_completion_length,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                do_sample=self.generation_config.do_sample,
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                top_k=self.generation_config.top_k,
            )
        
        # Move to CPU
        prompt_completion_ids = prompt_completion_ids.cpu()
        prompt_ids = generate_inputs["input_ids"].cpu()
        prompt_mask = generate_inputs["attention_mask"].cpu()
        
        # Extract fixed-length completions
        # Prompt is already max_prompt_length (padded above)
        completion_ids = prompt_completion_ids[:, self.max_prompt_length:]

        # # debug
        # prompt_lengths = [len(p) for p in prompt_ids]
        # completion_lengths = [len(c) for c in completion_ids]
        # assert len(set(prompt_lengths)) == 2, f"Variable prompt lengths: {prompt_lengths}"
        # assert len(set(completion_lengths)) == 1, f"Variable completion lengths: {completion_lengths}"
        # assert prompt_lengths[0] == self.max_prompt_length, f"Wrong prompt length: {prompt_lengths[0]}"
        # assert completion_lengths[0] == self.max_completion_length, f"Wrong completion length: {completion_lengths[0]}"
        
        # Ensure exactly max_completion_length
        if completion_ids.size(1) < self.max_completion_length:
            pad_length = self.max_completion_length - completion_ids.size(1)
            completion_ids = torch.nn.functional.pad(
                completion_ids, (0, pad_length), value=self.pad_token_id
            )
        else:
            completion_ids = completion_ids[:, :self.max_completion_length]
        
        # Create completion masks (1 until EOS, 0 after)
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), self.max_completion_length, dtype=torch.long)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        sequence_indices = torch.arange(self.max_completion_length).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # This ensures TRL's pad() function becomes a no-op
        prompt_ids_list = []
        completion_ids_list = []
        
        for p, pm, c, cm in zip(prompt_ids, prompt_mask, completion_ids, completion_mask):
            # Extract only non-padding tokens from prompts
            valid_prompt = p[pm.bool()].tolist()
            # Pad prompt back to max_prompt_length for consistency
            valid_prompt += [self.pad_token_id] * (self.max_prompt_length - len(valid_prompt))
            prompt_ids_list.append(valid_prompt)
            
            # Extract only valid tokens from completions  
            valid_completion = c[cm.bool()].tolist()
            # Pad completion back to max_completion_length for consistency
            valid_completion += [self.pad_token_id] * (self.max_completion_length - len(valid_completion))
            completion_ids_list.append(valid_completion)
        
        return prompt_ids_list, completion_ids_list, None, {}

    # Use GRPO methods that don't need modification
    _generate = _GRPO._generate
    _generate_and_score_completions = _GRPO._generate_and_score_completions
    _compute_loss = _GRPO._compute_loss
    
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        # Override reward calculation to compute and gather rewards for Neuron/XLA
        # Computes rewards on CPU -> move to XLA before gathering -> return tensor on XLA

        # Compute rewards on CPU to avoid triggering XLA compilation
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device="cpu", dtype=torch.float32)
        
        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        
        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state
        
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            if isinstance(reward_func, torch.nn.Module):
                # Module-based reward functions: TODO - support on accelerator device in future
                raise NotImplementedError(
                    "nn.Module reward functions not yet supported on Neuron. "
                    "Use functional reward functions that compute on CPU."
                )
            else:
                # Functional reward functions: compute on CPU
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                )
                # Convert None values to NaN
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device="cpu")
        
        # Validate rewards
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )
        
        # Move to XLA device before any collective operations
        xla_device = self.accelerator.device
        rewards_per_func = rewards_per_func.to(xla_device)
        
        # Gather across all data parallel processes using XLA's all_gather
        if self.dp_size > 1:
            # xm.all_gather signature: all_gather(value, dim=0, groups=None)
            # It concatenates tensors along the specified dimension from all processes
            rewards_per_func = xm.all_gather(
                rewards_per_func,
                dim=0,  # Concatenate along batch dimension
                groups=get_data_parallel_replica_groups(),
            )
        
        return rewards_per_func

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Override to pad inputs to Neuron's flash attention 2048 multiple requirement
        original_seq_len = input_ids.size(1)
        logger.info(f"[_get_per_token_logps] Original input_ids shape: {input_ids.shape}")
        input_ids_padded = pad_to_neuron_sequence_length(input_ids, self.pad_token_id)
        attention_mask_padded = pad_to_neuron_sequence_length(attention_mask, 0)
        logger.info(f"[_get_per_token_logps] Padded input_ids shape: {input_ids_padded.shape}")
        
        # Call parent method with padded inputs
        logps, entropies = _GRPO._get_per_token_logps_and_entropies(
            self,
            model,
            input_ids_padded,
            attention_mask_padded,
            logits_to_keep,
            batch_size=batch_size,
            compute_entropy=compute_entropy,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            num_images=num_images,
            pixel_attention_mask=pixel_attention_mask,
            image_sizes=image_sizes,
            token_type_ids=token_type_ids,
        )
        
        return logps, entropies

    def _set_signature_columns_if_needed(self):
        """Override to set GRPO-specific signature columns."""
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images"]

    def get_batch_samples(
        self,
        epoch_iterator,
        num_batches: int,
        device: torch.device | None = None,
        prefetch_size: int | None = None,
    ) -> tuple[list[dict[str, Any]], int | torch.Tensor | None]:
        # override to hadnel GRPO generation and buffering logic
        # Check if we need to generate new completions
        generate_every = self.args.steps_per_generation * self.num_iterations
        if self._generation_step_counter % generate_every == 0 or self._buffered_inputs is None:
            logger.info(f"Generating completions at generation step {self._generation_step_counter}")
            
            # Collect prompts for the full generation batch
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
            
            # Flatten batches into list of samples
            raw_samples = []
            for batch in raw_batches:
                if isinstance(batch, list):
                    raw_samples.extend(batch)
                else:
                    raw_samples.append(batch)
            
            # Generate completions (returns fixed-shape tensors)
            generation_batch = self._generate_and_score_completions(raw_samples)
            
            # Verify shapes are fixed
            if "prompt_ids" in generation_batch:
                assert generation_batch["prompt_ids"].size(1) == self.max_prompt_length, \
                    f"Prompt shape: {generation_batch['prompt_ids'].size(1)} != {self.max_prompt_length}"
            if "completion_ids" in generation_batch:
                assert generation_batch["completion_ids"].size(1) == self.max_completion_length, \
                    f"Completion shape: {generation_batch['completion_ids'].size(1)} != {self.max_completion_length}"
            
            # Shuffle and split for multiple optimizer steps
            generation_batch = shuffle_sequence_dict(generation_batch)
            generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
            self._buffered_inputs = generation_batches
            
            # Reset buffer index
            self._buffer_index = 0
        
        # Get the batch for this step
        current_batch = self._buffered_inputs[self._buffer_index]
        self._buffer_index += 1
        self._generation_step_counter += 1
        
        # If we've used all buffered batches, reset for next generation
        if self._buffer_index >= len(self._buffered_inputs):
            self._buffer_index = 0
        
        logger.info(f"[get_batch_samples] batch keys: {list(current_batch.keys())}")
        if "prompt_ids" in current_batch:
            logger.info(f"[get_batch_samples] prompt_ids shape: {current_batch['prompt_ids'].shape}")
        if "completion_ids" in current_batch:
            logger.info(f"[get_batch_samples] completion_ids shape: {current_batch['completion_ids'].shape}")
        
        # Move to XLA device
        if device is not None and device.type == "xla":
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

        logger.info(f"[train_step] model type: {type(model).__name__}")
        logger.info(f"[train_step] inputs keys: {list(inputs.keys())}")
        if "prompt_ids" in inputs:
            logger.info(f"[train_step] prompt_ids shape: {inputs['prompt_ids'].shape}")
        if "completion_ids" in inputs:
            logger.info(f"[train_step] completion_ids shape: {inputs['completion_ids'].shape}")
        if "input_ids" in inputs:
            logger.info(f"[train_step] input_ids shape: {inputs['input_ids'].shape}")

        if isinstance(model, NxDPPModel):
            # Transform GRPO inputs to standard model inputs for pipeline parallelism
            if "prompt_ids" in inputs and "completion_ids" in inputs:
                # Concatenate prompt and completion for model
                input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
                attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
                seq_len = input_ids.size(1)
                logger.info(f"[train_step NxDPPModel] Concatenated input_ids shape: {input_ids.shape}")
                
                # Pad to meet Neuron's flash attention requirement (multiples of 2048)
                pad_token_id = self.pad_token_id
                input_ids = pad_to_neuron_sequence_length(input_ids, pad_token_id)
                attention_mask = pad_to_neuron_sequence_length(attention_mask, 0)
                padded_seq_len = input_ids.size(1)
                logger.info(f"[train_step NxDPPModel] Padded input_ids shape: {input_ids.shape}")
                
                # Create labels: -100 for prompts (ignore), actual tokens for completions
                batch_size = input_ids.size(0)
                labels = torch.full_like(input_ids, -100)
                labels[:, :seq_len][:, self.max_prompt_length:] = inputs["completion_ids"]
                logger.info(f"[train_step NxDPPModel] labels shape: {labels.shape}")
                
                # Build model inputs dict
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                
                # Add any other standard model inputs if present
                for key in ["pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes", "token_type_ids"]:
                    if key in inputs:
                        model_inputs[key] = inputs[key]
                
                logger.info(f"[train_step NxDPPModel] Calling run_train with keys: {list(model_inputs.keys())}")
                
                # Pass model inputs to run_train
                with manager:
                    loss = model.run_train(**model_inputs)
            else:
                # Already in standard format
                with manager:
                    loss = model.run_train(**inputs)
            
            if self.pp_rank != self.pp_size - 1:
                dtype = torch.bfloat16 if self.args.bf16 else torch.float32
                loss = torch.tensor(0, dtype=dtype).to(xm.xla_device())
        else:
            # For non-pipeline models, _compute_loss handles concatenation and padding via _get_per_token_logps_and_entropies
            with manager:
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            
            self.accelerator.backward(loss)
        
        return loss

    def train(self, resume_from_checkpoint: str | bool | None = None):
        """Use NeuronTrainer's training loop."""
        return super().train(resume_from_checkpoint=resume_from_checkpoint)