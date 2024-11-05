import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from exporters.model_configs import get_exporter_config_constructor
from exporters.model_wrappers import (  # noqa: E402
    CONTEXT_ENCODING_MODEL_TAG,  # noqa: E402
    TOKEN_GENERATION_MODEL_TAG,  # noqa: E402
    DecoderModelWrapper,  # noqa: E402
)
from modules.checkpoint import load_state_dict
from modules.config import NeuronInferenceConfig
from modules.sampling import Sampler  # noqa: E402
from neuronx_distributed.trace.model_builder import ModelBuilder
from safetensors.torch import load_file
from transformers import AutoConfig, PretrainedConfig
from transformers.generation import (
    GenerationConfig,
    GenerationMixin,
    SampleDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput


SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]


@dataclass
class CheckPointLoader:
    """Loads model checkpoints

    A Pickable class to be used by compilation processes spawned
    by the ModelBuilder to load checkpoints.

    """

    model_path: Union[str, Path]
    model_prefix: str
    dtype: torch.dtype

    def load_checkpoint(self):
        # this function loads the model's state dictionary and weights from
        # the hf model, removes the prefix and casts to the correct dtype
        model_sd = load_state_dict(self.model_path)
        param_name_list = list(model_sd.keys())
        for param_name in param_name_list:
            if param_name.startswith(self.model_prefix):
                updated_param_name = param_name.replace(self.model_prefix, "", 1)
                model_sd[updated_param_name] = model_sd[param_name].to(self.dtype)
                del model_sd[param_name]
            else:
                model_sd[param_name] = model_sd[param_name].to(self.dtype)
        return model_sd


class NeuronModelForCausalLM(GenerationMixin):

    # Required by GenerationMixin, but present in PreTrainedModel
    main_input_name = "input_ids"
    _supports_cache_class = False
    # _supports_static_cache = False

    def __init__(self, config: PretrainedConfig, model: torch.jit.ScriptModule):
        super().__init__()

        self.config = config
        self.generation_config = GenerationConfig.from_model_config(config)
        self.vocab_size = config.vocab_size
        self.padding_side = config.padding_side
        self.kv_cache_populated = False

        self.sampler = None

        self.context_encoding_model = DecoderModelWrapper(model)
        self.token_generation_model = DecoderModelWrapper(model)

    def can_generate(self):
        # Not needed after transformers 4.50
        return True

    @classmethod
    def from_pretrained(cls, model_path: str, config: PretrainedConfig):
        return cls(model_path, config)

    @classmethod
    def export(cls, model_path: Union[str, Path], neuron_config: NeuronInferenceConfig, serialize_base_path=None):

        config = AutoConfig.from_pretrained(model_path)
        export_config_cls = get_exporter_config_constructor(config.model_type)

        if not os.path.exists(serialize_base_path):
            os.makedirs(serialize_base_path)

        neuron_config.attn_cls = export_config_cls._ATTN_CLS
        neuron_config.save_pretrained(serialize_base_path)
        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")

        checkpoint_loader = CheckPointLoader(
            model_path, export_config_cls._STATE_DICT_MODEL_PREFIX, neuron_config.torch_dtype
        )

        builder = ModelBuilder(
            router=None,
            tp_degree=neuron_config.tp_degree,
            checkpoint_loader=checkpoint_loader.load_checkpoint,
            compiler_workdir=base_compile_work_dir,
        )

        # The builder will create a single NxDModel that houses multiple SPMDBucketModel
        # sharing the same weights. Typically, one SPMDBucketModel is created for each
        # input shape that the NxDModel accepts.
        # The SPMDBucketModel used for inference will be selected dynamically at runtime
        # based on the inputs.
        # For LLM models, we typically use different sets of SPMDBucketModel for encoding and
        # token generation, each with its own list of buckets.
        export_configs = {
            CONTEXT_ENCODING_MODEL_TAG: export_config_cls(neuron_config, is_prefill=True),
            TOKEN_GENERATION_MODEL_TAG: export_config_cls(neuron_config, is_prefill=False),
        }
        for tag, export_config in export_configs.items():
            # We need a pickable object to provide the callbacks required by the Builder
            builder.add(
                key=tag,
                model_instance=export_config.get_model_instance(),
                example_inputs=export_config.input_generator(),
                bucket_config=export_config.bucket_config(),
                compiler_args=export_config.get_compiler_args(),
                priority_model_idx=None,
            )

        traced_model = builder.trace(initialize_model_weights=False)
        torch.jit.save(traced_model, NeuronModelForCausalLM.get_traced_model_path(serialize_base_path))
        del traced_model

        builder.shard_checkpoint(serialize_path=os.path.join(serialize_base_path, "weights/"))

    @staticmethod
    def get_traced_model_path(base_path: Union[str, Path]):
        return os.path.join(base_path, "model.pt")

    @classmethod
    def load(cls, serialize_base_path):

        config = NeuronInferenceConfig.from_pretrained(serialize_base_path)

        traced_model = torch.jit.load(NeuronModelForCausalLM.get_traced_model_path(serialize_base_path))

        SHARD_PREFIX = "tp"
        SHARD_SUFFIX = "_sharded_checkpoint.safetensors"
        sharded_checkpoints_root = os.path.join(serialize_base_path, "weights")
        sharded_checkpoints = glob.glob(SHARD_PREFIX + "*" + SHARD_SUFFIX, root_dir=sharded_checkpoints_root)

        def get_rank(shard):
            return int(shard[len(SHARD_PREFIX) : -len(SHARD_SUFFIX)])

        sharded_checkpoints = sorted(sharded_checkpoints, key=get_rank)
        weights = [load_file(os.path.join(sharded_checkpoints_root, ckpt_path)) for ckpt_path in sharded_checkpoints]

        traced_model.nxd_model.initialize(weights)

        return cls(config, traced_model)

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # We dont want HF to move parameters to device
        return torch.device("cpu")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        output_attentions, output_hidden_states, return_dict = self._setup_func_config(
            output_attentions, output_hidden_states, return_dict
        )

        # infer attention_mask from position_ids if not provided
        if attention_mask is None:
            attention_mask = self._infer_attention_mask(position_ids)

        self._log_input(input_ids, attention_mask, position_ids, seq_ids)

        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])

        logits_or_next_tokens = self._get_model_outputs(input_ids, attention_mask, position_ids, seq_ids)

        logging.debug("---output---")
        logging.debug(f"{'tokens' if self.config.on_device_sampling else 'logits'} = %s, ", logits_or_next_tokens)

        return self._construct_output(logits_or_next_tokens)

    def _setup_func_config(self, output_attentions, output_hidden_states, return_dict):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return output_attentions, output_hidden_states, return_dict

    def _infer_attention_mask(self, position_ids):
        assert position_ids is not None, "need to call forward with position_ids if attention_mask is not provided"
        batch_size, seq_len = position_ids.shape
        if position_ids.shape[-1] == 1:
            seq_len = self.config.n_positions
            position_ids_to_compare = position_ids.expand(batch_size, seq_len) - 1
        else:
            seq_len = position_ids.shape[-1]
            position_ids_to_compare = position_ids
        mask = torch.arange(seq_len).view(1, -1).expand(batch_size, seq_len)
        attention_mask = (position_ids_to_compare >= mask).to(dtype=position_ids.dtype)
        return attention_mask

    def _log_input(self, input_ids, attention_mask, position_ids, seq_ids):
        logging.debug("---input---")
        logging.debug("input_ids shape = %s type=%s", input_ids.shape, input_ids.type())
        logging.debug("attention_mask shape = %s type=%s", attention_mask.shape, attention_mask.type())
        logging.debug("position_ids shape = %s type=%s", position_ids.shape, position_ids.type())
        logging.debug("input_ids =%s", input_ids)
        logging.debug("attention_mask =%s", attention_mask)
        logging.debug("position_ids =%s", position_ids)
        logging.debug(f"seq_ids: {seq_ids}")

    def pad_to_batch_size(self, tensor):
            if tensor is None or tensor.shape[0] == self.config.batch_size:
                return tensor

            padded_shape = list(tensor.shape)
            padded_shape[0] = self.config.batch_size
            padded_tensor = torch.zeros(padded_shape, dtype=tensor.dtype)
            padded_tensor[: tensor.shape[0]] = tensor
            return padded_tensor

    def pad_to_max_compiled_seq(self, *args):
        if not self.kv_cache_populated:
            to_pad = args[:3]
            pad_lengths = [self.config.max_context_length - arg.shape[1] for arg in to_pad]
            tensor_pad_vals = [self.config.pad_token_id, 0, 1]
            padded_args = [
                torch.nn.functional.pad(arg, (0, pad_len), "constant", pad_val)
                for arg, pad_val, pad_len in zip(to_pad, tensor_pad_vals, pad_lengths)
            ]
            args = (*padded_args, *args[3:])
        else:
            input_ids, attention_mask, *rest_of_args = args
            pad_len = self.config.max_length - attention_mask.shape[1]
            padded_attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), "constant", 0)
            args = (input_ids, padded_attention_mask, *rest_of_args)

        return args

    def _get_model_outputs(self, input_ids, attention_mask, position_ids, seq_ids):
        # TODO: handle continuous batching here
        assert torch.equal(seq_ids, torch.tensor(range(self.config.max_batch_size)))
        input_ids, attention_mask, position_ids, seq_ids = self.pad_to_max_compiled_seq(input_ids, attention_mask, position_ids, seq_ids)
        input_ids = self.pad_to_batch_size(input_ids)
        attention_mask = self.pad_to_batch_size(attention_mask)
        position_ids = self.pad_to_batch_size(position_ids)
        if not self.kv_cache_populated:
            outputs = self.context_encoding_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
            )
            self.kv_cache_populated = True
        else:
            outputs = self.token_generation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
            )

        return outputs

    def _construct_output(self, logits_or_next_tokens):
        next_tokens = logits_or_next_tokens

        OutputParams = CausalLMOutputWithPast(
            logits=None if self.config.on_device_sampling else logits_or_next_tokens,
            hidden_states=logits_or_next_tokens,
            attentions=None,
        )

        OutputParams.tokens = next_tokens

        return OutputParams

    # We override this function because we want to change the way attention_mask
    # is updated each iteration.
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_for_token_generation: Optional[bool] = False,
        is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:

        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if is_for_token_generation:
                if self.padding_side == "left":
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                    )
                    attention_mask = attention_mask[:, 1:]
                else:
                    attention_mask = torch.cat(
                        [attention_mask.new_ones((attention_mask.shape[0], 1)), attention_mask], dim=-1
                    )
            model_kwargs["attention_mask"] = attention_mask
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if self.kv_cache_populated:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if self.kv_cache_populated:
                position_ids = torch.amax(position_ids, 1, keepdim=True)
                position_ids = position_ids + 1

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", False),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def reset(self):
        # We need to reset the KV cache flag for a new batch of inference.
        # When the flag is reset, the subsequent run will invoke the
        # context encoding model.
        self.kv_cache_populated = False

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        We override the GenerationMixin sample function (_sample for transformers>=4.39.0) to add support for right side padding.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        eos_token_id = 2
        pad_token_id = eos_token_id
        # pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        # eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False
        # auto-regressive generation
        while not this_peer_finished:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            is_for_token_generation = self.kv_cache_populated

            # forward pass to get next token
            outputs = self.forward(**model_inputs, return_dict=True)

            if not self.config.on_device_sampling:
                next_token_logits = outputs.logits[:, -1, :]

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)
                next_token_scores = logits_warper(input_ids, next_token_scores)

                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)

            if not self.config.on_device_sampling:
                if self.sampler is None:
                    self.config.do_sample = True
                    self.sampler = Sampler(self.config)
                next_tokens = self.sampler.sample(outputs.logits[:, -1, :])
            else:
                next_tokens = outputs.tokens

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                is_for_token_generation=is_for_token_generation,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0

        if return_dict_in_generate:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
            )
        else:
            return input_ids
