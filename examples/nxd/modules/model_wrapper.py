import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from modules.autobucketing import generate_buckets, get_context_encoder_bk, get_token_generation_bk
from neuronx_distributed.trace.model_builder import BaseModelInstance
from torch_neuronx import BucketModelConfig


CONTEXT_ENCODING_MODEL_TAG = "context_encoding_model"
TOKEN_GENERATION_MODEL_TAG = "token_generation_model"


class ModelWrapper(torch.nn.Module):
    def __init__(self, config, model_cls, tag="", max_input_tokens: int = 128, max_total_tokens: int = 128) -> None:
        super().__init__()
        self.config = config

        if not self.config.torch_dtype:
            self.config.torch_dtype = torch.float32

        if self.config.pad_token_id is None:
            self.config.pad_token_id = 0

        self.model_cls = model_cls
        self.model = None
        self.is_compiled = False
        self.serialize_base_path = None
        self.tag = tag
        self.enable_bucketing = config.enable_bucketing
        self.max_input_tokens = max_input_tokens
        self.max_total_tokens = max_total_tokens

    def _forward_with_pad(self, *args):
        seq_ids = args[3]

        # pad the inputs up to the compiled batch size in the end
        def pad_helper(tensor):
            if tensor is None or tensor.shape[0] == self.config.batch_size:
                return tensor

            padded_shape = list(tensor.shape)
            padded_shape[0] = self.config.batch_size
            padded_tensor = torch.zeros(padded_shape, dtype=tensor.dtype)
            padded_tensor[: tensor.shape[0]] = tensor
            return padded_tensor

        padded_args = []
        # pad input_ids, attn_mask and postition_ids
        for arg in args[0:3]:
            padded_args.append(pad_helper(arg))

        # need to handle seq_ids seperately, when compiled batch is 4, if we pad seq_ids from [0,2,1] to [0,2,1,0].
        # then the kv cache of padded input could be written into the first cache line, so we need to pad as [0, 2, 1, 3] instead

        seq_ids_list = seq_ids.tolist()
        padded_seq_ids = torch.tensor(
            seq_ids_list + [x for x in range(self.config.max_batch_size) if x not in seq_ids_list], dtype=seq_ids.dtype
        )
        padded_args.append(padded_seq_ids)

        outputs = self._forward(*padded_args)

        # note that we don't do index select here as it should already be handled, simply sliced out padding here
        logits = outputs
        return logits[: seq_ids.shape[0]]

    def reorder_helper(self, *args):
        # we then reorder the other inputs based on padded_seq_ids
        # because there are issue with compiler to do gather, we cannot fully support artibrary order of seq_ids for now
        seq_ids = args[3]

        reorder_args = []

        for arg in args:
            reorder_args.append(torch.index_select(arg, 0, seq_ids))

        return [seq_ids] + reorder_args

    def _forward(self, *args):
        if self.config.is_continuous_batching and self.config.batch_size == self.config.max_batch_size:
            logging.debug("running forward and reorder the inputs based on seq_ids")
            seq_ids, *args = self.reorder_helper(*args)

        logging.debug("Processed inputs to the model", self.tag, args)

        outputs = self.model(*args)

        if self.config.is_continuous_batching and self.config.batch_size == self.config.max_batch_size:
            return torch.index_select(outputs, 0, seq_ids)

        return outputs

    def pad_to_max_compiled_seq(self, *args):
        if self.tag == CONTEXT_ENCODING_MODEL_TAG:
            to_pad = args[:3]
            pad_lengths = [self.config.max_context_length - arg.shape[1] for arg in to_pad]
            tensor_pad_vals = [self.config.pad_token_id, 0, 1]
            padded_args = [
                F.pad(arg, (0, pad_len), "constant", pad_val)
                for arg, pad_val, pad_len in zip(to_pad, tensor_pad_vals, pad_lengths)
            ]
            args = (*padded_args, *args[3:])
        else:
            input_ids, attention_mask, *rest_of_args = args
            pad_len = self.config.max_length - attention_mask.shape[1]
            padded_attention_mask = F.pad(attention_mask, (0, pad_len), "constant", 0)
            args = (input_ids, padded_attention_mask, *rest_of_args)

        return args

    def forward(self, *args):
        logging.debug(f"calling forward on network {self.tag}")

        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() or load_state_dict() making calling forward")

        args = self.pad_to_max_compiled_seq(*args)

        seq_ids = args[3]

        input_batch_size = seq_ids.shape[0]

        if input_batch_size == self.config.batch_size:
            return self._forward(*args)

        cur_batch = 0
        output_logits = []

        logging.debug(
            f"get input_batch_size as {input_batch_size} but compiled batch_size as {self.config.batch_size}"
        )
        while cur_batch < input_batch_size:
            if cur_batch + self.config.batch_size <= input_batch_size:
                # we only process part of the input to run
                logging.debug(f"running foward on batch {cur_batch}:{cur_batch+self.config.batch_size}")
                outputs = self._forward(*[arg[cur_batch : cur_batch + self.config.batch_size] for arg in args])
            else:
                # we need to pad the input to run
                logging.debug(
                    f"running forward on batch {cur_batch}:{input_batch_size}, padded up to {self.config.batch_size}"
                )
                outputs = self._forward_with_pad(*[arg[cur_batch:input_batch_size] for arg in args])

            output_logits.append(outputs)
            cur_batch += self.config.batch_size

        return torch.cat(output_logits, dim=0)


class DecoderModelInstance(BaseModelInstance):

    def __init__(self, model_cls, config, buckets):
        self.model_cls = model_cls
        self.module = None
        self.input_output_aliases = None
        self.config = config
        self.buckets = buckets

    def load_module(self):
        float_model = self.model_cls(self.config)
        float_model.eval()

        if self.config.torch_dtype == torch.bfloat16:
            float_model.bfloat16()

        self.module = float_model

    def get(self, bucket_rank, **kwargs):
        if bucket_rank is not None:
            self.module.n_positions = self.buckets[bucket_rank]

        # Currently we have to init an input_output_aliases map for
        # each buckets, otherwise it will fail the aliasing setup when
        # generating HLO
        self.input_output_aliases = {}
        num_output_from_trace = 1
        for i in range(len(self.module.past_key_values)):
            self.input_output_aliases[self.module.past_key_values[i]] = num_output_from_trace + i
        return self.module, self.input_output_aliases


@dataclass
class ModelExporter(ABC):

    tag: str
    model_cls: type
    config: Any
    buckets: Tuple[int]
    max_input_tokens: int

    def input_generator(self):
        inputs = []
        for bucket in self.buckets:
            batch_size = self.config.batch_size
            n_active_tokens = min(self.max_input_tokens, bucket)

            input_ids = torch.zeros((batch_size, n_active_tokens), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, bucket), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, n_active_tokens), dtype=torch.int64)
            seq_ids = torch.zeros((batch_size), dtype=torch.int64)

            inputs.append((input_ids, attention_mask, position_ids, seq_ids))

        return inputs

    def get_model_instance(self):
        return DecoderModelInstance(model_cls=self.model_cls, config=self.config, buckets=self.buckets)

    @abstractmethod
    def bucket_config(self):
        raise NotImplementedError


class ContextEncodingModelExporter(ModelExporter):

    def __init__(self, model_cls, config):
        if config.enable_bucketing:
            buckets = generate_buckets(128, config.max_context_length)
        else:
            buckets = [config.max_context_length]
        print(buckets)
        super().__init__(
            tag=CONTEXT_ENCODING_MODEL_TAG,
            model_cls=model_cls,
            config=config,
            buckets=buckets,
            max_input_tokens=config.max_context_length,
        )

    def bucket_config(self):
        if not self.config.enable_bucketing:
            return None
        bucket_degree = len(self.buckets)
        return BucketModelConfig(
            bucket_kernel=get_context_encoder_bk,
            bucket_kernel_constant_args=(
                torch.tensor(self.buckets),
                self.config.padding_side,
                self.config.pad_token_id,
            ),
            shared_state_buffer=None,
            func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
        )


class TokenGenerationModelExporter(ModelExporter):

    def __init__(self, model_cls, config):
        if config.enable_bucketing:
            buckets = generate_buckets(128, config.max_length)
        else:
            buckets = [config.max_length]
        print(buckets)
        super().__init__(
            tag=TOKEN_GENERATION_MODEL_TAG,
            model_cls=model_cls,
            config=config,
            buckets=buckets,
            max_input_tokens=1,
        )

    def bucket_config(self):
        if not self.config.enable_bucketing:
            return None
        bucket_degree = len(self.buckets)
        return BucketModelConfig(
            bucket_kernel=get_token_generation_bk,
            bucket_kernel_constant_args=(torch.tensor(self.buckets), self.config.padding_side),
            shared_state_buffer=None,
            func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
        )
