import logging

import torch
import torch.nn.functional as F


CONTEXT_ENCODING_MODEL_TAG = "context_encoding_model"
TOKEN_GENERATION_MODEL_TAG = "token_generation_model"


class DecoderModelWrapper(torch.nn.Module):
    def __init__(self, config, model, tag="") -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.tag = tag

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

        logging.debug(f"Processed inputs to the model {self.tag} with args {args}")

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
