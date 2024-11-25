from typing import Union

import torch
from modules.sampling import Sampler  # noqa: E402
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import (
    SampleDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
)


SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]


class NeuronDecoderModel(PreTrainedModel):
    """
    Base model that NeuronXXXModel classes inherit from.

    The forward() function will be traced and compiled by NxD.
    """

    SEQ_DIM = 2

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.batch_size = config.batch_size
        self.n_positions = config.n_positions
        self.vocab_size = config.vocab_size
        self.padding_side = config.padding_side
        self.max_length = config.max_length

        self.setup_attr_for_model(config)
        self.init_model(config)
        self.init_inference_optimization(config)
        self.post_init()

    def setup_attr_for_model(self, config: PretrainedConfig):
        """
        Please provide model-specific definition for the following attributes
            self.on_device_sampling
            self.tp_degree
            self.hidden_size
            self.num_attention_heads
            self.num_key_value_heads
            self.max_batch_size
        """
        raise NotImplementedError("setup_attr_for_model() is not implemented")

    def init_model(self, config: PretrainedConfig):
        """
        Please provide definition for the following components:
            self.embed_tokens
            self.layers
            self.norm
            self.lm_head
        """
        raise NotImplementedError("init_model() is not implemented")

    def init_inference_optimization(self, config: PretrainedConfig):
        if config.on_device_sampling:
            self.sampler = Sampler(config)
        return

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        seq_ids,
    ):

        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device  # noqa
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        next_decoder_cache = ()

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=True,
            )

            hidden_states = layer_outputs[0]

            next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if self.padding_side == "left":
            index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            # simple token generation
            index = torch.max(position_ids, dim=1, keepdim=True).indices
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        res = logits
        if self.sampler is not None:
            # perform sampling on Neuron to get tokens
            res = self.sampler.sample(logits[:, -1, :])

        return res, next_decoder_cache

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
