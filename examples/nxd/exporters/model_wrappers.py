import torch


CONTEXT_ENCODING_MODEL_TAG = "context_encoding_model"
TOKEN_GENERATION_MODEL_TAG = "token_generation_model"


class DecoderModelWrapper(torch.nn.Module):
    """Eventually this wrapper should include the KV cache management
    That is now implemented in NeuronDecoderModel.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, seq_ids):
        return self.model(input_ids, attention_mask, position_ids, seq_ids)
