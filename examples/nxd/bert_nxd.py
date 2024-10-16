import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


name = "bert-base-cased-finetuned-mrpc"

model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
torch.save({"model": model.state_dict()}, "bert.pt")

import neuronx_distributed
import torch
from neuronx_distributed.parallel_layers import layers, parallel_state
from transformers import AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput


def encode(tokenizer, *inputs, max_length=128, batch_size=1):
    tokens = tokenizer.encode_plus(
        *inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return (
        torch.repeat_interleave(tokens["input_ids"], batch_size, 0),
        torch.repeat_interleave(tokens["attention_mask"], batch_size, 0),
        torch.repeat_interleave(tokens["token_type_ids"], batch_size, 0),
    )


# Create the tokenizer and model
name = "bert-base-cased-finetuned-mrpc"
tokenizer = AutoTokenizer.from_pretrained(name)


# Set up some example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

paraphrase = encode(tokenizer, sequence_1, sequence_2)
not_paraphrase = encode(tokenizer, sequence_1, sequence_1)


def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)

    # Here we build a model with tensor-parallel layers.
    # Note: If you already have a Model class that does this, we can use that directly
    # and load the checkpoint in it.
    class ParallelSelfAttention(BertSelfAttention):
        def __init__(self, config, position_embedding_type=None):
            super().__init__(config, position_embedding_type)
            self.query = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
            self.key = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
            self.value = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
            self.num_attention_heads = self.num_attention_heads // parallel_state.get_tensor_model_parallel_size()
            self.all_head_size = self.all_head_size // parallel_state.get_tensor_model_parallel_size()

    class ParallelSelfOutput(BertSelfOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = layers.RowParallelLinear(config.hidden_size, config.hidden_size, input_is_parallel=True)

    for layer in model.bert.encoder.layer:
        layer.attention.self = ParallelSelfAttention(model.config)
        layer.attention.output = ParallelSelfOutput(model.config)

    # Here we created a checkpoint as mentioned above. We pass sharded=False, since the checkpoint
    # we obtained is unsharded. In case you are using the checkpoint from the tensor-parallel training,
    # you can set the sharded=True, as that checkpoint will contain shards from each tp rank.
    neuronx_distributed.parallel_layers.load("bert.pt", model, sharded=False)

    # These io aliases would enable us to mark certain input tensors as state tensors. These
    # state tensors are going to be device tensors.
    io_aliases = {}
    return model, io_aliases


if __name__ == "__main__":

    # Note how we are passing a function that returns a model object, which needs to be traced.
    # This is mainly done, since the model initialization needs to happen within the processes
    # that get launched internally within the parallel_model_trace.
    model = neuronx_distributed.trace.parallel_model_trace(
        get_model, paraphrase, tp_degree=2, inline_weights_to_neff=False
    )

    # Once traced, we now save the trace model for future inference. This API takes care
    # of saving the checkpoint from each tensor parallel worker
    # neuronx_distributed.trace.parallel_model_save(model, "tp_models")

    # We now load the saved model and will run inference against it
    # model = neuronx_distributed.trace.parallel_model_load("tp_models")
    cpu_model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
    assert torch.argmax(model(*paraphrase)[0]) == torch.argmax(cpu_model(*paraphrase)[0])
