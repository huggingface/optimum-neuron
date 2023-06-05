import torch

# noqa
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    set_seed,
)

from optimum.neuron import (
    NeuronModelForMultipleChoice,
)


# model_id = "hf-internal-testing/tiny-random-RobertaModel"
# model_id = "hf-internal-testing/tiny-random-BertModel"
model_id = "hf-internal-testing/tiny-random-AlbertModel"
# model_id = "hf-internal-testing/tiny-random-flaubert"
# model_id = "hf-internal-testing/tiny-random-XLMModel"
# model_id = "flaubert/flaubert_small_cased"
# model_id = "hf-internal-testing/tiny-xlm-roberta"
input_shapes = {
    "batch_size": 1,
    "sequence_length": 32,
    "num_choices": 4,
}

# from_trfrs
neuron_model = NeuronModelForMultipleChoice.from_pretrained(
    model_id, export=True, dynamic_batch_size=True, **input_shapes
)
set_seed(42)
transformers_model = AutoModelForMultipleChoice.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# neuron_model.save_pretrained("test_bert/")


# from local
# model_id = "test_bert/"
# neuron_model = NeuronModelForSequenceClassification.from_pretrained(model_id)
# print(neuron_model.config.neuron_batch_size)
# print(neuron_model.config.neuron_sequence_length)
# print(neuron_model.config.dynamic_batch_size)
# print(neuron_model.neuron_config.dynamic_batch_size)

# dummy_input_shapes = {
#     "batch_size": 5,
#     "sequence_length": 62,
# }
# dummy = neuron_model.neuron_config.generate_dummy_inputs(return_tuple=False, **dummy_input_shapes)

# input for multiple choice
num_choices = 4
first_sentence = ["The sky is blue due to the shorter wavelength of blue light."] * num_choices
start = "The color of the sky is"
second_sentence = [start + "blue", start + "green", start + "red", start + "yellow"]
inputs = tokenizer(first_sentence, second_sentence, truncation=True, padding=True)

# Unflatten the tokenized inputs values expanding it to the shape [batch_size, num_choices, seq_length]
for k, v in inputs.items():
    inputs[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]

pt_inputs = dict(inputs.convert_to_tensors(tensor_type="pt"))
with torch.no_grad():
    trfrs_outputs = transformers_model(**pt_inputs)
neuron_outputs = neuron_model(**pt_inputs)
import pdb


pdb.set_trace()


text = "This is a sample output"
tokens = tokenizer(text, return_tensors="pt")
input_with_pad = tokenizer.encode_plus(text, padding="max_length", max_length=32, return_tensors="pt")
import pdb


pdb.set_trace()
neuron_outputs, pad_input = neuron_model(**tokens)
padded_inputs = {"input_ids": pad_input[0], "attention_mask": pad_input[1], "token_type_ids": pad_input[2]}
with torch.no_grad():
    transformers_outputs = transformers_model(**padded_inputs)
