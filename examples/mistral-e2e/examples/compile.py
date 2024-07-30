from optimum.neuron import NeuronModelForCausalLM

# num_cores is the number of neuron cores. Find this with the command neuron-ls
compiler_args = {"num_cores": 12, "auto_cast_type": 'bf16'}
input_shapes = {"batch_size": 1, "sequence_length": 4096}

# Compiles an Optimum Neuron model from the previously trained (uncompiled) model
model = NeuronModelForCausalLM.from_pretrained(
    "mistral",
    export=True,
    **compiler_args,
    **input_shapes)

# Saves the compiled model to the directory mistral_neuron
model.save_pretrained("mistral_neuron")
