
#!/bin/bash

model=${1:-meta-llama/Meta-Llama-3-8B-Instruct}
batch_size=${2:-32}
tasks=${3:-lambada_openai,hellaswag,gsm8k}

# Export model locally
eval_model=./data/${model}-lm_eval
optimum-cli export neuron -m ${model} --batch_size ${batch_size} \
                                      --auto_cast_type bf16 \
                                      --sequence_length 4096 \
                                      ${eval_model}

lm_eval --model neuronx \
        --tasks ${tasks} \
        --model_args pretrained=${eval_model} --batch_size ${batch_size}
