# Sample LLama 1B

This is sample code to show how NxD can be used to run a Llama 1B model. 

The model is defined in model.py 

`run.py` has some functions which you can use to, 
1. run the model on CPU
2. compile the model
3. run the model on neuron

I recommend reading the code, especially the comments to understand what needs 
to be done. This sample will eventually make it into the NxD/examples directory. 

Note: The ModelBuilder API is BETA. We are working on a official version, which will 
be a lot easier to use. But regardless, the 'development' flow will remain the same. 
Moreover this does not stop you from developing with the current API, as the functionality
will remain the same.


# How do I run this sample?

1. Install NXD
2. Download llama 1B from https://www.llama.com/llama-downloads/


## Run on CPU

sample
```
python run.py generate_cpu \
--model_path ~/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth \
--tokenizer_path /home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model
```

output
```
<|begin_of_text|>I will just count till 20 - 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20.<|eot_id|>
<|begin_of_text|>I will just count till 20 - 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20.<|eot_id|>
```
## Compile Model 
```
python run.py compile --batch_size 2 --seq_len 128 \
--model_path /home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth
```


## Run model on Neuron 

```
python run.py generate_nxd
```

output
```
(venv) ubuntu@ip-10-0-8-215:~/workspace-trn1/DROP$ python run.py generate_nxd
<|begin_of_text|>I will just count till 20 - 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20.<|eot_id|>
<|begin_of_text|>I will just count till 20 - 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20.<|eot_id|>
```


# More options to play with

Note this example was only tested for limited configurations. 
```
python run.py generate_cpu --help

NAME
    run.py generate_cpu

SYNOPSIS
    run.py generate_cpu <flags>

FLAGS
    -b, --batch_size=BATCH_SIZE
        Default: 2
    -s, --seq_len=SEQ_LEN
        Default: 128
    -m, --model_path=MODEL_PATH
        Default: '/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/consol...
    -t, --tokenizer_path=TOKENIZER_PATH
        Default: '/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/tokeni...
    -p, --prompts=PROMPTS
        Default: ['I will just count till 20 - 1,2,3,4', 'I will just count t...q
```
```
python run.py compile --help

NAME
    run.py compile

SYNOPSIS
    run.py compile <flags>

FLAGS
    -b, --batch_size=BATCH_SIZE
        Default: 2
    -s, --seq_len=SEQ_LEN
        Default: 128
    -t, --tp_degree=TP_DEGREE
        Default: 32
    -m, --model_path=MODEL_PATH
        Default: '/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/consol...
    -o, --output_path=OUTPUT_PATH
        Default: 'traced_model/'
```

```
python run.py generate_nxd --help

NAME
    run.py generate_nxd

SYNOPSIS
    run.py generate_nxd <flags>

FLAGS
    -w, --world_size=WORLD_SIZE
        Default: 32
    --traced_model_path=TRACED_MODEL_PATH
        Default: '/home/ubuntu/workspace-trn1/src/Aazhiko-workplace/scripts/m...
    --tokenizer_path=TOKENIZER_PATH
        Default: '/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/tokeni...
    -p, --prompts=PROMPTS
        Default: ['I will just count till 20 - 1,2,3,4', 'I will just count t...
```