# Fine-Tune Qwen3

This example demonstrates how to fine-tune the Qwen3 model using the `optimum-neuron` library. 
It includes the necessary steps to load the dataset, prepare the model, and run the training process.

TIP: This example can be easily adapted to other datasets and/or models, the only restriction is that there is support for this model in the `optimum-neuron` library.

## Usage

### Compilation

To train any model on AWS Trainium, you need to compile it first. It is very simple to do so, just run the following command:

```bash
neuron_parallel_compile ./finetune_qwen3.sh
```

**Note**: It is not necessary to perform this step if you are using a pre-compiled model hosted on the [cache on the Hugging Face Hub](https://huggingface.co/aws-neuron/optimum-neuron-cache).

### Fine-Tuning

Once the model is compiled, you can start the fine-tuning process by running the following command:

```bash
./finetune_qwen3.sh
```

## Result for `Qwen/Qwen3-8B`



