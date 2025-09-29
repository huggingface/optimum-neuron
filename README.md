<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Optimum Neuron

🤗 Optimum Neuron is the interface between the 🤗 Transformers library and AWS Accelerators including [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/?nc1=h_ls) and [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/?nc1=h_ls).
**Key Features:**
- 🔄 **Drop-in replacement** for standard Transformers training and inference
- ⚡ **Distributed training** support with minimal code changes
- 🎯 **Optimized models** for AWS accelerators
- 📈 **Production-ready** inference with compiled models

## Install
To install the latest release of this package:

* For AWS Trainium (trn1) or AWS inferentia2 (inf2)

```bash
pip install --upgrade-strategy eager optimum-neuron[neuronx]
```

* For AWS inferentia (inf1)

```bash
pip install --upgrade-strategy eager optimum-neuron[neuron]
```

Optimum Neuron is a fast-moving project, and you may want to install it from source:

```bash
pip install git+https://github.com/huggingface/optimum-neuron.git
```

*Make sure that you have installed the Neuron driver and tools before installing `optimum-neuron`, [more extensive guide here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx).*

## Quick Start

Optimum Neuron makes AWS accelerator adoption seamless for Transformers users.

### Training

Training on AWS Trainium requires minimal changes to your existing code:

```python
import torch
import torch_xla.runtime as xr

from datasets import load_dataset
from transformers import AutoTokenizer

# Optimum Neuron's drop-in replacements for standard training components
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM


def format_dolly_dataset(example):
    """Format Dolly dataset into instruction-following format."""
    instruction = f"### Instruction\n{example['instruction']}"
    context = f"### Context\n{example['context']}" if example["context"] else None
    response = f"### Answer\n{example['response']}"

    # Combine all parts with double newlines
    parts = [instruction, context, response]
    return "\n\n".join(part for part in parts if part)


def main():
    # Load instruction-following dataset
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    # Model configuration
    model_id = "Qwen/Qwen3-1.7B"
    output_dir = "qwen3-1.7b-finetuned"

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure training for Trainium
    training_args = NeuronTrainingArguments(
        learning_rate=1e-4,
        tensor_parallel_size=8,  # Split model across 8 accelerators
        per_device_train_batch_size=1,  # Batch size per device
        gradient_accumulation_steps=8,
        logging_steps=1,
        output_dir=output_dir,
    )

    # Load model optimized for Trainium
    model = NeuronModelForCausalLM.from_pretrained(
        model_id,
        training_args.trn_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # Enable flash attention
    )

    # Setup supervised fine-tuning
    sft_config = NeuronSFTConfig(
        max_seq_length=2048,
        packing=True,  # Pack multiple samples for efficiency
        **training_args.to_dict(),
    )

    # Initialize trainer and start training
    trainer = NeuronSFTTrainer(
        model=model,
        args=sft_config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_dolly_dataset,
    )

    trainer.train()

    # Share your model with the community
    trainer.push_to_hub(
        commit_message="Fine-tuned on Databricks Dolly dataset",
        blocking=True,
        model_name=output_dir,
    )

    if xr.local_ordinal() == 0:
        print(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
```

This example demonstrates supervised fine-tuning on the [Databricks Dolly dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k) using `NeuronSFTTrainer` and `NeuronModelForCausalLM` - the Trainium-optimized versions of standard Transformers components.


**Compilation** (optional for first run):
```bash
NEURON_CC_FLAGS="--model-type transformer" neuron_parallel_compile torchrun --nproc_per_node 32 sft_finetune_qwen3.py
```

**Training:**
```bash
NEURON_CC_FLAGS="--model-type transformer" torchrun --nproc_per_node 32 sft_finetune_qwen3.py
```


### Inference

You can compile and export your 🤗 Transformers models to a serialized format before inference on Neuron devices:

```bash
optimum-cli export neuron \
  --model distilbert-base-uncased-finetuned-sst-2-english \
  --batch_size 1 \
  --sequence_length 32 \
  --auto_cast matmul \
  --auto_cast_type bf16 \
  distilbert_base_uncased_finetuned_sst2_english_neuron/
```

The command above will export `distilbert-base-uncased-finetuned-sst-2-english` with static shapes: `batch_size=1` and `sequence_length=32`, and cast all `matmul` operations from FP32 to BF16. Check out the [exporter guide](https://huggingface.co/docs/optimum-neuron/guides/export_model) for more compilation options.

Then you can run the exported Neuron model on Neuron devices with `NeuronModelForXXX` classes which are similar to `AutoModelForXXX` classes in 🤗 Transformers:

```diff
from transformers import AutoTokenizer
-from transformers import AutoModelForSequenceClassification
+from optimum.neuron import NeuronModelForSequenceClassification

# PyTorch checkpoint
-model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
+model = NeuronModelForSequenceClassification.from_pretrained("distilbert_base_uncased_finetuned_sst2_english_neuron")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
inputs = tokenizer("Hamilton is considered to be the best musical of past years.", return_tensors="pt")

logits = model(**inputs).logits
print(model.config.id2label[logits.argmax().item()])
# 'POSITIVE'
```

### Documentation

Check out [the documentation of Optimum Neuron](https://huggingface.co/docs/optimum-neuron/index) for more advanced usage.

<!---

## Validated Models

The following model architectures, tasks and device distributions have been validated for 🤗 Optimum Neuron:

<div align="center">

| Architecture     | State | <center>Tasks</center>                                                                                                                                                                                                                                                                                                                                 |
| ---------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| BERT             | ✅     | <li>[text classification](https://github.com/huggingface/optimum-neuron/tree/main/examples/text-classification)</li><li>[question answering](https://github.com/huggingface/optimum-neuron/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-neuron/tree/main/examples/language-modeling)</li> |
| RoBERTa          | ❌     | <li>[question answering](https://github.com/huggingface/optimum-neuron/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-neuron/tree/main/examples/language-modeling)</li>                                                                                                                     |
| ALBERT           | ❌     | <li>[question answering](https://github.com/huggingface/optimum-neuron/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-neuron/tree/main/examples/language-modeling)</li>                                                                                                                     |
| DistilBERT       | ❌     | <li>[question answering](https://github.com/huggingface/optimum-neuron/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-neuron/tree/main/examples/language-modeling)</li>                                                                                                                     |
| GPT2             | ❌     | <li>[language modeling](https://github.com/huggingface/optimum-neuron/tree/main/examples/language-modeling)</li>                                                                                                                                                                                                                                       |
| T5               | ❌     | <li>[summarization](https://github.com/huggingface/optimum-neuron/tree/main/examples/summarization)</li><li>[translation](https://github.com/huggingface/optimum-neuron/tree/main/examples/translation)</li>                                                                                                                                           |
| ViT              | ❌     | <li>[image classification](https://github.com/huggingface/optimum-neuron/tree/main/examples/image-classification)</li>                                                                                                                                                                                                                                 |
| Swin             | ❌     | <li>[image classification](https://github.com/huggingface/optimum-neuron/tree/main/examples/image-classification)</li>                                                                                                                                                                                                                                 |
| Wav2Vec2         | ❌     | <li>[audio classification](https://github.com/huggingface/optimum-neuron/tree/main/examples/audio-classification)</li><li>[speech recognition](https://github.com/huggingface/optimum-neuron/tree/main/examples/speech-recognition)</li>                                                                                                               |
| Stable Diffusion | ❌     | <li>[text-to-image generation](https://github.com/huggingface/optimum-neuron/tree/main/examples/stable-diffusion)</li>                                                                                                                                                                                                                                 |
| CLIP             | ❌     | <li>[contrastive image-text training](https://github.com/huggingface/optimum-neuron/tree/main/examples/contrastive-image-text)</li>                                                                                                                                                                                                                    |

</div>

Other models and tasks supported by the 🤗 Transformers library may also work. You can refer to this [section](https://github.com/huggingface/optimum-neuron#how-to-use-it) for using them with 🤗 Optimum Neuron. Besides, [this page](https://github.com/huggingface/optimum-neuron/tree/main/examples) explains how to modify any [example](https://github.com/huggingface/transformers/tree/main/examples/pytorch) from the 🤗 Transformers library to make it work with 🤗 Optimum Neuron.

-->

If you find any issue while using those, please open an issue or a pull request.
