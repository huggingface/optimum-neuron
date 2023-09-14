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
It provides a set of tools enabling easy model loading, training and inference on single- and multi-Accelerator settings for different downstream tasks.
The list of officially validated models and tasks is available [here](TODO:). Users can try other models and tasks with only few changes.

## Install
To install the latest release of this package:

* For AWS Trainium (trn1) or AWS inferentia2 (inf2)

```bash
pip install optimum[neuronx]
```

* For AWS inferentia (inf1)

```bash
pip install optimum[neuron]
```

Optimum Neuron is a fast-moving project, and you may want to install it from source:

```bash
pip install git+https://github.com/huggingface/optimum-neuron.git
```

> Alternatively, you can install the package without pip as follows:
> ```bash
> git clone https://github.com/huggingface/optimum-neuron.git
> cd optimum-neuron
> python setup.py install
> ```

*Make sure that you have installed the Neuron driver and tools before installing `optimum-neuron`, [more extensive guide here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx).*

Last but not least, don't forget to install the requirements for every example:

```bash
cd <example-folder>
pip install -r requirements.txt
```


## Quick Start

🤗 Optimum Neuron was designed with one goal in mind: **to make training and inference straightforward for any 🤗 Transformers user while leveraging the complete power of AWS Accelerators**.

### Training

There are two main classes one needs to know:
- TrainiumArgumentParser: inherits the original [HfArgumentParser](https://huggingface.co/docs/transformers/main/en/internal/trainer_utils#transformers.HfArgumentParser) in Transformers with additional checks on the argument values to make sure that they will work well with AWS Trainium instances.
- [NeuronTrainer](https://huggingface.co/docs/optimum/neuron/package_reference/trainer): this version trainer takes care of doing the proper checks and changes to the supported models to make them trainable on AWS Trainium instances.

The [NeuronTrainer](https://huggingface.co/docs/optimum/neuron/package_reference/trainer) is very similar to the [🤗 Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), and adapting a script using the Trainer to make it work with Trainium will mostly consist in simply swapping the Trainer class for the NeuronTrainer one.
That's how most of the [example scripts](https://github.com/huggingface/optimum-neuron/tree/main/examples) were adapted from their [original counterparts](https://github.com/huggingface/transformers/tree/main/examples/pytorch).

```diff
from transformers import TrainingArguments
+from optimum.neuron import NeuronTrainer as Trainer

training_args = TrainingArguments(
  # training arguments...
)

# A lot of code here

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,  # Original training arguments.
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
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
