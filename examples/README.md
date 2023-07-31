# 🤗 Transformers example scripts on AWS Trainium/Inferentia with Optimum Neuron

Most of the following example scripts have been taken from the [official 🤗 Transformers example directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch).

## Training

The training scripts for the following tasks have been adapted to use the `NeuronTrainer`:

- image-classification,
- language-modeling,
- multiple-choice,
- question-answering,
- summarization,
- text-classification,
- token-classification,
- translation.

The required change is as simple as:

```diff
- from transformers import Trainer
+ from optimum.neuron import NeuronTrainer as Trainer
```

While this is not *mandatory*, using the `NeuronTrainer` over the regular `Trainer` offers some advantages:

- Sanity checks are done preventing you to provide training argument values that do not behave well with Trainium. This can be disabled by setting the environment variable `DISABLE_STRICT_MODE=false`.

- Some modifications are performed during the precompilation phase that can help avoiding recompilation during training.

- Some models will not work out of the box, and the `NeuronTrainer` takes care of patching them accordingly, at least for the supported models.


That being said, you can use those examples exactly as you would use the official examples from the 🤗 Transformers library.
Feel free to check there if you have any usage related questions!

## Inference

The inference scripts for the following tasks have been adapted to use Neuron models:

- text-generation.
