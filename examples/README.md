# 🤗 Transformers example scripts on AWS Trainium with Optimum Neuron

The following example scripts have been taken from the [official 🤗 Transformers example directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch) and adapted to use the `TrainiumTrainer`:

```diff
- from transformers import Trainer
+ from optimum.neuron import TrainiumTrainer as Trainer
```

While this is not *mandatory*, using the `TraniumTrainer` offers some advantages:

- Sanity checks are done preventing you to provide training argument values that do not behave well with Trainium. This can be disabled by setting the environment variable `DISABLE_STRICT_MODE=false`.

- Some modifications are performed during the precompilation phase that can help avoiding recompilation during training.

- Some models will not work out of the box, and the `TraniumTrainer` takes care of patching them accordingly, at least for the supported models.


That being said, you use those examples exactly as you would use the official examples from the 🤗 Transformers library.
Feel free to check there if you have any usage related questions!
