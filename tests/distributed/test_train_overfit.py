import importlib
from functools import partial

import datasets
import pytest
import torch
from transformers import AutoTokenizer, TrainerCallback

from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.utils.import_utils import (
    is_neuronx_distributed_available,
    is_torch_xla_available,
)

from .. import launch_procs


if is_torch_xla_available():
    import torch_xla.runtime as xr

if is_neuronx_distributed_available():
    pass


def _overfit_causal_lm(
    model_class_name,
    model_name_or_path,
    training_kwargs,
    tp_size,
    pp_size,
    output_dir,
):
    # Dataset creation.
    sample_to_overfit = "Paris is the most beautiful city in the world."
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    # We use a big sequence length to test that the model trains when there is padding since it can be tricky with
    # `recompute_causal_mask=True` or when using flash attention.
    inputs = tokenizer(sample_to_overfit, return_tensors="pt", padding="max_length", max_length=512)
    inputs["labels"] = inputs["input_ids"].clone()
    # We basically remove the batch dimension to have a single example, the batch dimension is added when creating the
    # dataset.
    inputs = {k: v[0, :] for k, v in inputs.items()}

    def gen():
        yield inputs

    dataset = datasets.Dataset.from_generator(gen)
    dataset = dataset.select([0] * 1000)

    # Training args creation.
    training_args = NeuronTrainingArguments(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        do_train=True,
        do_eval=False,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_grad_norm=1,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        max_steps=30,
        output_dir=output_dir,
        **training_kwargs,
    )

    # Model creation.
    training_mod = importlib.import_module("optimum.neuron.models.training")
    model_class = getattr(training_mod, model_class_name)
    model = model_class.from_pretrained(
        model_name_or_path,
        training_args.mp_config,
        torch_dtype=torch.bfloat16,
    )

    stored_logs = []

    class StoreLogsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                stored_logs.append(logs)

    # Training
    trainer = NeuronTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[StoreLogsCallback()],
    )

    trainer.train()

    # The master worker checks the logs, since it is the only worker to have access to them, to retrieve the last logged
    # loss. It then checks if it is equal to 0.0.
    if xr.global_ordinal() == 0:
        last_loss = None
        for logs in reversed(stored_logs):
            if "loss" in logs:
                last_loss = logs["loss"]
                break
        if last_loss is None:
            raise ValueError("No loss found in the logs.")
        print("Last loss", last_loss)
        assert last_loss == 0.0, "The model did not overfit the dataset."


@pytest.mark.parametrize(
    "model_class_name,model_name_or_path,training_kwargs",
    [
        [
            "LlamaForCausalLM",
            "meta-llama/Llama-3.2-1B-Instruct",
            {
                "use_flash_attention": True,
            },
        ],
    ],
    ids=["meta-llama/Llama-3.2-1B-Instruct"],
)
@pytest.mark.parametrize(
    "world_size,tp_size,pp_size",
    [[8, 8, 1]],
    ids=["dp=1,tp=8,pp=1"],
)
def test_overfit_causal_lm(
    model_class_name,
    model_name_or_path,
    training_kwargs,
    world_size,
    tp_size,
    pp_size,
    tmpdir,
    set_cache_for_ci,  # This fixture will handle setting the remote cache to make this test faster.
):
    run_fn = partial(
        _overfit_causal_lm,
        model_class_name,
        model_name_or_path,
        training_kwargs,
        tp_size,
        pp_size,
        tmpdir,
    )
    launch_procs(run_fn, world_size, tp_size, pp_size)
