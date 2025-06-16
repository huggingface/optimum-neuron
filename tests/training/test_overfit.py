import importlib
import os
from datetime import datetime
from functools import partial

import datasets
import pytest
import torch
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback

from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.peft import get_peft_model
from optimum.neuron.utils.import_utils import (
    is_neuronx_distributed_available,
)
from optimum.neuron.utils.misc import is_precompilation
from optimum.neuron.utils.testing_utils import is_trainium_test
from optimum.neuron.utils.training_utils import is_main_worker_for_metrics

from .. import launch_procs


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers.parallel_state import get_data_parallel_size


def _overfit_causal_lm(
    model_class_name,
    model_name_or_path,
    learning_rate,
    warmup_ratio,
    training_kwargs,
    max_length,
    max_expected_loss,
    num_steps,
    tp_size,
    pp_size,
    use_flash_attention_2,
    output_dir,
    peft_config=None,
):
    training_mod = importlib.import_module("optimum.neuron.models.training")
    model_class = getattr(training_mod, model_class_name)

    if pp_size > 1 and not model_class.supports_pipeline_parallelism():
        pytest.skip(f"The model {model_class_name} does not support pipeline parallelism, skipping the test.")

    # Dataset creation.
    sample_to_overfit = "Paris is the most beautiful city in the world."
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(sample_to_overfit, return_tensors="pt", padding="max_length", max_length=max_length)
    inputs["labels"] = inputs["input_ids"].clone()
    # We basically remove the batch dimension to have a single example, the batch dimension is added when creating the
    # dataset.
    inputs = {k: v[0, :] for k, v in inputs.items()}

    def gen():
        yield inputs

    dataset = datasets.Dataset.from_generator(gen)
    dataset = dataset.select([0] * 10000)

    # Wandb setup.
    os.environ["WANDB_MODE"] = "online" if not is_precompilation() else "disabled"
    os.environ["WANDB_PROJECT"] = "test-train-overfit"
    dp_size = get_data_parallel_size()
    wandb_run_name = f"{model_name_or_path}-dp={dp_size},tp={tp_size},pp={pp_size}"
    if peft_config is not None:
        wandb_run_name += "-lora"
    date_str = datetime.now().strftime("%d%m%y-%H-%M")
    wandb_run_name += f"-{date_str}"

    # Training args creation.
    training_args = NeuronTrainingArguments(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        do_train=True,
        do_eval=False,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        lr_scheduler_type="cosine",
        gradient_checkpointing=False,
        max_grad_norm=1,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        max_steps=10 if is_precompilation() else num_steps,
        output_dir=output_dir,
        run_name=wandb_run_name,
        **training_kwargs,
    )

    # Model creation.
    model = model_class.from_pretrained(
        model_name_or_path,
        training_args.trn_config,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=use_flash_attention_2,
    )

    if peft_config is not None:
        model = get_peft_model(model, peft_config)

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

    # If the test was run with neuron_parallel_compile, we stop after training since the goal of this run was only
    # to compile the model, no test can actually be done here.
    if is_precompilation():
        return

    # The master worker checks the logs, since it is the only worker to have access to them, to retrieve the last logged
    # loss. It then checks if it is lower or equal to max_expected_loss.
    if is_main_worker_for_metrics():
        last_loss = None
        for logs in reversed(stored_logs):
            if "loss" in logs:
                last_loss = logs["loss"]
                break
        if last_loss is None:
            raise ValueError("No loss found in the logs.")
        print("Last loss", last_loss)
        assert last_loss <= max_expected_loss, "The model did not overfit the dataset."


@pytest.mark.parametrize(
    "model_class_name,model_name_or_path,learning_rate,warmup_ratio,training_kwargs,use_flash_attention_2,max_expected_loss,max_length,num_steps",
    [
        [
            "LlamaForCausalLM",
            "meta-llama/Llama-3.2-1B-Instruct",
            1e-4,
            0.03,
            {},
            True,
            0.0,
            2048,
            30,
        ],
        [
            "GraniteForCausalLM",
            "ibm-granite/granite-3.2-2b-instruct",
            1e-4,
            0,
            {},
            # For now we disable flash attention because the default configuration has a dropout for the attention
            # which is broken with the flash attention kernel in the current Neuron SDK.
            False,
            0.07,
            512,  # Do 2048 once we have flash_attention enabled.
            30,
        ],
        [
            "Qwen3ForCausalLM",
            "Qwen/Qwen3-0.6B",
            1e-4,
            0.04,
            {},
            True,
            0.0,
            2048,
            50,
        ],
    ],
    ids=["meta-llama/Llama-3.2-1B-Instruct", "ibm-granite/granite-3.2-2b-instruct", "Qwen/Qwen3-0.6B"],
)
@pytest.mark.parametrize(
    "world_size,tp_size,pp_size",
    [
        # It is important to make it so that the DP size is the same otherwise the tests are not equivalent with the provided hyperparameters.
        [32, 8, 1],
        [32, 2, 4],
    ],
    ids=[
        "dp=4,tp=8",
        "dp=4,tp=2,pp=4",
    ],
)
@pytest.mark.neuron_parallel_compile
@is_trainium_test
def test_overfit_causal_lm(
    model_class_name,
    model_name_or_path,
    learning_rate,
    warmup_ratio,
    training_kwargs,
    use_flash_attention_2,
    max_expected_loss,
    max_length,
    num_steps,
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
        learning_rate,
        warmup_ratio,
        training_kwargs,
        max_length,
        max_expected_loss,
        num_steps,
        tp_size,
        pp_size,
        use_flash_attention_2,
        tmpdir,
    )
    launch_procs(run_fn, world_size, tp_size, pp_size)


@pytest.mark.parametrize(
    "world_size,tp_size,pp_size",
    [
        [8, 8, 1],
        [32, 32, 1],
    ],
    ids=[
        "dp=1,tp=8",
        # This is to test the case where we have more than 8 TP workers, which will use GQAGQAColumnParallelLinear.
        "dp=1,tp=32",
    ],
)
@pytest.mark.neuron_parallel_compile
@is_trainium_test
def test_overfit_lora_causal_lm(world_size, tp_size, pp_size, tmpdir, set_cache_for_ci):
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["embed_tokens", "q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    run_fn = partial(
        _overfit_causal_lm,
        "LlamaForCausalLM",
        "meta-llama/Llama-3.2-1B-Instruct",
        1e-4,
        0.03,
        {},
        2048,
        0.0,
        30,
        tp_size,
        pp_size,
        True,
        tmpdir,
        peft_config=peft_config,
    )
    launch_procs(run_fn, world_size, tp_size, pp_size)
