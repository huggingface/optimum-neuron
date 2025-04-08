import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.neuron.models.training.config import TrainingNeuronConfig
from optimum.neuron.models.training.granite.modeling_granite import GraniteForCausalLM
from optimum.neuron.utils.import_utils import (
    is_torch_xla_available,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from .. import launch_procs


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


@torch.no_grad()
def _get_expected_output(model_id, inputs, torch_dtype):
    # Get the expected output. Inference will run on CPU
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device="xla")
    model = model.eval()
    outputs = model(**inputs)
    return outputs.logits.detach()


def sample_greedy(logits):
    next_logits = logits.to("cpu")[:, -1]
    next_token_id = torch.argmax(next_logits, dim=-1)[:, None].int()
    return next_token_id


@torch.no_grad()
def _test_parallel_granite():
    model_id = "ibm-granite/granite-3.2-2b-instruct"
    prompt = "What is Deep Learning?"
    torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(prompt, return_tensors="pt").to("xla")

    # Expected output is the one loaded from transformers "vanilla" modeling on XLA
    expected_output = _get_expected_output(model_id, inputs, torch_dtype)
    xm.mark_step()

    # Note that model is init on CPU, then moved  to XLA
    mp_config = TrainingNeuronConfig(
        sequence_parallel_enabled=False,
    )
    model = GraniteForCausalLM.from_pretrained(model_id, mp_config, torch_dtype=torch_dtype).to(device="xla")
    model.eval()
    outputs = model(**inputs)
    xm.mark_step()

    # It would be better to have this lower, like torch.finfo(torch_dtype).resolution, ( that is 0.1 in bfloat16),
    # but apparently sharded model results are different from unsharded ones.
    atol = 0.2
    outputs_match = torch.allclose(outputs.logits.to("cpu"), expected_output.to("cpu"), atol=atol)
    assert outputs_match, "Sharded model output does not match unsharded one"

    # It is possible to verify that untokenized output is the same when using greedy sampling
    expected_text_output = tokenizer.batch_decode(sample_greedy(expected_output), skip_special_tokens=True)
    text_output = tokenizer.batch_decode(sample_greedy(outputs.logits), skip_special_tokens=True)
    assert expected_text_output == text_output, "Sharded model output does not match unsharded one"


@is_trainium_test
def _test_parallel_granite():
    launch_procs(
        _test_parallel_granite,
        num_procs=2,
        tp_size=2,
        pp_size=1,
    )


def _test_training_granite0():
    model_id = "ibm-granite/granite-3.2-2b-instruct"
    torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = GraniteForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device="xla")
    model.train()

    # Sample training data
    prompt = "What is Artificial Intelligence?"
    target = "Artificial Intelligence is the simulation of human intelligence in machines."
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("xla")
    labels = tokenizer(target, return_tensors="pt", padding=True).input_ids.to("xla")

    # Adjust labels to match input shape
    labels = torch.cat([labels, torch.full((labels.size(0), inputs.input_ids.size(1) - labels.size(1)), -100, device="xla")], dim=1)
    return
    # Get initial output for comparison
    model.eval()
    with torch.no_grad():
        initial_outputs = model(**inputs).logits
    model.train()

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop (1 step for demonstration purposes)
    for _ in range(1):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        print(f"Training loss: {loss.item()}")
        loss.backward()
        xm.optimizer_step(optimizer)
        xm.mark_step()

    # Get output after training
    model.eval()
    with torch.no_grad():
        trained_outputs = model(**inputs).logits

    # Check if the outputs are different
    outputs_differ = not torch.allclose(initial_outputs.to("cpu"), trained_outputs.to("cpu"), atol=1e-4)
    print(f"Outputs differ after training: {outputs_differ}")
    assert outputs_differ, "Model outputs did not change after training"

from datasets import load_dataset
from transformers import set_seed
from optimum.neuron import NeuronTrainer, NeuronTrainingArguments


def format_dolly(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = f"### Instruction\n{examples['instruction'][i]}"
        context = f"### Context\n{examples['context'][i]}" if len(examples["context"][i]) > 0 else None
        response = f"### Answer\n{examples['response'][i]}"
        prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
        output_text.append(prompt)
    return output_text


def _test_training_granite():
    # set seed
    set_seed(42)
    torch_dtype = torch.bfloat16

    # Load the dataset
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    model_id = "ibm-granite/granite-3.2-2b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Tokenize the dataset
    def tokenize_function(example):
        ret = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=24,
        )
        return ret

    # Load the model
    model = GraniteForCausalLM.from_pretrained(model_id,
                                               torch_dtype=torch_dtype).to(device="xla")

    output_dir = f"{model_id}-finetuned"
    # Define training arguments
    training_args = NeuronTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,
        num_train_epochs=1,
        max_steps=1,
        do_train=True,
        bf16=True,
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="epoch",
        save_total_limit=2,
        formatting_func=format_dolly,
    )

    # Initialize the Trainer, only for training (no validation in this test)
    trainer = NeuronTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
    )
    xm.mark_step()
    # Train the model
    train_result = trainer.train()
    return
    metrics = train_result.metrics

    eval_dataset = tokenized_emotions["validation"]
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    metrics.update(eval_metrics)
    trainer.log_metrics("train", metrics)

    trainer.save_model(output_dir)
    trainer.create_model_card()
    if args.repository_id:
        trainer.push_to_hub(repository_id=args.repository_id, token=args.hf_token)



@is_trainium_test
def test_training_granite():
    launch_procs(
        _test_training_granite,
        num_procs=2,
        tp_size=2,
        pp_size=1,
    )


