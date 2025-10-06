# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from optimum.neuron.models.training.llama.modeling_llama import LlamaForCausalLM
from optimum.neuron.trainers import NeuronTrainer
from optimum.neuron.trainers.metrics import MovingAverageWindow, TrainingMetricsCollector
from optimum.neuron.trainers.training_args import NeuronTrainingArguments
from optimum.neuron.utils.testing_utils import is_trainium_test

from .distributed_utils import distributed_test
from .utils import MODEL_NAME


@is_trainium_test
@distributed_test(world_size=2, tp_size=2, pp_size=1)
def test_metrics_calculations_real_functionality(tmpdir):
    args = NeuronTrainingArguments(
        output_dir=tmpdir,
        enable_throughput_metrics=True,
        enable_mfu_metrics=True,
        enable_efficiency_metrics=True,
        metrics_window_size=3,
    )

    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, trn_config=args.trn_config)
    collector = TrainingMetricsCollector(model, args)

    # Test MovingAverageWindow standalone
    window = MovingAverageWindow(window_size=2)
    window.add_step(tokens=1000, samples=4, step_time=0.5)
    window.add_step(tokens=2000, samples=8, step_time=1.0)

    stats = window.get_window_stats()
    assert stats["total_tokens"] == 3000
    assert stats["total_samples"] == 12
    assert stats["total_time"] == 1.5
    assert stats["avg_tokens_per_step"] == 1500
    assert stats["avg_time_per_step"] == 0.75

    # Test throughput calculation using proper collector API
    inputs = {"input_ids": torch.randint(0, 1000, (1, 1000))}  # 1000 tokens
    collector.start_metric("throughput", inputs)
    time.sleep(0.5)
    collector.stop_metric("throughput", step_number=1)

    throughput_metrics = collector.calculate_metric("throughput")
    expected_global_rate = 2000.0 * collector.dp_size  # 1000 tokens / 0.5 sec * dp_size
    assert abs(throughput_metrics["train/tokens_per_sec"] - expected_global_rate) < 100  # Allow timing variance

    # Test MFU calculation using proper collector API
    collector.start_metric("mfu", inputs)
    time.sleep(0.5)
    collector.stop_metric("mfu", step_number=2)

    mfu_metrics = collector.calculate_metric("mfu")
    theoretical_flops = 18 * collector.model_params * 1000
    actual_flops_per_sec = theoretical_flops / 0.5
    peak_flops_per_sec = collector.peak_tflops_per_core * 1e12 * collector.total_neuron_cores
    expected_mfu = (actual_flops_per_sec / peak_flops_per_sec) * 100
    assert abs(mfu_metrics["train/mfu"] - round(expected_mfu, 2)) < 1.0  # Allow more variance for real timing

    # Test efficiency calculation using gradient accumulation cycle
    collector.start_gradient_accumulation_cycle()

    collector.start_metric("forward_pass", inputs)
    time.sleep(0.1)
    collector.stop_metric("forward_pass")

    collector.start_metric("backward_pass", inputs)
    time.sleep(0.2)
    collector.stop_metric("backward_pass")

    collector.start_metric("optimizer_step", inputs)
    time.sleep(0.05)
    collector.stop_metric("optimizer_step")

    collector.end_gradient_accumulation_cycle(step_number=3)

    collector.start_metric("total_step", inputs)
    time.sleep(0.4)
    collector.stop_metric("total_step", step_number=4)

    efficiency_metrics = collector.calculate_metric("efficiency")
    # With real timing, just verify metrics exist and are reasonable
    assert "train/efficiency" in efficiency_metrics
    assert "train/forward_time_percent" in efficiency_metrics
    assert "train/backward_time_percent" in efficiency_metrics
    assert "train/optimizer_time_percent" in efficiency_metrics
    assert "train/overhead_time_percent" in efficiency_metrics
    assert 0 <= efficiency_metrics["train/efficiency"] <= 100

    # Test context manager
    initial_tokens = collector.metric_windows["throughput"].get_window_stats()["total_tokens"]
    inputs_ctx = {"input_ids": torch.randint(0, 1000, (2, 32))}
    with collector.time_metric("throughput", inputs_ctx, step_number=99):
        time.sleep(0.01)

    ctx_stats = collector.metric_windows["throughput"].get_window_stats()
    assert ctx_stats["total_tokens"] == initial_tokens + 64  # Added 2*32 tokens
    assert ctx_stats["total_time"] > 0.008


@is_trainium_test
@distributed_test(world_size=8, tp_size=2, pp_size=1)
def test_metrics_basic_functionality(tmpdir):
    args = NeuronTrainingArguments(
        output_dir=tmpdir,
        enable_throughput_metrics=True,
        enable_efficiency_metrics=True,
        metrics_window_size=5,
        gradient_accumulation_steps=2,
    )
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, trn_config=args.trn_config)
    collector = TrainingMetricsCollector(model, args)

    inputs = {"input_ids": torch.randint(0, 1000, (2, 64))}
    inputs["labels"] = inputs["input_ids"].clone()

    collector.start_metric("throughput", inputs)
    time.sleep(0.01)
    collector.stop_metric("throughput", step_number=1)

    assert collector.metric_windows["throughput"].size == 1
    stats = collector.metric_windows["throughput"].get_window_stats()
    assert stats["total_tokens"] == 2 * 64
    assert stats["total_samples"] == 2
    assert stats["total_time"] > 0

    collector.start_gradient_accumulation_cycle()

    for i in range(2):
        collector.start_metric("forward_pass", inputs)
        time.sleep(0.005)
        collector.stop_metric("forward_pass")

        collector.start_metric("backward_pass", inputs)
        time.sleep(0.005)
        collector.stop_metric("backward_pass")

    collector.end_gradient_accumulation_cycle(step_number=2)

    assert collector.metric_windows["forward_pass"].size == 1
    assert collector.metric_windows["backward_pass"].size == 1

    forward_stats = collector.metric_windows["forward_pass"].get_window_stats()
    backward_stats = collector.metric_windows["backward_pass"].get_window_stats()

    tokens_per_call = 2 * 64  # 128 tokens per inputs
    total_calls = 4  # 2 forward + 2 backward calls
    expected_cycle_tokens = tokens_per_call * total_calls  # 512 total tokens processed in cycle
    assert forward_stats["total_tokens"] == expected_cycle_tokens
    assert backward_stats["total_tokens"] == expected_cycle_tokens

    assert forward_stats["total_time"] > 0.008
    assert backward_stats["total_time"] > 0.008

    # Add optimizer step measurement for efficiency calculation
    collector.start_metric("optimizer_step", inputs)
    time.sleep(0.01)
    collector.stop_metric("optimizer_step", step_number=3)

    if collector.metric_windows["total_step"].size > 0:
        collector.start_metric("total_step")
        time.sleep(0.02)
        collector.stop_metric("total_step", step_number=4)

        efficiency_metrics = collector.calculate_metric("efficiency")
        if efficiency_metrics:
            forward_pct = efficiency_metrics.get("train/forward_time_percent", 0)
            backward_pct = efficiency_metrics.get("train/backward_time_percent", 0)
            optimizer_pct = efficiency_metrics.get("train/optimizer_time_percent", 0)
            unaccounted_pct = efficiency_metrics.get("train/overhead_time_percent", 0)
            total_efficiency = efficiency_metrics.get("train/efficiency", 0)

            assert abs((forward_pct + backward_pct + optimizer_pct) - total_efficiency) < 0.1
            assert abs((total_efficiency + unaccounted_pct) - 100.0) < 0.1

    inputs_ctx = {"input_ids": torch.randint(0, 1000, (2, 64))}
    inputs_ctx["labels"] = inputs_ctx["input_ids"].clone()
    initial_size = collector.metric_windows["throughput"].size

    with collector.time_metric("throughput", inputs_ctx, step_number=10):
        time.sleep(0.005)

    assert collector.metric_windows["throughput"].size == initial_size + 1
    latest_stats = collector.metric_windows["throughput"].get_window_stats()
    # Check that new tokens were added (previous tokens + new context manager tokens)
    assert latest_stats["total_tokens"] >= 128 + 128, (
        f"Expected >= {128 + 128} tokens, got {latest_stats['total_tokens']}"
    )

    collector.start_gradient_accumulation_cycle()
    initial_forward_size = collector.metric_windows["forward_pass"].size

    for i in range(2):
        with collector.time_metric("forward_pass", inputs_ctx):
            time.sleep(0.003)

    collector.end_gradient_accumulation_cycle(step_number=11)

    assert collector.metric_windows["forward_pass"].size == initial_forward_size + 1
    forward_stats_ctx = collector.metric_windows["forward_pass"].get_window_stats()
    # Check that tokens from both accumulation cycles are present
    # Previous cycle: 512 tokens, new cycle: 256 tokens (2 calls * 128 tokens each)
    expected_min_tokens = 512 + 256
    assert forward_stats_ctx["total_tokens"] >= expected_min_tokens, (
        f"Expected >= {expected_min_tokens} tokens, got {forward_stats_ctx['total_tokens']}"
    )


@is_trainium_test
@distributed_test(world_size=8, tp_size=2, pp_size=4)
def test_metrics_calculations_accuracy(tmpdir):
    args = NeuronTrainingArguments(
        output_dir=tmpdir,
        enable_throughput_metrics=True,
        enable_mfu_metrics=True,
        enable_efficiency_metrics=True,
        metrics_window_size=3,
    )

    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, args.trn_config)
    collector = TrainingMetricsCollector(model, args)

    inputs = {"input_ids": torch.randint(0, 1000, (4, 32))}
    inputs["labels"] = inputs["input_ids"].clone()

    for step in range(3):
        collector.start_metric("throughput", inputs)
        start_time = time.perf_counter()
        time.sleep(0.02)
        elapsed = time.perf_counter() - start_time
        collector.stop_metric("throughput", step_number=step)

        window_stats = collector.metric_windows["throughput"].get_window_stats()
        if step == 0:
            assert window_stats["total_tokens"] == 128
            assert abs(window_stats["total_time"] - elapsed) < 0.005

    throughput_metrics = collector.calculate_metric("throughput")
    assert "train/tokens_per_sec" in throughput_metrics

    window_stats = collector.metric_windows["throughput"].get_window_stats()
    expected_local_rate = window_stats["total_tokens"] / window_stats["total_time"]
    expected_global_rate = expected_local_rate * collector.dp_size
    actual_global_rate = throughput_metrics["train/tokens_per_sec"]

    relative_error = abs(actual_global_rate - expected_global_rate) / expected_global_rate
    assert relative_error < 0.05, (
        f"Distributed scaling validation failed: expected {expected_global_rate:.2f}, got {actual_global_rate:.2f} (dp_size={collector.dp_size}, error={relative_error:.3f})"
    )

    if args.enable_mfu_metrics and collector.model_params:
        mfu_metrics = collector.calculate_metric("mfu")
        if mfu_metrics:
            total_tokens = window_stats["total_tokens"]
            total_time = window_stats["total_time"]
            theoretical_flops = 18 * collector.model_params * total_tokens
            actual_flops_per_sec = theoretical_flops / total_time
            peak_flops_per_sec = collector.peak_tflops_per_core * 1e12 * collector.total_neuron_cores
            expected_mfu = (actual_flops_per_sec / peak_flops_per_sec) * 100

            mfu_diff = abs(mfu_metrics["train/mfu"] - round(expected_mfu, 2))
            assert mfu_diff < 0.1, (
                f"MFU calculation failed: expected {round(expected_mfu, 2):.2f}%, got {mfu_metrics['train/mfu']:.2f}% (diff={mfu_diff:.3f}, cores={collector.total_neuron_cores})"
            )

    summary_metrics = collector.calculate_summary_metrics()
    if "summary/tokens_per_sec_avg" in summary_metrics:
        throughput_data = collector.summary_metrics["throughput"]
        if throughput_data["step_times"]:
            manual_rates = [
                (tokens / time * collector.dp_size)
                for tokens, time in zip(throughput_data["tokens_per_step"], throughput_data["step_times"])
                if time > 0
            ]
            expected_summary = sum(manual_rates) / len(manual_rates)
            actual_summary = summary_metrics["summary/tokens_per_sec_avg"]
            assert abs(actual_summary - expected_summary) < 0.01


@is_trainium_test
@distributed_test(world_size=2, tp_size=1, pp_size=1)
def test_metrics_integration_with_trainer(tmpdir):
    args = NeuronTrainingArguments(
        output_dir=tmpdir,
        do_train=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        logging_steps=1,
        enable_throughput_metrics=True,
        enable_efficiency_metrics=True,
        metrics_logging_steps=1,
        metrics_window_size=2,
        save_strategy="no",
    )

    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, args.trn_config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=32)

    train_dataset = Dataset.from_dict({"text": ["Hello world this is a test"] * 8}).map(
        tokenize_function, batched=True
    )

    trainer = NeuronTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    assert hasattr(trainer, "metrics_collector")
    assert trainer.metrics_collector.enabled

    trainer.train()

    collector = trainer.metrics_collector
    summary_metrics = collector.calculate_summary_metrics()

    assert len(summary_metrics) > 0
    if "summary/tokens_per_sec_avg" in summary_metrics:
        throughput_data = collector.summary_metrics.get("throughput", {})
        if throughput_data.get("step_times") and throughput_data.get("tokens_per_step"):
            manual_rates = []
            for tokens, time in zip(throughput_data["tokens_per_step"], throughput_data["step_times"]):
                if time > 0:
                    manual_rates.append((tokens / time) * collector.dp_size)

            if manual_rates:
                expected_avg = sum(manual_rates) / len(manual_rates)
                actual_avg = summary_metrics["summary/tokens_per_sec_avg"]
                assert abs(actual_avg - expected_avg) < 0.01, f"Expected {expected_avg}, got {actual_avg}"

    if "summary/total_steps" in summary_metrics:
        total_steps = summary_metrics["summary/total_steps"]
        expected_steps = len(train_dataset) // (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * collector.dp_size
        )
        assert total_steps == expected_steps, (
            f"Expected {expected_steps} steps, got {total_steps} (dp_size={collector.dp_size})"
        )

    if "summary/total_tokens_processed" in summary_metrics:
        total_tokens = summary_metrics["summary/total_tokens_processed"]
        expected_tokens = len(train_dataset) * 32
        assert total_tokens == expected_tokens, f"Expected {expected_tokens} tokens, got {total_tokens}"
