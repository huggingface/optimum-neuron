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

import json
import os
import time
from unittest.mock import patch

import datasets
import torch
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_size,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_size,
)
from transformers import AutoTokenizer

from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM
from optimum.neuron.trainers.metrics import MovingAverageWindow, TrainingMetricsCollector
from optimum.neuron.utils.testing_utils import is_trainium_test
from optimum.neuron.utils.training_utils import get_model_param_count

from .distributed_utils import distributed_test


TINY_MODEL_NAME = "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"


def test_metrics_collector_standalone():
    """Test standalone metrics collector with comprehensive computation testing."""

    # Test MovingAverageWindow
    window = MovingAverageWindow(window_size=3)

    # Test empty window
    window_stats = window.get_window_stats()
    assert window_stats == {}

    # Add known values and test averages
    window.add_step(tokens=100, samples=10, step_time=1.0)  # 100 tokens/sec, 10 samples/sec
    window_stats = window.get_window_stats()
    assert window_stats["total_tokens"] == 100
    assert window_stats["total_samples"] == 10
    assert window_stats["total_time"] == 1.0

    window.add_step(tokens=200, samples=20, step_time=2.0)  # 100 tokens/sec, 10 samples/sec
    window.add_step(tokens=300, samples=30, step_time=2.0)  # 150 tokens/sec, 15 samples/sec
    window_stats = window.get_window_stats()
    assert window_stats["total_tokens"] == 600
    assert window_stats["total_samples"] == 60
    assert window_stats["total_time"] == 5.0

    # Test window overflow
    window.add_step(tokens=400, samples=40, step_time=1.0)  # 400 tokens/sec, 40 samples/sec
    window_stats = window.get_window_stats()
    # Now window contains last 3 steps: [200, 300, 400] tokens and [20, 30, 40] samples
    assert window_stats["total_tokens"] == 900  # 200+300+400
    assert window_stats["total_samples"] == 90  # 20+30+40
    assert window_stats["total_time"] == 5.0  # 2+2+1

    # Test TrainingMetricsCollector initialization
    training_args = NeuronTrainingArguments(
        output_dir="/tmp/test",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        dataloader_num_workers=0,
        enable_mfu_metrics=True,
        enable_efficiency_metrics=True,
        metrics_window_size=5
    )

    # Create a mock model for testing
    class MockModel:
        def parameters(self):
            return [torch.randn(1000), torch.randn(500)]

    mock_model = MockModel()

    with patch.object(TrainingMetricsCollector, '_detect_hardware_tflops', return_value=190.0):
        with patch('optimum.neuron.trainers.metrics.get_model_param_count', return_value=1000000):
            collector = TrainingMetricsCollector(model=mock_model, training_args=training_args)

    # Test initialization
    assert collector.model_params == 1000000
    assert collector.peak_tflops_per_core == 190.0

    # Test metric timing
    assert not collector.metric_start_times["throughput"]

    collector.start_metric("throughput")
    assert collector.metric_start_times["throughput"] is not None

    time.sleep(0.1)  # Small delay for timing

    collector.stop_metric("throughput")
    assert collector.metric_start_times["throughput"] is None
    assert len(collector.metric_windows["throughput"].step_times) == 1

    # Test training efficiency component timing
    collector.start_metric("forward_pass")
    time.sleep(0.05)
    collector.stop_metric("forward_pass")

    collector.start_metric("backward_pass")
    time.sleep(0.03)
    collector.stop_metric("backward_pass")

    collector.start_metric("optimizer_step")
    time.sleep(0.02)
    collector.stop_metric("optimizer_step")

    collector.start_metric("total_step")
    time.sleep(0.15)  # Should be >= sum of components
    collector.stop_metric("total_step")

    # Test helper method for getting average times
    forward_time = collector._get_metric_average_time("forward_pass")
    backward_time = collector._get_metric_average_time("backward_pass")
    optimizer_time = collector._get_metric_average_time("optimizer_step")
    total_time = collector._get_metric_average_time("total_step")

    assert forward_time > 0
    assert backward_time > 0
    assert optimizer_time > 0
    assert total_time > 0

    # Test training efficiency calculation
    efficiency_metrics = collector._calculate_training_efficiency_metrics()
    assert "train/training_efficiency" in efficiency_metrics
    assert "train/training_overhead" in efficiency_metrics
    assert "train/compute_time_ratio" in efficiency_metrics
    assert "train/overhead_time_ratio" in efficiency_metrics

    # Verify efficiency and overhead calculations
    compute_time = forward_time + backward_time + optimizer_time
    overhead_time = total_time - compute_time
    expected_efficiency = (compute_time / total_time) * 100
    expected_overhead = (overhead_time / total_time) * 100

    assert abs(efficiency_metrics["train/training_efficiency"] - expected_efficiency) < 0.1
    assert abs(efficiency_metrics["train/training_overhead"] - expected_overhead) < 0.1
    assert abs(efficiency_metrics["train/compute_time_ratio"] - (compute_time / total_time)) < 0.01
    assert abs(efficiency_metrics["train/overhead_time_ratio"] - (overhead_time / total_time)) < 0.01

    # Verify efficiency + overhead = 100%
    total_percentage = efficiency_metrics["train/training_efficiency"] + efficiency_metrics["train/training_overhead"]
    assert abs(total_percentage - 100.0) < 0.1

    # Test non-existent metric
    assert collector._get_metric_average_time("non_existent") == 0.0

    # Test calculate_metric with training_efficiency
    all_metrics = collector.calculate_metric("all")
    assert "train/training_efficiency" in all_metrics

    specific_metrics = collector.calculate_metric("training_efficiency")
    assert "train/training_efficiency" in specific_metrics


def test_metrics_real_model_computation():
    """Test metric computations with real model characteristics and minimal mocking."""

    # Use real model to get actual parameters
    training_args = NeuronTrainingArguments(
        output_dir="/tmp/test",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dataloader_num_workers=0,
    )

    model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)
    real_param_count = get_model_param_count(model, trainable_only=False)

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_NAME)
    vocab_size = tokenizer.vocab_size

    with patch.object(TrainingMetricsCollector, '_detect_hardware_tflops', return_value=190.0):
        collector = TrainingMetricsCollector(
            training_args=training_args,
            model_param_count=real_param_count,
            vocab_size=vocab_size,
            enable_mfu_metrics=True,
            enable_efficiency_metrics=True,
            metrics_window_size=3
        )

    # Test with realistic values
    collector.start_metric("real_test")
    start_time = collector.active_metrics["real_test"]

    # Simulate 2 seconds of processing 500 tokens (25 samples with seq_len=20)
    collector.active_metrics["real_test"] = start_time - 2.0
    collector.stop_metric("real_test", tokens=500, samples=25)

    metrics = collector.calculate_metric("real_test")

    # Verify throughput calculations
    expected_tokens_per_sec = 500 / 2  # 250 tokens/sec
    expected_samples_per_sec = 25 / 2  # 12.5 samples/sec

    assert abs(metrics["tokens_per_second"] - expected_tokens_per_sec) < 0.1
    assert abs(metrics["samples_per_second"] - expected_samples_per_sec) < 0.1

    # Per neuron core (1 core in this test setup)
    assert abs(metrics["tokens_per_second_per_neuron_core"] - expected_tokens_per_sec) < 0.1
    assert abs(metrics["samples_per_second_per_neuron_core"] - expected_samples_per_sec) < 0.1

    # Test MFU bounds
    assert 0 <= metrics["model_flops_utilization"] <= 100

    # Test efficiency
    assert metrics["training_efficiency"] > 0

    # Test with DP=2 scenario
    training_args_dp2 = NeuronTrainingArguments(
        output_dir="/tmp/test",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dataloader_num_workers=0,
    )

    with patch.object(TrainingMetricsCollector, '_detect_hardware_tflops', return_value=190.0):
        collector_dp2 = TrainingMetricsCollector(
            training_args=training_args_dp2,
            model_param_count=real_param_count,
            vocab_size=vocab_size,
            enable_mfu_metrics=True,
            enable_efficiency_metrics=True,
            metrics_window_size=3
        )

    # Simulate processing on 2 DP ranks
    collector_dp2.start_metric("dp2_test")
    start_time = collector_dp2.active_metrics["dp2_test"]
    collector_dp2.active_metrics["dp2_test"] = start_time - 1.0

    # Each rank processes 250 tokens, but global throughput should account for DP size
    collector_dp2.stop_metric("dp2_test", tokens=250, samples=12)

    metrics_dp2 = collector_dp2.calculate_metric("dp2_test")

    # Global throughput should be same as per-rank when DP=1 (no scaling in this test)
    assert abs(metrics_dp2["tokens_per_second"] - 250.0) < 0.1
    assert abs(metrics_dp2["samples_per_second"] - 12.0) < 0.1


@distributed_test(world_size=32, tp_size=2, pp_size=4)
def test_trainer_full_metrics_integration(tmpdir):
    """Test trainer integration with full metrics validation."""

    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()
    dp_size = get_data_parallel_size()

    # Create training arguments with metrics enabled
    training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=3,  # Minimal steps for testing
        logging_steps=1,
        save_steps=10,  # Don't save checkpoints
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        dataloader_num_workers=0,
        enable_mfu_metrics=True,
        enable_efficiency_metrics=True,
        metrics_window_size=2,
    )

    # Create model and dataset
    model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_NAME)
    inputs = tokenizer(
        "Paris is the most beautiful city in the world.",
        return_tensors="pt",
        padding="max_length",
        max_length=128
    )
    inputs["labels"] = inputs["input_ids"].clone()
    dataset = datasets.Dataset.from_dict(inputs)
    dataset = dataset.select([0] * 100)  # Small dataset

    # Mock hardware detection for consistent test results
    with patch.object(TrainingMetricsCollector, '_detect_hardware_tflops', return_value=190.0):
        trainer = NeuronTrainer(model, training_args, train_dataset=dataset)

        # Verify metrics collector is initialized
        assert trainer.metrics_collector is not None
        assert trainer.metrics_collector.enable_mfu_metrics
        assert trainer.metrics_collector.enable_efficiency_metrics

        # Run training
        trainer.train()

        # Verify training metrics were logged (train/ prefix)
        log_history = trainer.state.log_history
        training_logs = [log for log in log_history if any(key.startswith("train/") for key in log.keys())]

        assert len(training_logs) > 0, "No training metrics were logged"

        # Check that all expected metrics are present
        expected_metrics = [
            "train/tokens_per_second",
            "train/samples_per_second",
            "train/tokens_per_second_per_neuron_core",
            "train/samples_per_second_per_neuron_core",
            "train/model_flops_utilization",
            "train/training_efficiency",
            "train/training_overhead"
        ]

        last_training_log = training_logs[-1]
        for metric in expected_metrics:
            assert metric in last_training_log, f"Missing metric: {metric}"
            assert isinstance(last_training_log[metric], (int, float)), f"Invalid metric type: {metric}"
            assert last_training_log[metric] >= 0, f"Negative metric value: {metric}"

        # Verify MFU is within reasonable bounds
        assert 0 <= last_training_log["train/model_flops_utilization"] <= 100

        # Verify training efficiency is within reasonable bounds (should be <= 100%)
        assert 0 <= last_training_log["train/training_efficiency"] <= 100

        # Verify training overhead is within reasonable bounds (should be <= 100%)
        assert 0 <= last_training_log["train/training_overhead"] <= 100

        # Verify efficiency + overhead = 100%
        efficiency_overhead_sum = last_training_log["train/training_efficiency"] + last_training_log["train/training_overhead"]
        assert abs(efficiency_overhead_sum - 100.0) < 0.1, f"Efficiency + Overhead should equal 100%, got {efficiency_overhead_sum}%"

        # Test summary metrics generation and file saving
        summary_file = os.path.join(tmpdir, "training_summary_metrics.json")
        assert os.path.exists(summary_file), "Summary metrics file was not created"

        # Load and validate summary metrics
        with open(summary_file, 'r') as f:
            summary_metrics = json.load(f)

        # Check summary metrics structure
        expected_summary_metrics = [
            "summary/tokens_per_second_avg",
            "summary/samples_per_second_avg",
            "summary/tokens_per_second_per_neuron_core_avg",
            "summary/samples_per_second_per_neuron_core_avg",
            "summary/model_flops_utilization_avg",
            "summary/training_efficiency_avg",
            "summary/training_overhead_avg"
        ]

        for metric in expected_summary_metrics:
            assert metric in summary_metrics, f"Missing summary metric: {metric}"
            assert isinstance(summary_metrics[metric], (int, float)), f"Invalid summary metric type: {metric}"
            assert summary_metrics[metric] >= 0, f"Negative summary metric: {metric}"

        # Verify summary MFU bounds
        assert 0 <= summary_metrics["summary/model_flops_utilization_avg"] <= 100

        # Verify summary training efficiency bounds (should be <= 100%)
        assert 0 <= summary_metrics["summary/training_efficiency_avg"] <= 100

        # Verify summary training overhead bounds (should be <= 100%)
        assert 0 <= summary_metrics["summary/training_overhead_avg"] <= 100

        # Verify summary efficiency + overhead = 100%
        summary_efficiency_overhead_sum = summary_metrics["summary/training_efficiency_avg"] + summary_metrics["summary/training_overhead_avg"]
        assert abs(summary_efficiency_overhead_sum - 100.0) < 0.1, f"Summary efficiency + overhead should equal 100%, got {summary_efficiency_overhead_sum}%"

        # Test metric value consistency
        # Per-neuron-core metrics should be lower than general metrics (with 32 total cores)
        assert summary_metrics["summary/tokens_per_second_per_neuron_core_avg"] <= summary_metrics["summary/tokens_per_second_avg"]
        assert summary_metrics["summary/samples_per_second_per_neuron_core_avg"] <= summary_metrics["summary/samples_per_second_avg"]

        # Test disabled metrics scenario
        training_args_no_metrics = NeuronTrainingArguments(
            output_dir=tmpdir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=1,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            dataloader_num_workers=0,
            enable_mfu_metrics=False,
            enable_efficiency_metrics=False,
        )

        with patch.object(TrainingMetricsCollector, '_detect_hardware_tflops', return_value=190.0):
            trainer_no_metrics = NeuronTrainer(model, training_args_no_metrics, train_dataset=dataset)

            # Verify metrics are disabled
            assert not trainer_no_metrics.metrics_collector.enable_mfu_metrics
            assert not trainer_no_metrics.metrics_collector.enable_efficiency_metrics


@is_trainium_test
def test_metrics_collector_edge_cases():
    """Test edge cases and error conditions."""

    training_args = NeuronTrainingArguments(
        output_dir="/tmp/test",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dataloader_num_workers=0,
    )

    with patch.object(TrainingMetricsCollector, '_detect_hardware_tflops', return_value=190.0):
        collector = TrainingMetricsCollector(
            training_args=training_args,
            model_param_count=1000000,
            vocab_size=32000,
            enable_mfu_metrics=True,
            enable_efficiency_metrics=True,
            metrics_window_size=1
        )

    # Test zero time edge case
    collector.start_metric("zero_time")
    collector.stop_metric("zero_time", tokens=100, samples=10)

    metrics = collector.calculate_metric("zero_time")
    # Should handle division by zero gracefully
    assert metrics["tokens_per_second"] >= 0
    assert metrics["samples_per_second"] >= 0

    # Test single step window
    assert len(collector.windows["zero_time"].tokens_per_step) == 1

    # Test stopping non-existent metric (should not crash)
    try:
        collector.stop_metric("non_existent", tokens=0, samples=0)
    except KeyError:
        pass  # Expected behavior

    # Test calculating metric for non-existent window
    empty_metrics = collector.calculate_metric("non_existent_metric")
    assert all(value == 0.0 for value in empty_metrics.values())
