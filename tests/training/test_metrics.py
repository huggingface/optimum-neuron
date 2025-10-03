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
from unittest.mock import Mock, patch

import pytest
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer

from optimum.neuron.models.training.config import TrainingNeuronConfig
from optimum.neuron.models.training.llama.modeling_llama import LlamaForCausalLM
from optimum.neuron.trainers import NeuronTrainer
from optimum.neuron.trainers.metrics import MovingAverageWindow, TrainingMetricsCollector
from optimum.neuron.trainers.training_args import NeuronTrainingArguments
from optimum.neuron.utils.testing_utils import is_trainium_test

from .distributed_utils import distributed_test
from .utils import MODEL_NAME

@is_trainium_test
@distributed_test(world_size=1, tp_size=1, pp_size=1)
def test_metrics_calculations_with_mocks(tmpdir):
    # Mock all distributed and hardware dependencies
    with patch("optimum.neuron.trainers.metrics.get_data_parallel_size", return_value=2), \
         patch("torch_xla.runtime.world_size", return_value=4), \
         patch("optimum.neuron.trainers.metrics.get_model_param_count", return_value=1000000), \
         patch.dict("os.environ", {"NEURON_HARDWARE_TYPE": "trn1"}):

        args = NeuronTrainingArguments(
            output_dir=tmpdir,
            enable_throughput_metrics=True,
            enable_mfu_metrics=True,
            enable_efficiency_metrics=True,
            metrics_window_size=3,
        )

        model = Mock()
        collector = TrainingMetricsCollector(model, args)

        # Test MovingAverageWindow
        window = MovingAverageWindow(window_size=2)
        window.add_step(tokens=1000, samples=4, step_time=0.5)
        window.add_step(tokens=2000, samples=8, step_time=1.0)

        stats = window.get_window_stats()
        assert stats["total_tokens"] == 3000
        assert stats["total_samples"] == 12
        assert stats["total_time"] == 1.5
        assert stats["avg_tokens_per_step"] == 1500
        assert stats["avg_time_per_step"] == 0.75

        # Test throughput calculations
        collector.metric_windows["throughput"].add_step(tokens=1000, samples=4, step_time=0.5)
        throughput_metrics = collector._calculate_throughput_metrics_from_window("throughput")

        # Expected: 1000 tokens / 0.5 sec = 2000 tokens/sec local, * 2 (DP) = 4000 global
        assert throughput_metrics["train/global_tokens_per_sec"] == 4000.0
        assert throughput_metrics["train/local_step_time"] == 0.5

        # Test MFU calculations
        collector.metric_windows["mfu"].add_step(tokens=1000, samples=4, step_time=0.5)
        mfu_metrics = collector._calculate_mfu_metrics_from_window("mfu")

        # Expected: 18 * 1M params * 1000 tokens = 18B theoretical FLOPs
        # 18B FLOPs / 0.5 sec = 36 GFLOPS/sec actual
        # TRN1: 95 TFLOPS/core * 4 cores = 380 TFLOPS = 380e12 FLOPS/sec peak
        # MFU = (36e9 / 380e12) * 100 = 0.0095%
        expected_mfu = (18 * 1000000 * 1000 / 0.5) / (95 * 1e12 * 4) * 100
        assert abs(mfu_metrics["train/mfu"] - round(expected_mfu, 2)) < 0.01

        # Test training efficiency calculations
        collector.metric_windows["forward_pass"].add_step(tokens=1000, samples=4, step_time=0.1)
        collector.metric_windows["backward_pass"].add_step(tokens=1000, samples=4, step_time=0.2)
        collector.metric_windows["optimizer_step"].add_step(tokens=1000, samples=4, step_time=0.05)
        collector.metric_windows["total_step"].add_step(tokens=1000, samples=4, step_time=0.4)

        efficiency_metrics = collector._calculate_training_efficiency_metrics()

        # Expected: (0.1 + 0.2 + 0.05) / 0.4 = 0.35 / 0.4 = 87.5%
        assert efficiency_metrics["train/training_efficiency"] == 87.5
        assert efficiency_metrics["train/forward_time_percent"] == 25.0  # 0.1/0.4 * 100
        assert efficiency_metrics["train/backward_time_percent"] == 50.0  # 0.2/0.4 * 100
        assert efficiency_metrics["train/optimizer_time_percent"] == 12.5  # 0.05/0.4 * 100
        assert efficiency_metrics["train/unaccounted_time_percent"] == 12.5  # 100 - 87.5



@is_trainium_test
@distributed_test(world_size=4, tp_size=2, pp_size=1)
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

    # Test basic metric lifecycle
    inputs = {"input_ids": torch.randint(0, 1000, (2, 64))}

    collector.start_metric("throughput", inputs)
    time.sleep(0.01)  # Simulate work
    collector.stop_metric("throughput", step_number=1)

    # Verify metric was recorded
    assert collector.metric_windows["throughput"].size == 1
    stats = collector.metric_windows["throughput"].get_window_stats()
    assert stats["total_tokens"] == 128  # 2 * 64
    assert stats["total_samples"] == 2
    assert stats["total_time"] > 0

    # Test gradient accumulation cycle
    collector.start_gradient_accumulation_cycle()

    # Accumulate multiple forward/backward passes
    for i in range(2):
        collector.start_metric("forward_pass", inputs)
        time.sleep(0.005)
        collector.stop_metric("forward_pass")

        collector.start_metric("backward_pass", inputs)
        time.sleep(0.005)
        collector.stop_metric("backward_pass")

    collector.end_gradient_accumulation_cycle(step_number=2)

    # Verify accumulation worked
    assert collector.metric_windows["forward_pass"].size == 1
    assert collector.metric_windows["backward_pass"].size == 1

    forward_stats = collector.metric_windows["forward_pass"].get_window_stats()
    backward_stats = collector.metric_windows["backward_pass"].get_window_stats()

    # Should have accumulated tokens from both micro-batches
    assert forward_stats["total_tokens"] == 256  # 2 * 128
    assert backward_stats["total_tokens"] == 256

    # Times should be cumulative
    assert forward_stats["total_time"] > 0.008  # Should be > 2 * 0.005
    assert backward_stats["total_time"] > 0.008



@is_trainium_test
@distributed_test(world_size=4, tp_size=2, pp_size=2)
def test_metrics_calculations_accuracy():
    """Test calculation accuracy in distributed setting."""
    args = NeuronTrainingArguments(
        output_dir="test_output",
        enable_throughput_metrics=True,
        enable_mfu_metrics=True,
        enable_efficiency_metrics=True,
        metrics_window_size=3,
    )

    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, args.trn_config)
    collector = TrainingMetricsCollector(model, args)

    # Test with known inputs
    inputs = {"input_ids": torch.randint(0, 1000, (4, 32))}  # 128 tokens

    # Record multiple steps with timing
    for step in range(3):
        collector.start_metric("throughput", inputs)
        start_time = time.perf_counter()
        time.sleep(0.02)  # Controlled timing
        elapsed = time.perf_counter() - start_time
        collector.stop_metric("throughput", step_number=step)

        # Verify the measurement matches expected
        window_stats = collector.metric_windows["throughput"].get_window_stats()
        if step == 0:
            # First measurement
            assert window_stats["total_tokens"] == 128
            assert abs(window_stats["total_time"] - elapsed) < 0.005

    # Test throughput calculation
    throughput_metrics = collector.calculate_metric("throughput")
    assert "train/global_tokens_per_sec" in throughput_metrics
    assert throughput_metrics["train/global_tokens_per_sec"] > 0

    # Test MFU calculation if enabled
    if args.enable_mfu_metrics:
        mfu_metrics = collector.calculate_metric("mfu")
        if mfu_metrics:  # May be empty if insufficient data
            assert "train/mfu" in mfu_metrics
            assert 0 <= mfu_metrics["train/mfu"] <= 100

    # Test summary vs moving window consistency
    summary_metrics = collector.calculate_summary_metrics()
    if "summary/global_tokens_per_sec_avg" in summary_metrics:
        # Summary should be close to current moving average
        window_throughput = throughput_metrics.get("train/global_tokens_per_sec", 0)
        summary_throughput = summary_metrics["summary/global_tokens_per_sec_avg"]
        if window_throughput > 0 and summary_throughput > 0:
            ratio = summary_throughput / window_throughput
            assert 0.5 < ratio < 2.0  # Should be reasonably close



@is_trainium_test
@distributed_test(world_size=2, tp_size=1, pp_size=1)
def test_metrics_integration_with_trainer():
    args = NeuronTrainingArguments(
        output_dir="test_output",
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

    # Create minimal dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=32)

    train_dataset = Dataset.from_dict({
        "text": ["Hello world this is a test"] * 8
    }).map(tokenize_function, batched=True)

    trainer = NeuronTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Verify metrics collector is initialized
    assert hasattr(trainer, "metrics_collector")
    assert trainer.metrics_collector.enabled

    # Run minimal training to test integration
    trainer.train()

    # Verify metrics were collected
    collector = trainer.metrics_collector
    summary_metrics = collector.calculate_summary_metrics()

    # Should have throughput metrics
    assert len(summary_metrics) > 0
    if "summary/global_tokens_per_sec_avg" in summary_metrics:
        assert summary_metrics["summary/global_tokens_per_sec_avg"] > 0

    # Should have training steps recorded
    if "summary/total_training_steps" in summary_metrics:
        assert summary_metrics["summary/total_training_steps"] > 0
