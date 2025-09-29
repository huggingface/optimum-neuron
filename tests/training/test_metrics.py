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

import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

from optimum.neuron.trainers.metrics import TrainingMetricsCollector
from optimum.neuron.trainers.training_args import NeuronTrainingArguments
from optimum.neuron.utils.testing_utils import is_trainium_test


def create_mock_model():
    """Create a mock model with basic attributes for testing."""
    model = MagicMock()
    model.__class__.__name__ = "TestModel"
    return model


def create_test_training_args(
    enable_training_metrics=True,
    enable_throughput_metrics=True,
    enable_mfu_metrics=False,
    enable_efficiency_metrics=False,
    metrics_logging_steps=None,
    logging_steps=10,
):
    """Create test training arguments with specified metrics configuration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        return NeuronTrainingArguments(
            output_dir=tmp_dir,
            enable_training_metrics=enable_training_metrics,
            enable_throughput_metrics=enable_throughput_metrics,
            enable_mfu_metrics=enable_mfu_metrics,
            enable_efficiency_metrics=enable_efficiency_metrics,
            metrics_logging_steps=metrics_logging_steps,
            logging_steps=logging_steps,
        )


def create_sample_batch(batch_size=2, seq_length=128):
    """Create a sample batch for testing metrics recording."""
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask": torch.ones((batch_size, seq_length)),
        "labels": torch.randint(0, 1000, (batch_size, seq_length)),
    }


class TestTrainingMetricsCollector:
    """Test suite for TrainingMetricsCollector class."""

    def test_metrics_collector_initialization(self):
        """Test that TrainingMetricsCollector initializes correctly with different configurations."""
        model = create_mock_model()
        args = create_test_training_args()

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=2), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=2), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_model_param_count', return_value=1000000):

            collector = TrainingMetricsCollector(model, args)

            assert collector.dp_size == 2
            assert collector.tp_size == 2
            assert collector.pp_size == 1
            assert collector.total_neuron_cores == 4
            assert collector.args == args
            assert collector.model == model

    def test_metrics_collector_disabled(self):
        """Test that metrics collection can be disabled."""
        model = create_mock_model()
        args = create_test_training_args(enable_training_metrics=False)

        # This should not cause any issues even if metrics are disabled
        # The actual disabling happens at the trainer level, but the collector should handle gracefully
        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)
            assert collector is not None

    def test_record_batch_metrics_basic(self):
        """Test basic batch metrics recording functionality."""
        model = create_mock_model()
        args = create_test_training_args()

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Record some batches
            batch = create_sample_batch(batch_size=4, seq_length=128)
            collector.record_batch_metrics(batch, step=1)

            # Check that metrics were recorded
            assert collector.total_tokens_processed == 4 * 128  # batch_size * seq_length
            assert collector.total_samples_processed == 4
            assert collector.step_count == 1

    def test_record_batch_metrics_accumulation(self):
        """Test that batch metrics accumulate correctly over multiple batches."""
        model = create_mock_model()
        args = create_test_training_args()

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Record multiple batches
            batch1 = create_sample_batch(batch_size=2, seq_length=64)
            batch2 = create_sample_batch(batch_size=3, seq_length=128)

            collector.record_batch_metrics(batch1, step=1)
            collector.record_batch_metrics(batch2, step=2)

            # Check accumulated metrics
            expected_tokens = (2 * 64) + (3 * 128)  # 128 + 384 = 512
            expected_samples = 2 + 3  # 5

            assert collector.total_tokens_processed == expected_tokens
            assert collector.total_samples_processed == expected_samples
            assert collector.step_count == 2

    def test_throughput_metrics_calculation(self):
        """Test throughput metrics calculation."""
        model = create_mock_model()
        args = create_test_training_args(enable_throughput_metrics=True)

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=2), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=2), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Simulate some processing
            collector.total_tokens_processed = 1000
            collector.total_samples_processed = 10

            # Test throughput calculation
            elapsed_time = 2.0  # 2 seconds
            metrics = collector.calculate_throughput_metrics(elapsed_time)

            # Expected values
            expected_tokens_per_sec = 1000 / 2.0  # 500
            expected_samples_per_sec = 10 / 2.0   # 5
            expected_tokens_per_core = 1000 / (2.0 * 4)  # 125 (4 cores total)
            expected_samples_per_core = 10 / (2.0 * 4)    # 1.25

            assert metrics["tokens_per_sec"] == expected_tokens_per_sec
            assert metrics["samples_per_sec"] == expected_samples_per_sec
            assert metrics["tokens_per_sec_per_neuron_core"] == expected_tokens_per_core
            assert metrics["samples_per_sec_per_neuron_core"] == expected_samples_per_core

    def test_mfu_metrics_calculation(self):
        """Test Model FLOPs Utilization (MFU) metrics calculation."""
        model = create_mock_model()
        args = create_test_training_args(enable_mfu_metrics=True, peak_tflops_per_core=100.0)

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_model_param_count', return_value=1000000):

            collector = TrainingMetricsCollector(model, args)

            # Simulate processing
            collector.total_tokens_processed = 1000
            elapsed_time = 1.0

            metrics = collector.calculate_mfu_metrics(elapsed_time)

            # Check that MFU metrics are present
            assert "model_flops_utilization" in metrics
            assert "theoretical_flops_per_sec" in metrics
            assert "peak_flops_per_sec" in metrics

            # Verify peak FLOPs calculation (1 core * 100 TFLOPS)
            expected_peak_flops = 100.0 * 1e12
            assert metrics["peak_flops_per_sec"] == expected_peak_flops

    def test_should_calculate_metrics(self):
        """Test the logic for when metrics should be calculated."""
        model = create_mock_model()
        args = create_test_training_args(logging_steps=5, metrics_logging_steps=None)

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Test various steps
            assert not collector.should_calculate_metrics(0)  # Step 0 should not calculate
            assert not collector.should_calculate_metrics(1)  # Step 1, not divisible by 5
            assert not collector.should_calculate_metrics(4)  # Step 4, not divisible by 5
            assert collector.should_calculate_metrics(5)     # Step 5, divisible by 5
            assert collector.should_calculate_metrics(10)    # Step 10, divisible by 5

    def test_should_calculate_metrics_custom_logging_steps(self):
        """Test metrics calculation with custom metrics_logging_steps."""
        model = create_mock_model()
        args = create_test_training_args(logging_steps=5, metrics_logging_steps=3)

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Should use metrics_logging_steps=3 instead of logging_steps=5
            assert not collector.should_calculate_metrics(2)
            assert collector.should_calculate_metrics(3)
            assert not collector.should_calculate_metrics(4)
            assert not collector.should_calculate_metrics(5)  # Not divisible by 3
            assert collector.should_calculate_metrics(6)

    def test_reset_window(self):
        """Test that reset_window properly clears metrics."""
        model = create_mock_model()
        args = create_test_training_args()

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Record some metrics
            batch = create_sample_batch()
            collector.record_batch_metrics(batch, step=1)

            # Verify metrics are recorded
            assert collector.total_tokens_processed > 0
            assert collector.total_samples_processed > 0
            assert collector.step_count > 0

            # Reset and verify
            collector.reset_window()
            assert collector.total_tokens_processed == 0
            assert collector.total_samples_processed == 0
            assert collector.step_count == 0
            assert collector.window_start_time is None
            assert collector.window_start_step is None

    def test_calculate_metrics_integration(self):
        """Test end-to-end metrics calculation."""
        model = create_mock_model()
        args = create_test_training_args(
            enable_throughput_metrics=True,
            enable_mfu_metrics=True,
            enable_efficiency_metrics=True
        )

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=2), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_model_param_count', return_value=500000):

            collector = TrainingMetricsCollector(model, args)

            # Simulate batch processing
            collector.start_timing_window(1)
            time.sleep(0.1)  # Small delay to have measurable time

            batch1 = create_sample_batch(batch_size=2, seq_length=64)
            batch2 = create_sample_batch(batch_size=2, seq_length=64)
            collector.record_batch_metrics(batch1, step=1)
            collector.record_batch_metrics(batch2, step=2)

            # Calculate all metrics
            metrics = collector.calculate_metrics()

            # Verify that metrics are present
            assert "tokens_per_sec" in metrics
            assert "samples_per_sec" in metrics
            assert "tokens_per_sec_per_neuron_core" in metrics
            assert "samples_per_sec_per_neuron_core" in metrics
            assert "model_flops_utilization" in metrics
            assert "training_efficiency" in metrics
            assert "metrics_window_steps" in metrics
            assert "metrics_window_duration" in metrics

            # Verify values are reasonable
            assert metrics["tokens_per_sec"] > 0
            assert metrics["samples_per_sec"] > 0
            assert metrics["metrics_window_steps"] == 2
            assert metrics["metrics_window_duration"] >= 0.1


@is_trainium_test
class TestTrainingMetricsIntegration:
    """Integration tests that require Trainium hardware."""

    def test_metrics_with_real_parallelism(self):
        """Test metrics calculation with real distributed training setup."""
        # This test would run on actual Trainium hardware with real parallelism
        # For now, it's a placeholder that could be expanded when running on hardware
        pass

    def test_metrics_with_real_model(self):
        """Test metrics with a real model and training setup."""
        # This test would use a real model and verify metrics are calculated correctly
        # during actual training steps on Trainium hardware
        pass


class TestMetricsConfiguration:
    """Test metrics configuration in NeuronTrainingArguments."""

    def test_default_metrics_configuration(self):
        """Test default metrics configuration values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = NeuronTrainingArguments(output_dir=tmp_dir)

            assert args.enable_training_metrics is True
            assert args.enable_throughput_metrics is True
            assert args.enable_mfu_metrics is False
            assert args.enable_efficiency_metrics is False
            assert args.metrics_logging_steps is None
            assert args.peak_tflops_per_core == 100.0

    def test_custom_metrics_configuration(self):
        """Test custom metrics configuration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = NeuronTrainingArguments(
                output_dir=tmp_dir,
                enable_training_metrics=False,
                enable_throughput_metrics=False,
                enable_mfu_metrics=True,
                enable_efficiency_metrics=True,
                metrics_logging_steps=20,
                peak_tflops_per_core=150.0
            )

            assert args.enable_training_metrics is False
            assert args.enable_throughput_metrics is False
            assert args.enable_mfu_metrics is True
            assert args.enable_efficiency_metrics is True
            assert args.metrics_logging_steps == 20
            assert args.peak_tflops_per_core == 150.0


if __name__ == "__main__":
    pytest.main([__file__])