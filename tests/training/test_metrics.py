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

from optimum.neuron.trainers.metrics import TrainingMetricsCollector, MetricsClock, MovingAverageWindow
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


class TestMetricsClock:
    """Test suite for MetricsClock class."""

    def test_wall_time_clock(self):
        """Test wall time clock functionality."""
        clock = MetricsClock('wall_time')
        assert clock.clock_type == 'wall_time'

        # Test initial state
        assert clock.elapsed() is None

        # Test timing
        clock.start()
        time.sleep(0.01)  # Small delay
        elapsed = clock.elapsed()
        assert elapsed is not None
        assert elapsed >= 0.01

        # Test reset
        clock.reset()
        assert clock.elapsed() is None

    def test_process_time_clock(self):
        """Test process time clock functionality."""
        clock = MetricsClock('process_time')
        assert clock.clock_type == 'process_time'

        clock.start()
        # Do some CPU work
        _ = sum(range(1000))
        elapsed = clock.elapsed()
        assert elapsed is not None
        assert elapsed >= 0

    def test_perf_counter_clock(self):
        """Test performance counter clock functionality."""
        clock = MetricsClock('perf_counter')
        assert clock.clock_type == 'perf_counter'

        clock.start()
        time.sleep(0.01)
        elapsed = clock.elapsed()
        assert elapsed is not None
        assert elapsed >= 0.01

    def test_invalid_clock_type(self):
        """Test that invalid clock types raise an error."""
        with pytest.raises(ValueError, match="Unsupported clock type"):
            MetricsClock('invalid_clock')


class TestMovingAverageWindow:
    """Test suite for MovingAverageWindow class."""

    def test_basic_window_functionality(self):
        """Test basic moving window operations."""
        window = MovingAverageWindow(window_size=3)

        # Test initial state
        assert window.size == 0
        assert not window.is_full
        assert window.get_window_stats() == {}

        # Add some steps
        window.add_step(tokens=100, samples=2, step_time=0.5, step_number=1)
        window.add_step(tokens=200, samples=4, step_time=0.6, step_number=2)

        assert window.size == 2
        assert not window.is_full

        # Fill the window
        window.add_step(tokens=150, samples=3, step_time=0.4, step_number=3)
        assert window.size == 3
        assert window.is_full

    def test_window_stats_calculation(self):
        """Test window statistics calculation."""
        window = MovingAverageWindow(window_size=3)

        window.add_step(tokens=100, samples=2, step_time=0.5, step_number=1)
        window.add_step(tokens=200, samples=4, step_time=0.6, step_number=2)
        window.add_step(tokens=300, samples=6, step_time=0.7, step_number=3)

        stats = window.get_window_stats()

        assert stats['total_tokens'] == 600
        assert stats['total_samples'] == 12
        assert stats['total_time'] == 1.8
        assert stats['window_steps'] == 3
        assert stats['avg_tokens_per_step'] == 200.0
        assert stats['avg_samples_per_step'] == 4.0
        assert stats['avg_time_per_step'] == 0.6

    def test_window_overflow(self):
        """Test that window properly handles overflow with maxlen."""
        window = MovingAverageWindow(window_size=2)

        # Add 3 steps to a window of size 2
        window.add_step(tokens=100, samples=2, step_time=0.5, step_number=1)
        window.add_step(tokens=200, samples=4, step_time=0.6, step_number=2)
        window.add_step(tokens=300, samples=6, step_time=0.7, step_number=3)  # Should evict first step

        assert window.size == 2
        assert window.is_full

        stats = window.get_window_stats()
        # Should only contain steps 2 and 3
        assert stats['total_tokens'] == 500  # 200 + 300
        assert stats['total_samples'] == 10  # 4 + 6

    def test_window_clear(self):
        """Test window clearing functionality."""
        window = MovingAverageWindow(window_size=3)

        window.add_step(tokens=100, samples=2, step_time=0.5, step_number=1)
        window.add_step(tokens=200, samples=4, step_time=0.6, step_number=2)

        assert window.size == 2

        window.clear()
        assert window.size == 0
        assert not window.is_full
        assert window.get_window_stats() == {}


class TestTrainingMetricsCollectorEnhanced:
    """Enhanced test suite for TrainingMetricsCollector with moving averages and multiple clocks."""

    def test_moving_window_initialization(self):
        """Test that moving window is properly initialized."""
        model = create_mock_model()
        args = create_test_training_args(metrics_window_size=25)

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            assert collector.window_size == 25
            assert collector.moving_window.window_size == 25
            assert collector.moving_window.size == 0

    def test_multiple_clocks_initialization(self):
        """Test that multiple clocks are properly initialized."""
        model = create_mock_model()
        args = create_test_training_args()

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Check default clocks
            assert 'throughput' in collector.clocks
            assert 'mfu' in collector.clocks
            assert 'efficiency' in collector.clocks

            assert collector.clocks['throughput'].clock_type == 'perf_counter'
            assert collector.clocks['mfu'].clock_type == 'wall_time'
            assert collector.clocks['efficiency'].clock_type == 'process_time'

    def test_custom_clocks_configuration(self):
        """Test custom clock configuration."""
        model = create_mock_model()

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = NeuronTrainingArguments(
                output_dir=tmp_dir,
                metrics_clocks={'custom_clock': 'wall_time', 'another_clock': 'process_time'}
            )

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Check that custom clocks are added
            assert 'custom_clock' in collector.clocks
            assert 'another_clock' in collector.clocks
            assert collector.clocks['custom_clock'].clock_type == 'wall_time'
            assert collector.clocks['another_clock'].clock_type == 'process_time'

    def test_step_timing_and_finalization(self):
        """Test step timing and finalization process."""
        model = create_mock_model()
        args = create_test_training_args()

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Record metrics for step 1
            batch1 = create_sample_batch(batch_size=2, seq_length=64)
            collector.record_batch_metrics(batch1, step=1)

            # Should have started timing for step 1
            assert collector.last_step_number == 1
            assert hasattr(collector, 'current_step_tokens')
            assert collector.current_step_tokens == 2 * 64

            # Small delay to ensure measurable timing
            time.sleep(0.01)

            # Record metrics for step 2 (should finalize step 1)
            batch2 = create_sample_batch(batch_size=3, seq_length=64)
            collector.record_batch_metrics(batch2, step=2)

            # Step 1 should be finalized and added to moving window
            assert collector.moving_window.size == 1
            assert collector.last_step_number == 2

    def test_moving_average_metrics_calculation(self):
        """Test metrics calculation using moving averages."""
        model = create_mock_model()
        args = create_test_training_args(
            enable_throughput_metrics=True,
            metrics_window_size=3
        )

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=2), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Record several steps
            for step in range(1, 4):
                batch = create_sample_batch(batch_size=2, seq_length=128)
                collector.record_batch_metrics(batch, step=step)
                time.sleep(0.01)  # Small delay for timing

            # Finalize the last step
            collector._finalize_step_metrics()

            # Calculate metrics
            metrics = collector.calculate_throughput_metrics()

            # Should have metrics from moving window
            assert "tokens_per_sec" in metrics
            assert "samples_per_sec" in metrics
            assert "tokens_per_sec_per_neuron_core" in metrics
            assert "samples_per_sec_per_neuron_core" in metrics
            assert "metrics_window_steps" in metrics
            assert "avg_tokens_per_step" in metrics
            assert "window_is_full" in metrics

            assert metrics["metrics_window_steps"] == 3
            assert metrics["avg_tokens_per_step"] == 2 * 128  # 256 tokens per step

    def test_mfu_calculation_with_moving_window(self):
        """Test MFU calculation using moving averages."""
        model = create_mock_model()
        args = create_test_training_args(
            enable_mfu_metrics=True,
            metrics_window_size=2
        )

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_model_param_count', return_value=1000000):

            collector = TrainingMetricsCollector(model, args)

            # Record steps
            for step in range(1, 3):
                batch = create_sample_batch(batch_size=4, seq_length=64)
                collector.record_batch_metrics(batch, step=step)
                time.sleep(0.02)

            collector._finalize_step_metrics()

            # Calculate MFU metrics
            metrics = collector.calculate_mfu_metrics()

            assert "model_flops_utilization" in metrics
            assert "theoretical_flops_per_sec" in metrics
            assert "peak_flops_per_sec" in metrics

    def test_efficiency_metrics_with_consistency(self):
        """Test efficiency metrics including step time consistency."""
        model = create_mock_model()
        args = create_test_training_args(
            enable_efficiency_metrics=True,
            enable_throughput_metrics=True,
            metrics_window_size=5,
            expected_tokens_per_core=1000.0
        )

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Record steps with varying timing (to test consistency)
            sleep_times = [0.01, 0.015, 0.012, 0.011, 0.013]
            for step, sleep_time in enumerate(sleep_times, 1):
                batch = create_sample_batch(batch_size=2, seq_length=128)
                collector.record_batch_metrics(batch, step=step)
                time.sleep(sleep_time)

            collector._finalize_step_metrics()

            # Calculate efficiency metrics
            metrics = collector.calculate_efficiency_metrics()

            assert "training_efficiency" in metrics
            assert "step_time_consistency" in metrics

            # Efficiency should be calculated based on expected_tokens_per_core
            assert 0 <= metrics["training_efficiency"] <= 100
            assert 0 <= metrics["step_time_consistency"] <= 100

    def test_reset_window_with_new_features(self):
        """Test that reset_window properly clears all new components."""
        model = create_mock_model()
        args = create_test_training_args(metrics_window_size=3)

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Record some metrics
            batch = create_sample_batch()
            collector.record_batch_metrics(batch, step=1)

            # Should have some state
            assert collector.moving_window.size > 0 or collector.last_step_number is not None

            # Reset
            collector.reset_window()

            # All state should be cleared
            assert collector.moving_window.size == 0
            assert collector.last_step_number is None
            assert collector.current_step_start_times == {}

    def test_configuration_integration(self):
        """Test integration of new configuration options."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = NeuronTrainingArguments(
                output_dir=tmp_dir,
                metrics_window_size=25,
                expected_tokens_per_core=750.0,
                metrics_clocks={'test_clock': 'perf_counter'}
            )

            assert args.metrics_window_size == 25
            assert args.expected_tokens_per_core == 750.0
            assert args.metrics_clocks == {'test_clock': 'perf_counter'}

    def test_individual_metric_control(self):
        """Test individual metric start/stop control."""
        model = create_mock_model()
        args = create_test_training_args(
            enable_throughput_metrics=True,
            enable_mfu_metrics=True,
            enable_efficiency_metrics=True,
            metrics_window_size=3
        )

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_model_param_count', return_value=1000000):

            collector = TrainingMetricsCollector(model, args)

            # Test starting and stopping individual metrics
            batch = create_sample_batch(batch_size=2, seq_length=128)

            # Start throughput timing
            collector.start_metric('throughput', batch)
            assert collector.metric_start_times['throughput'] is not None
            assert collector.current_batch_data['throughput']['tokens'] == 2 * 128

            time.sleep(0.01)  # Small delay

            # Stop throughput timing
            collector.stop_metric('throughput', step_number=1)
            assert collector.metric_start_times['throughput'] is None
            assert collector.metric_windows['throughput'].size == 1

            # Start MFU timing separately
            collector.start_metric('mfu', batch)
            time.sleep(0.01)
            collector.stop_metric('mfu', step_number=1)

            # Calculate individual metrics
            throughput_metrics = collector.finalize_metric('throughput')
            assert 'tokens_per_sec' in throughput_metrics
            assert 'samples_per_sec' in throughput_metrics

            mfu_metrics = collector.finalize_metric('mfu')
            assert 'model_flops_utilization' in mfu_metrics

    def test_start_stop_metric_workflow(self):
        """Test complete start/stop workflow for metrics."""
        model = create_mock_model()
        args = create_test_training_args(
            enable_throughput_metrics=True,
            metrics_window_size=2
        )

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=2), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Simulate multiple training steps with individual metric control
            for step in range(1, 4):
                batch = create_sample_batch(batch_size=4, seq_length=64)

                # Start throughput metric
                collector.start_metric('throughput', batch)

                # Simulate processing time
                time.sleep(0.005)

                # Stop throughput metric
                collector.stop_metric('throughput', step_number=step)

            # Should have 2 measurements (window size = 2, so oldest evicted)
            assert collector.metric_windows['throughput'].size == 2

            # Calculate final metrics
            metrics = collector.finalize_metric('throughput')
            assert metrics['tokens_per_sec'] > 0
            assert metrics['tokens_per_sec_per_neuron_core'] > 0
            assert metrics['metrics_window_steps'] == 2

    def test_finalize_metric_custom_clock(self):
        """Test finalizing metrics with custom clock names."""
        model = create_mock_model()

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = NeuronTrainingArguments(
                output_dir=tmp_dir,
                enable_throughput_metrics=True,
                metrics_clocks={'custom_timer': 'wall_time'}
            )

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Record some steps
            batch = create_sample_batch(batch_size=2, seq_length=128)
            collector.record_batch_metrics(batch, step=1)
            time.sleep(0.01)

            # Should be able to use custom clock name
            custom_metrics = collector.finalize_metric('custom_timer')
            assert 'tokens_per_sec' in custom_metrics  # Should fall back to throughput calculation

    def test_finalize_metric_invalid_name(self):
        """Test that invalid metric names raise appropriate errors."""
        model = create_mock_model()
        args = create_test_training_args()

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            with pytest.raises(ValueError, match="Unsupported metric name"):
                collector.finalize_metric('invalid_metric_name')

    def test_finalize_metric_disabled_metrics(self):
        """Test finalizing metrics when they are disabled."""
        model = create_mock_model()
        args = create_test_training_args(
            enable_throughput_metrics=False,
            enable_mfu_metrics=False,
            enable_efficiency_metrics=False
        )

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Should return empty dict when metrics are disabled
            throughput_metrics = collector.finalize_metric('throughput')
            assert throughput_metrics == {}

            mfu_metrics = collector.finalize_metric('mfu')
            assert mfu_metrics == {}

            efficiency_metrics = collector.finalize_metric('efficiency')
            assert efficiency_metrics == {}

    def test_update_metric_batch_data(self):
        """Test updating batch data for gradient accumulation steps."""
        model = create_mock_model()
        args = create_test_training_args()

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Start a metric
            batch1 = create_sample_batch(batch_size=2, seq_length=64)
            collector.start_metric('throughput', batch1)

            # Update with more batch data (simulating gradient accumulation)
            batch2 = create_sample_batch(batch_size=3, seq_length=64)
            collector.update_metric_batch_data('throughput', batch2)

            # Should have accumulated both batches
            assert collector.current_batch_data['throughput']['tokens'] == (2 * 64) + (3 * 64)
            assert collector.current_batch_data['throughput']['samples'] == 2 + 3

    def test_start_metric_errors(self):
        """Test error handling in start_metric."""
        model = create_mock_model()
        args = create_test_training_args()

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Should raise error for unknown metric
            with pytest.raises(ValueError, match="Unknown metric"):
                collector.start_metric('unknown_metric')

    def test_stop_metric_without_start(self):
        """Test stopping a metric that wasn't started."""
        model = create_mock_model()
        args = create_test_training_args()

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Should not raise error, should just ignore
            collector.stop_metric('throughput')  # Should not crash

    def test_metric_initialization(self):
        """Test that metrics are properly initialized with individual windows."""
        model = create_mock_model()
        args = create_test_training_args(metrics_window_size=10)

        with patch('optimum.neuron.trainers.metrics.get_data_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_tensor_model_parallel_size', return_value=1), \
             patch('optimum.neuron.trainers.metrics.get_pipeline_model_parallel_size', return_value=1):

            collector = TrainingMetricsCollector(model, args)

            # Check that individual metric systems are initialized
            assert 'throughput' in collector.metric_windows
            assert 'mfu' in collector.metric_windows
            assert 'efficiency' in collector.metric_windows

            assert 'throughput' in collector.metric_clocks
            assert 'mfu' in collector.metric_clocks
            assert 'efficiency' in collector.metric_clocks

            # Check window sizes
            assert collector.metric_windows['throughput'].window_size == 10
            assert collector.metric_windows['mfu'].window_size == 10


if __name__ == "__main__":
    pytest.main([__file__])