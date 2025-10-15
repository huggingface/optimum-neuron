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

import pytest
import torch

from optimum.neuron.models.training.llama.modeling_llama import LlamaForCausalLM
from optimum.neuron.trainers.metrics import TrainingMetricsCollector
from optimum.neuron.trainers.training_args import NeuronTrainingArguments
from optimum.neuron.utils.testing_utils import is_trainium_test

from .distributed_utils import run_distributed_test
from .utils import MODEL_NAME


@pytest.mark.parametrize(
    "world_size,tp_size,pp_size",
    [
        (8, 1, 1),
        (32, 8, 1),
        (32, 1, 4),
        (32, 8, 4),
    ],
    ids=["8_1_1", "32_8_1", "32_1_4", "32_8_4"],
)
@is_trainium_test
def test_metrics_distributed_correctness(world_size, tp_size, pp_size, tmpdir):
    def _test_metrics_computation():
        args = NeuronTrainingArguments(
            output_dir=tmpdir,
            enable_throughput_metrics=True,
            enable_mfu_metrics=True,
            enable_efficiency_metrics=True,
            metrics_window_size=3,
        )

        model = LlamaForCausalLM.from_pretrained(MODEL_NAME, args.trn_config)
        collector = TrainingMetricsCollector(model, args)

        # Test unit system first
        units = collector.get_all_metric_units()
        assert units["train/tokens_per_sec"] == "tokens/s"
        assert units["train/mfu"] == "%"
        assert units["train/step_time"] == "s"
        assert units["train/efficiency"] == "%"
        assert units["train/forward_time_percent"] == "%"
        assert units["train/backward_time_percent"] == "%"
        assert units["train/optimizer_time_percent"] == "%"
        assert units["train/overhead_time_percent"] == "%"

        inputs = {"input_ids": torch.randint(0, 1000, (4, 32))}
        inputs["labels"] = inputs["input_ids"].clone()

        # Test accumulation cycle with all component timings
        collector.start_gradient_accumulation_cycle()
        collector.start_metric("total_step")

        # Simulate gradient_accumulation_steps=2
        for _ in range(2):
            collector.start_metric("forward_pass", inputs)
            time.sleep(0.005)
            collector.stop_metric("forward_pass")

            collector.start_metric("backward_pass", inputs)
            time.sleep(0.005)
            collector.stop_metric("backward_pass")

        collector.start_metric("optimizer_step", inputs)
        time.sleep(0.003)
        collector.stop_metric("optimizer_step")

        collector.stop_metric("total_step")
        collector.end_gradient_accumulation_cycle(step_number=1)

        # Test throughput and MFU metrics together
        for step in range(3):
            collector.start_metric("throughput", inputs)
            collector.start_metric("mfu", inputs)
            start_time = time.perf_counter()
            time.sleep(0.02)
            elapsed = time.perf_counter() - start_time
            collector.stop_metric("throughput", step_number=step)
            collector.stop_metric("mfu", step_number=step)

            window_stats = collector.metric_windows["throughput"].get_window_stats()
            
            assert window_stats["total_tokens"] == 128 * (step + 1)
            assert abs(window_stats["total_time"] - elapsed) < 0.05

        # Validate throughput computation
        throughput_metrics = collector.calculate_metric("throughput")
        assert "train/tokens_per_sec" in throughput_metrics

        window_stats = collector.metric_windows["throughput"].get_window_stats()
        expected_local_rate = window_stats["total_tokens"] / window_stats["total_time"]
        expected_global_rate = expected_local_rate * collector.dp_size
        actual_global_rate = throughput_metrics["train/tokens_per_sec"]

        relative_error = abs(actual_global_rate - expected_global_rate) / expected_global_rate
        assert relative_error < 0.05, f"Throughput calculation failed: expected {expected_global_rate}, got {actual_global_rate} " \
                                      f"(relative error={relative_error:.3f})"

        # Validate MFU computation
        mfu_metrics = collector.calculate_metric("mfu")
        assert mfu_metrics != {}, "MFU metrics should not be empty"

        total_tokens = window_stats["total_tokens"]
        total_time = window_stats["total_time"]

        # Use exact same formula as mfu.py implementation
        assert collector.model_params is not None, "Model params should be set in the collector"
        assert collector.seq_length is not None, "Sequence length should be set in the collector"

        N = collector.model_params
        L, H, Q, T = collector.num_layers, collector.num_heads, collector.head_dim, collector.seq_length
        flops_per_token = 6 * N + 12 * L * H * Q * T

        system_tokens = total_tokens * collector.dp_size
        system_flops_per_iter = flops_per_token * system_tokens
        system_actual_flops_per_sec = system_flops_per_iter / total_time
        system_peak_flops_per_sec = collector.peak_tflops_per_core * collector.total_neuron_cores * 1e12
        expected_system_mfu = (system_actual_flops_per_sec / system_peak_flops_per_sec) * 100

        system_mfu_diff = abs(mfu_metrics["train/mfu"] - round(expected_system_mfu, 2))
        assert system_mfu_diff < 0.1, (
            f"System MFU calculation failed: expected {round(expected_system_mfu, 2):.2f}%, got "
            f"{mfu_metrics['train/mfu']:.2f}% (diff={system_mfu_diff:.3f}, cores={collector.total_neuron_cores})"
        )

        # Validate efficiency computation (test the total_step timing fix)
        efficiency_metrics = collector.calculate_metric("efficiency")
        assert efficiency_metrics != {}, "Efficiency metrics should not be empty"

        forward_pct = efficiency_metrics.get("train/forward_time_percent", 0)
        backward_pct = efficiency_metrics.get("train/backward_time_percent", 0)
        optimizer_pct = efficiency_metrics.get("train/optimizer_time_percent", 0)
        overhead_pct = efficiency_metrics.get("train/overhead_time_percent", 0)
        total_efficiency = efficiency_metrics.get("train/efficiency", 0)

        # Test that percentages are reasonable
        assert forward_pct > 0, "Forward time percentage should be > 0"
        assert backward_pct > 0, "Backward time percentage should be > 0"
        assert optimizer_pct > 0, "Optimizer time percentage should be > 0"

        # Test that efficiency calculation is correct
        assert abs((forward_pct + backward_pct + optimizer_pct) - total_efficiency) < 0.01
        assert abs((total_efficiency + overhead_pct) - 100.0) < 0.01

        # Validate summary metrics
        summary_metrics = collector.calculate_summary_metrics()
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

    run_distributed_test(_test_metrics_computation, world_size, tp_size, pp_size)
