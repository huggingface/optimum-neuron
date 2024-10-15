import time
from functools import partial

import numpy as np

BENCHMARK_REPORT_FILENAME = "benchmark_report.json"


class Benchmark:
    def __init__(self, benchmark_func, input_param, config, num_runs=20, preprocess_func=None, post_warmup_func=None) -> None:
        if isinstance(input_param, (tuple, list)):
            self.benchmark_func = partial(benchmark_func, *input_param)
        elif isinstance(input_param, dict):
            self.benchmark_func = partial(benchmark_func, **input_param)
        else:
            self.benchmark_func = partial(benchmark_func, input_param)

        self.config = config
        self.num_runs = num_runs
        self.preprocess_func = preprocess_func
        self.post_warmup_func = post_warmup_func
        self.latency_list = None

    def run(self):
        # Warm up
        if self.preprocess_func:
            self.preprocess_func()
        self.benchmark_func()

        if self.post_warmup_func:
            self.post_warmup_func()

        latency_collector = LatencyCollector()
        for _ in range(self.num_runs):
            latency_collector.pre_hook()
            if self.preprocess_func:
                self.preprocess_func()
            self.benchmark_func()
            latency_collector.hook()
        self.latency_list = latency_collector.latency_list


class LatencyCollector:
    def __init__(self):
        self.start = None
        self.latency_list = []

    def pre_hook(self, *args):
        self.start = time.time()

    def hook(self, *args):
        self.latency_list.append(time.time() - self.start)


def generate_report(latency_list, config):
    latency_array = np.array(latency_list)

    n_runs = len(latency_list)
    max_length = config.max_length
    batch_size = config.max_batch_size
    total_time = np.sum(latency_array)
    throughput = (n_runs * max_length * batch_size) / total_time

    return {
        "latency_ms_p50": np.percentile(latency_array, 50) * 1000,
        "latency_ms_p90": np.percentile(latency_array, 90) * 1000,
        "latency_ms_p95": np.percentile(latency_array, 95) * 1000,
        "latency_ms_p99": np.percentile(latency_array, 99) * 1000,
        "latency_ms_p100": np.percentile(latency_array, 100) * 1000,
        "latency_ms_avg": np.average(latency_array) * 1000,
        "throughput": throughput,
    }
