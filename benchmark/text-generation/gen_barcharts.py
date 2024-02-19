import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_bar_chart(title, labels, ylabel, series, save_path):
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")
    fig.set_figwidth(10)

    max_value = 0

    for attribute, measurement in series.items():
        max_value = max(max_value, max(measurement))
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=5)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width, labels)
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, max_value * 1.2)

    plt.savefig(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=str, nargs="*", help="A list of benchmark results files (.json).")
    args = parser.parse_args()
    inputs = args.inputs
    if len(inputs) == 0:
        inputs = glob.glob("*.json")
    benchmarks = {}
    for input in inputs:
        model_name = Path(input).stem
        with open(input) as f:
            benchmarks[model_name] = json.load(f)
    model_names = benchmarks.keys()
    # Generate encoding barchart
    input_length = []
    encoding_times = {}
    for name in model_names:
        results = benchmarks[name]["results"]
        cur_input_length = [result["input_length"] for result in results]
        if len(input_length) == 0:
            input_length = cur_input_length
        else:
            assert cur_input_length == input_length, f"{name} does not have the same number of results"
        encoding_times[name] = [round(result["encoding_time"], 1) for result in results]
    save_bar_chart(
        title="Encoding time per input token",
        labels=input_length,
        series=encoding_times,
        ylabel="Encoding time (s)",
        save_path="encoding_times.png",
    )
    # Generate latency and throughput barcharts (for the first input length only)
    new_tokens = []
    latencies = {}
    throughputs = {}
    for name in model_names:
        generations = benchmarks[name]["results"][0]["generations"]
        cur_new_tokens = [generation["new_tokens"] for generation in generations]
        if len(new_tokens) == 0:
            new_tokens = cur_new_tokens
        else:
            assert cur_new_tokens == new_tokens, f"{name} does not have the same number of results"
        latencies[name] = [round(generation["latency"], 1) for generation in generations]
        throughputs[name] = [round(generation["throughput"], 0) for generation in generations]
    save_bar_chart(
        title="End-to-end latency per generated tokens for 256 input tokens",
        labels=new_tokens,
        series=latencies,
        ylabel="Latency (s)",
        save_path="latencies.png",
    )
    save_bar_chart(
        title="Throughput per generated tokens for 256 input tokens",
        labels=new_tokens,
        series=throughputs,
        ylabel="Throughput (tokens/s)",
        save_path="throughputs.png",
    )


if __name__ == "__main__":
    main()
