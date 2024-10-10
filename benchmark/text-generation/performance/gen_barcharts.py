import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_bar_chart(title, labels, xlabel, ylabel, series, save_path):
    x = np.arange(len(labels))  # the label locations
    width = 0.18  # the width of the bars
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + 2 * width, labels)
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
    ttft_s = {}
    latency_ms = {}
    throughput_t_per_s = {}
    for name in model_names:
        results = benchmarks[name]["results"]
        cur_input_length = [result["input_length"] for result in results]
        if len(input_length) == 0:
            input_length = cur_input_length
        else:
            assert cur_input_length == input_length, f"{name} does not have the same number of results"
        ttft_s[name] = [round(result["encoding_time"], 1) for result in results]
        latency_ms[name] = [round(result["latency"], 0) for result in results]
        throughput_t_per_s[name] = [round(result["throughput"], 0) for result in results]
    save_bar_chart(
        title="Time to generate the first token in seconds",
        labels=input_length,
        series=ttft_s,
        xlabel="Input tokens",
        ylabel="Time to first token (s)",
        save_path="ttft.png",
    )
    save_bar_chart(
        title="Inter-token latency in milliseconds",
        labels=input_length,
        series=latency_ms,
        xlabel="Input tokens",
        ylabel="Latency (ms)",
        save_path="latency.png",
    )
    save_bar_chart(
        title="Generated tokens per second (end-to-end)",
        labels=input_length,
        series=throughput_t_per_s,
        xlabel="Input tokens",
        ylabel="Throughput (tokens/s)",
        save_path="throughput.png",
    )


if __name__ == "__main__":
    main()
