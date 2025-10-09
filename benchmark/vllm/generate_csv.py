import argparse
import glob
import os

import pandas as pd
from guidellm.core import GuidanceReport, TextGenerationBenchmark


def _benchmark_rate_id(benchmark: TextGenerationBenchmark) -> str:
    """
    Generate a string identifier for a benchmark rate.

    :param benchmark: The benchmark for which to generate the rate ID.
    :type benchmark: TextGenerationBenchmark
    :return: A string representing the benchmark rate ID.
    :rtype: str
    """
    rate_id = f"{benchmark.mode}@{benchmark.rate:.2f} req/sec" if benchmark.rate else f"{benchmark.mode}"
    return rate_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=".", help="The directory containing the benchmark results")
    args = parser.parse_args()
    suffix = "_guidellm_report.json"
    paths = glob.glob(f"{args.dir}/*{suffix}")
    if len(paths) == 0:
        exit()

    results = []

    for path in paths:
        filename = os.path.basename(path)
        # Extract model_id
        model_id, date = filename.replace(suffix, "").split("#")
        with open(path) as f:
            report = GuidanceReport.from_json(f.read())
            for benchmark in report.benchmarks:
                for b in benchmark.benchmarks_sorted:
                    d = {
                        "model_id": model_id,
                        "Date": date,
                        "Input type": _benchmark_rate_id(b),
                        "Requests per Second": b.completed_request_rate,
                        "Request Latency (s)": b.request_latency,
                        "Time-to-first-token (ms)": b.time_to_first_token,
                        "Inter Token Latency (ms)": b.inter_token_latency,
                        "Output Token Throughput (t/s)": b.output_token_throughput,
                    }
                    results.append(pd.DataFrame.from_dict(d, orient="index").transpose())

    df = pd.concat(results).sort_values(by="Date")
    df.to_csv(f"{args.dir}/vllm-results.csv", index=False)


if __name__ == "__main__":
    main()
