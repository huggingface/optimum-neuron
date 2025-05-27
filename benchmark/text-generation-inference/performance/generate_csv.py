import argparse
import glob
import os

import pandas as pd
from guidellm.benchmark import GenerativeBenchmarksReport


def _benchmark_rate_id(benchmark) -> str:
    """
    Generate a string identifier for a benchmark rate.

    :param benchmark: The benchmark for which to generate the rate ID.
    :return: A string representing the benchmark rate ID.
    :rtype: str
    """
    strategy = benchmark.args.strategy
    strategy_type = strategy.type_
    
    if hasattr(strategy, 'rate') and strategy.rate:
        rate_id = f"{strategy_type}@{strategy.rate:.2f} req/sec"
    else:
        rate_id = f"{strategy_type}"
    
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
            report = GenerativeBenchmarksReport.model_validate_json(f.read())
            for benchmark in report.benchmarks:
                d = {
                    "model_id": model_id,
                    "Date": date,
                    "Input type": _benchmark_rate_id(benchmark),
                    "Requests per Second": benchmark.metrics.requests_per_second.successful.mean,
                    "Request Latency (s)": benchmark.metrics.request_latency.successful.mean,
                    "Time-to-first-token (ms)": benchmark.metrics.time_to_first_token_ms.successful.mean,
                    "Inter Token Latency (ms)": benchmark.metrics.inter_token_latency_ms.successful.mean,
                    "Output Token Throughput (t/s)": benchmark.metrics.output_tokens_per_second.successful.mean,
                }
                results.append(pd.DataFrame.from_dict(d, orient="index").transpose())

    df = pd.concat(results).sort_values(by="Date")
    df.to_csv("tgi-results.csv", index=False)


if __name__ == "__main__":
    main()
