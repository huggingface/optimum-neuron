import glob
import json

import pandas as pd


filenames = glob.glob("tgi_bench_results/*/*summary.json")

results = []

for filename in filenames:
    with open(filename) as f:
        summary = json.load(f)
        d = {
            "model_id": summary["model"],
            "concurrent requests": summary["num_concurrent_requests"],
            "throughput (t/s)": summary["results_mean_output_throughput_token_per_s"],
            "Time-to-first-token @ P50 (s)": summary["results_ttft_s_quantiles_p50"],
            "average latency (ms)": summary["results_inter_token_latency_s_quantiles_p50"] * 1000,
        }
        results.append(pd.DataFrame.from_dict(d, orient="index").transpose())

df = pd.concat(results).sort_values(by="concurrent requests")
df.to_csv("tgi-results.csv", index=False)
