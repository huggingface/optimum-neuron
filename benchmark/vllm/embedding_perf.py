#!/usr/bin/env python3
"""Embedding throughput benchmark for vLLM servers.

Sends concurrent embedding requests and measures throughput and latency.
"""

import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp


def generate_prompt(target_tokens: int, variance: int = 150) -> str:
    """Generate a random prompt of approximately target_tokens tokens.

    Uses ~1.3 chars per token as a rough estimate for English text.
    """
    actual_tokens = max(10, target_tokens + random.randint(-variance, variance))
    words = []
    token_count = 0
    # Common English words (~1 token each)
    vocab = [
        "the",
        "of",
        "and",
        "to",
        "in",
        "a",
        "is",
        "that",
        "for",
        "it",
        "was",
        "on",
        "are",
        "as",
        "with",
        "his",
        "they",
        "at",
        "be",
        "this",
        "have",
        "from",
        "or",
        "one",
        "had",
        "by",
        "but",
        "not",
        "what",
        "all",
        "were",
        "we",
        "when",
        "your",
        "can",
        "said",
        "there",
        "each",
        "which",
        "their",
        "time",
        "will",
        "way",
        "about",
        "many",
        "then",
        "them",
        "would",
        "like",
        "so",
        "these",
        "her",
        "long",
        "make",
        "thing",
        "see",
        "him",
        "two",
        "has",
        "look",
        "more",
        "day",
        "could",
        "go",
        "come",
        "did",
        "my",
        "no",
        "most",
        "who",
        "over",
        "know",
        "than",
        "call",
        "first",
        "may",
        "down",
        "should",
        "people",
        "been",
        "now",
        "find",
        "any",
        "new",
        "part",
        "take",
        "get",
        "place",
        "made",
        "after",
        "back",
        "only",
        "use",
        "work",
    ]
    while token_count < actual_tokens:
        words.append(random.choice(vocab))
        token_count += 1
    return " ".join(words)


async def send_embedding_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
) -> dict:
    """Send a single embedding request and return timing info."""
    payload = {"input": prompt, "model": model}
    t0 = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        result = await resp.json()
        elapsed = time.perf_counter() - t0
        if resp.status != 200:
            return {"success": False, "latency": elapsed, "error": result}
        usage = result.get("usage", {})
        return {
            "success": True,
            "latency": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }


async def run_benchmark(
    target: str,
    model: str,
    concurrent: int,
    total: int,
    prompt_tokens: int,
) -> dict:
    """Run the embedding benchmark."""
    url = f"{target}/embeddings"
    semaphore = asyncio.Semaphore(concurrent)
    results = []

    # Pre-generate all prompts
    prompts = [generate_prompt(prompt_tokens) for _ in range(total)]

    async def bounded_request(session, prompt):
        async with semaphore:
            return await send_embedding_request(session, url, model, prompt)

    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=concurrent + 10)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Warmup: send a few requests first
        warmup_count = min(concurrent, total, 10)
        print(f"Warming up with {warmup_count} requests...")
        warmup_tasks = [bounded_request(session, prompts[i]) for i in range(warmup_count)]
        await asyncio.gather(*warmup_tasks)

        # Benchmark
        print(f"Running {total} requests with {concurrent} concurrent...")
        wall_start = time.perf_counter()
        tasks = [bounded_request(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)
        wall_elapsed = time.perf_counter() - wall_start

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        print("ERROR: All requests failed!")
        if failed:
            print(f"Sample error: {failed[0].get('error', 'unknown')}")
        return {}

    latencies = [r["latency"] for r in successful]
    total_prompt_tokens = sum(r["prompt_tokens"] for r in successful)

    summary = {
        "model": model,
        "concurrent_users": concurrent,
        "total_requests": total,
        "target_prompt_tokens": prompt_tokens,
        "successful": len(successful),
        "failed": len(failed),
        "wall_time_s": round(wall_elapsed, 3),
        "requests_per_second": round(len(successful) / wall_elapsed, 2),
        "tokens_per_second": round(total_prompt_tokens / wall_elapsed, 2),
        "total_prompt_tokens": total_prompt_tokens,
        "latency_p50_ms": round(statistics.median(latencies) * 1000, 1),
        "latency_p90_ms": round(sorted(latencies)[int(len(latencies) * 0.9)] * 1000, 1),
        "latency_p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)] * 1000, 1),
        "latency_mean_ms": round(statistics.mean(latencies) * 1000, 1),
        "latency_min_ms": round(min(latencies) * 1000, 1),
        "latency_max_ms": round(max(latencies) * 1000, 1),
    }

    print("\n" + "=" * 60)
    print(f"  Embedding Benchmark Results — {model}")
    print("=" * 60)
    print(f"  Requests:   {summary['successful']} ok / {summary['failed']} failed")
    print(f"  Wall time:  {summary['wall_time_s']}s")
    print(f"  Throughput: {summary['requests_per_second']} req/s")
    print(f"  Throughput: {summary['tokens_per_second']} tok/s")
    print(f"  Latency p50:  {summary['latency_p50_ms']}ms")
    print(f"  Latency p90:  {summary['latency_p90_ms']}ms")
    print(f"  Latency p99:  {summary['latency_p99_ms']}ms")
    print(f"  Latency mean: {summary['latency_mean_ms']}ms")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Embedding throughput benchmark")
    parser.add_argument("--target", required=True, help="vLLM base URL (e.g. http://localhost:8080/v1)")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--concurrent", type=int, default=32, help="Concurrent requests")
    parser.add_argument("--total", type=int, default=1000, help="Total requests")
    parser.add_argument("--prompt-tokens", type=int, default=1500, help="Target prompt tokens per request")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    result = asyncio.run(
        run_benchmark(
            target=args.target,
            model=args.model,
            concurrent=args.concurrent,
            total=args.total,
            prompt_tokens=args.prompt_tokens,
        )
    )

    if args.output and result:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
