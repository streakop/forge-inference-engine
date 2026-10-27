"""Tiny repeatable benchmark client. Run after starting the server."""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/generate")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt", default="Explain autoregressive language-model inference in three concise points.")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    latencies, throughputs = [], []
    for i in range(args.runs):
        payload = json.dumps({"prompt": args.prompt, "max_tokens": args.max_tokens, "temperature": 0.0}).encode()
        request = urllib.request.Request(args.url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read())
        except urllib.error.URLError as exc:
            raise SystemExit(f"Could not reach Forge at {args.url}: {exc}") from exc
        client_elapsed = time.perf_counter() - started
        latencies.append(result["latency_seconds"])
        throughputs.append(result["tokens_per_second"])
        print(f"run={i+1} client_s={client_elapsed:.3f} engine_s={result['latency_seconds']:.3f} output_tokens={result['completion_tokens']} tok_s={result['tokens_per_second']:.2f}")
    print("\nSummary (small samples; first run may include model load/warm-up effects)")
    print(f"runs={args.runs} median_engine_latency_s={statistics.median(latencies):.3f} median_tokens_per_second={statistics.median(throughputs):.2f}")


if __name__ == "__main__":
    main()
