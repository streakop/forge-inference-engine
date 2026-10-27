#!/usr/bin/env python3
"""Measure client-observed latency under concurrent streaming requests.

This is a load probe, not a production load-testing framework. Run only against
an instance you own, preferably bound to localhost.
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_once(url: str, prompt: str, max_tokens: int, temperature: float, timeout: float) -> dict:
    payload = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}).encode()
    request = Request(url.rstrip("/") + "/v1/generate/stream", data=payload,
                      headers={"Content-Type": "application/json", "Accept": "text/event-stream"}, method="POST")
    started = time.perf_counter()
    events = 0
    got_done = False
    got_metrics = False
    error = None
    try:
        with urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if "error" in event:
                    error = event["error"]
                if "text" in event:
                    events += 1
                if "metrics" in event:
                    got_metrics = True
                if event.get("done"):
                    got_done = True
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {"latency_seconds": time.perf_counter() - started, "events": events,
            "done": got_done, "metrics_received": got_metrics, "error": error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=8, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=2, help="Maximum simultaneous clients")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--prompt", default="Explain autoregressive inference briefly.")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.max_tokens < 1:
        parser.error("requests, concurrency, and max-tokens must be positive")

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(request_once, args.url, args.prompt, args.max_tokens,
                               args.temperature, args.timeout) for _ in range(args.requests)]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            status = "OK" if not result["error"] and result["done"] else "FAIL"
            print(f"request {index}: {status} latency={result['latency_seconds']:.3f}s "
                  f"events={result['events']} metrics={result['metrics_received']} "
                  f"error={result['error'] or '-'}")
    wall = time.perf_counter() - started
    latencies = [r["latency_seconds"] for r in results]
    successful = sum(not r["error"] and r["done"] for r in results)
    print("\\n=== Concurrency summary ===")
    print(f"requests={args.requests} concurrency={args.concurrency} max_tokens={args.max_tokens}")
    print(f"success={successful}/{len(results)} success_rate={successful / len(results) * 100:.1f}%")
    print(f"wall_time={wall:.3f}s completed_requests_per_second={successful / wall:.2f}")
    print(f"client_latency_mean={statistics.mean(latencies):.3f}s "
          f"p50={statistics.median(latencies):.3f}s "
          f"p95={sorted(latencies)[min(len(latencies)-1, int(0.95 * (len(latencies)-1)))]:.3f}s")
    print("Note: these are client-observed results; runtime throughput is not aggregated here.")


if __name__ == "__main__":
    main()
