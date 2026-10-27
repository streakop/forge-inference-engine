
#!/usr/bin/env python3
"""Benchmark Forge's streaming inference API."""

import argparse
import json
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def percentile(values: list[float], p: float) -> float:
    """Return a percentile using linear interpolation."""
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def mean_or_zero(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def benchmark_stream(
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict:
    """Run one streaming request and collect timing and runtime metrics."""

    url = f"{base_url.rstrip('/')}/v1/generate/stream"

    payload = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")

    request = Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    start = time.perf_counter()
    first_event_time = None
    first_text_time = None
    last_text_time = None
    server_done = False

    event_times: list[float] = []
    inter_event_times: list[float] = []
    token_ids: list[int] = []
    text_parts: list[str] = []
    runtime_metrics = None

    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")

            if "text/event-stream" not in content_type:
                body = response.read().decode(
                    "utf-8", errors="replace"
                )
                raise RuntimeError(
                    f"Expected SSE response, got {content_type!r}: "
                    f"{body[:500]}"
                )

            event_lines: list[str] = []

            def process_event(lines: list[str]) -> None:
                nonlocal first_event_time
                nonlocal first_text_time
                nonlocal last_text_time
                nonlocal server_done
                nonlocal runtime_metrics

                data_lines = [
                    line[5:].lstrip()
                    for line in lines
                    if line.startswith("data:")
                ]

                if not data_lines:
                    return

                raw_data = "\n".join(data_lines)

                if raw_data.strip() == "[DONE]":
                    server_done = True
                    return

                try:
                    event = json.loads(raw_data)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid SSE JSON: {raw_data[:300]}"
                    ) from exc

                now = time.perf_counter()

                if first_event_time is None:
                    first_event_time = now

                if "error" in event:
                    raise RuntimeError(
                        f"Server error: {event['error']}"
                    )

                # The server sends runtime metrics after generation.
                if "metrics" in event:
                    runtime_metrics = event["metrics"]
                    return

                if event.get("done"):
                    server_done = True
                    return

                if "text" in event:
                    if first_text_time is None:
                        first_text_time = now

                    if last_text_time is not None:
                        inter_event_times.append(
                            now - last_text_time
                        )

                    last_text_time = now
                    event_times.append(now)
                    text_parts.append(event["text"])

                    token_id = event.get("token_id")
                    if isinstance(token_id, int):
                        token_ids.append(token_id)

            for raw_line in response:
                line = raw_line.decode(
                    "utf-8", errors="replace"
                ).rstrip("\r\n")

                if line == "":
                    process_event(event_lines)
                    event_lines = []
                else:
                    event_lines.append(line)

            # Process a final event if the response did not end with
            # a blank line.
            if event_lines:
                process_event(event_lines)

    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from {url}: {body[:1000]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach Forge at {url}: {exc.reason}"
        ) from exc

    end = time.perf_counter()

    total_latency = end - start
    ttfe = (
        first_text_time - start
        if first_text_time is not None
        else None
    )

    text = "".join(text_parts)
    observed_events = len(event_times)

    first_to_last = (
        last_text_time - first_text_time
        if first_text_time is not None
        and last_text_time is not None
        else 0.0
    )

    event_rate = (
        observed_events / first_to_last
        if first_to_last > 0
        else 0.0
    )

    return {
        "total_latency_seconds": total_latency,
        "ttfe_seconds": ttfe,
        "first_to_last_seconds": first_to_last,
        "inter_event_times": inter_event_times,
        "observed_events": observed_events,
        "token_ids": token_ids,
        "text": text,
        "characters": len(text),
        "server_done": server_done,
        "runtime_metrics": runtime_metrics,
        "event_rate_per_second": event_rate,
    }


def print_run_result(result: dict, run_number: int) -> None:
    """Print details for a single request."""

    ttfe = result["ttfe_seconds"]
    ttfe_display = (
        f"{ttfe:.4f}s" if ttfe is not None else "N/A"
    )

    print(
        f"Run {run_number}: "
        f"latency={result['total_latency_seconds']:.4f}s, "
        f"TTFE={ttfe_display}, "
        f"events={result['observed_events']}, "
        f"chars={result['characters']}, "
        f"event_rate={result['event_rate_per_second']:.2f}/s, "
        f"done={result['server_done']}"
    )

    metrics = result["runtime_metrics"] or {}

    prompt_tokens = metrics.get("prompt_tokens")
    prompt_tps = metrics.get("prompt_tokens_per_second")
    generated_tokens = metrics.get("generated_tokens")
    generation_tps = metrics.get("generation_tokens_per_second")
    peak_memory = metrics.get("peak_memory_gb")
    finish_reason = metrics.get("finish_reason")

    if metrics:
        print(
            f"  prompt={prompt_tokens} tokens "
            f"at {prompt_tps or 0:.1f} tok/s; "
            f"generated={generated_tokens} tokens "
            f"at {generation_tps or 0:.1f} tok/s; "
            f"peak_MLX_memory={peak_memory or 0:.3f} GB; "
            f"finish_reason={finish_reason}"
        )
    else:
        print("  Runtime metrics: not received")

    token_ids = result["token_ids"]
    print(
        f"  token IDs: {len(token_ids)}/"
        f"{result['observed_events']} events"
    )


def print_summary(results: list[dict], max_tokens: int) -> None:
    """Print aggregate statistics across measured runs."""

    latencies = [
        r["total_latency_seconds"] for r in results
    ]
    ttfe_values = [
        r["ttfe_seconds"]
        for r in results
        if r["ttfe_seconds"] is not None
    ]
    inter_event_values = [
        gap
        for r in results
        for gap in r["inter_event_times"]
    ]
    event_rates = [
        r["event_rate_per_second"]
        for r in results
        if r["event_rate_per_second"] > 0
    ]

    total_events = sum(
        r["observed_events"] for r in results
    )
    total_token_ids = sum(
        len(r["token_ids"]) for r in results
    )

    print("\n=== Benchmark summary ===")
    print(f"Measured runs: {len(results)}")
    print(f"Requested max tokens: {max_tokens}")
    print(f"Total text events: {total_events}")
    print(
        f"Events with token IDs: {total_token_ids}/"
        f"{total_events} "
        f"({100 * total_token_ids / total_events:.1f}%)"
        if total_events
        else "Events with token IDs: 0"
    )

    print("\nClient-observed performance:")
    print(
        f"Latency mean: {mean_or_zero(latencies):.4f}s"
    )
    print(
        f"Latency p50:  {percentile(latencies, 0.50):.4f}s"
    )
    print(
        f"Latency p95:  {percentile(latencies, 0.95):.4f}s"
    )

    if ttfe_values:
        print(
            f"TTFE mean:   {mean_or_zero(ttfe_values):.4f}s"
        )
        print(
            f"TTFE p50:    {percentile(ttfe_values, 0.50):.4f}s"
        )

    if inter_event_values:
        print(
            "Inter-event mean: "
            f"{mean_or_zero(inter_event_values):.4f}s"
        )
        print(
            "Inter-event p50:  "
            f"{percentile(inter_event_values, 0.50):.4f}s"
        )

    if event_rates:
        print(
            "Observed event rate mean: "
            f"{mean_or_zero(event_rates):.2f} events/s"
        )

    print("\nRuntime-reported metrics:")
    metric_rows = [
        r["runtime_metrics"]
        for r in results
        if r["runtime_metrics"] is not None
    ]

    if not metric_rows:
        print("No runtime metrics received.")
        return

    prompt_rates = [
        m["prompt_tokens_per_second"]
        for m in metric_rows
        if m.get("prompt_tokens_per_second") is not None
    ]
    generation_rates = [
        m["generation_tokens_per_second"]
        for m in metric_rows
        if m.get("generation_tokens_per_second") is not None
    ]
    peak_memories = [
        m["peak_memory_gb"]
        for m in metric_rows
        if m.get("peak_memory_gb") is not None
    ]
    prompt_counts = [
        m["prompt_tokens"]
        for m in metric_rows
        if m.get("prompt_tokens") is not None
    ]
    generated_counts = [
        m["generated_tokens"]
        for m in metric_rows
        if m.get("generated_tokens") is not None
    ]

    if prompt_counts:
        print(
            f"Prompt tokens mean: "
            f"{mean_or_zero(prompt_counts):.1f}"
        )
    if prompt_rates:
        print(
            f"Prompt processing mean: "
            f"{mean_or_zero(prompt_rates):.2f} tokens/s"
        )
    if generated_counts:
        print(
            f"Generated tokens mean: "
            f"{mean_or_zero(generated_counts):.1f}"
        )
    if generation_rates:
        print(
            f"Generation throughput mean: "
            f"{mean_or_zero(generation_rates):.2f} tokens/s"
        )
    if peak_memories:
        print(
            f"Peak MLX memory max: "
            f"{max(peak_memories):.3f} GB"
        )

    print(
        f"Requests with runtime metrics: "
        f"{len(metric_rows)}/{len(results)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Forge's SSE streaming inference API."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Forge server base URL",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of measured runs per token length",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs per token length",
    )
    parser.add_argument(
        "--lengths",
        default="32,128,256",
        help="Comma-separated max-token lengths",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Explain how a local language model generates text, "
            "including tokenization, inference, and streaming. "
            "Be clear and technically accurate."
        ),
        help="Prompt sent to the model",
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    try:
        lengths = [
            int(item.strip())
            for item in args.lengths.split(",")
            if item.strip()
        ]
    except ValueError:
        parser.error("--lengths must contain comma-separated integers")

    if not lengths or any(length < 1 for length in lengths):
        parser.error("All token lengths must be positive integers")

    for max_tokens in lengths:
        print(f"\n=== max_tokens={max_tokens} ===")

        for warmup_number in range(1, args.warmup + 1):
            print(
                f"Warmup {warmup_number}/{args.warmup}...",
                flush=True,
            )
            benchmark_stream(
                args.url,
                args.prompt,
                max_tokens,
                args.temperature,
                args.timeout,
            )

        results = []

        for run_number in range(1, args.runs + 1):
            result = benchmark_stream(
                args.url,
                args.prompt,
                max_tokens,
                args.temperature,
                args.timeout,
            )
            results.append(result)
            print_run_result(result, run_number)

        print_summary(results, max_tokens)


if __name__ == "__main__":
    main()
