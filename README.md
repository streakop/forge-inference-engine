# Forge — Local AI Inference Engine

Forge is a small local LLM serving project built to explore inference orchestration, streaming APIs, runtime metrics, and repeatable performance measurement on Apple Silicon. It uses **MLX-LM** for model execution and **FastAPI** for its HTTP interface.

> **Scope:** Forge is an inference server and orchestration layer built on MLX-LM. It is not a from-scratch transformer implementation, custom GPU-kernel project, or production multi-tenant serving platform.

## Highlights

- Local inference with an MLX-LM model on Apple Silicon
- FastAPI endpoints for health, complete generation, and SSE streaming
- Stream events include generated text and token IDs
- Terminal SSE metrics include prompt processing speed, generation throughput, peak MLX memory, and finish reason
- Benchmark client records TTFE, end-to-end latency, inter-event timing, event rate, token IDs, and runtime metrics
- Optional JSON and CSV benchmark exports
- A concurrent-client probe for observing queueing and service behavior
- Request validation through Pydantic and serialized access to the MLX runtime

## Requirements

- Apple Silicon Mac with a supported macOS version
- Python 3.11+
- Network access for the first model download
- Enough unified memory and disk space for the selected model

MLX-LM is installed only on Apple Silicon macOS by the project dependency marker. The default model is `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
cp .env.example .env
uvicorn forge.app:app --host 127.0.0.1 --port 8000
```

The model is loaded when the app module initializes. The first start may download model files and take longer than subsequent requests. Keep the server bound to localhost unless you add authentication and appropriate network controls.

## API

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Non-streaming generation

```bash
curl http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain KV caching briefly.","max_tokens":128,"temperature":0.2}'
```

### Streaming generation (SSE)

```bash
curl -N http://127.0.0.1:8000/v1/generate/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain tokenization briefly.","max_tokens":128,"temperature":0.2}'
```

The stream sends `data:` JSON events for text chunks (`text`, `token_id`), a terminal `metrics` event, and a `done` event. Streaming errors are sent as SSE error events. Clients should inspect those events rather than assuming HTTP 200 guarantees successful generation.

Request fields:

| Field | Constraints | Default |
|---|---|---|
| `prompt` | 1–20,000 characters | Required |
| `max_tokens` | 1–2,048 | 128 |
| `temperature` | 0–2 | 0.7 |

## Tests

Run on a compatible environment with dependencies installed:

```bash
python -m pytest
```

The API tests use a fake runtime to test request validation and response/event shape without requiring a model generation for each test. A real smoke test should also be run on the target Apple Silicon machine.

## Benchmarks

Start the server in one terminal. In another, run:

```bash
python benchmarks/benchmark.py --runs 5 --warmup 1 --lengths 32,128,256 \
  --json-out benchmark-results/forge-benchmark.json \
  --csv-out benchmark-results/forge-benchmark.csv
```

The JSON output includes per-request detail (including response text and token IDs); treat it as potentially sensitive if your prompts contain private data. The CSV is intended for spreadsheet analysis and stores nested runtime metrics/token IDs as JSON strings.

To probe concurrent clients:

```bash
python benchmarks/concurrency.py --requests 8 --concurrency 2 --max-tokens 64
```

Repeat with concurrency values such as `1`, `2`, and `4`. This probe measures client-observed completion latency, success rate, and aggregate completed requests per second. It does not claim that MLX performs parallel decoding; the runtime serializes model access to avoid concurrent calls into the same model instance. Higher concurrency may therefore increase queueing latency rather than throughput.

### Recorded local baseline

The following results were recorded by the project author on a MacBook Air M5 using `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, one warm-up and five measured runs per length. They are a machine-specific baseline, not a universal performance claim.

| Requested output length | Mean end-to-end latency | Mean runtime generation throughput | Mean TTFE | Peak MLX-reported memory |
|---:|---:|---:|---:|---:|
| 32 tokens | 0.2866 s | 131.99 tokens/s | 38.4 ms | 0.938 GB |
| 128 tokens | 1.0426 s | 128.29 tokens/s | 38.8 ms | 0.938 GB |
| 256 tokens | 2.0745 s | 126.19 tokens/s | 39.8 ms | 0.938 GB |

Prompt processing averaged about 653–675 tokens/s across these runs. Every measured text event had a token ID (2,080/2,080 events total).

**Metric definitions and caveats:**

- **TTFE** is client-observed time until the first text event; it includes client/server transport and scheduling.
- **Runtime generation throughput** is reported by MLX-LM. It is not a measurement of raw GPU-kernel throughput.
- **Event rate** is computed from client-observed text event timestamps and is distinct from the runtime-reported throughput metric.
- **Peak MLX memory** is the runtime's reported peak, not total system RAM or total process resident memory.
- Results depend on prompt, output length, model, software versions, temperature, power/thermal state, and warm-up policy. Compare only runs with controlled conditions.

## Architecture

```text
HTTP client
    |
FastAPI + Pydantic validation
    |-------------------------------|
POST /v1/generate          POST /v1/generate/stream
    |                               |
MLXRuntime.generate        MLXRuntime.stream
    |                               |
    +--------- MLX-LM model --------+
                  |
      response / SSE text + metrics
```

## Limitations and future work

- Apple Silicon / MLX-LM runtime only; another backend would need a separate adapter.
- No authentication, quotas, persistent request queue, cancellation, or continuous batching.
- Runtime calls are serialized; the concurrency script measures queueing and service behavior rather than parallel decoding.
- Streaming errors are SSE events; disconnect cancellation and standardized error codes can be improved.
- Benchmark results are local and small-sample; broader workloads and longer prompts should be tested before drawing general conclusions.
- JSON benchmark output stores full prompt and generated text; review it before publishing or sharing.

## Project status

Forge is a learning and portfolio project focused on local inference serving and measurement. The baseline implementation and local benchmark results are available; concurrent-load behavior should be measured on the target machine before making claims about production readiness.

## License

No license has been added yet. Add a `LICENSE` file before inviting reuse or contributions, and choose a license that matches your intentions.
