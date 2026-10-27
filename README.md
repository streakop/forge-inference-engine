# Forge — Local AI Inference Engine

Forge is a small, measurable LLM serving system for Apple Silicon. It wraps a real MLX-LM model runtime with a typed HTTP API, server-sent-event streaming, and a repeatable benchmark client.

> **Scope honesty:** this is an inference *server and orchestration layer* built on MLX-LM, not a from-scratch transformer/GPU-kernel implementation. The next engineering milestones are request scheduling, continuous batching, KV-cache inspection, and a custom sampler/runtime study.

## Requirements

- Apple Silicon Mac and a supported recent macOS version
- Python 3.11+
- Network access for the first model download
- Several GB of free disk space and enough unified memory for the selected model

## Quick start (macOS / Apple Silicon)

```bash
cd forge-inference-engine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[test]'
cp .env.example .env
uvicorn forge.app:app --host 127.0.0.1 --port 8000
```

The model is loaded lazily on the first generation request. The initial request can be slow because model files may need downloading and the runtime needs to warm up. Start with the small 4-bit Qwen model in `.env.example`; change `FORGE_MODEL` only after confirming the first run works.

## Try it

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Generate:

```bash
curl http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain KV caching in simple terms.","max_tokens":128,"temperature":0.2}'
```

Stream generation (SSE):

```bash
curl -N http://127.0.0.1:8000/v1/generate/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a short explanation of tokenization.","max_tokens":128,"temperature":0.7}'
```

## Tests and benchmark

```bash
pytest
python scripts/benchmark.py --runs 5 --max-tokens 128
```

The benchmark prints measured wall-clock latency and throughput returned by the runtime. Do not compare results unless model, prompt, token limit, runtime versions, power mode, and warm-up policy are controlled. The first run may include download/load/warm-up effects. The current throughput estimate is output-token count divided by end-to-end generation wall time, so it includes runtime overhead and is not a pure GPU-kernel metric.

## API

- `GET /health` — process and model-load status
- `POST /v1/generate` — one complete generation and basic metrics
- `POST /v1/generate/stream` — Server-Sent Events containing text chunks and a terminal event

Request JSON:

```json
{"prompt":"Hello","max_tokens":128,"temperature":0.7}
```

Validation: prompt 1–20,000 characters, `max_tokens` 1–2,048, temperature 0–2.

## Architecture

```text
HTTP client
   |
FastAPI + Pydantic validation
   |---------------------------|
/v1/generate              /v1/generate/stream
   |                           |
MLXRuntime.generate       MLXRuntime.stream
   |                           |
MLX-LM model + tokenizer on Apple Silicon
   |
Measured response / streamed text
```

## Known limitations / next milestones

1. One generation at a time: a lock protects the model runtime. No queue, cancellation, or continuous batching yet.
2. Streaming errors are represented as SSE error events; production systems should standardize error codes and disconnect cancellation.
3. Metrics are intentionally basic. Add TTFT, prompt prefill throughput, decode throughput, peak-memory measurement, and structured benchmark output next.
4. No authentication. Keep the server bound to localhost; do not expose it to the public internet as-is.
5. The MLX adapter is Apple Silicon-specific. A CPU/CUDA runtime would be a separate adapter.

## Today's definition of done

- [ ] Install dependencies on the Mac
- [ ] `pytest` passes
- [ ] `/health` returns 200
- [ ] Generate endpoint returns a real model response
- [ ] Streaming endpoint emits incremental text
- [ ] Run a benchmark and save actual results
- [ ] Record model/runtime versions and known limitations
