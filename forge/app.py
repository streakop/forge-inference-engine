from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from forge.config import settings
from forge.runtime import MLXRuntime
from forge.schemas import GenerateRequest, GenerateResponse

app = FastAPI(
    title="Forge Inference Engine",
    version="0.1.0",
    description="A local language-model serving API with measured generation metrics.",
)
runtime = MLXRuntime(settings.model_id)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "engine": "forge",
        "runtime": "mlx",
        "model": settings.model_id,
        "model_loaded": runtime.loaded,
        "platform": f"{platform.system()}-{platform.machine()}",
    }


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> dict:
    try:
        result = runtime.generate(request.prompt, request.max_tokens, request.temperature)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Keep internal exception details out of the HTTP response; server logs retain them.
        raise HTTPException(status_code=500, detail=f"Generation failed: {type(exc).__name__}") from exc
    return {"model": settings.model_id, **result}



@app.post("/v1/generate/stream")
def generate_stream(request: GenerateRequest) -> StreamingResponse:
    def events() -> Iterator[str]:
        started = time.perf_counter()
        last_chunk = None

        try:
            for chunk in runtime.stream(
                request.prompt,
                request.max_tokens,
                request.temperature,
            ):
                last_chunk = chunk
                yield (
                    f"data: {json.dumps({'text': chunk.text, 'token_id': chunk.token_id})}\n\n"
                )

            if last_chunk is not None:
                metrics = {
                    "prompt_tokens": last_chunk.prompt_tokens,
                    "prompt_tokens_per_second": last_chunk.prompt_tps,
                    "generated_tokens": last_chunk.generation_tokens,
                    "generation_tokens_per_second": last_chunk.generation_tps,
                    "peak_memory_gb": last_chunk.peak_memory_gb,
                    "finish_reason": last_chunk.finish_reason,
                    "server_elapsed_seconds": time.perf_counter() - started,
                }
                yield f"data: {json.dumps({'metrics': metrics})}\n\n"

            yield (
                f"data: {json.dumps({'done': True, 'elapsed_seconds': time.perf_counter() - started})}\n\n"
            )

        except Exception as exc:
            yield (
                f"data: {json.dumps({'error': type(exc).__name__, 'message': str(exc)})}\n\n"
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
