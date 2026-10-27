
import time
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from mlx_lm import generate, load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler


@dataclass
class TokenChunk:
    text: str
    token_id: int | None = None
    prompt_tokens: int | None = None
    prompt_tps: float | None = None
    generation_tokens: int | None = None
    generation_tps: float | None = None
    peak_memory_gb: float | None = None
    finish_reason: str | None = None

    
class MLXRuntime:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        self.loaded = True
        # MLX model access is serialized until concurrent execution is explicitly supported.
        self._generation_lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.2,
    ) -> dict:
        sampler = make_sampler(temp=temperature)
        prompt_tokens = len(self.tokenizer.encode(prompt))

        start = time.perf_counter()

        with self._generation_lock:
            text = generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )

        elapsed = time.perf_counter() - start
        completion_tokens = len(self.tokenizer.encode(text))

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_seconds": round(elapsed, 4),
            "tokens_per_second": round(
                completion_tokens / elapsed, 2
            ) if elapsed > 0 else 0.0,
        }

  
   
    def stream(self, prompt: str, max_tokens: int = 128, temperature: float = 0.2):
        sampler = make_sampler(temp=temperature)

        with self._generation_lock:
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                yield TokenChunk(
                    text=response.text,
                    token_id=int(response.token) if response.token is not None else None,
                    prompt_tokens=response.prompt_tokens,
                    prompt_tps=response.prompt_tps,
                    generation_tokens=response.generation_tokens,
                    generation_tps=response.generation_tps,
                    peak_memory_gb=response.peak_memory,
                    finish_reason=response.finish_reason,
                )
