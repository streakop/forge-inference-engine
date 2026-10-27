from fastapi.testclient import TestClient

from forge.app import app, runtime


class FakeRuntime:
    model_id = "test-model"
    loaded = True

    def generate(self, prompt, max_tokens, temperature):
        return {
            "text": f"echo: {prompt}",
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "latency_seconds": 0.05,
            "tokens_per_second": 60.0,
        }

    def stream(self, prompt, max_tokens, temperature):
        from forge.runtime import TokenChunk
        yield TokenChunk(text="hello", token_id=1)
        yield TokenChunk(text=" world", token_id=2, prompt_tokens=2, prompt_tps=100.0, generation_tokens=2, generation_tps=50.0, peak_memory_gb=0.5, finish_reason="stop")


client = TestClient(app)
_original_runtime = runtime


def setup_function():
    import forge.app as app_module
    app_module.runtime = FakeRuntime()


def teardown_function():
    import forge.app as app_module
    app_module.runtime = _original_runtime


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["engine"] == "forge"


def test_generate_returns_measured_metrics_shape():
    response = client.post("/v1/generate", json={"prompt": "hi"})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "echo: hi"
    assert body["completion_tokens"] == 3
    assert body["tokens_per_second"] == 60.0


def test_rejects_invalid_request():
    response = client.post("/v1/generate", json={"prompt": "", "max_tokens": 0})
    assert response.status_code == 422


def test_stream_returns_server_sent_events():
    response = client.post("/v1/generate/stream", json={"prompt": "hi"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"text": "hello"' in response.text
    assert '"done": true' in response.text
    assert '"metrics"' in response.text
    assert '"generation_tokens_per_second": 50.0' in response.text
