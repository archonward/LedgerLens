from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from ledgerlens.generation import InferenceConfig
from ledgerlens.generation.ollama import OLLAMA_MODEL, OllamaBackend, OllamaBackendError

DIGEST = "a" * 64


@dataclass
class MockTransport:
    responses: list[dict[str, object]]
    calls: list[tuple[str, str, dict[str, object] | None, float]] = field(default_factory=list)

    def request(
        self,
        method: str,
        url: str,
        payload: dict[str, object] | None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        self.calls.append((method, url, payload, timeout_seconds))
        return self.responses.pop(0)


def test_ollama_backend_requires_installed_model_and_maps_inference_config() -> None:
    transport = MockTransport(
        [
            {"models": [{"name": OLLAMA_MODEL, "digest": DIGEST}]},
            {"model": OLLAMA_MODEL, "response": "Revenue increased.", "done": True},
        ]
    )
    backend = OllamaBackend(
        timeout_seconds=45,
        keep_alive="2m",
        transport=transport,
    )

    result = backend.generate("controlled prompt", InferenceConfig(72, 0.25))

    assert result == "Revenue increased."
    assert backend.identity.to_dict() == {
        "backend": "ollama",
        "model": "qwen3:4b-instruct",
        "revision": f"sha256-{DIGEST}",
    }
    assert transport.calls[0] == ("GET", "http://127.0.0.1:11434/api/tags", None, 45)
    method, url, payload, timeout = transport.calls[1]
    assert (method, url, timeout) == ("POST", "http://127.0.0.1:11434/api/generate", 45)
    assert payload == {
        "model": "qwen3:4b-instruct",
        "prompt": "controlled prompt",
        "stream": False,
        "think": False,
        "keep_alive": "2m",
        "options": {"num_predict": 72, "temperature": 0.25},
    }
    assert "pull" not in json.dumps(transport.calls)


def test_ollama_backend_refuses_to_generate_when_exact_model_is_absent() -> None:
    transport = MockTransport(
        [{"models": [{"name": "some-other-model:latest", "digest": "b" * 64}]}]
    )
    backend = OllamaBackend(transport=transport)

    with pytest.raises(OllamaBackendError, match="is not installed.*will not download"):
        backend.generate("prompt", InferenceConfig())

    assert len(transport.calls) == 1
    assert transport.calls[0][1].endswith("/api/tags")
    assert backend.identity.revision is None


@pytest.mark.parametrize(
    "response, message",
    [
        ({"model": "wrong-model", "response": "Revenue increased.", "done": True}, "model"),
        ({"model": OLLAMA_MODEL, "response": "Revenue increased.", "done": False}, "complete"),
        ({"model": OLLAMA_MODEL, "response": "", "done": True}, "generated text"),
    ],
)
def test_ollama_backend_rejects_malformed_generation_response(
    response: dict[str, object], message: str
) -> None:
    transport = MockTransport([{"models": [{"model": OLLAMA_MODEL, "digest": DIGEST}]}, response])

    with pytest.raises(OllamaBackendError, match=message):
        OllamaBackend(transport=transport).generate("prompt", InferenceConfig())


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:11434",
        "http://example.com:11434",
        "http://user:password@localhost:11434",
        "http://localhost:11434/api",
    ],
)
def test_ollama_backend_accepts_only_loopback_http_origin(url: str) -> None:
    with pytest.raises(ValueError, match="loopback HTTP origin"):
        OllamaBackend(base_url=url, transport=MockTransport([]))


def test_ollama_backend_rejects_invalid_inventory_digest() -> None:
    backend = OllamaBackend(
        transport=MockTransport([{"models": [{"name": OLLAMA_MODEL, "digest": "not-a-hash"}]}])
    )

    with pytest.raises(OllamaBackendError, match="invalid digest"):
        backend.generate("prompt", InferenceConfig())
