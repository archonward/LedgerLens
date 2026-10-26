"""Optional local Ollama adapter for the reviewed LedgerLens generation boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ledgerlens.generation import InferenceConfig, ModelIdentity

OLLAMA_MODEL = "qwen3:4b-instruct"
OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_MAX_RESPONSE_BYTES = 1_000_000
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class OllamaBackendError(RuntimeError):
    """Ollama is unavailable or returned an unsafe or invalid response."""


class OllamaJsonTransport(Protocol):
    """Small injectable HTTP boundary used by the Ollama backend."""

    def request(
        self,
        method: str,
        url: str,
        payload: dict[str, object] | None,
        timeout_seconds: float,
    ) -> dict[str, object]: ...


class UrllibOllamaTransport:
    """Standard-library JSON transport for a loopback Ollama server."""

    def request(
        self,
        method: str,
        url: str,
        payload: dict[str, object] | None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise OllamaBackendError(f"local Ollama request failed: {exc}") from exc
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise OllamaBackendError("local Ollama response exceeded the size limit")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaBackendError("local Ollama returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise OllamaBackendError("local Ollama response must be a JSON object")
        return value


@dataclass
class OllamaBackend:
    """Local-only backend fixed to the already-installed qwen3:4b-instruct tag."""

    base_url: str = OLLAMA_DEFAULT_BASE_URL
    timeout_seconds: float = 120.0
    keep_alive: str = "5m"
    transport: OllamaJsonTransport = field(default_factory=UrllibOllamaTransport, repr=False)
    _identity: ModelIdentity = field(
        default_factory=lambda: ModelIdentity("ollama", OLLAMA_MODEL), init=False, repr=False
    )

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in _LOOPBACK_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("Ollama base_url must be an unauthenticated loopback HTTP origin")
        if (
            not isinstance(self.timeout_seconds, int | float)
            or isinstance(self.timeout_seconds, bool)
            or not 0 < self.timeout_seconds <= 3600
        ):
            raise ValueError("timeout_seconds must be numeric and between 0 and 3600")
        if not isinstance(self.keep_alive, str) or not self.keep_alive.strip():
            raise ValueError("keep_alive must be a non-empty string")
        self.base_url = self.base_url.rstrip("/")

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def generate(self, prompt: str, config: InferenceConfig) -> str:
        """Verify the exact local tag and request one non-streaming explanation."""
        self.verify_installed_model()
        response = self.transport.request(
            "POST",
            f"{self.base_url}/api/generate",
            {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "num_predict": config.max_new_tokens,
                    "temperature": config.temperature,
                },
            },
            self.timeout_seconds,
        )
        if response.get("model") != OLLAMA_MODEL:
            raise OllamaBackendError("Ollama response model does not match the required model")
        if response.get("done") is not True:
            raise OllamaBackendError("Ollama response did not complete")
        text = response.get("response")
        if not isinstance(text, str) or not text.strip():
            raise OllamaBackendError("Ollama response is missing generated text")
        return text

    def verify_installed_model(self) -> ModelIdentity:
        """Pin the exact installed tag digest without loading or generating from the model."""
        digest = self._installed_model_digest()
        self._identity = ModelIdentity("ollama", OLLAMA_MODEL, f"sha256-{digest}")
        return self._identity

    def _installed_model_digest(self) -> str:
        response = self.transport.request(
            "GET", f"{self.base_url}/api/tags", None, self.timeout_seconds
        )
        models = response.get("models")
        if not isinstance(models, list):
            raise OllamaBackendError("Ollama model inventory is malformed")
        for item in models:
            if not isinstance(item, dict):
                continue
            if item.get("name") != OLLAMA_MODEL and item.get("model") != OLLAMA_MODEL:
                continue
            digest = item.get("digest")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise OllamaBackendError("installed Ollama model has an invalid digest")
            return digest
        raise OllamaBackendError(
            f"required local model {OLLAMA_MODEL!r} is not installed; "
            "LedgerLens will not download it"
        )


__all__ = [
    "OLLAMA_DEFAULT_BASE_URL",
    "OLLAMA_MODEL",
    "OllamaBackend",
    "OllamaBackendError",
    "OllamaJsonTransport",
    "UrllibOllamaTransport",
]
