"""Ollama model adapter — REST API via httpx."""

from __future__ import annotations

import time

import httpx

from apex.models.base import ModelAdapter
from apex.types import ChatMessage, ChatResponse, ModelInfo


class OllamaAdapter(ModelAdapter):
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **info_overrides,
    ) -> None:
        self._model = model_name
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._info_overrides = info_overrides
        self._client = httpx.Client(base_url=self._base_url, timeout=600.0)

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self._info_overrides.get("model_id", self._model),
            backend="ollama",
            model_name=self._model,
            architecture=self._info_overrides.get("architecture", "unknown"),
            parameters=self._info_overrides.get("parameters", "unknown"),
            quantization=self._info_overrides.get("quantization", "none"),
            max_context_window=self._info_overrides.get("max_context_window", 4096),
            tokenizer=self._info_overrides.get("tokenizer", "approximate"),
        )

    def single_turn(self, system: str, user: str) -> ChatResponse:
        return self.chat([
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ])

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        if self._max_tokens is not None:
            payload["options"]["num_predict"] = self._max_tokens
        start = time.monotonic()
        resp = self._client.post("/api/chat", json=payload)
        latency = int((time.monotonic() - start) * 1000)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return ChatResponse(
            content=content,
            latency_ms=latency,
            finish_reason=data.get("done_reason", "stop"),
        )
