"""OpenAI API adapter."""

from __future__ import annotations

import time

from apex.models.base import ModelAdapter
from apex.types import ChatMessage, ChatResponse, ModelInfo


class OpenAIAdapter(ModelAdapter):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **info_overrides,
    ) -> None:
        from openai import OpenAI

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._info_overrides = info_overrides

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self._info_overrides.get("model_id", self._model),
            backend="openai",
            model_name=self._model,
            architecture=self._info_overrides.get("architecture", "unknown"),
            parameters=self._info_overrides.get("parameters", "unknown"),
            quantization=self._info_overrides.get("quantization", "none"),
            max_context_window=self._info_overrides.get("max_context_window", 128000),
            tokenizer=self._info_overrides.get("tokenizer", f"tiktoken:{self._model}"),
        )

    def single_turn(self, system: str, user: str) -> ChatResponse:
        return self.chat([
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ])

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        start = time.monotonic()
        kwargs = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        resp = self._client.chat.completions.create(**kwargs)
        latency = int((time.monotonic() - start) * 1000)
        choice = resp.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            latency_ms=latency,
            finish_reason=choice.finish_reason or "stop",
        )
