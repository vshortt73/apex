"""Anthropic API adapter."""

from __future__ import annotations

import time

from apex.models.base import ModelAdapter
from apex.types import ChatMessage, ChatResponse, ModelInfo


class AnthropicAdapter(ModelAdapter):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **info_overrides,
    ) -> None:
        from anthropic import Anthropic

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = Anthropic(**kwargs)
        self._model = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens or 4096
        self._info_overrides = info_overrides

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self._info_overrides.get("model_id", self._model),
            backend="anthropic",
            model_name=self._model,
            architecture=self._info_overrides.get("architecture", "unknown"),
            parameters=self._info_overrides.get("parameters", "unknown"),
            quantization=self._info_overrides.get("quantization", "none"),
            max_context_window=self._info_overrides.get("max_context_window", 200000),
            tokenizer=self._info_overrides.get("tokenizer", "anthropic"),
        )

    def single_turn(self, system: str, user: str) -> ChatResponse:
        start = time.monotonic()
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=self._temperature,
        )
        latency = int((time.monotonic() - start) * 1000)
        content = resp.content[0].text if resp.content else ""
        return ChatResponse(
            content=content,
            latency_ms=latency,
            finish_reason=resp.stop_reason or "stop",
        )

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        # Extract system message if present
        system = ""
        chat_msgs = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat_msgs.append({"role": m.role, "content": m.content})

        start = time.monotonic()
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system if system else "You are a helpful assistant.",
            messages=chat_msgs,
            temperature=self._temperature,
        )
        latency = int((time.monotonic() - start) * 1000)
        content = resp.content[0].text if resp.content else ""
        return ChatResponse(
            content=content,
            latency_ms=latency,
            finish_reason=resp.stop_reason or "stop",
        )
