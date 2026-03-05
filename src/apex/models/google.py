"""Google Gemini API adapter."""

from __future__ import annotations

import time

from apex.models.base import ModelAdapter
from apex.types import ChatMessage, ChatResponse, ModelInfo


class GoogleAdapter(ModelAdapter):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **info_overrides,
    ) -> None:
        from google import genai

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = genai.Client(**kwargs)
        self._model = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._info_overrides = info_overrides

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self._info_overrides.get("model_id", self._model),
            backend="google",
            model_name=self._model,
            architecture=self._info_overrides.get("architecture", "unknown"),
            parameters=self._info_overrides.get("parameters", "unknown"),
            quantization=self._info_overrides.get("quantization", "none"),
            max_context_window=self._info_overrides.get("max_context_window", 1000000),
            tokenizer=self._info_overrides.get("tokenizer", "approximate"),
        )

    def single_turn(self, system: str, user: str) -> ChatResponse:
        return self.chat([
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ])

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        from google.genai import types

        system_text = ""
        contents = []
        for m in messages:
            if m.role == "system":
                system_text = m.content
            else:
                role = "user" if m.role == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part(text=m.content)]))

        config_kwargs = {
            "temperature": self._temperature,
            "system_instruction": system_text if system_text else None,
        }
        if self._max_tokens is not None:
            config_kwargs["max_output_tokens"] = self._max_tokens
        config = types.GenerateContentConfig(**config_kwargs)

        start = time.monotonic()
        resp = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        latency = int((time.monotonic() - start) * 1000)

        content = resp.text or ""
        finish_reason = "stop"
        if resp.candidates and resp.candidates[0].finish_reason:
            finish_reason = resp.candidates[0].finish_reason.name.lower()

        return ChatResponse(
            content=content,
            latency_ms=latency,
            finish_reason=finish_reason,
        )
