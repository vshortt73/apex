"""Model adapter abstract base class."""

from __future__ import annotations

import abc

from apex.types import ChatMessage, ChatResponse, ModelInfo


class ModelAdapter(abc.ABC):
    """Abstract interface for model inference backends."""

    @abc.abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return metadata about the model."""
        ...

    @abc.abstractmethod
    def single_turn(self, system: str, user: str) -> ChatResponse:
        """Send a single user message with system prompt. Returns model response."""
        ...

    @abc.abstractmethod
    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        """Send a multi-turn conversation. Returns model response."""
        ...


def get_adapter(
    backend: str,
    model_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    model_info_overrides: dict | None = None,
) -> ModelAdapter:
    """Factory: create the right adapter for a backend string."""
    overrides = model_info_overrides or {}

    if backend == "ollama":
        from apex.models.ollama import OllamaAdapter

        return OllamaAdapter(
            model_name=model_name,
            base_url=base_url or "http://localhost:11434",
            temperature=temperature,
            max_tokens=max_tokens,
            **overrides,
        )
    elif backend == "llamacpp":
        from apex.models.llamacpp import LlamaCppAdapter

        return LlamaCppAdapter(
            model_name=model_name,
            base_url=base_url or "http://localhost:8080",
            temperature=temperature,
            max_tokens=max_tokens,
            **overrides,
        )
    elif backend == "sglang":
        from apex.models.sglang import SGLangAdapter

        return SGLangAdapter(
            model_name=model_name,
            base_url=base_url or "http://localhost:30000",
            temperature=temperature,
            max_tokens=max_tokens,
            **overrides,
        )
    elif backend == "openai":
        from apex.models.openai import OpenAIAdapter

        return OpenAIAdapter(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **overrides,
        )
    elif backend == "anthropic":
        from apex.models.anthropic import AnthropicAdapter

        return AnthropicAdapter(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **overrides,
        )
    elif backend == "google":
        from apex.models.google import GoogleAdapter

        return GoogleAdapter(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **overrides,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
