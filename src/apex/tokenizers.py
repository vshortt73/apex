"""Tokenizer abstraction for cross-model token counting."""

from __future__ import annotations

import abc
import logging

logger = logging.getLogger(__name__)


class TokenizerBackend(abc.ABC):
    """Abstract tokenizer interface."""

    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        ...

    @abc.abstractmethod
    def name(self) -> str:
        ...


class ApproximateTokenizer(TokenizerBackend):
    """Fallback: ~4 chars per token."""

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def name(self) -> str:
        return "approximate"


class TiktokenTokenizer(TokenizerBackend):
    """OpenAI tiktoken-based tokenizer."""

    def __init__(self, model: str = "gpt-4o") -> None:
        import tiktoken

        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")
        self._model = model

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def name(self) -> str:
        return f"tiktoken:{self._model}"


class HuggingFaceTokenizer(TokenizerBackend):
    """HuggingFace transformers tokenizer."""

    def __init__(self, model_path: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model_path = model_path

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def name(self) -> str:
        return f"hf:{self._model_path}"


class AnthropicTokenizer(TokenizerBackend):
    """Anthropic token counter via the SDK."""

    def __init__(self) -> None:
        from anthropic import Anthropic

        self._client = Anthropic()

    def count_tokens(self, text: str) -> int:
        result = self._client.count_tokens(text)
        return result

    def name(self) -> str:
        return "anthropic"


def get_tokenizer(spec: str) -> TokenizerBackend:
    """Factory: create tokenizer from config spec string.

    Formats:
      - "approximate"
      - "tiktoken:<model>"  e.g. "tiktoken:gpt-4o"
      - "anthropic"
      - anything else treated as HuggingFace model path
    """
    if spec == "approximate":
        return ApproximateTokenizer()
    if spec.startswith("tiktoken:"):
        model = spec.split(":", 1)[1]
        return TiktokenTokenizer(model)
    if spec == "anthropic":
        try:
            return AnthropicTokenizer()
        except Exception:
            logger.warning("Anthropic tokenizer unavailable, falling back to approximate")
            return ApproximateTokenizer()
    # Treat as HuggingFace model path
    try:
        return HuggingFaceTokenizer(spec)
    except Exception:
        logger.warning("HF tokenizer %s unavailable, falling back to approximate", spec)
        return ApproximateTokenizer()
