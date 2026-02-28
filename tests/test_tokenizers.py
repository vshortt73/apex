"""Tests for tokenizer abstraction."""

from apex.tokenizers import ApproximateTokenizer, get_tokenizer


def test_approximate_tokenizer():
    tok = ApproximateTokenizer()
    count = tok.count_tokens("Hello world, this is a test.")
    assert count > 0
    assert tok.name() == "approximate"


def test_approximate_is_roughly_correct():
    tok = ApproximateTokenizer()
    text = "a" * 400
    count = tok.count_tokens(text)
    assert 90 <= count <= 110  # ~4 chars per token = 100


def test_get_tokenizer_approximate():
    tok = get_tokenizer("approximate")
    assert isinstance(tok, ApproximateTokenizer)


def test_get_tokenizer_unknown_fallback():
    tok = get_tokenizer("some/nonexistent/model")
    # Should fall back to approximate
    assert tok.name() == "approximate"


def test_get_tokenizer_tiktoken():
    try:
        tok = get_tokenizer("tiktoken:gpt-4o")
        assert tok.name().startswith("tiktoken")
        count = tok.count_tokens("Hello world")
        assert count > 0
    except ImportError:
        pass  # tiktoken not installed
