"""Tests for the probe runner (with mocked model calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from apex.config import ModelConfig, RunConfig, load_config
from apex.libraries import ProbeLibrary
from apex.runner import ProbeRunner, _is_refusal
from apex.storage import ResultStore
from apex.types import ChatResponse, ModelInfo


def test_is_refusal_empty():
    assert _is_refusal(ChatResponse(content="", latency_ms=0))


def test_is_refusal_whitespace():
    assert _is_refusal(ChatResponse(content="   ", latency_ms=0))


def test_is_refusal_phrase():
    assert _is_refusal(ChatResponse(content="I cannot help with that request.", latency_ms=0))


def test_is_not_refusal():
    assert not _is_refusal(ChatResponse(content="The answer is 42.", latency_ms=0))


def test_runner_with_mock(tmp_data_dir, tmp_path):
    """Test runner with mocked adapter — no real model calls."""
    config = RunConfig(
        seed=42,
        temperature=0.0,
        repetitions=1,
        filler_type="neutral",
        data_dir=str(tmp_data_dir),
        output_db=str(tmp_path / "test.db"),
        positions=[0.5],
        context_lengths=[2048],
        probe_select=["F-001"],
        models=[
            ModelConfig(
                name="mock-model",
                backend="ollama",
                model_name="mock",
                tokenizer="approximate",
                max_context_window=4096,
            )
        ],
    )

    library = ProbeLibrary(config.data_dir)
    store = ResultStore(config.output_db)

    mock_adapter = MagicMock()
    mock_adapter.get_model_info.return_value = ModelInfo(
        model_id="mock-model",
        backend="ollama",
        model_name="mock",
        max_context_window=4096,
    )
    mock_adapter.single_turn.return_value = ChatResponse(
        content="I read about the Millau Viaduct.", latency_ms=100
    )
    mock_adapter.chat.return_value = ChatResponse(
        content="Michel Virlogeux designed the Millau Viaduct.", latency_ms=100
    )

    runner = ProbeRunner(config, library, store)

    with patch("apex.runner.get_adapter", return_value=mock_adapter):
        with patch("apex.runner.ScoringDispatcher") as mock_dispatcher_cls:
            mock_dispatcher = MagicMock()
            mock_dispatcher.score.return_value = (1.0, None, None)
            mock_dispatcher_cls.return_value = mock_dispatcher
            runner.run()

    assert store.count_results() == 1
    results = store.query_results()
    assert results[0]["probe_id"] == "F-001"
    assert results[0]["score"] == 1.0
    store.close()


def test_runner_resume(tmp_data_dir, tmp_path):
    """Test that runner skips already-completed probes."""
    config = RunConfig(
        seed=42,
        temperature=0.0,
        repetitions=1,
        filler_type="neutral",
        data_dir=str(tmp_data_dir),
        output_db=str(tmp_path / "test.db"),
        positions=[0.5],
        context_lengths=[2048],
        probe_select=["F-001"],
        models=[
            ModelConfig(
                name="mock-model",
                backend="ollama",
                model_name="mock",
                tokenizer="approximate",
                max_context_window=4096,
            )
        ],
    )

    library = ProbeLibrary(config.data_dir)
    store = ResultStore(config.output_db)

    mock_adapter = MagicMock()
    mock_adapter.get_model_info.return_value = ModelInfo(
        model_id="mock-model", backend="ollama", model_name="mock", max_context_window=4096,
    )
    mock_adapter.single_turn.return_value = ChatResponse(content="Response.", latency_ms=100)
    mock_adapter.chat.return_value = ChatResponse(content="Michel Virlogeux.", latency_ms=100)

    runner = ProbeRunner(config, library, store)

    # Run once
    with patch("apex.runner.get_adapter", return_value=mock_adapter):
        with patch("apex.runner.ScoringDispatcher") as mock_d:
            mock_d.return_value.score.return_value = (1.0, None, None)
            runner.run()

    assert store.count_results() == 1
    call_count_1 = mock_adapter.single_turn.call_count

    # Run again — should skip
    mock_adapter.reset_mock()
    with patch("apex.runner.get_adapter", return_value=mock_adapter):
        with patch("apex.runner.ScoringDispatcher") as mock_d:
            mock_d.return_value.score.return_value = (1.0, None, None)
            runner.run()

    assert mock_adapter.single_turn.call_count == 0
    assert store.count_results() == 1
    store.close()
