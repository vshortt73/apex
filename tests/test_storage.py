"""Tests for SQLite storage."""

from datetime import datetime, timezone

import pytest

from apex.storage import ResultStore
from apex.types import ProbeResult


def _make_result(**overrides) -> ProbeResult:
    defaults = dict(
        model_id="test-model",
        model_architecture="transformer",
        model_parameters="7B",
        quantization="Q4_K_M",
        max_context_window=4096,
        context_length=2048,
        context_fill_ratio=0.5,
        target_position=512,
        target_position_percent=0.25,
        dimension="factual_recall",
        content_type="factual",
        probe_id="F-001",
        probe_content="Test probe content.",
        filler_type="neutral",
        test_query_id="FT-001",
        temperature=0.0,
        run_number=1,
        total_runs=3,
        score=1.0,
        score_method="exact_match",
        raw_response="Model response",
        raw_test_response="Test response",
        latency_ms=500,
        timestamp=datetime.now(timezone.utc).isoformat(),
        library_version="0.1.0-test",
        framework_version="0.1.0",
    )
    defaults.update(overrides)
    return ProbeResult(**defaults)


def test_create_and_write(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    result = _make_result()
    store.write_result(result)
    assert store.count_results() == 1
    store.close()


def test_write_multiple(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    for i in range(5):
        store.write_result(_make_result(run_number=i + 1))
    assert store.count_results() == 5
    store.close()


def test_unique_constraint(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    r = _make_result()
    store.write_result(r)
    # Writing the same key again should replace (INSERT OR REPLACE)
    store.write_result(_make_result(score=0.5))
    assert store.count_results() == 1
    results = store.query_results()
    assert results[0]["score"] == 0.5
    store.close()


def test_get_completed_runs(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    store.write_result(_make_result(probe_id="F-001", target_position_percent=0.1, run_number=1))
    store.write_result(_make_result(probe_id="F-001", target_position_percent=0.5, run_number=1))
    store.write_result(_make_result(probe_id="F-002", target_position_percent=0.1, run_number=1))

    completed = store.get_completed_runs("test-model")
    assert ("F-001", 0.1, 2048, 1) in completed
    assert ("F-001", 0.5, 2048, 1) in completed
    assert ("F-002", 0.1, 2048, 1) in completed
    assert len(completed) == 3
    store.close()


def test_query_results_filtered(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    store.write_result(_make_result(probe_id="F-001", dimension="factual_recall", run_number=1))
    store.write_result(_make_result(probe_id="A-001", dimension="application", run_number=1))

    factual = store.query_results(dimension="factual_recall")
    assert len(factual) == 1
    assert factual[0]["probe_id"] == "F-001"

    all_results = store.query_results()
    assert len(all_results) == 2
    store.close()


def test_export_json(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    store.write_result(_make_result(run_number=1))
    store.write_result(_make_result(run_number=2))

    out = tmp_path / "export.json"
    count = store.export_json(out)
    assert count == 2
    assert out.exists()

    import json

    data = json.loads(out.read_text())
    assert len(data) == 2
    store.close()


def test_refused_result(tmp_path):
    store = ResultStore(tmp_path / "test.db")
    store.write_result(_make_result(score=None, refused=True))
    results = store.query_results()
    assert results[0]["score"] is None
    assert results[0]["refused"] == 1
    store.close()
