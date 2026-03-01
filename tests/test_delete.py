"""Tests for run UUID tracking and delete operations."""

import sqlite3
from datetime import datetime, timezone

import pytest

from apex.db import COLUMN_NAMES, SqliteBackend
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


def test_run_uuid_written(tmp_path):
    """Write result with run_uuid, verify it's stored and queryable."""
    store = ResultStore(tmp_path / "test.db")
    result = _make_result(run_uuid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    store.write_result(result)

    rows = store.query_results()
    assert len(rows) == 1
    assert rows[0]["run_uuid"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    store.close()


def test_run_uuid_null_by_default(tmp_path):
    """Results without run_uuid should have NULL."""
    store = ResultStore(tmp_path / "test.db")
    store.write_result(_make_result())
    rows = store.query_results()
    assert rows[0]["run_uuid"] is None
    store.close()


def test_delete_by_run_uuid(tmp_path):
    """Write 3 results with same UUID + 1 with different UUID, delete by first UUID."""
    store = ResultStore(tmp_path / "test.db")
    uuid_a = "aaaaaaaa-1111-1111-1111-111111111111"
    uuid_b = "bbbbbbbb-2222-2222-2222-222222222222"

    store.write_result(_make_result(run_uuid=uuid_a, run_number=1))
    store.write_result(_make_result(run_uuid=uuid_a, run_number=2))
    store.write_result(_make_result(run_uuid=uuid_a, run_number=3))
    store.write_result(_make_result(run_uuid=uuid_b, run_number=1, probe_id="F-002"))

    assert store.count_results() == 4

    deleted = store.delete_by_run_uuid(uuid_a)
    assert deleted == 3
    assert store.count_results() == 1

    # The remaining result should be the one with uuid_b
    rows = store.query_results()
    assert rows[0]["run_uuid"] == uuid_b
    assert rows[0]["probe_id"] == "F-002"
    store.close()


def test_delete_by_model(tmp_path):
    """Write results for 2 models, delete one, verify other remains."""
    store = ResultStore(tmp_path / "test.db")
    store.write_result(_make_result(model_id="model-A", probe_id="F-001", run_number=1))
    store.write_result(_make_result(model_id="model-A", probe_id="F-002", run_number=1))
    store.write_result(_make_result(model_id="model-B", probe_id="F-001", run_number=1))

    assert store.count_results() == 3

    deleted = store.delete_by_model("model-A")
    assert deleted == 2
    assert store.count_results() == 1

    rows = store.query_results()
    assert rows[0]["model_id"] == "model-B"
    store.close()


def test_delete_by_filters(tmp_path):
    """Delete with combined model+dimension filter."""
    store = ResultStore(tmp_path / "test.db")
    store.write_result(_make_result(model_id="m1", dimension="factual_recall", probe_id="F-001", run_number=1))
    store.write_result(_make_result(model_id="m1", dimension="application", probe_id="A-001", run_number=1))
    store.write_result(_make_result(model_id="m2", dimension="factual_recall", probe_id="F-001", run_number=1))

    deleted = store.delete_by_filters(model_id="m1", dimension="factual_recall")
    assert deleted == 1
    assert store.count_results() == 2

    # m1/application and m2/factual_recall should remain
    rows = store.query_results()
    ids = {(r["model_id"], r["dimension"]) for r in rows}
    assert ("m1", "application") in ids
    assert ("m2", "factual_recall") in ids
    store.close()


def test_delete_by_filters_no_filters(tmp_path):
    """delete_by_filters with no filters returns 0 (safety: won't delete everything)."""
    store = ResultStore(tmp_path / "test.db")
    store.write_result(_make_result())
    deleted = store.delete_by_filters()
    assert deleted == 0
    assert store.count_results() == 1
    store.close()


def test_get_run_uuids(tmp_path):
    """get_run_uuids returns correct summary."""
    store = ResultStore(tmp_path / "test.db")
    uuid_a = "aaaaaaaa-1111-1111-1111-111111111111"
    uuid_b = "bbbbbbbb-2222-2222-2222-222222222222"

    store.write_result(_make_result(run_uuid=uuid_a, run_number=1, model_id="m1"))
    store.write_result(_make_result(run_uuid=uuid_a, run_number=2, model_id="m1"))
    store.write_result(_make_result(run_uuid=uuid_b, run_number=1, model_id="m2", probe_id="F-002"))
    # Result without UUID should not appear
    store.write_result(_make_result(run_number=3, model_id="m3", probe_id="F-003"))

    uuids = store.get_run_uuids()
    assert len(uuids) == 2

    uuid_map = {u["run_uuid"]: u for u in uuids}
    assert uuid_map[uuid_a]["model_id"] == "m1"
    assert uuid_map[uuid_a]["count"] == 2
    assert uuid_map[uuid_b]["model_id"] == "m2"
    assert uuid_map[uuid_b]["count"] == 1
    store.close()


def test_delete_dry_run_cli(tmp_path, capsys, monkeypatch):
    """Test CLI --list-runs and --dry-run output."""
    from apex.cli import main

    # Ensure env var doesn't override our test db path
    monkeypatch.delenv("APEX_DATABASE_URL", raising=False)

    db_path = str(tmp_path / "test.db")
    store = ResultStore(db_path)
    uuid_a = "aaaaaaaa-1111-1111-1111-111111111111"
    store.write_result(_make_result(run_uuid=uuid_a, model_id="test-model"))
    store.write_result(_make_result(run_uuid=uuid_a, model_id="test-model", run_number=2))
    store.close()

    # --list-runs
    main(["delete", db_path, "--list-runs"])
    captured = capsys.readouterr()
    assert uuid_a in captured.out

    # --dry-run
    main(["delete", db_path, "--model", "test-model", "--dry-run"])
    captured = capsys.readouterr()
    assert "2 result(s) would be deleted" in captured.out


def test_schema_migration(tmp_path):
    """Create table without run_uuid, then call create_schema again, verify column exists."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    # Create a table without the run_uuid column
    conn.execute("""
        CREATE TABLE probe_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            probe_id TEXT NOT NULL,
            target_position_percent REAL NOT NULL,
            context_length INTEGER NOT NULL,
            run_number INTEGER NOT NULL,
            UNIQUE(model_id, probe_id, target_position_percent, context_length, run_number)
        )
    """)
    conn.commit()
    conn.close()

    # Now open via SqliteBackend — create_schema should add missing columns
    backend = SqliteBackend(db_path)
    backend.create_schema()

    # Verify run_uuid column exists
    cursor = backend.connection.execute("PRAGMA table_info(probe_results)")
    columns = {row[1] for row in cursor.fetchall()}
    assert "run_uuid" in columns
    backend.close()
