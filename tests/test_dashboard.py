"""Tests for the APEX dashboard: QueryManager and app creation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from apex.db import SqliteBackend
from apex.dashboard.queries import QueryManager

# ---- Fixtures ----


def _insert_result(conn, **overrides):
    """Insert a test result with sensible defaults."""
    defaults = dict(
        model_id="test-model",
        model_architecture="transformer",
        model_parameters="7B",
        quantization="Q4_K_M",
        max_context_window=8192,
        context_length=4096,
        context_fill_ratio=0.95,
        target_position=2048,
        target_position_percent=50.0,
        dimension="factual_recall",
        content_type="fact",
        probe_id="probe-001",
        probe_content="The capital of France is Paris.",
        filler_type="neutral",
        test_query_id="query-001",
        temperature=0.0,
        run_number=1,
        total_runs=1,
        score=0.8,
        score_method="exact_match",
        raw_response="Acknowledged.",
        raw_test_response="Paris",
        evaluator_model_id=None,
        evaluator_justification=None,
        latency_ms=500,
        timestamp=datetime.now().isoformat(),
        library_version="1.0",
        framework_version="0.1.0",
        refused=0,
    )
    defaults.update(overrides)
    cols = ", ".join(defaults.keys())
    placeholders = ", ".join("?" for _ in defaults)
    conn.execute(f"INSERT INTO probe_results ({cols}) VALUES ({placeholders})", list(defaults.values()))
    conn.commit()


@pytest.fixture
def db_path(tmp_path):
    """Create a test database with known data."""
    db = tmp_path / "test_results.db"
    backend = SqliteBackend(db)
    backend.create_schema()
    conn = backend.connection

    # Positions stored as fractions (0-1), matching real APEX data
    positions = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]

    # Model A: factual_recall and application at 4096 tokens
    for pos in positions:
        pos_pct = pos * 100
        _insert_result(conn, model_id="model-A", probe_id=f"fr-{int(pos_pct)}",
                       dimension="factual_recall", target_position_percent=pos,
                       score=max(0.0, 1.0 - abs(pos_pct - 50) / 100),
                       context_length=4096)
        _insert_result(conn, model_id="model-A", probe_id=f"app-{int(pos_pct)}",
                       dimension="application", target_position_percent=pos,
                       score=max(0.0, 0.9 - abs(pos_pct - 30) / 100),
                       context_length=4096)

    # Model A: factual_recall at 8192 tokens
    for pos in positions:
        pos_pct = pos * 100
        _insert_result(conn, model_id="model-A", probe_id=f"fr8k-{int(pos_pct)}",
                       dimension="factual_recall", target_position_percent=pos,
                       score=max(0.0, 0.8 - abs(pos_pct - 50) / 100),
                       context_length=8192)

    # Model B: factual_recall at 4096 tokens
    for pos in positions:
        pos_pct = pos * 100
        _insert_result(conn, model_id="model-B", probe_id=f"fr-b-{int(pos_pct)}",
                       model_architecture="mamba", model_parameters="3B",
                       dimension="factual_recall", target_position_percent=pos,
                       score=max(0.0, 0.7 - abs(pos_pct - 50) / 100),
                       context_length=4096)

    # One refused result
    _insert_result(conn, model_id="model-A", probe_id="refused-1",
                   dimension="salience", target_position_percent=0.50,
                   score=None, refused=1, context_length=4096)

    # One null-score result
    _insert_result(conn, model_id="model-A", probe_id="null-score-1",
                   dimension="salience", target_position_percent=0.50,
                   score=None, refused=0, context_length=4096)

    backend.close()
    return db


@pytest.fixture
def qm(db_path):
    return QueryManager(db_path)


# ---- Discovery tests ----

class TestDiscovery:
    def test_get_models(self, qm):
        models = qm.get_models()
        ids = [m["model_id"] for m in models]
        assert "model-A" in ids
        assert "model-B" in ids

    def test_get_models_metadata(self, qm):
        models = {m["model_id"]: m for m in qm.get_models()}
        assert models["model-B"]["architecture"] == "mamba"
        assert models["model-B"]["parameters"] == "3B"

    def test_get_dimensions(self, qm):
        dims = qm.get_dimensions()
        assert "factual_recall" in dims
        assert "application" in dims

    def test_get_dimensions_for_model(self, qm):
        dims = qm.get_dimensions("model-B")
        assert dims == ["factual_recall"]

    def test_get_context_lengths(self, qm):
        cls = qm.get_context_lengths("model-A")
        assert 4096 in cls
        assert 8192 in cls

    def test_get_probe_ids(self, qm):
        probes = qm.get_probe_ids("model-A", "factual_recall")
        assert len(probes) > 0
        assert all(p.startswith("fr") for p in probes)


# ---- Summary tests ----

class TestSummary:
    def test_run_summary(self, qm):
        summary = qm.get_run_summary()
        assert len(summary) == 2  # model-A and model-B
        model_a = summary[summary["model_id"] == "model-A"].iloc[0]
        assert model_a["refused_count"] == 1

    def test_dimension_breakdown(self, qm):
        breakdown = qm.get_dimension_breakdown("model-A")
        dims = set(breakdown["dimension"])
        assert "factual_recall" in dims
        assert "application" in dims


# ---- Curve data tests ----

class TestCurveData:
    def test_get_curve_data(self, qm):
        df = qm.get_curve_data("model-A", 4096)
        assert len(df) > 0
        assert "score" in df.columns
        assert "target_position_percent" in df.columns

    def test_get_curve_data_with_dimension_filter(self, qm):
        df = qm.get_curve_data("model-A", 4096, "factual_recall")
        dims = df["dimension"].unique()
        assert list(dims) == ["factual_recall"]


# ---- Aggregation tests ----

class TestAggregation:
    def test_aggregate_curve(self, qm):
        df = qm.get_curve_data("model-A", 4096, "factual_recall")
        agg = qm.aggregate_curve(df, "dimension")
        assert "mean" in agg.columns
        assert "ci_lower" in agg.columns
        assert "ci_upper" in agg.columns
        assert (agg["ci_lower"] >= 0).all()
        assert (agg["ci_upper"] <= 1).all()

    def test_aggregate_curve_empty(self, qm):
        import pandas as pd
        empty = pd.DataFrame(columns=["dimension", "target_position_percent", "score", "refused"])
        agg = qm.aggregate_curve(empty)
        assert agg.empty

    def test_aggregate_curve_n1(self, qm):
        """Single result per position: std should be 0, CI degenerates."""
        df = qm.get_curve_data("model-A", 4096, "factual_recall")
        agg = qm.aggregate_curve(df, "dimension")
        # All n=1, so std should be NaN→0
        assert (agg["std"] == 0).all() or (agg["count"] == 1).all()

    def test_aggregate_curve_all_refused(self, qm):
        """If all rows are refused, aggregation returns empty."""
        import pandas as pd
        df = pd.DataFrame({
            "dimension": ["factual_recall"] * 3,
            "target_position_percent": [0, 50, 100],
            "score": [0.5, 0.6, 0.7],
            "refused": [1, 1, 1],
        })
        agg = qm.aggregate_curve(df, "dimension")
        assert agg.empty

    def test_aggregate_curve_all_null_scores(self, qm):
        """If all scores are null, aggregation returns empty."""
        import pandas as pd
        df = pd.DataFrame({
            "dimension": ["factual_recall"] * 3,
            "target_position_percent": [0, 50, 100],
            "score": [None, None, None],
            "refused": [0, 0, 0],
        })
        agg = qm.aggregate_curve(df, "dimension")
        assert agg.empty


# ---- Cross-model tests ----

class TestCrossModel:
    def test_get_cross_model_data(self, qm):
        df = qm.get_cross_model_data(["model-A", "model-B"], "factual_recall", 4096)
        models = set(df["model_id"])
        assert "model-A" in models
        assert "model-B" in models


# ---- Probe detail tests ----

class TestProbeDetail:
    def test_get_probe_detail(self, qm):
        df = qm.get_probe_detail("fr-50", "model-A")
        assert len(df) == 1
        assert df.iloc[0]["score"] == pytest.approx(1.0, abs=0.01)
        # Position should be returned as percentage (0-100)
        assert df.iloc[0]["target_position_percent"] == pytest.approx(50.0)

    def test_get_probe_metadata(self, qm):
        meta = qm.get_probe_metadata("fr-50")
        assert meta is not None
        assert meta["dimension"] == "factual_recall"
        assert meta["score_method"] == "exact_match"


# ---- Analysis tests ----

class TestAnalysis:
    def test_compute_correlations(self, qm):
        df = qm.get_curve_data("model-A", 4096)
        agg = qm.aggregate_curve(df, "dimension")
        corrs = qm.compute_dimension_correlations(agg)
        # At least one pair should exist (factual_recall vs application)
        assert len(corrs) >= 1
        for key, val in corrs.items():
            assert -1.0 <= val <= 1.0

    def test_find_sweet_spots(self, qm):
        df = qm.get_curve_data("model-A", 4096)
        agg = qm.aggregate_curve(df, "dimension")
        spots = qm.find_sweet_spots(agg, threshold=0.5)
        # Should be a DataFrame with target_position_percent column
        assert "target_position_percent" in spots.columns


# ---- Launch history tests ----

class TestLaunchHistory:
    def test_record_and_retrieve_launch(self, qm):
        qm.record_launch(
            launch_id="launch-001",
            node="node1",
            model_path="/models/test.gguf",
            port=8080,
            requested_ctx_per_slot=8192,
            parallel=2,
            total_ctx=16384,
            gpu_layers=999,
            threads=None,
            flash_attn=True,
            llama_server_bin="/usr/bin/llama-server",
            pid=12345,
            status="unknown",
            launched_at="2026-02-28T12:00:00+00:00",
        )
        df = qm.get_launch_history()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["launch_id"] == "launch-001"
        assert row["node"] == "node1"
        assert row["model_path"] == "/models/test.gguf"
        assert row["port"] == 8080
        assert row["requested_ctx_per_slot"] == 8192
        assert row["parallel"] == 2
        assert row["total_ctx"] == 16384
        assert row["gpu_layers"] == 999
        assert row["flash_attn"] == 1
        assert row["pid"] == 12345
        assert row["status"] == "unknown"

    def test_update_launch_actual(self, qm):
        qm.record_launch(
            launch_id="launch-002",
            node="node1",
            model_path="/models/test.gguf",
            port=8080,
            requested_ctx_per_slot=8192,
            parallel=2,
            total_ctx=16384,
            gpu_layers=999,
            threads=None,
            flash_attn=True,
            llama_server_bin="/usr/bin/llama-server",
            pid=12346,
            status="unknown",
            launched_at="2026-02-28T12:01:00+00:00",
        )
        qm.update_launch_actual(
            launch_id="launch-002",
            actual_ctx_per_slot=4096,
            model_id_reported="test-model-7b",
            n_params=7000000000,
            n_ctx_train=8192,
            notes="Context truncated: requested 8,192/slot but server allocated 4,096/slot",
        )
        rec = qm.get_launch_by_id("launch-002")
        assert rec is not None
        assert rec["actual_ctx_per_slot"] == 4096
        assert rec["model_id_reported"] == "test-model-7b"
        assert rec["n_params"] == 7000000000
        assert rec["n_ctx_train"] == 8192
        assert "truncated" in rec["notes"].lower()
        assert rec["status"] == "running"

    def test_launch_history_empty(self, db_path):
        """Fresh QM with no launches returns empty DataFrame."""
        fresh_qm = QueryManager(db_path)
        df = fresh_qm.get_launch_history()
        assert df.empty

    def test_get_launch_by_id_not_found(self, qm):
        assert qm.get_launch_by_id("nonexistent-id") is None


# ---- Per-run curve data tests ----

class TestPerRunCurve:
    """Tests for per-run curve explorer features."""

    @pytest.fixture
    def db_with_runs(self, tmp_path):
        """Create a database with known run_uuid data."""
        db = tmp_path / "runs_test.db"
        backend = SqliteBackend(db)
        backend.create_schema()
        conn = backend.connection

        positions = [0.0, 0.25, 0.50, 0.75, 1.0]
        uuid_a = "aaaaaaaa-1111-2222-3333-444444444444"
        uuid_b = "bbbbbbbb-1111-2222-3333-444444444444"

        for pos in positions:
            # Run A: factual_recall at 4096
            _insert_result(conn, model_id="model-A", probe_id=f"fr-a-{int(pos*100)}",
                           dimension="factual_recall", target_position_percent=pos,
                           score=max(0.0, 1.0 - pos), context_length=4096,
                           run_uuid=uuid_a, filler_type="neutral",
                           timestamp="2026-03-01T10:00:00")
            # Run B: factual_recall at 4096
            _insert_result(conn, model_id="model-A", probe_id=f"fr-b-{int(pos*100)}",
                           dimension="factual_recall", target_position_percent=pos,
                           score=max(0.0, 0.8 - pos * 0.5), context_length=4096,
                           run_uuid=uuid_b, filler_type="adversarial",
                           timestamp="2026-03-01T11:00:00")
            # Run A: factual_recall at 8192
            _insert_result(conn, model_id="model-A", probe_id=f"fr-a8k-{int(pos*100)}",
                           dimension="factual_recall", target_position_percent=pos,
                           score=max(0.0, 0.9 - pos), context_length=8192,
                           run_uuid=uuid_a, filler_type="neutral",
                           timestamp="2026-03-01T10:05:00")

        backend.close()
        return db

    @pytest.fixture
    def qm_runs(self, db_with_runs):
        return QueryManager(db_with_runs)

    def test_get_curve_data_includes_run_uuid(self, qm_runs):
        """get_curve_data should include run_uuid column."""
        df = qm_runs.get_curve_data("model-A", 4096)
        assert "run_uuid" in df.columns
        assert df["run_uuid"].notna().all()

    def test_get_run_uuids_for_model(self, qm_runs):
        """get_run_uuids_for_model returns grouped run info."""
        df = qm_runs.get_run_uuids_for_model("model-A")
        assert len(df) == 2  # uuid_a has 2 context lengths but groups by filler too
        uuids = set(df["run_uuid"])
        assert "aaaaaaaa-1111-2222-3333-444444444444" in uuids
        assert "bbbbbbbb-1111-2222-3333-444444444444" in uuids
        assert "result_count" in df.columns
        assert "filler_type" in df.columns

    def test_get_run_uuids_for_model_with_context_filter(self, qm_runs):
        """Filtering by context_length narrows results."""
        df = qm_runs.get_run_uuids_for_model("model-A", context_length=8192)
        assert len(df) == 1
        assert df.iloc[0]["run_uuid"] == "aaaaaaaa-1111-2222-3333-444444444444"
        assert df.iloc[0]["result_count"] == 5

    def test_get_run_uuids_for_model_empty(self, qm_runs):
        """Non-existent model returns empty DataFrame."""
        df = qm_runs.get_run_uuids_for_model("nonexistent-model")
        assert df.empty
        assert "run_uuid" in df.columns

    def test_aggregate_curve_by_run_uuid(self, qm_runs):
        """aggregate_curve works with 'run_uuid' as group_col."""
        df = qm_runs.get_curve_data("model-A", 4096)
        agg = QueryManager.aggregate_curve(df, "run_uuid")
        assert "run_uuid" in agg.columns
        assert "mean" in agg.columns
        uuids = set(agg["run_uuid"])
        assert len(uuids) == 2


# ---- App creation test ----

class TestAppCreation:
    def test_create_app(self, db_path):
        from apex.dashboard import create_app
        app = create_app(str(db_path))
        assert app is not None
        assert app.title == "APEX Console"
