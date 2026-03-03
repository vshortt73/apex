"""Tests for the calibration subsystem."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from apex.calibration import (
    BaselineRunner,
    CalibrationGenerator,
    CalibrationValidator,
    content_hash,
    probe_hash,
)
from apex.calibration_store import CalibrationStore
from apex.cli import main
from apex.config import ModelConfig, RunConfig
from apex.libraries import ProbeLibrary
from apex.runner import ProbeRunner
from apex.storage import ResultStore
from apex.types import CalibrationBaseline, CalibrationPrompt, ChatResponse, ModelInfo

import json


# ---------------------------------------------------------------------------
# Store tests (unit, no model)
# ---------------------------------------------------------------------------


class TestCalibrationStore:
    def test_calibration_schema_created(self, tmp_path):
        """Both calibration tables exist on fresh DB."""
        db = str(tmp_path / "cal.db")
        store = CalibrationStore(db)
        import sqlite3

        conn = sqlite3.connect(db)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "calibration_prompts" in tables
        assert "calibration_baselines" in tables
        conn.close()
        store.close()

    def test_write_and_read_prompt(self, tmp_path):
        """Roundtrip write/read of a calibration prompt."""
        store = CalibrationStore(str(tmp_path / "cal.db"))
        prompt = CalibrationPrompt(
            probe_id="F-001",
            dimension="factual_recall",
            position_percent=0.5,
            context_length=4096,
            seed=42,
            full_text="some text with probe",
            actual_token_count=100,
            target_position_tokens=50,
            filler_ids_before="NF-001,NF-002",
            filler_ids_after="NF-003",
            probe_hash="abc123",
            content_hash="def456",
            generated_at="2025-01-01T00:00:00Z",
        )
        store.write_prompt(prompt)
        results = store.get_prompts()
        assert len(results) == 1
        assert results[0]["probe_id"] == "F-001"
        assert results[0]["position_percent"] == 0.5
        assert results[0]["content_hash"] == "def456"
        store.close()

    def test_prompt_upsert(self, tmp_path):
        """Same unique key → count stays 1 (upsert)."""
        store = CalibrationStore(str(tmp_path / "cal.db"))
        prompt = CalibrationPrompt(
            probe_id="F-001",
            dimension="factual_recall",
            position_percent=0.5,
            context_length=4096,
            seed=42,
            full_text="original text",
            actual_token_count=100,
            target_position_tokens=50,
            filler_ids_before="NF-001",
            filler_ids_after="NF-002",
            probe_hash="abc123",
            content_hash="hash1",
            generated_at="2025-01-01T00:00:00Z",
        )
        store.write_prompt(prompt)
        assert store.count_prompts() == 1

        # Write again with different content but same key
        prompt2 = CalibrationPrompt(
            probe_id="F-001",
            dimension="factual_recall",
            position_percent=0.5,
            context_length=4096,
            seed=42,
            full_text="updated text",
            actual_token_count=110,
            target_position_tokens=55,
            filler_ids_before="NF-001",
            filler_ids_after="NF-002",
            probe_hash="abc123",
            content_hash="hash2",
            generated_at="2025-01-02T00:00:00Z",
        )
        store.write_prompt(prompt2)
        assert store.count_prompts() == 1
        results = store.get_prompts()
        assert results[0]["content_hash"] == "hash2"
        store.close()

    def test_write_and_read_baseline(self, tmp_path):
        """Roundtrip write/read of a calibration baseline."""
        store = CalibrationStore(str(tmp_path / "cal.db"))
        baseline = CalibrationBaseline(
            probe_id="F-001",
            dimension="factual_recall",
            model_id="test-model",
            baseline_type="bare",
            score=0.85,
            score_method="exact_match",
            justification="matched",
            raw_response="response1",
            raw_test_response="response2",
            error=None,
            timestamp="2025-01-01T00:00:00Z",
        )
        store.write_baseline(baseline)
        results = store.get_baselines()
        assert len(results) == 1
        assert results[0]["score"] == 0.85
        assert results[0]["baseline_type"] == "bare"
        store.close()

    def test_baseline_upsert(self, tmp_path):
        """Same unique key → count stays 1 (upsert)."""
        store = CalibrationStore(str(tmp_path / "cal.db"))
        baseline = CalibrationBaseline(
            probe_id="F-001",
            dimension="factual_recall",
            model_id="test-model",
            baseline_type="bare",
            score=0.85,
            score_method="exact_match",
            timestamp="2025-01-01T00:00:00Z",
        )
        store.write_baseline(baseline)
        assert store.count_baselines() == 1

        baseline2 = CalibrationBaseline(
            probe_id="F-001",
            dimension="factual_recall",
            model_id="test-model",
            baseline_type="bare",
            score=0.90,
            score_method="exact_match",
            timestamp="2025-01-02T00:00:00Z",
        )
        store.write_baseline(baseline2)
        assert store.count_baselines() == 1
        results = store.get_baselines()
        assert results[0]["score"] == 0.90
        store.close()

    def test_delete_prompts(self, tmp_path):
        """delete_prompts clears all prompts."""
        store = CalibrationStore(str(tmp_path / "cal.db"))
        prompt = CalibrationPrompt(
            probe_id="F-001", dimension="factual_recall",
            position_percent=0.5, context_length=4096, seed=42,
            full_text="text", actual_token_count=10, target_position_tokens=5,
            filler_ids_before="", filler_ids_after="",
            probe_hash="a", content_hash="b", generated_at="2025-01-01",
        )
        store.write_prompt(prompt)
        assert store.count_prompts() == 1
        deleted = store.delete_prompts()
        assert deleted == 1
        assert store.count_prompts() == 0
        store.close()

    def test_delete_baselines_by_model(self, tmp_path):
        """delete_baselines with model filter."""
        store = CalibrationStore(str(tmp_path / "cal.db"))
        for model in ["model-a", "model-b"]:
            store.write_baseline(CalibrationBaseline(
                probe_id="F-001", dimension="factual_recall",
                model_id=model, baseline_type="bare",
                score=0.5, score_method="exact_match",
                timestamp="2025-01-01",
            ))
        assert store.count_baselines() == 2
        deleted = store.delete_baselines(model_id="model-a")
        assert deleted == 1
        assert store.count_baselines() == 1
        remaining = store.get_baselines()
        assert remaining[0]["model_id"] == "model-b"
        store.close()

    def test_get_baseline_for_probe(self, tmp_path):
        """get_baseline_for_probe returns correct score."""
        store = CalibrationStore(str(tmp_path / "cal.db"))
        store.write_baseline(CalibrationBaseline(
            probe_id="F-001", dimension="factual_recall",
            model_id="test-model", baseline_type="bare",
            score=0.75, score_method="exact_match",
            timestamp="2025-01-01",
        ))
        score = store.get_baseline_for_probe("F-001", "test-model", "bare")
        assert score == 0.75
        missing = store.get_baseline_for_probe("F-999", "test-model", "bare")
        assert missing is None
        store.close()


    def test_get_filler_factor(self, tmp_path):
        """get_filler_factor computes anchored / bare."""
        store = CalibrationStore(str(tmp_path / "cal.db"))
        store.write_baseline(CalibrationBaseline(
            probe_id="F-001", dimension="factual_recall",
            model_id="test-model", baseline_type="bare",
            score=1.0, score_method="exact_match",
            timestamp="2025-01-01",
        ))
        store.write_baseline(CalibrationBaseline(
            probe_id="F-001", dimension="factual_recall",
            model_id="test-model", baseline_type="anchored",
            score=0.8, score_method="exact_match",
            timestamp="2025-01-01",
        ))
        factor = store.get_filler_factor("F-001", "test-model")
        assert factor == pytest.approx(0.8)

        # Missing anchored → None
        assert store.get_filler_factor("F-999", "test-model") is None

        # Bare score of zero → None (avoid division by zero)
        store.write_baseline(CalibrationBaseline(
            probe_id="F-002", dimension="factual_recall",
            model_id="test-model", baseline_type="bare",
            score=0.0, score_method="exact_match",
            timestamp="2025-01-01",
        ))
        store.write_baseline(CalibrationBaseline(
            probe_id="F-002", dimension="factual_recall",
            model_id="test-model", baseline_type="anchored",
            score=0.5, score_method="exact_match",
            timestamp="2025-01-01",
        ))
        assert store.get_filler_factor("F-002", "test-model") is None
        store.close()


# ---------------------------------------------------------------------------
# Generator tests (unit, no model)
# ---------------------------------------------------------------------------


class TestCalibrationGenerator:
    def test_generate_correct_count(self, tmp_data_dir):
        """3 probes x 3 positions x 1 context length = 9 prompts."""
        library = ProbeLibrary(tmp_data_dir)
        gen = CalibrationGenerator(library)
        prompts = gen.generate(
            positions=[0.1, 0.5, 0.9],
            context_lengths=[2048],
        )
        assert len(prompts) == 3 * 3 * 1

    def test_generate_deterministic(self, tmp_data_dir):
        """Two runs produce identical content_hashes."""
        library = ProbeLibrary(tmp_data_dir)
        gen = CalibrationGenerator(library)
        prompts1 = gen.generate(positions=[0.5], context_lengths=[2048])
        prompts2 = gen.generate(positions=[0.5], context_lengths=[2048])
        hashes1 = [p.content_hash for p in prompts1]
        hashes2 = [p.content_hash for p in prompts2]
        assert hashes1 == hashes2

    def test_generate_fixed_filler_across_positions(self, tmp_data_dir):
        """Same filler IDs used across all positions for a given probe+context."""
        library = ProbeLibrary(tmp_data_dir)
        gen = CalibrationGenerator(library)
        prompts = gen.generate(positions=[0.1, 0.5, 0.9], context_lengths=[2048])

        # Group by probe_id — all positions should use the same filler set
        by_probe: dict[str, list[CalibrationPrompt]] = {}
        for p in prompts:
            by_probe.setdefault(p.probe_id, []).append(p)

        for probe_id, group in by_probe.items():
            # Collect all filler IDs (before + after) for each position
            all_filler_sets = []
            for p in group:
                before = set(p.filler_ids_before.split(",")) if p.filler_ids_before else set()
                after = set(p.filler_ids_after.split(",")) if p.filler_ids_after else set()
                all_filler_sets.append(before | after)

            # All positions should have the same filler passages
            for filler_set in all_filler_sets[1:]:
                assert filler_set == all_filler_sets[0], (
                    f"Filler differs across positions for {probe_id}"
                )

    def test_generate_probe_present(self, tmp_data_dir):
        """Probe content is a substring of each full_text."""
        library = ProbeLibrary(tmp_data_dir)
        gen = CalibrationGenerator(library)
        prompts = gen.generate(positions=[0.5], context_lengths=[2048])
        for p in prompts:
            probe = library.probes[p.probe_id]
            assert probe.content in p.full_text

    def test_generate_content_hash_valid(self, tmp_data_dir):
        """Stored hash matches recomputed hash."""
        library = ProbeLibrary(tmp_data_dir)
        gen = CalibrationGenerator(library)
        prompts = gen.generate(positions=[0.5], context_lengths=[2048])
        for p in prompts:
            assert p.content_hash == content_hash(p.full_text)


# ---------------------------------------------------------------------------
# Validator tests (unit, no model)
# ---------------------------------------------------------------------------


class TestCalibrationValidator:
    def test_validate_all_pass(self, tmp_data_dir, tmp_path):
        """Generate then validate — all pass."""
        library = ProbeLibrary(tmp_data_dir)
        gen = CalibrationGenerator(library)
        prompts = gen.generate(positions=[0.5], context_lengths=[2048])

        store = CalibrationStore(str(tmp_path / "cal.db"))
        store.write_prompts(prompts)
        prompt_dicts = store.get_prompts()

        validator = CalibrationValidator(library)
        results = validator.validate(prompt_dicts)
        assert all(r.passed for r in results)
        store.close()

    def test_validate_corrupted_hash(self, tmp_data_dir, tmp_path):
        """Modified content_hash → fails validation."""
        library = ProbeLibrary(tmp_data_dir)
        gen = CalibrationGenerator(library)
        prompts = gen.generate(positions=[0.5], context_lengths=[2048])

        store = CalibrationStore(str(tmp_path / "cal.db"))
        store.write_prompts(prompts)
        prompt_dicts = store.get_prompts()

        # Corrupt hash
        prompt_dicts[0]["content_hash"] = "corrupted"

        validator = CalibrationValidator(library)
        results = validator.validate(prompt_dicts)
        assert not results[0].passed
        assert not results[0].checks["content_hash"]

        store.close()

    def test_validate_missing_probe(self, tmp_data_dir, tmp_path):
        """Unknown probe_id → fails validation."""
        library = ProbeLibrary(tmp_data_dir)
        gen = CalibrationGenerator(library)
        prompts = gen.generate(positions=[0.5], context_lengths=[2048])

        store = CalibrationStore(str(tmp_path / "cal.db"))
        store.write_prompts(prompts)
        prompt_dicts = store.get_prompts()

        # Change probe_id to unknown
        prompt_dicts[0]["probe_id"] = "UNKNOWN-999"

        validator = CalibrationValidator(library)
        results = validator.validate(prompt_dicts)
        assert not results[0].passed
        assert not results[0].checks["probe_present"]

        store.close()


# ---------------------------------------------------------------------------
# Baseline tests (mocked adapter)
# ---------------------------------------------------------------------------


class TestBaselineRunner:
    def _make_mock_adapter(self):
        adapter = MagicMock()
        adapter.get_model_info.return_value = ModelInfo(
            model_id="mock-model", backend="ollama",
            model_name="mock", max_context_window=4096,
        )
        adapter.single_turn.return_value = ChatResponse(
            content="I read the information.", latency_ms=50,
        )
        adapter.chat.return_value = ChatResponse(
            content="Michel Virlogeux designed the Millau Viaduct.", latency_ms=50,
        )
        return adapter

    def _make_mock_dispatcher(self, score=1.0):
        dispatcher = MagicMock()
        dispatcher.score.return_value = (score, None, "matched")
        return dispatcher

    def test_bare_baseline_with_mock(self, tmp_data_dir, tmp_path):
        """Bare baseline: mock returns predictable responses, scored and stored."""
        library = ProbeLibrary(tmp_data_dir)
        store = CalibrationStore(str(tmp_path / "cal.db"))
        adapter = self._make_mock_adapter()
        dispatcher = self._make_mock_dispatcher()

        runner = BaselineRunner(library, dispatcher, adapter, store)
        baselines = runner.run_baselines(baseline_type="bare", probe_ids=["F-001"])

        assert len(baselines) == 1
        assert baselines[0].score == 1.0
        assert baselines[0].baseline_type == "bare"
        assert baselines[0].error is None

        # Verify stored
        assert store.count_baselines() == 1
        stored = store.get_baselines()
        assert stored[0]["score"] == 1.0
        store.close()

    def test_anchored_baseline_with_mock(self, tmp_data_dir, tmp_path):
        """Anchored baseline: runs both endpoints (0.05 and 0.95), keeps higher score."""
        library = ProbeLibrary(tmp_data_dir)
        store = CalibrationStore(str(tmp_path / "cal.db"))

        # Generate frozen prompts at both endpoints
        gen = CalibrationGenerator(library)
        prompts = gen.generate(positions=[0.05, 0.95], context_lengths=[2048])
        store.write_prompts(prompts)

        adapter = self._make_mock_adapter()
        # Return different scores for the two calls — 0.8 then 0.9
        dispatcher = MagicMock()
        dispatcher.score.side_effect = [(0.8, None, "partial"), (0.9, None, "better")]

        runner = BaselineRunner(library, dispatcher, adapter, store)
        baselines = runner.run_baselines(baseline_type="anchored", probe_ids=["F-001"])

        # One result per context length, with the higher score kept
        assert len(baselines) == 1
        assert baselines[0].baseline_type == "anchored"
        assert baselines[0].score == 0.9  # higher of 0.8 and 0.9

        # Adapter was called twice (once per endpoint)
        assert adapter.single_turn.call_count == 2

        # Turn 1 input should be frozen prompt text, not bare content
        for call in adapter.single_turn.call_args_list:
            turn1_input = call[0][1]
            assert len(turn1_input) > len(library.probes["F-001"].content)
        store.close()

    def test_baseline_turn1_failure(self, tmp_data_dir, tmp_path):
        """Turn 1 failure → error recorded gracefully."""
        library = ProbeLibrary(tmp_data_dir)
        store = CalibrationStore(str(tmp_path / "cal.db"))
        adapter = self._make_mock_adapter()
        adapter.single_turn.side_effect = RuntimeError("connection failed")

        dispatcher = self._make_mock_dispatcher()
        runner = BaselineRunner(library, dispatcher, adapter, store)
        baselines = runner.run_baselines(baseline_type="bare", probe_ids=["F-001"])

        assert len(baselines) == 1
        assert baselines[0].score is None
        assert baselines[0].error is not None
        assert "connection failed" in baselines[0].error

        stored = store.get_baselines()
        assert stored[0]["error"] is not None
        store.close()

    def test_baseline_type_stored(self, tmp_data_dir, tmp_path):
        """Verify baseline_type field is stored correctly."""
        library = ProbeLibrary(tmp_data_dir)
        store = CalibrationStore(str(tmp_path / "cal.db"))
        adapter = self._make_mock_adapter()
        dispatcher = self._make_mock_dispatcher()

        runner = BaselineRunner(library, dispatcher, adapter, store)

        # Run bare
        runner.run_baselines(baseline_type="bare", probe_ids=["F-001"])
        bare = store.get_baselines(baseline_type="bare")
        assert len(bare) == 1
        assert bare[0]["baseline_type"] == "bare"

        # Run anchored (need frozen prompts at both endpoints)
        gen = CalibrationGenerator(library)
        prompts = gen.generate(positions=[0.05, 0.95], context_lengths=[2048])
        store.write_prompts(prompts)
        runner.run_baselines(baseline_type="anchored", probe_ids=["F-001"])
        anchored = store.get_baselines(baseline_type="anchored")
        assert len(anchored) == 1
        assert anchored[0]["baseline_type"] == "anchored"
        store.close()


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCalibrateCLI:
    def test_calibrate_generate_cli(self, tmp_data_dir, tmp_path, monkeypatch):
        """Generate via CLI entry point."""
        monkeypatch.delenv("APEX_DATABASE_URL", raising=False)
        db = str(tmp_path / "cli_cal.db")
        main([
            "calibrate", "generate",
            "--db", db,
            "--data-dir", str(tmp_data_dir),
            "--context-lengths", "2048",
        ])
        store = CalibrationStore(db)
        # 3 probes * 19 positions * 1 ctx = 57
        assert store.count_prompts() == 3 * 19
        store.close()

    def test_calibrate_validate_cli(self, tmp_data_dir, tmp_path, monkeypatch):
        """Generate then validate via CLI."""
        monkeypatch.delenv("APEX_DATABASE_URL", raising=False)
        db = str(tmp_path / "cli_cal.db")
        main([
            "calibrate", "generate",
            "--db", db,
            "--data-dir", str(tmp_data_dir),
            "--context-lengths", "2048",
        ])
        # Validate should succeed (exit 0 — no exception)
        main([
            "calibrate", "validate",
            "--db", db,
            "--data-dir", str(tmp_data_dir),
            "--verbose",
        ])


# ---------------------------------------------------------------------------
# Calibrated run integration tests (frozen prompts in ProbeRunner)
# ---------------------------------------------------------------------------


class TestCalibratedRun:
    def _make_config(self, tmp_data_dir, db_path, use_calibration=True):
        return RunConfig(
            seed=42,
            temperature=0.0,
            repetitions=1,
            filler_type="neutral",
            data_dir=str(tmp_data_dir),
            output_db=db_path,
            positions=[0.5],
            context_lengths=[2048],
            use_calibration=use_calibration,
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

    def _make_mock_adapter(self):
        adapter = MagicMock()
        adapter.get_model_info.return_value = ModelInfo(
            model_id="mock-model", backend="ollama",
            model_name="mock", max_context_window=4096,
        )
        adapter.single_turn.return_value = ChatResponse(
            content="I read the information.", latency_ms=50,
        )
        adapter.chat.return_value = ChatResponse(
            content="Michel Virlogeux designed the Millau Viaduct.", latency_ms=50,
        )
        return adapter

    def test_calibrated_run_uses_frozen_prompts(self, tmp_data_dir, tmp_path, monkeypatch):
        """When use_calibration=True, runner uses frozen prompt text."""
        monkeypatch.delenv("APEX_DATABASE_URL", raising=False)
        db = str(tmp_path / "cal_run.db")

        # Generate frozen prompts
        library = ProbeLibrary(str(tmp_data_dir))
        cal_store = CalibrationStore(db)
        gen = CalibrationGenerator(library)
        prompts = gen.generate(positions=[0.5], context_lengths=[2048])
        cal_store.write_prompts(prompts)

        # Get the frozen text for F-001 at pos=0.5, ctx=2048
        frozen = cal_store.get_prompts(probe_id="F-001")
        frozen_text = next(
            fp["full_text"] for fp in frozen
            if abs(fp["position_percent"] - 0.5) < 0.001
            and fp["context_length"] == 2048
        )
        cal_store.close()

        config = self._make_config(tmp_data_dir, db)
        store = ResultStore(db)
        mock_adapter = self._make_mock_adapter()
        runner = ProbeRunner(config, library, store)

        with patch("apex.runner.get_adapter", return_value=mock_adapter):
            with patch("apex.runner.ScoringDispatcher") as mock_d:
                mock_d.return_value.score.return_value = (1.0, None, None)
                runner.run()

        # Turn 1 should have received the frozen prompt text
        call_args = mock_adapter.single_turn.call_args
        turn1_input = call_args[0][1]
        assert turn1_input == frozen_text

        store.close()

    def test_calibrated_run_sets_filler_type(self, tmp_data_dir, tmp_path, monkeypatch):
        """Calibrated run records filler_type as 'calibrated'."""
        monkeypatch.delenv("APEX_DATABASE_URL", raising=False)
        db = str(tmp_path / "cal_run.db")

        library = ProbeLibrary(str(tmp_data_dir))
        cal_store = CalibrationStore(db)
        gen = CalibrationGenerator(library)
        prompts = gen.generate(positions=[0.5], context_lengths=[2048])
        cal_store.write_prompts(prompts)
        cal_store.close()

        config = self._make_config(tmp_data_dir, db)
        store = ResultStore(db)
        mock_adapter = self._make_mock_adapter()
        runner = ProbeRunner(config, library, store)

        with patch("apex.runner.get_adapter", return_value=mock_adapter):
            with patch("apex.runner.ScoringDispatcher") as mock_d:
                mock_d.return_value.score.return_value = (1.0, None, None)
                runner.run()

        results = store.query_results()
        assert len(results) == 1
        assert results[0]["filler_type"] == "calibrated"
        store.close()

    def test_calibrated_fallback_to_assembler(self, tmp_data_dir, tmp_path, monkeypatch):
        """When no frozen prompt exists for a position, falls back to assembler."""
        monkeypatch.delenv("APEX_DATABASE_URL", raising=False)
        db = str(tmp_path / "cal_run.db")

        # Generate frozen prompts for pos=0.1 only — NOT 0.5
        library = ProbeLibrary(str(tmp_data_dir))
        cal_store = CalibrationStore(db)
        gen = CalibrationGenerator(library)
        prompts = gen.generate(positions=[0.1], context_lengths=[2048])
        cal_store.write_prompts(prompts)
        cal_store.close()

        # Config asks for pos=0.5 which has no frozen prompt
        config = self._make_config(tmp_data_dir, db)
        store = ResultStore(db)
        mock_adapter = self._make_mock_adapter()
        runner = ProbeRunner(config, library, store)

        with patch("apex.runner.get_adapter", return_value=mock_adapter):
            with patch("apex.runner.ScoringDispatcher") as mock_d:
                mock_d.return_value.score.return_value = (1.0, None, None)
                runner.run()

        results = store.query_results()
        assert len(results) == 1
        # Fell back to assembler — filler_type should be "neutral" (config default)
        assert results[0]["filler_type"] == "neutral"
        store.close()

    def test_dynamic_run_unaffected(self, tmp_data_dir, tmp_path, monkeypatch):
        """When use_calibration=False, runner assembles fresh (unchanged behavior)."""
        monkeypatch.delenv("APEX_DATABASE_URL", raising=False)
        db = str(tmp_path / "dyn_run.db")

        library = ProbeLibrary(str(tmp_data_dir))
        config = self._make_config(tmp_data_dir, db, use_calibration=False)
        store = ResultStore(db)
        mock_adapter = self._make_mock_adapter()
        runner = ProbeRunner(config, library, store)

        with patch("apex.runner.get_adapter", return_value=mock_adapter):
            with patch("apex.runner.ScoringDispatcher") as mock_d:
                mock_d.return_value.score.return_value = (1.0, None, None)
                runner.run()

        results = store.query_results()
        assert len(results) == 1
        assert results[0]["filler_type"] == "neutral"
        store.close()


# ---------------------------------------------------------------------------
# Export / Import tests
# ---------------------------------------------------------------------------


class TestCalibrateExportImport:
    """Tests for CalibrationStore.export_json() and import_json()."""

    def _seed_store(self, store: CalibrationStore) -> None:
        """Populate a store with sample prompts and baselines."""
        for probe_id, dim in [("F-001", "factual_recall"), ("A-001", "application")]:
            store.write_prompt(CalibrationPrompt(
                probe_id=probe_id, dimension=dim,
                position_percent=0.5, context_length=4096, seed=42,
                full_text=f"text for {probe_id}",
                actual_token_count=100, target_position_tokens=50,
                filler_ids_before="NF-001", filler_ids_after="NF-002",
                probe_hash="ph1", content_hash=f"ch-{probe_id}",
                generated_at="2025-01-01T00:00:00Z",
            ))
        for model in ["model-a", "model-b"]:
            store.write_baseline(CalibrationBaseline(
                probe_id="F-001", dimension="factual_recall",
                model_id=model, baseline_type="bare",
                score=0.85, score_method="exact_match",
                justification="matched", raw_response="resp",
                raw_test_response="test_resp", error=None,
                timestamp="2025-01-01T00:00:00Z",
            ))
        store.write_baseline(CalibrationBaseline(
            probe_id="A-001", dimension="application",
            model_id="model-a", baseline_type="anchored",
            score=0.70, score_method="programmatic",
            justification="partial", raw_response="resp2",
            raw_test_response="test_resp2", error=None,
            timestamp="2025-01-01T00:00:00Z",
        ))

    def test_export_creates_valid_json(self, tmp_path):
        """Exported file has correct envelope with format, version, counts."""
        store = CalibrationStore(str(tmp_path / "cal.db"))
        self._seed_store(store)
        out = tmp_path / "export.json"
        counts = store.export_json(out)
        store.close()

        data = json.loads(out.read_text())
        assert data["format"] == "apex-calibration"
        assert data["format_version"] == 1
        assert data["apex_version"] == "1.2.0"
        assert "exported_at" in data
        assert data["counts"]["prompts"] == 2
        assert data["counts"]["baselines"] == 3
        assert counts == {"prompts": 2, "baselines": 3}

    def test_export_excludes_id(self, tmp_path):
        """No 'id' key in exported prompt or baseline rows."""
        store = CalibrationStore(str(tmp_path / "cal.db"))
        self._seed_store(store)
        out = tmp_path / "export.json"
        store.export_json(out)
        store.close()

        data = json.loads(out.read_text())
        for row in data["prompts"]:
            assert "id" not in row
        for row in data["baselines"]:
            assert "id" not in row

    def test_export_with_model_filter(self, tmp_path):
        """model_id filter narrows baselines but not prompts."""
        store = CalibrationStore(str(tmp_path / "cal.db"))
        self._seed_store(store)
        out = tmp_path / "export.json"
        counts = store.export_json(out, model_id="model-a")
        store.close()

        data = json.loads(out.read_text())
        # Prompts unaffected by model filter
        assert data["counts"]["prompts"] == 2
        # Only model-a baselines (2: bare F-001 + anchored A-001)
        assert data["counts"]["baselines"] == 2
        assert all(b["model_id"] == "model-a" for b in data["baselines"])
        assert counts == {"prompts": 2, "baselines": 2}

    def test_export_with_dimension_filter(self, tmp_path):
        """dimension filter narrows both prompts and baselines."""
        store = CalibrationStore(str(tmp_path / "cal.db"))
        self._seed_store(store)
        out = tmp_path / "export.json"
        counts = store.export_json(out, dimension="factual_recall")
        store.close()

        data = json.loads(out.read_text())
        assert data["counts"]["prompts"] == 1
        assert data["prompts"][0]["dimension"] == "factual_recall"
        # Only baselines with factual_recall dimension (2: model-a bare + model-b bare)
        assert data["counts"]["baselines"] == 2
        assert all(b["dimension"] == "factual_recall" for b in data["baselines"])
        assert counts == {"prompts": 1, "baselines": 2}

    def test_import_roundtrip(self, tmp_path):
        """Export from DB A, import to DB B — data matches."""
        store_a = CalibrationStore(str(tmp_path / "a.db"))
        self._seed_store(store_a)
        out = tmp_path / "export.json"
        store_a.export_json(out)
        store_a.close()

        store_b = CalibrationStore(str(tmp_path / "b.db"))
        counts = store_b.import_json(out)
        assert counts == {"prompts": 2, "baselines": 3}

        assert store_b.count_prompts() == 2
        assert store_b.count_baselines() == 3

        prompts_b = store_b.get_prompts()
        assert prompts_b[0]["probe_id"] in ("F-001", "A-001")
        store_b.close()

    def test_import_idempotent(self, tmp_path):
        """Importing same file twice doesn't duplicate records."""
        store_a = CalibrationStore(str(tmp_path / "a.db"))
        self._seed_store(store_a)
        out = tmp_path / "export.json"
        store_a.export_json(out)
        store_a.close()

        store_b = CalibrationStore(str(tmp_path / "b.db"))
        store_b.import_json(out)
        store_b.import_json(out)  # second import

        assert store_b.count_prompts() == 2
        assert store_b.count_baselines() == 3
        store_b.close()

    def test_import_invalid_format(self, tmp_path):
        """Rejects JSON with wrong format field."""
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"format": "wrong", "format_version": 1}))

        store = CalibrationStore(str(tmp_path / "cal.db"))
        with pytest.raises(ValueError, match="Invalid format"):
            store.import_json(bad)
        store.close()

    def test_import_invalid_version(self, tmp_path):
        """Rejects unsupported format_version."""
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"format": "apex-calibration", "format_version": 99}))

        store = CalibrationStore(str(tmp_path / "cal.db"))
        with pytest.raises(ValueError, match="Unsupported format_version"):
            store.import_json(bad)
        store.close()

    def test_import_missing_file(self, tmp_path):
        """Raises FileNotFoundError for nonexistent file."""
        store = CalibrationStore(str(tmp_path / "cal.db"))
        with pytest.raises(FileNotFoundError):
            store.import_json(tmp_path / "nonexistent.json")
        store.close()

    def test_cli_export_import_roundtrip(self, tmp_path, monkeypatch):
        """CLI export then import roundtrip."""
        monkeypatch.delenv("APEX_DATABASE_URL", raising=False)
        db_a = str(tmp_path / "a.db")
        db_b = str(tmp_path / "b.db")
        out = str(tmp_path / "cal.json")

        # Seed source DB
        store = CalibrationStore(db_a)
        self._seed_store(store)
        store.close()

        # Export via CLI
        main(["calibrate", "export", "--db", db_a, "-o", out])
        assert (tmp_path / "cal.json").exists()

        # Import via CLI
        main(["calibrate", "import", out, "--db", db_b])

        store_b = CalibrationStore(db_b)
        assert store_b.count_prompts() == 2
        assert store_b.count_baselines() == 3
        store_b.close()
