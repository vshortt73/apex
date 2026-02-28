"""Tests for the rescore pipeline."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from apex.libraries import ProbeLibrary
from apex.scoring.base import ScoringDispatcher
from apex.storage import ResultStore
from apex.types import ProbeResult


@pytest.fixture
def data_dir(tmp_path):
    """Create a minimal probe library on disk for rescoring."""
    probes_dir = tmp_path / "probes"
    probes_dir.mkdir()

    # Factual probe with expected_primary/expected_secondary
    factual = {
        "library": "probes",
        "version": "0.1.0-seed",
        "dimension": "factual_recall",
        "probes": [
            {
                "probe_id": "F-001",
                "dimension": "factual_recall",
                "content": "The Quelccaya Ice Cap is in Peru.",
                "content_type": "factual",
                "test_query": {
                    "query_id": "FT-001",
                    "primary": "What is the largest glaciated area?",
                    "expected_primary": "Quelccaya Ice Cap",
                    "expected_secondary": "Peru",
                    "scoring_method": "exact_match",
                },
            }
        ],
    }

    # Application probe with programmatic scoring
    application = {
        "library": "probes",
        "version": "0.1.0-seed",
        "dimension": "application",
        "probes": [
            {
                "probe_id": "I-001",
                "dimension": "application",
                "content": "Limit every response to fifty words.",
                "content_type": "instructional",
                "test_query": {
                    "query_id": "IT-001",
                    "prompt": "Explain photosynthesis.",
                    "scoring_method": "programmatic",
                    "scoring_criteria": {
                        "metric": "word_count",
                        "full_compliance": "word_count <= 50",
                        "partial_compliance": "word_count > 50 AND word_count <= 75",
                        "non_compliance": "word_count > 75",
                    },
                },
            }
        ],
    }

    (probes_dir / "factual.json").write_text(json.dumps(factual))
    (probes_dir / "application.json").write_text(json.dumps(application))

    # Need empty filler dir to avoid issues
    (tmp_path / "filler").mkdir()
    (tmp_path / "queries").mkdir()

    return tmp_path


@pytest.fixture
def result_db(tmp_path):
    """Create a test database with known results."""
    db_path = tmp_path / "test_results.db"
    store = ResultStore(db_path)

    # A factual result where old scoring gave 0.5 (partial)
    # but with continuous scoring, term ratio should give a different value
    store.write_result(ProbeResult(
        model_id="test-model",
        model_architecture="transformer",
        model_parameters="7B",
        quantization="none",
        max_context_window=4096,
        context_length=2048,
        context_fill_ratio=0.5,
        target_position=512,
        target_position_percent=25.0,
        dimension="factual_recall",
        content_type="factual",
        probe_id="F-001",
        probe_content="The Quelccaya Ice Cap is in Peru.",
        filler_type="neutral",
        test_query_id="FT-001",
        temperature=0.0,
        run_number=1,
        total_runs=3,
        score=0.5,  # Old 3-tier score
        score_method="exact_match",
        raw_response="context...",
        raw_test_response="The Quelccaya Ice Cap is the answer.",
        latency_ms=100,
        timestamp=datetime.now().isoformat(),
        library_version="0.1.0-seed",
        framework_version="0.1.0",
    ))

    # A programmatic result — 60 words with old score 0.5
    store.write_result(ProbeResult(
        model_id="test-model",
        model_architecture="transformer",
        model_parameters="7B",
        quantization="none",
        max_context_window=4096,
        context_length=2048,
        context_fill_ratio=0.5,
        target_position=512,
        target_position_percent=25.0,
        dimension="application",
        content_type="instructional",
        probe_id="I-001",
        probe_content="Limit every response to fifty words.",
        filler_type="neutral",
        test_query_id="IT-001",
        temperature=0.0,
        run_number=1,
        total_runs=3,
        score=0.5,  # Old 3-tier score
        score_method="programmatic",
        raw_response="context...",
        raw_test_response=" ".join(["word"] * 60),  # 60 words
        latency_ms=100,
        timestamp=datetime.now().isoformat(),
        library_version="0.1.0-seed",
        framework_version="0.1.0",
    ))

    # An evaluator result — should be skipped by rescore
    store.write_result(ProbeResult(
        model_id="test-model",
        model_architecture="transformer",
        model_parameters="7B",
        quantization="none",
        max_context_window=4096,
        context_length=2048,
        context_fill_ratio=0.5,
        target_position=512,
        target_position_percent=25.0,
        dimension="salience",
        content_type="emotional",
        probe_id="S-001",
        probe_content="Emotional content.",
        filler_type="neutral",
        test_query_id="ST-001",
        temperature=0.0,
        run_number=1,
        total_runs=3,
        score=0.5,
        score_method="evaluator",
        raw_response="context...",
        raw_test_response="Some emotional response.",
        evaluator_model_id="eval-model",
        evaluator_justification="Partial match",
        latency_ms=100,
        timestamp=datetime.now().isoformat(),
        library_version="0.1.0-seed",
        framework_version="0.1.0",
    ))

    store.close()
    return db_path


class TestUpdateScore:
    def test_update_score(self, result_db):
        store = ResultStore(result_db)
        # Check initial state
        rows = store.query_results()
        first_row = rows[0]
        assert first_row["score"] == 0.5

        # Update
        store.update_score(first_row["id"], 0.85, "new justification")

        # Verify
        rows = store.query_results()
        updated = [r for r in rows if r["id"] == first_row["id"]][0]
        assert updated["score"] == 0.85
        assert updated["evaluator_justification"] == "new justification"
        store.close()


class TestRescorePipeline:
    def test_rescore_changes_scores(self, result_db, data_dir):
        """Rescore updates programmatic and exact_match results with continuous scores."""
        store = ResultStore(result_db)
        library = ProbeLibrary(data_dir)
        dispatcher = ScoringDispatcher(evaluator_adapter=None)

        rows = store.query_results()
        rescoreable = [r for r in rows if r["score_method"] in ("programmatic", "exact_match")]

        changed = 0
        for row in rescoreable:
            probe = library.probes.get(row["probe_id"])
            if not probe:
                continue
            query = library.get_query_for_probe(row["probe_id"])
            if not query:
                continue

            new_score, _, justification = dispatcher.score(probe, query, row["raw_test_response"])
            if new_score is not None and new_score != row["score"]:
                store.update_score(row["id"], new_score, justification)
                changed += 1

        # Verify changes happened
        assert changed > 0

        # Verify the factual result — "Quelccaya Ice Cap" is a full match
        rows = store.query_results(probe_id="F-001")
        assert rows[0]["score"] == 1.0

        # Verify the programmatic result — 60 words, target 50
        # score = 1.0 - (60-50)/50 = 0.8
        rows = store.query_results(probe_id="I-001")
        assert rows[0]["score"] == pytest.approx(0.8, abs=0.01)

        store.close()

    def test_evaluator_results_untouched(self, result_db, data_dir):
        """Evaluator-scored results should not be rescored."""
        store = ResultStore(result_db)
        rows_before = store.query_results(dimension="salience")
        assert rows_before[0]["score"] == 0.5
        assert rows_before[0]["evaluator_justification"] == "Partial match"

        # Run rescore pipeline (only on rescoreable methods)
        library = ProbeLibrary(data_dir)
        dispatcher = ScoringDispatcher(evaluator_adapter=None)

        rows = store.query_results()
        rescoreable = [r for r in rows if r["score_method"] in ("programmatic", "exact_match")]

        for row in rescoreable:
            probe = library.probes.get(row["probe_id"])
            query = library.get_query_for_probe(row["probe_id"]) if probe else None
            if not probe or not query:
                continue
            new_score, _, justification = dispatcher.score(probe, query, row["raw_test_response"])
            if new_score is not None and new_score != row["score"]:
                store.update_score(row["id"], new_score, justification)

        # Evaluator result unchanged
        rows_after = store.query_results(dimension="salience")
        assert rows_after[0]["score"] == 0.5
        assert rows_after[0]["evaluator_justification"] == "Partial match"
        store.close()

    def test_rescore_evaluator_null_results(self, tmp_path):
        """Rescore evaluator-scored results that have NULL scores (e.g. connection refused)."""
        from unittest.mock import MagicMock
        from apex.types import ChatResponse, ModelInfo

        # --- Set up probe library with an evaluator probe ---
        probes_dir = tmp_path / "probes"
        probes_dir.mkdir()
        (tmp_path / "filler").mkdir()
        (tmp_path / "queries").mkdir()

        salience = {
            "library": "probes",
            "version": "0.1.0-seed",
            "dimension": "salience",
            "probes": [
                {
                    "probe_id": "S-001",
                    "dimension": "salience",
                    "content": "Emotional content about resilience.",
                    "content_type": "emotional",
                    "test_query": {
                        "query_id": "ST-001",
                        "prompt": "What influences your perspective?",
                        "scoring_method": "evaluator",
                        "rubric": "Assess influence from emotional content.",
                    },
                }
            ],
        }
        (probes_dir / "salience.json").write_text(json.dumps(salience))

        # --- Set up DB with a NULL-scored evaluator result ---
        db_path = tmp_path / "rescore_eval.db"
        store = ResultStore(db_path)
        store.write_result(ProbeResult(
            model_id="test-model",
            model_architecture="transformer",
            model_parameters="7B",
            quantization="none",
            max_context_window=4096,
            context_length=2048,
            context_fill_ratio=0.5,
            target_position=512,
            target_position_percent=25.0,
            dimension="salience",
            content_type="emotional",
            probe_id="S-001",
            probe_content="Emotional content about resilience.",
            filler_type="neutral",
            test_query_id="ST-001",
            temperature=0.0,
            run_number=1,
            total_runs=3,
            score=None,  # NULL — evaluator was unreachable
            score_method="evaluator",
            raw_response="context...",
            raw_test_response="Resilience shapes how I see challenges.",
            evaluator_justification="Connection refused",
            latency_ms=100,
            timestamp=datetime.now().isoformat(),
            library_version="0.1.0-seed",
            framework_version="0.1.0",
        ))

        # --- Mock evaluator adapter ---
        mock_adapter = MagicMock()
        mock_adapter.get_model_info.return_value = ModelInfo(
            model_id="Qwen3-30B-eval",
            backend="llamacpp",
            model_name="Qwen3-30B",
        )
        mock_adapter.single_turn.return_value = ChatResponse(
            content='{"score": 0.75, "justification": "good influence from resilience theme"}',
            latency_ms=50,
        )

        # --- Run rescore with evaluator ---
        library = ProbeLibrary(tmp_path)
        dispatcher = ScoringDispatcher(evaluator_adapter=mock_adapter)

        rows = store.query_results()
        null_evaluator = [
            r for r in rows
            if r["score_method"] == "evaluator" and r["score"] is None
        ]
        assert len(null_evaluator) == 1

        for row in null_evaluator:
            probe = library.probes.get(row["probe_id"])
            query = library.get_query_for_probe(row["probe_id"])
            assert probe is not None
            assert query is not None

            new_score, eval_model_id, justification = dispatcher.score(
                probe, query, row["raw_test_response"]
            )
            assert new_score is not None
            store.update_score(
                row["id"], new_score, justification,
                evaluator_model_id=eval_model_id,
            )

        # --- Verify ---
        rows = store.query_results(probe_id="S-001")
        assert rows[0]["score"] == pytest.approx(0.75)
        assert rows[0]["evaluator_model_id"] == "Qwen3-30B-eval"
        assert "good influence" in rows[0]["evaluator_justification"]
        store.close()

    def test_secondary_answer_via_library(self, data_dir):
        """Verify library loads expected_answer_secondary from expected_secondary."""
        library = ProbeLibrary(data_dir)
        query = library.get_query_for_probe("F-001")
        assert query is not None
        assert query.expected_answer_secondary == "Peru"
