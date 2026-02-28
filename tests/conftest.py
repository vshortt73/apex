"""Shared fixtures for APEX tests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from apex.types import (
    Dimension,
    FillerPassage,
    FillerType,
    Probe,
    ScoreMethod,
    TestQuery,
)


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with sample library files."""
    filler_dir = tmp_path / "filler"
    filler_dir.mkdir()
    probe_dir = tmp_path / "probes"
    probe_dir.mkdir()
    query_dir = tmp_path / "queries"
    query_dir.mkdir()

    # Sample filler
    filler_data = {
        "library": "filler",
        "version": "0.1.0-test",
        "tier": "neutral",
        "passages": [
            {
                "filler_id": f"NF-{i:03d}",
                "content": f"This is neutral filler passage number {i}. "
                "It contains mundane factual information about various topics "
                "that serves as background context without attracting attention. "
                "The passage is designed to fill space in the context window." * 2,
                "domain": "general",
                "token_count_estimate": 80,
                "flesch_kincaid_grade": 10.0,
                "panas_positive": 0.1,
                "panas_negative": 0.0,
            }
            for i in range(1, 21)
        ],
    }
    (filler_dir / "test_filler.json").write_text(json.dumps(filler_data))

    # Sample probes
    probe_data = {
        "library": "probes",
        "version": "0.1.0-test",
        "probes": [
            {
                "probe_id": "F-001",
                "dimension": "factual_recall",
                "content": "The architect who designed the Millau Viaduct was Michel Virlogeux.",
                "content_type": "factual",
                "token_counts": {"approximate": 15},
                "intrinsic_salience": {"factual_importance": 0.6},
                "domain": "engineering",
                "confounding_factors": "none",
                "evaluation_query_id": "FT-001",
                "score_method": "exact_match",
            },
            {
                "probe_id": "A-001",
                "dimension": "application",
                "content": "Limit all responses to exactly 50 words or fewer.",
                "content_type": "instructional",
                "token_counts": {"approximate": 10},
                "intrinsic_salience": {"contextual_influence": 0.8},
                "domain": "format",
                "confounding_factors": "none",
                "evaluation_query_id": "AT-001",
                "score_method": "programmatic",
            },
            {
                "probe_id": "S-001",
                "dimension": "salience",
                "content": "The old man sat alone on the park bench, watching children play, remembering when his own had been that small.",
                "content_type": "emotional",
                "token_counts": {"approximate": 25},
                "intrinsic_salience": {"emotional_weight": 0.7},
                "domain": "nostalgia",
                "confounding_factors": "none",
                "evaluation_query_id": "ST-001",
                "score_method": "evaluator",
            },
        ],
    }
    (probe_dir / "test_probes.json").write_text(json.dumps(probe_data))

    # Sample queries
    query_data = {
        "library": "queries",
        "version": "0.1.0-test",
        "queries": [
            {
                "query_id": "FT-001",
                "probe_id": "F-001",
                "dimension": "factual_recall",
                "query": "Who designed the Millau Viaduct?",
                "expected_answer": "Michel Virlogeux",
                "score_method": "exact_match",
            },
            {
                "query_id": "AT-001",
                "probe_id": "A-001",
                "dimension": "application",
                "query": "Explain the water cycle in detail.",
                "rubric": '{"check": "word_count_lte", "max": 50}',
                "score_method": "programmatic",
            },
            {
                "query_id": "ST-001",
                "probe_id": "S-001",
                "dimension": "salience",
                "query": "Write a short paragraph about a typical weekend.",
                "rubric": "Assess whether the response shows nostalgic or reflective emotional tone influenced by the probe content.",
                "score_method": "evaluator",
            },
        ],
    }
    (query_dir / "test_queries.json").write_text(json.dumps(query_data))

    return tmp_path


@pytest.fixture
def sample_fillers() -> list[FillerPassage]:
    return [
        FillerPassage(
            filler_id=f"NF-{i:03d}",
            content=f"Neutral filler passage {i}. " * 20,
            domain="general",
            token_count_estimate=80,
            flesch_kincaid_grade=10.0,
            panas_positive=0.1,
            panas_negative=0.0,
        )
        for i in range(1, 11)
    ]


@pytest.fixture
def sample_probe() -> Probe:
    return Probe(
        probe_id="F-001",
        dimension=Dimension.FACTUAL_RECALL,
        content="The architect who designed the Millau Viaduct was Michel Virlogeux.",
        content_type="factual",
        token_counts={"approximate": 15},
        intrinsic_salience={"factual_importance": 0.6},
        domain="engineering",
        confounding_factors="none",
        evaluation_query_id="FT-001",
        score_method=ScoreMethod.EXACT_MATCH,
    )


@pytest.fixture
def sample_query() -> TestQuery:
    return TestQuery(
        query_id="FT-001",
        probe_id="F-001",
        dimension=Dimension.FACTUAL_RECALL,
        query="Who designed the Millau Viaduct?",
        expected_answer="Michel Virlogeux",
        score_method=ScoreMethod.EXACT_MATCH,
    )


@pytest.fixture
def tmp_config(tmp_data_dir: Path, tmp_path: Path) -> Path:
    """Create a temporary YAML config file."""
    config = {
        "run": {
            "seed": 42,
            "temperature": 0.0,
            "repetitions": 1,
            "filler_type": "neutral",
        },
        "data": {
            "directory": str(tmp_data_dir),
            "output_db": str(tmp_path / "test_results.db"),
        },
        "probes": {"select": "all"},
        "positions": [0.1, 0.5, 0.9],
        "context_lengths": [2048],
        "models": [
            {
                "name": "test-model",
                "backend": "ollama",
                "model_name": "test:latest",
                "tokenizer": "approximate",
                "max_context_window": 4096,
            }
        ],
    }
    import yaml

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path
