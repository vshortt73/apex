"""Tests for probe/filler/query library loading."""

from apex.libraries import ProbeLibrary
from apex.types import Dimension, FillerType, ScoreMethod


def test_load_all(tmp_data_dir):
    lib = ProbeLibrary(tmp_data_dir)
    assert len(lib.fillers) == 20
    assert len(lib.probes) == 3
    assert len(lib.queries) == 3


def test_fillers_by_tier(tmp_data_dir):
    lib = ProbeLibrary(tmp_data_dir)
    neutral = lib.get_fillers(FillerType.NEUTRAL)
    assert len(neutral) == 20
    emotional = lib.get_fillers(FillerType.EMOTIONAL)
    assert len(emotional) == 0


def test_probes_by_dimension(tmp_data_dir):
    lib = ProbeLibrary(tmp_data_dir)
    factual = lib.get_probes(dimension=Dimension.FACTUAL_RECALL)
    assert len(factual) == 1
    assert factual[0].probe_id == "F-001"


def test_probes_by_id(tmp_data_dir):
    lib = ProbeLibrary(tmp_data_dir)
    selected = lib.get_probes(probe_ids=["A-001", "S-001"])
    assert len(selected) == 2


def test_query_for_probe(tmp_data_dir):
    lib = ProbeLibrary(tmp_data_dir)
    q = lib.get_query_for_probe("F-001")
    assert q is not None
    assert q.query_id == "FT-001"
    assert q.expected_answer == "Michel Virlogeux"


def test_missing_probe_query(tmp_data_dir):
    lib = ProbeLibrary(tmp_data_dir)
    q = lib.get_query_for_probe("NONEXISTENT")
    assert q is None


def test_probe_score_method(tmp_data_dir):
    lib = ProbeLibrary(tmp_data_dir)
    assert lib.probes["F-001"].score_method == ScoreMethod.EXACT_MATCH
    assert lib.probes["A-001"].score_method == ScoreMethod.PROGRAMMATIC
    assert lib.probes["S-001"].score_method == ScoreMethod.EVALUATOR


def test_empty_data_dir(tmp_path):
    (tmp_path / "filler").mkdir()
    (tmp_path / "probes").mkdir()
    (tmp_path / "queries").mkdir()
    lib = ProbeLibrary(tmp_path)
    assert len(lib.fillers) == 0
    assert len(lib.probes) == 0
    assert len(lib.queries) == 0
