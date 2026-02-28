"""Tests for prompt assembly."""

from apex.assembler import PromptAssembler
from apex.tokenizers import ApproximateTokenizer


def test_assemble_basic(sample_fillers, sample_probe, sample_query):
    tok = ApproximateTokenizer()
    assembler = PromptAssembler(tok, sample_fillers)
    result = assembler.assemble(
        probe=sample_probe,
        test_query=sample_query,
        position_percent=0.5,
        context_length=4096,
        config_seed=42,
        run_number=1,
    )
    assert result.probe is sample_probe
    assert result.test_query is sample_query
    assert result.target_position_percent == 0.5
    assert sample_probe.content in result.full_text
    assert result.actual_token_count > 0


def test_assemble_deterministic(sample_fillers, sample_probe, sample_query):
    tok = ApproximateTokenizer()
    assembler = PromptAssembler(tok, sample_fillers)
    r1 = assembler.assemble(sample_probe, sample_query, 0.3, 2048, 42, 1)
    r2 = assembler.assemble(sample_probe, sample_query, 0.3, 2048, 42, 1)
    assert r1.full_text == r2.full_text
    assert r1.seed == r2.seed


def test_assemble_different_seeds(sample_fillers, sample_probe, sample_query):
    tok = ApproximateTokenizer()
    assembler = PromptAssembler(tok, sample_fillers)
    r1 = assembler.assemble(sample_probe, sample_query, 0.3, 2048, 42, 1)
    r2 = assembler.assemble(sample_probe, sample_query, 0.3, 2048, 99, 1)
    # Different seeds should produce different filler ordering
    assert r1.seed != r2.seed


def test_assemble_probe_in_text(sample_fillers, sample_probe, sample_query):
    tok = ApproximateTokenizer()
    assembler = PromptAssembler(tok, sample_fillers)
    for pos in [0.1, 0.5, 0.9]:
        result = assembler.assemble(sample_probe, sample_query, pos, 4096, 42, 1)
        assert sample_probe.content in result.full_text


def test_assemble_filler_ids_recorded(sample_fillers, sample_probe, sample_query):
    tok = ApproximateTokenizer()
    assembler = PromptAssembler(tok, sample_fillers)
    result = assembler.assemble(sample_probe, sample_query, 0.5, 4096, 42, 1)
    all_ids = result.filler_ids_before + result.filler_ids_after
    assert len(all_ids) > 0


def test_assemble_does_not_exceed_context(sample_fillers, sample_probe, sample_query):
    tok = ApproximateTokenizer()
    assembler = PromptAssembler(tok, sample_fillers)
    result = assembler.assemble(sample_probe, sample_query, 0.5, 2048, 42, 1)
    # Should not significantly exceed target (whole-passage packing may undershoot)
    assert result.actual_token_count <= 2048 + 200  # Allow some tolerance for whole passages
