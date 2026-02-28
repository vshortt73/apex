"""Tests for scoring modules."""

import json

import pytest

from apex.scoring.exact_match import ExactMatchScorer
from apex.scoring.programmatic import ProgrammaticScorer
from apex.types import Dimension, Probe, ScoreMethod, TestQuery


@pytest.fixture
def factual_probe():
    return Probe(
        probe_id="F-001",
        dimension=Dimension.FACTUAL_RECALL,
        content="The Millau Viaduct was designed by Michel Virlogeux.",
        content_type="factual",
        token_counts={},
        intrinsic_salience={},
        domain="engineering",
        confounding_factors="",
        evaluation_query_id="FT-001",
        score_method=ScoreMethod.EXACT_MATCH,
    )


@pytest.fixture
def factual_query():
    return TestQuery(
        query_id="FT-001",
        probe_id="F-001",
        dimension=Dimension.FACTUAL_RECALL,
        query="Who designed the Millau Viaduct?",
        expected_answer="Michel Virlogeux",
    )


@pytest.fixture
def factual_query_with_secondary():
    return TestQuery(
        query_id="FT-001",
        probe_id="F-001",
        dimension=Dimension.FACTUAL_RECALL,
        query="What is the largest glaciated area in the tropics?",
        expected_answer="Quelccaya Ice Cap",
        expected_answer_secondary="Peru",
    )


@pytest.fixture
def programmatic_query_word_count():
    return TestQuery(
        query_id="AT-001",
        probe_id="A-001",
        dimension=Dimension.APPLICATION,
        query="Explain the water cycle.",
        rubric=json.dumps({"check": "word_count", "target": 50, "partial_max": 75}),
        score_method=ScoreMethod.PROGRAMMATIC,
    )


@pytest.fixture
def programmatic_query_sentence_count():
    return TestQuery(
        query_id="AT-002",
        probe_id="A-002",
        dimension=Dimension.APPLICATION,
        query="What are the main causes of soil erosion?",
        rubric=json.dumps({"check": "sentence_count", "target": 3, "partial_off_by": 1}),
        score_method=ScoreMethod.PROGRAMMATIC,
    )


@pytest.fixture
def programmatic_query_legacy():
    """Legacy rubric format with word_count_lte check name."""
    return TestQuery(
        query_id="AT-001",
        probe_id="A-001",
        dimension=Dimension.APPLICATION,
        query="Explain the water cycle.",
        rubric=json.dumps({"check": "word_count_lte", "max": 50}),
        score_method=ScoreMethod.PROGRAMMATIC,
    )


class TestExactMatch:
    def test_exact_match_full(self, factual_probe, factual_query):
        scorer = ExactMatchScorer()
        score, justification = scorer.score(factual_probe, factual_query, "Michel Virlogeux designed it.")
        assert score == 1.0
        assert "exact_match" in justification

    def test_exact_match_case_insensitive(self, factual_probe, factual_query):
        scorer = ExactMatchScorer()
        score, _ = scorer.score(factual_probe, factual_query, "MICHEL VIRLOGEUX was the architect.")
        assert score == 1.0

    def test_exact_match_no_match(self, factual_probe, factual_query):
        scorer = ExactMatchScorer()
        score, _ = scorer.score(factual_probe, factual_query, "I don't know anything about that.")
        assert score == 0.0

    def test_exact_match_partial_continuous(self, factual_probe, factual_query):
        """Partial match now returns continuous term ratio."""
        scorer = ExactMatchScorer()
        # "Michel" matches (>3 chars), "Virlogeux" does not -> 1/2 terms
        score, justification = scorer.score(factual_probe, factual_query, "Michel designed the bridge.")
        assert score == 0.5
        assert "term_match" in justification

    def test_no_expected_answer(self, factual_probe):
        scorer = ExactMatchScorer()
        q = TestQuery(
            query_id="X", probe_id="X", dimension=Dimension.FACTUAL_RECALL, query="?",
        )
        score, justification = scorer.score(factual_probe, q, "anything")
        assert score is None

    def test_secondary_answer_match(self, factual_probe, factual_query_with_secondary):
        scorer = ExactMatchScorer()
        # Primary doesn't match but secondary does
        score, justification = scorer.score(
            factual_probe, factual_query_with_secondary,
            "It is located in Peru, in the southeastern region.",
        )
        assert score == 1.0
        assert "secondary_match" in justification

    def test_term_ratio_granularity(self, factual_probe):
        """Verify continuous scores for different match ratios."""
        scorer = ExactMatchScorer()
        q = TestQuery(
            query_id="X", probe_id="X", dimension=Dimension.FACTUAL_RECALL,
            query="?",
            expected_answer="alpha beta gamma delta",
        )
        # All terms > 3 chars: alpha, beta, gamma, delta (4 terms)
        # Match 3 of 4
        score, _ = scorer.score(factual_probe, q, "alpha beta gamma somewhere")
        assert score == 0.75

        # Match 1 of 4
        score, _ = scorer.score(factual_probe, q, "only alpha here")
        assert score == 0.25

        # Match 0 of 4
        score, _ = scorer.score(factual_probe, q, "nothing relevant here")
        assert score == 0.0


class TestProgrammatic:
    def test_word_count_pass(self, factual_probe, programmatic_query_word_count):
        scorer = ProgrammaticScorer()
        response = " ".join(["word"] * 30)
        score, justification = scorer.score(factual_probe, programmatic_query_word_count, response)
        assert score == 1.0
        assert "word_count=30" in justification

    def test_word_count_at_target(self, factual_probe, programmatic_query_word_count):
        scorer = ProgrammaticScorer()
        response = " ".join(["word"] * 50)
        score, _ = scorer.score(factual_probe, programmatic_query_word_count, response)
        assert score == 1.0

    def test_word_count_continuous_decay(self, factual_probe, programmatic_query_word_count):
        """Word count above target decays linearly to 0 at 2*target."""
        scorer = ProgrammaticScorer()
        # 62 words: (62-50)/50 = 0.24, score = 0.76
        response = " ".join(["word"] * 62)
        score, _ = scorer.score(factual_probe, programmatic_query_word_count, response)
        assert score == pytest.approx(0.76, abs=0.01)

        # 75 words: (75-50)/50 = 0.50, score = 0.50
        response = " ".join(["word"] * 75)
        score, _ = scorer.score(factual_probe, programmatic_query_word_count, response)
        assert score == pytest.approx(0.50, abs=0.01)

        # 100 words: (100-50)/50 = 1.0, score = 0.0
        response = " ".join(["word"] * 100)
        score, _ = scorer.score(factual_probe, programmatic_query_word_count, response)
        assert score == 0.0

    def test_word_count_beyond_double(self, factual_probe, programmatic_query_word_count):
        scorer = ProgrammaticScorer()
        response = " ".join(["word"] * 150)
        score, _ = scorer.score(factual_probe, programmatic_query_word_count, response)
        assert score == 0.0

    def test_word_count_legacy_format(self, factual_probe, programmatic_query_legacy):
        """Legacy word_count_lte check name still works with continuous scoring."""
        scorer = ProgrammaticScorer()
        response = " ".join(["word"] * 30)
        score, _ = scorer.score(factual_probe, programmatic_query_legacy, response)
        assert score == 1.0

        response = " ".join(["word"] * 75)
        score, _ = scorer.score(factual_probe, programmatic_query_legacy, response)
        assert score == pytest.approx(0.50, abs=0.01)

    def test_sentence_count_exact(self, factual_probe, programmatic_query_sentence_count):
        scorer = ProgrammaticScorer()
        response = "First sentence. Second sentence. Third sentence."
        score, justification = scorer.score(factual_probe, programmatic_query_sentence_count, response)
        assert score == 1.0
        assert "sentence_count=3" in justification

    def test_sentence_count_off_by_one(self, factual_probe, programmatic_query_sentence_count):
        scorer = ProgrammaticScorer()
        # 4 sentences: abs(4-3) = 1, score = 1.0 - 0.25 = 0.75
        response = "One. Two. Three. Four."
        score, _ = scorer.score(factual_probe, programmatic_query_sentence_count, response)
        assert score == 0.75

    def test_sentence_count_off_by_two(self, factual_probe, programmatic_query_sentence_count):
        scorer = ProgrammaticScorer()
        # 5 sentences: abs(5-3) = 2, score = 1.0 - 0.50 = 0.50
        response = "One. Two. Three. Four. Five."
        score, _ = scorer.score(factual_probe, programmatic_query_sentence_count, response)
        assert score == 0.5

    def test_sentence_count_far_off(self, factual_probe, programmatic_query_sentence_count):
        scorer = ProgrammaticScorer()
        # 7 sentences: abs(7-3) = 4, score = 1.0 - 1.0 = 0.0
        response = "A. B. C. D. E. F. G."
        score, _ = scorer.score(factual_probe, programmatic_query_sentence_count, response)
        assert score == 0.0

    def test_contains_check(self, factual_probe):
        scorer = ProgrammaticScorer()
        q = TestQuery(
            query_id="X", probe_id="X", dimension=Dimension.APPLICATION, query="?",
            rubric=json.dumps({"check": "contains", "terms": ["alpha", "beta", "gamma", "delta"]}),
        )
        # 4/4 terms
        score, _ = scorer.score(factual_probe, q, "This has alpha, beta, gamma and delta.")
        assert score == 1.0

        # 3/4 terms
        score, _ = scorer.score(factual_probe, q, "This has alpha, beta and gamma.")
        assert score == 0.75

        # 1/4 terms
        score, _ = scorer.score(factual_probe, q, "Only alpha here.")
        assert score == 0.25

    def test_not_contains_continuous(self, factual_probe):
        scorer = ProgrammaticScorer()
        q = TestQuery(
            query_id="X", probe_id="X", dimension=Dimension.APPLICATION, query="?",
            rubric=json.dumps({"check": "not_contains", "terms": ["cost", "price", "money", "budget"]}),
        )
        # No violations
        score, _ = scorer.score(factual_probe, q, "This is clean text.")
        assert score == 1.0

        # 1/4 violations -> 0.75
        score, _ = scorer.score(factual_probe, q, "The cost is high.")
        assert score == 0.75

        # 2/4 violations -> 0.5
        score, _ = scorer.score(factual_probe, q, "The cost and price are high.")
        assert score == 0.5

        # 4/4 violations -> 0.0
        score, _ = scorer.score(factual_probe, q, "The cost, price, money and budget are factors.")
        assert score == 0.0

    def test_starts_with_match(self, factual_probe):
        scorer = ProgrammaticScorer()
        q = TestQuery(
            query_id="X", probe_id="X", dimension=Dimension.APPLICATION, query="?",
            rubric=json.dumps({"check": "starts_with", "prefix": "Interestingly"}),
        )
        score, _ = scorer.score(factual_probe, q, "Interestingly, this works.")
        assert score == 1.0

    def test_starts_with_elsewhere(self, factual_probe):
        scorer = ProgrammaticScorer()
        q = TestQuery(
            query_id="X", probe_id="X", dimension=Dimension.APPLICATION, query="?",
            rubric=json.dumps({"check": "starts_with", "prefix": "Interestingly"}),
        )
        score, _ = scorer.score(factual_probe, q, "Well, interestingly enough, it does.")
        assert score == 0.5

    def test_starts_with_absent(self, factual_probe):
        scorer = ProgrammaticScorer()
        q = TestQuery(
            query_id="X", probe_id="X", dimension=Dimension.APPLICATION, query="?",
            rubric=json.dumps({"check": "starts_with", "prefix": "Interestingly"}),
        )
        score, _ = scorer.score(factual_probe, q, "This is a normal response.")
        assert score == 0.0

    def test_no_rubric(self, factual_probe):
        scorer = ProgrammaticScorer()
        q = TestQuery(
            query_id="X", probe_id="X", dimension=Dimension.APPLICATION, query="?",
        )
        score, _ = scorer.score(factual_probe, q, "anything")
        assert score is None
