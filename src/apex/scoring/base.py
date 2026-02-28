"""Scorer ABC and dispatch system."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from apex.types import Probe, ScoreMethod, TestQuery

if TYPE_CHECKING:
    from apex.models.base import ModelAdapter


class Scorer(abc.ABC):
    """Abstract scorer interface."""

    @abc.abstractmethod
    def score(self, probe: Probe, query: TestQuery, response: str) -> tuple[float | None, str | None]:
        """Score a response. Returns (score, justification)."""
        ...


class ScoringDispatcher:
    """Routes scoring to the appropriate scorer based on probe.score_method."""

    def __init__(self, evaluator_adapter: ModelAdapter | None = None) -> None:
        from apex.scoring.evaluator import EvaluatorScorer
        from apex.scoring.exact_match import ExactMatchScorer
        from apex.scoring.programmatic import ProgrammaticScorer
        from apex.scoring.semantic import SemanticScorer

        self._scorers: dict[ScoreMethod, Scorer] = {
            ScoreMethod.EXACT_MATCH: ExactMatchScorer(),
            ScoreMethod.SEMANTIC: SemanticScorer(),
            ScoreMethod.PROGRAMMATIC: ProgrammaticScorer(),
            ScoreMethod.EVALUATOR: EvaluatorScorer(evaluator_adapter),
        }

    def score(self, probe: Probe, query: TestQuery, response: str) -> tuple[float | None, str | None, str | None]:
        """Score a response using the probe's declared method.

        Returns (score, evaluator_model_id, justification).
        """
        scorer = self._scorers[probe.score_method]
        score_val, justification = scorer.score(probe, query, response)

        evaluator_model_id = None
        if probe.score_method == ScoreMethod.EVALUATOR:
            from apex.scoring.evaluator import EvaluatorScorer
            ev = self._scorers[ScoreMethod.EVALUATOR]
            if isinstance(ev, EvaluatorScorer) and ev.adapter:
                evaluator_model_id = ev.adapter.get_model_info().model_id

        return score_val, evaluator_model_id, justification
