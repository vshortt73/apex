"""Exact match scorer — continuous string matching for factual recall."""

from __future__ import annotations

from apex.scoring.base import Scorer
from apex.types import Probe, TestQuery


class ExactMatchScorer(Scorer):
    """Score by matching expected answer in model response.

    Continuous scoring:
      1.0 — exact expected answer found as substring (case-insensitive),
             or expected_answer_secondary matches
      (0, 1) — term match ratio for partial matches
      0.0 — no significant terms matched
    """

    def score(self, probe: Probe, query: TestQuery, response: str) -> tuple[float | None, str | None]:
        if not query.expected_answer:
            return None, "No expected answer defined"

        response_lower = response.lower().strip()
        expected_lower = query.expected_answer.lower().strip()

        # Exact substring match
        if expected_lower in response_lower:
            return 1.0, f"exact_match='{query.expected_answer}'"

        # Check secondary answer — also grants 1.0
        if query.expected_answer_secondary:
            secondary_lower = query.expected_answer_secondary.lower().strip()
            if secondary_lower in response_lower:
                return 1.0, f"secondary_match='{query.expected_answer_secondary}'"

        # Partial match: use term ratio directly as continuous score
        terms = [t for t in expected_lower.split() if len(t) > 3]
        if not terms:
            return 0.0, "No significant terms to match"

        matched = [t for t in terms if t in response_lower]
        ratio = len(matched) / len(terms)

        justification = f"term_match={len(matched)}/{len(terms)}"
        if matched:
            justification += f", matched: {matched}"

        return round(ratio, 4), justification
