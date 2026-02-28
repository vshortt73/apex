"""Programmatic scorer — continuous scoring for word count, format checks, etc."""

from __future__ import annotations

import json
import re

from apex.scoring.base import Scorer
from apex.types import Probe, TestQuery


class ProgrammaticScorer(Scorer):
    """Score by programmatic checks defined in the query rubric.

    All checks produce continuous scores in [0.0, 1.0] with raw measurements
    included in the justification string.

    Rubric is a JSON string with a "check" field and parameters:
      - word_count: {"check": "word_count", "target": 50, "partial_max": 75}
      - sentence_count: {"check": "sentence_count", "target": 3, "partial_off_by": 1}
      - contains: {"check": "contains", "terms": ["word1", "word2"]}
      - not_contains: {"check": "not_contains", "terms": ["word1"], "partial_max_violations": 2}
      - format_check: {"check": "format_check", "pattern": "regex"}
      - starts_with: {"check": "starts_with", "prefix": "Dear"}

    Legacy check names (word_count_lte, sentence_count_lte) are supported for
    backward compatibility with existing rubrics.
    """

    def score(self, probe: Probe, query: TestQuery, response: str) -> tuple[float | None, str | None]:
        if not query.rubric:
            return None, "No rubric defined for programmatic scoring"

        try:
            rubric = json.loads(query.rubric) if isinstance(query.rubric, str) else query.rubric
        except json.JSONDecodeError:
            return None, f"Invalid rubric JSON: {query.rubric}"

        check = rubric.get("check", "")
        response_stripped = response.strip()

        if check in ("word_count", "word_count_lte"):
            return self._score_word_count(rubric, response_stripped)

        elif check in ("sentence_count", "sentence_count_lte"):
            return self._score_sentence_count(rubric, response_stripped)

        elif check == "contains":
            return self._score_contains(rubric, response_stripped)

        elif check == "not_contains":
            return self._score_not_contains(rubric, response_stripped)

        elif check == "format_check":
            return self._score_format_check(rubric, response_stripped)

        elif check == "starts_with":
            return self._score_starts_with(rubric, response_stripped)

        return None, f"Unknown check type: {check}"

    @staticmethod
    def _score_word_count(rubric: dict, response: str) -> tuple[float, str]:
        """Continuous word count scoring.

        1.0 if <= target; linear decay 1.0 -> 0.0 over [target, 2*target].
        """
        target = rubric.get("target", rubric.get("max", 50))
        word_count = len(response.split())

        if word_count <= target:
            score = 1.0
        else:
            score = max(0.0, 1.0 - (word_count - target) / target)

        return round(score, 4), f"word_count={word_count}, target={target}"

    @staticmethod
    def _score_sentence_count(rubric: dict, response: str) -> tuple[float, str]:
        """Continuous sentence count scoring.

        max(0, 1.0 - 0.25 * abs(actual - target)).
        """
        target = rubric.get("target", rubric.get("max", 3))
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        count = len(sentences)

        score = max(0.0, 1.0 - 0.25 * abs(count - target))

        return round(score, 4), f"sentence_count={count}, target={target}"

    @staticmethod
    def _score_contains(rubric: dict, response: str) -> tuple[float, str]:
        """Continuous contains scoring — ratio of matched terms."""
        terms = rubric["terms"]
        response_lower = response.lower()
        found = [t for t in terms if t.lower() in response_lower]
        ratio = len(found) / len(terms) if terms else 0.0

        return round(ratio, 4), f"terms_found={len(found)}/{len(terms)}"

    @staticmethod
    def _score_not_contains(rubric: dict, response: str) -> tuple[float, str]:
        """Continuous not_contains scoring.

        1.0 - (violations / total_terms).
        """
        terms = rubric["terms"]
        response_lower = response.lower()
        violations = [t for t in terms if t.lower() in response_lower]
        total = len(terms)
        score = 1.0 - (len(violations) / total) if total else 1.0

        justification = f"violations={len(violations)}/{total}"
        if violations:
            justification += f", found: {violations}"

        return round(score, 4), justification

    @staticmethod
    def _score_format_check(rubric: dict, response: str) -> tuple[float, str]:
        """Continuous format check scoring — ratio of bulleted items to total items.

        Counts lines that match the bullet pattern vs total non-empty lines that
        appear to be list items (heuristic: lines starting with bullet, number, or
        short content lines in a sequence).
        """
        pattern = rubric["pattern"]
        lines = [line.strip() for line in response.split("\n") if line.strip()]

        bullet_matches = [line for line in lines if re.match(pattern, line)]
        # Heuristic: lines that look like list items (bullet, numbered, or short standalone)
        list_item_pattern = r'^(?:[-•*]\s+|\d+[.)]\s+|\w)'
        list_items = [line for line in lines if re.match(list_item_pattern, line)]

        if not list_items:
            # No discernible list structure
            if bullet_matches:
                return 1.0, f"bullet_lines={len(bullet_matches)}, total_lines={len(lines)}"
            return 0.0, "no_list_items_detected"

        ratio = len(bullet_matches) / len(list_items)

        return round(ratio, 4), f"bullet_items={len(bullet_matches)}, list_items={len(list_items)}"

    @staticmethod
    def _score_starts_with(rubric: dict, response: str) -> tuple[float, str]:
        """Starts-with scoring.

        1.0 if starts with prefix; 0.5 if word found elsewhere; 0.0 if absent.
        """
        prefix = rubric["prefix"]

        if response.lower().startswith(prefix.lower()):
            return 1.0, f"starts_with='{prefix}'"

        if prefix.lower() in response.lower():
            return 0.5, f"contains '{prefix}' but not at start"

        return 0.0, f"'{prefix}' not found"
