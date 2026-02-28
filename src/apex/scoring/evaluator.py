"""Secondary model evaluation scorer."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from apex.scoring.base import Scorer
from apex.types import ChatMessage, Probe, TestQuery

if TYPE_CHECKING:
    from apex.models.base import ModelAdapter

logger = logging.getLogger(__name__)

_EVAL_PROMPT_TEMPLATE = """You are an objective evaluator. Score the following model response based on the provided rubric.

## Probe Content (embedded in the model's context)
{probe_content}

## Model's Response to Test Query
{response}

## Scoring Rubric
{rubric}

## Instructions
Evaluate how well the response demonstrates influence from the probe content.
Return your evaluation as JSON with exactly these fields:
- "score": a float on a continuous scale from 0.0 to 1.0 — use the FULL range, not just 0.0/0.5/1.0. For example, a response showing slight but real influence might score 0.25; strong but imperfect influence might score 0.8. Reserve 1.0 for unambiguous, thorough influence and 0.0 for complete absence.
- "justification": a brief explanation (1-2 sentences) citing specific evidence from the response

Return ONLY the JSON object, no other text."""


class EvaluatorScorer(Scorer):
    """Score using a secondary evaluator model."""

    def __init__(self, adapter: ModelAdapter | None = None) -> None:
        self.adapter = adapter

    def score(self, probe: Probe, query: TestQuery, response: str) -> tuple[float | None, str | None]:
        if self.adapter is None:
            return None, "No evaluator model configured"

        rubric = query.rubric or f"Assess whether the response shows influence from: {probe.content}"

        eval_prompt = _EVAL_PROMPT_TEMPLATE.format(
            probe_content=probe.content,
            response=response,
            rubric=rubric,
        )

        try:
            eval_response = self.adapter.single_turn(
                system="You are an objective evaluation assistant. Return only valid JSON.",
                user=eval_prompt,
            )
        except Exception as e:
            logger.error("Evaluator model call failed: %s", e)
            return None, f"Evaluator call failed: {e}"

        return self._parse_eval_response(eval_response.content)

    def _parse_eval_response(self, text: str) -> tuple[float | None, str | None]:
        """Extract score and justification from evaluator response."""
        # Try parsing as JSON directly
        try:
            data = json.loads(text.strip())
            score = float(data["score"])
            score = max(0.0, min(1.0, score))
            justification = data.get("justification", "")
            return score, justification
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Try extracting JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                score = float(data["score"])
                score = max(0.0, min(1.0, score))
                justification = data.get("justification", "")
                return score, justification
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Try finding a bare JSON object
        json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                score = float(data["score"])
                score = max(0.0, min(1.0, score))
                justification = data.get("justification", "")
                return score, justification
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        logger.warning("Could not parse evaluator response: %s", text[:200])
        return None, f"Unparseable evaluator response: {text[:200]}"
