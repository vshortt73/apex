"""Semantic similarity scorer using sentence-transformers."""

from __future__ import annotations

import logging

from apex.scoring.base import Scorer
from apex.types import Probe, TestQuery

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            logger.warning("sentence-transformers unavailable; semantic scoring will return None")
    return _model


class SemanticScorer(Scorer):
    """Score by cosine similarity between expected answer and response embeddings."""

    def score(self, probe: Probe, query: TestQuery, response: str) -> tuple[float | None, str | None]:
        if not query.expected_answer:
            return None, "No expected answer defined"

        model = _get_model()
        if model is None:
            return None, "Semantic model unavailable"

        embeddings = model.encode([query.expected_answer, response])
        # Cosine similarity
        import numpy as np

        a, b = embeddings[0], embeddings[1]
        cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        return round(max(0.0, cos_sim), 4), f"cosine_similarity={cos_sim:.4f}"
