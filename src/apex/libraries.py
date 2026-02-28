"""Load JSON probe/filler/query libraries."""

from __future__ import annotations

import json
import re
from pathlib import Path

from apex.types import (
    Dimension,
    FillerPassage,
    FillerType,
    Probe,
    ScoreMethod,
    TestQuery,
)


class ProbeLibrary:
    """Loads and indexes probe, filler, and query libraries from JSON files."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.fillers: dict[str, FillerPassage] = {}
        self.fillers_by_tier: dict[FillerType, list[FillerPassage]] = {
            t: [] for t in FillerType
        }
        self.probes: dict[str, Probe] = {}
        self.probes_by_dimension: dict[Dimension, list[Probe]] = {
            d: [] for d in Dimension
        }
        self.queries: dict[str, TestQuery] = {}
        self.queries_by_probe: dict[str, TestQuery] = {}
        self.library_version: str = "unknown"
        self._load_all()

    def _load_all(self) -> None:
        filler_dir = self.data_dir / "filler"
        if filler_dir.is_dir():
            for fp in sorted(filler_dir.glob("*.json")):
                self._load_filler_file(fp)

        probe_dir = self.data_dir / "probes"
        if probe_dir.is_dir():
            for fp in sorted(probe_dir.glob("*.json")):
                self._load_probe_file(fp)

        query_dir = self.data_dir / "queries"
        if query_dir.is_dir():
            for fp in sorted(query_dir.glob("*.json")):
                self._load_query_file(fp)

    def _load_filler_file(self, path: Path) -> None:
        data = json.loads(path.read_text())
        version = data.get("version", "unknown")
        if self.library_version == "unknown":
            self.library_version = version
        tier_str = data.get("tier", "neutral")
        tier = FillerType(tier_str)
        for entry in data.get("passages", []):
            fp = FillerPassage(
                filler_id=entry["filler_id"],
                content=entry["content"],
                domain=entry.get("domain", "general"),
                token_count_estimate=entry.get("token_count_estimate", 100),
                flesch_kincaid_grade=entry.get("flesch_kincaid_grade", 10.0),
                panas_positive=entry.get("panas_positive", 0.0),
                panas_negative=entry.get("panas_negative", 0.0),
                notes=entry.get("notes", ""),
            )
            self.fillers[fp.filler_id] = fp
            self.fillers_by_tier[tier].append(fp)

    def _load_probe_file(self, path: Path) -> None:
        data = json.loads(path.read_text())
        version = data.get("version", "unknown")
        if self.library_version == "unknown":
            self.library_version = version
        for entry in data.get("probes", []):
            # Score method may be on the probe or inside the inline test_query
            tq = entry.get("test_query", {})
            score_method_str = entry.get(
                "score_method", tq.get("scoring_method", tq.get("score_method", "exact_match"))
            )
            score_method = ScoreMethod(score_method_str)
            query_id = tq.get("query_id", entry.get("evaluation_query_id", ""))

            probe = Probe(
                probe_id=entry["probe_id"],
                dimension=Dimension(entry["dimension"]),
                content=entry["content"],
                content_type=entry.get("content_type", "factual"),
                token_counts=entry.get("token_counts", {}),
                intrinsic_salience=entry.get("intrinsic_salience", {}),
                domain=entry.get("domain", "general"),
                confounding_factors=entry.get("confounding_factors", ""),
                evaluation_query_id=query_id,
                score_method=score_method,
                version=entry.get("version", "1.0"),
            )
            self.probes[probe.probe_id] = probe
            self.probes_by_dimension[probe.dimension].append(probe)

            # Extract inline test_query into the queries index
            if tq and query_id:
                query = self._parse_inline_query(
                    tq, probe.probe_id, probe.dimension, score_method,
                )
                if query:
                    self.queries[query.query_id] = query
                    self.queries_by_probe[query.probe_id] = query

    def _parse_inline_query(
        self,
        tq: dict,
        probe_id: str,
        dimension: Dimension,
        score_method: ScoreMethod,
    ) -> TestQuery | None:
        """Parse an inline test_query dict from a probe entry.

        Handles three seed formats:
          - Factual: primary/secondary questions, expected_primary/expected_secondary
          - Application (programmatic): prompt + scoring_criteria with metric
          - Application/Salience (evaluator): prompt + scoring_criteria with rubric
        """
        query_id = tq.get("query_id", "")
        if not query_id:
            return None

        # Determine the query text
        query_text = tq.get("query", tq.get("prompt", tq.get("primary", "")))

        # Determine expected answer (factual recall)
        expected_answer = tq.get("expected_answer", tq.get("expected_primary"))
        expected_answer_secondary = tq.get("expected_answer_secondary", tq.get("expected_secondary"))

        # Build rubric from scoring_criteria
        rubric = tq.get("rubric")
        criteria = tq.get("scoring_criteria")
        if criteria and not rubric:
            rubric = self._translate_scoring_criteria(criteria, score_method)

        return TestQuery(
            query_id=query_id,
            probe_id=probe_id,
            dimension=dimension,
            query=query_text,
            expected_answer=expected_answer,
            expected_answer_secondary=expected_answer_secondary,
            rubric=rubric,
            score_method=score_method,
        )

    @staticmethod
    def _translate_scoring_criteria(criteria: dict, score_method: ScoreMethod) -> str:
        """Translate seed scoring_criteria into the rubric format scorers expect.

        For programmatic: returns JSON string with "check" field and full compliance
        thresholds so the scorer can produce continuous scores.
        For evaluator: returns the rubric text string.
        """
        if score_method == ScoreMethod.EVALUATOR:
            return criteria.get("rubric", json.dumps(criteria))

        # Programmatic — translate metric to ProgrammaticScorer format
        metric = criteria.get("metric", "")

        if metric == "word_count":
            # Parse target from full_compliance like "word_count <= 50"
            fc = criteria.get("full_compliance", "")
            target = 50
            for part in fc.replace("<=", " ").split():
                try:
                    target = int(part)
                except ValueError:
                    continue
            # Parse partial_max from partial_compliance like "word_count > 50 AND word_count <= 75"
            pc = criteria.get("partial_compliance", "")
            partial_max = target
            for part in pc.replace("<=", " ").replace(">", " ").split():
                try:
                    val = int(part)
                    if val > target:
                        partial_max = val
                except ValueError:
                    continue
            return json.dumps({
                "check": "word_count",
                "target": target,
                "partial_max": partial_max,
            })

        elif metric == "sentence_count":
            # Parse target from full_compliance like "sentence_count == 3"
            fc = criteria.get("full_compliance", "")
            target = 3
            for part in fc.replace("==", " ").split():
                try:
                    target = int(part)
                except ValueError:
                    continue
            # Parse off-by from partial_compliance like "sentence_count == 2 OR sentence_count == 4"
            pc = criteria.get("partial_compliance", "")
            partial_off_by = 1
            pc_vals = []
            for part in pc.replace("==", " ").replace("OR", " ").split():
                try:
                    pc_vals.append(int(part))
                except ValueError:
                    continue
            if pc_vals:
                partial_off_by = max(abs(v - target) for v in pc_vals)
            return json.dumps({
                "check": "sentence_count",
                "target": target,
                "partial_off_by": partial_off_by,
            })

        elif metric == "starts_with_word":
            target = criteria.get("target_word", "")
            return json.dumps({"check": "starts_with", "prefix": target})

        elif metric == "contains_bullet_points":
            return json.dumps({"check": "format_check", "pattern": r"[-•*]\s+"})

        elif metric == "absence_of_terms":
            terms = criteria.get("forbidden_terms", [])
            # Parse partial_max_violations from partial_compliance like "1-2 financial terms"
            pc = criteria.get("partial_compliance", "")
            partial_max_violations = 2
            m = re.search(r"(\d+)-(\d+)", pc)
            if m:
                partial_max_violations = int(m.group(2))
            return json.dumps({
                "check": "not_contains",
                "terms": terms,
                "partial_max_violations": partial_max_violations,
            })

        # Fallback: store the criteria as-is
        return json.dumps(criteria)

    def _load_query_file(self, path: Path) -> None:
        data = json.loads(path.read_text())
        for entry in data.get("queries", []):
            query = TestQuery(
                query_id=entry["query_id"],
                probe_id=entry["probe_id"],
                dimension=Dimension(entry["dimension"]),
                query=entry["query"],
                expected_answer=entry.get("expected_answer"),
                rubric=entry.get("rubric"),
                score_method=ScoreMethod(entry.get("score_method", "exact_match")),
            )
            self.queries[query.query_id] = query
            self.queries_by_probe[query.probe_id] = query

    def get_fillers(self, filler_type: FillerType = FillerType.NEUTRAL) -> list[FillerPassage]:
        return self.fillers_by_tier[filler_type]

    def get_probes(
        self,
        dimension: Dimension | None = None,
        probe_ids: list[str] | None = None,
    ) -> list[Probe]:
        if probe_ids is not None:
            return [self.probes[pid] for pid in probe_ids if pid in self.probes]
        if dimension is not None:
            return self.probes_by_dimension[dimension]
        return list(self.probes.values())

    def get_query_for_probe(self, probe_id: str) -> TestQuery | None:
        return self.queries_by_probe.get(probe_id)
