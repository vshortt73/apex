"""Probe result storage with resume support."""

from __future__ import annotations

import json
from pathlib import Path

from apex.db import get_backend
from apex.types import ProbeResult


class ResultStore:
    """Database-backed probe result storage with resume support."""

    def __init__(self, dsn_or_path: str | Path) -> None:
        self._backend = get_backend(str(dsn_or_path))
        self._backend.create_schema()

    @property
    def _conn(self):
        return self._backend.connection

    @property
    def _ph(self) -> str:
        return self._backend.placeholder

    def write_result(self, result: ProbeResult) -> None:
        self._conn.execute(
            self._backend.upsert_sql(),
            (
                result.model_id,
                result.model_architecture,
                result.model_parameters,
                result.quantization,
                result.max_context_window,
                result.context_length,
                result.context_fill_ratio,
                result.target_position,
                result.target_position_percent,
                result.dimension,
                result.content_type,
                result.probe_id,
                result.probe_content,
                result.filler_type,
                result.test_query_id,
                result.temperature,
                result.run_number,
                result.total_runs,
                result.score,
                result.score_method,
                result.raw_response,
                result.raw_test_response,
                result.evaluator_model_id,
                result.evaluator_justification,
                result.latency_ms,
                result.timestamp,
                result.library_version,
                result.framework_version,
                1 if result.refused else 0,
            ),
        )
        self._conn.commit()

    def get_completed_runs(
        self, model_id: str
    ) -> set[tuple[str, float, int, int]]:
        """Return set of (probe_id, position_percent, context_length, run_number) already completed."""
        cursor = self._conn.execute(
            f"SELECT probe_id, target_position_percent, context_length, run_number"
            f" FROM probe_results WHERE model_id = {self._ph}",
            (model_id,),
        )
        return {(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()}

    def query_results(
        self,
        model_id: str | None = None,
        dimension: str | None = None,
        probe_id: str | None = None,
        context_length: int | None = None,
    ) -> list[dict]:
        """Query results with optional filters."""
        conditions = []
        params: list = []
        if model_id:
            conditions.append(f"model_id = {self._ph}")
            params.append(model_id)
        if dimension:
            conditions.append(f"dimension = {self._ph}")
            params.append(dimension)
        if probe_id:
            conditions.append(f"probe_id = {self._ph}")
            params.append(probe_id)
        if context_length:
            conditions.append(f"context_length = {self._ph}")
            params.append(context_length)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        cursor = self._conn.execute(
            f"SELECT * FROM probe_results{where} ORDER BY id", params
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def export_json(self, output_path: str | Path, **filters) -> int:
        """Export results to JSON file. Returns count of exported records."""
        results = self.query_results(**filters)
        Path(output_path).write_text(json.dumps(results, indent=2, default=str))
        return len(results)

    def count_results(self, model_id: str | None = None) -> int:
        if model_id:
            cursor = self._conn.execute(
                f"SELECT COUNT(*) FROM probe_results WHERE model_id = {self._ph}",
                (model_id,),
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(*) FROM probe_results")
        return cursor.fetchone()[0]

    def update_score(self, row_id: int, score: float | None, justification: str | None) -> None:
        """Update score and justification for a result by primary key."""
        self._conn.execute(
            f"UPDATE probe_results SET score = {self._ph},"
            f" evaluator_justification = {self._ph} WHERE id = {self._ph}",
            (score, justification, row_id),
        )
        self._conn.commit()

    def close(self) -> None:
        self._backend.close()
