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
                result.run_uuid,
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

    def update_score(
        self,
        row_id: int,
        score: float | None,
        justification: str | None,
        evaluator_model_id: str | None = None,
    ) -> None:
        """Update score and justification for a result by primary key.

        When evaluator_model_id is provided, also update the evaluator_model_id column.
        """
        if evaluator_model_id is not None:
            self._conn.execute(
                f"UPDATE probe_results SET score = {self._ph},"
                f" evaluator_justification = {self._ph},"
                f" evaluator_model_id = {self._ph} WHERE id = {self._ph}",
                (score, justification, evaluator_model_id, row_id),
            )
        else:
            self._conn.execute(
                f"UPDATE probe_results SET score = {self._ph},"
                f" evaluator_justification = {self._ph} WHERE id = {self._ph}",
                (score, justification, row_id),
            )
        self._conn.commit()

    def delete_by_run_uuid(self, run_uuid: str) -> int:
        """Delete all results from a specific run UUID. Returns row count."""
        cursor = self._conn.execute(
            f"DELETE FROM probe_results WHERE run_uuid = {self._ph}",
            (run_uuid,),
        )
        self._conn.commit()
        return cursor.rowcount

    def delete_by_model(self, model_id: str) -> int:
        """Delete all results for a model. Returns row count."""
        cursor = self._conn.execute(
            f"DELETE FROM probe_results WHERE model_id = {self._ph}",
            (model_id,),
        )
        self._conn.commit()
        return cursor.rowcount

    def delete_by_filters(
        self,
        model_id: str | None = None,
        dimension: str | None = None,
        probe_id: str | None = None,
        context_length: int | None = None,
    ) -> int:
        """Delete results matching optional filters. Returns row count."""
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
        if context_length is not None:
            conditions.append(f"context_length = {self._ph}")
            params.append(context_length)

        if not conditions:
            return 0

        where = " WHERE " + " AND ".join(conditions)
        cursor = self._conn.execute(
            f"DELETE FROM probe_results{where}", params
        )
        self._conn.commit()
        return cursor.rowcount

    def get_run_uuids(self) -> list[dict]:
        """Return distinct run UUIDs with model_id, count, and timestamp range."""
        cursor = self._conn.execute(
            "SELECT run_uuid, model_id, COUNT(*) AS count,"
            " MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts"
            " FROM probe_results"
            " WHERE run_uuid IS NOT NULL"
            " GROUP BY run_uuid, model_id"
            " ORDER BY last_ts DESC"
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        self._backend.close()
