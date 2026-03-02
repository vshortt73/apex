"""Storage layer for calibration prompts and baselines."""

from __future__ import annotations

from pathlib import Path

from apex.db import (
    CALIBRATION_BASELINE_COLUMN_NAMES,
    CALIBRATION_PROMPT_COLUMN_NAMES,
    create_calibration_schema,
    get_backend,
)
from apex.types import CalibrationBaseline, CalibrationPrompt


class CalibrationStore:
    """Database-backed storage for calibration prompts and baselines."""

    def __init__(self, dsn_or_path: str | Path) -> None:
        self._backend = get_backend(str(dsn_or_path))
        create_calibration_schema(self._backend)

    @property
    def _conn(self):
        return self._backend.connection

    @property
    def _ph(self) -> str:
        return self._backend.placeholder

    # --- Prompt operations ---

    def write_prompt(self, prompt: CalibrationPrompt) -> None:
        cols = ", ".join(CALIBRATION_PROMPT_COLUMN_NAMES)
        phs = ", ".join(self._ph for _ in CALIBRATION_PROMPT_COLUMN_NAMES)
        sql = self._upsert_prompt_sql(cols, phs)
        self._conn.execute(sql, self._prompt_values(prompt))
        self._conn.commit()

    def write_prompts(self, prompts: list[CalibrationPrompt]) -> None:
        if not prompts:
            return
        cols = ", ".join(CALIBRATION_PROMPT_COLUMN_NAMES)
        phs = ", ".join(self._ph for _ in CALIBRATION_PROMPT_COLUMN_NAMES)
        sql = self._upsert_prompt_sql(cols, phs)
        for prompt in prompts:
            self._conn.execute(sql, self._prompt_values(prompt))
        self._conn.commit()

    def get_prompts(
        self,
        probe_id: str | None = None,
        context_length: int | None = None,
    ) -> list[dict]:
        conditions: list[str] = []
        params: list = []
        if probe_id:
            conditions.append(f"probe_id = {self._ph}")
            params.append(probe_id)
        if context_length is not None:
            conditions.append(f"context_length = {self._ph}")
            params.append(context_length)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        cursor = self._conn.execute(
            f"SELECT * FROM calibration_prompts{where} ORDER BY id", params
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def count_prompts(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM calibration_prompts")
        return cursor.fetchone()[0]

    def delete_prompts(self) -> int:
        cursor = self._conn.execute("DELETE FROM calibration_prompts")
        self._conn.commit()
        return cursor.rowcount

    # --- Baseline operations ---

    def write_baseline(self, baseline: CalibrationBaseline) -> None:
        cols = ", ".join(CALIBRATION_BASELINE_COLUMN_NAMES)
        phs = ", ".join(self._ph for _ in CALIBRATION_BASELINE_COLUMN_NAMES)
        sql = self._upsert_baseline_sql(cols, phs)
        self._conn.execute(sql, self._baseline_values(baseline))
        self._conn.commit()

    def get_baselines(
        self,
        model_id: str | None = None,
        probe_id: str | None = None,
        baseline_type: str | None = None,
    ) -> list[dict]:
        conditions: list[str] = []
        params: list = []
        if model_id:
            conditions.append(f"model_id = {self._ph}")
            params.append(model_id)
        if probe_id:
            conditions.append(f"probe_id = {self._ph}")
            params.append(probe_id)
        if baseline_type:
            conditions.append(f"baseline_type = {self._ph}")
            params.append(baseline_type)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        cursor = self._conn.execute(
            f"SELECT * FROM calibration_baselines{where} ORDER BY id", params
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_baseline_for_probe(
        self,
        probe_id: str,
        model_id: str,
        baseline_type: str = "bare",
    ) -> float | None:
        cursor = self._conn.execute(
            f"SELECT score FROM calibration_baselines"
            f" WHERE probe_id = {self._ph}"
            f" AND model_id = {self._ph}"
            f" AND baseline_type = {self._ph}",
            (probe_id, model_id, baseline_type),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def count_baselines(
        self,
        model_id: str | None = None,
        baseline_type: str | None = None,
    ) -> int:
        conditions: list[str] = []
        params: list = []
        if model_id:
            conditions.append(f"model_id = {self._ph}")
            params.append(model_id)
        if baseline_type:
            conditions.append(f"baseline_type = {self._ph}")
            params.append(baseline_type)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        cursor = self._conn.execute(
            f"SELECT COUNT(*) FROM calibration_baselines{where}", params
        )
        return cursor.fetchone()[0]

    def delete_baselines(
        self,
        model_id: str | None = None,
        baseline_type: str | None = None,
    ) -> int:
        conditions: list[str] = []
        params: list = []
        if model_id:
            conditions.append(f"model_id = {self._ph}")
            params.append(model_id)
        if baseline_type:
            conditions.append(f"baseline_type = {self._ph}")
            params.append(baseline_type)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        cursor = self._conn.execute(
            f"DELETE FROM calibration_baselines{where}", params
        )
        self._conn.commit()
        return cursor.rowcount

    def get_filler_factor(self, probe_id: str, model_id: str) -> float | None:
        """Compute filler_factor = anchored / bare for a probe+model pair.

        Returns None if either baseline is missing or bare is zero.
        """
        bare = self.get_baseline_for_probe(probe_id, model_id, "bare")
        anchored = self.get_baseline_for_probe(probe_id, model_id, "anchored")
        if bare is None or anchored is None or bare == 0.0:
            return None
        return anchored / bare

    def close(self) -> None:
        self._backend.close()

    # --- Internal helpers ---

    def _upsert_prompt_sql(self, cols: str, phs: str) -> str:
        from apex.db import SqliteBackend

        if isinstance(self._backend, SqliteBackend):
            return f"INSERT OR REPLACE INTO calibration_prompts ({cols}) VALUES ({phs})"
        # PostgreSQL
        unique = "probe_id, position_percent, context_length"
        data_cols = [n for n in CALIBRATION_PROMPT_COLUMN_NAMES if n not in ("probe_id", "position_percent", "context_length")]
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in data_cols)
        return (
            f"INSERT INTO calibration_prompts ({cols}) VALUES ({phs}) "
            f"ON CONFLICT ({unique}) DO UPDATE SET {set_clause}"
        )

    def _upsert_baseline_sql(self, cols: str, phs: str) -> str:
        from apex.db import SqliteBackend

        if isinstance(self._backend, SqliteBackend):
            return f"INSERT OR REPLACE INTO calibration_baselines ({cols}) VALUES ({phs})"
        # PostgreSQL
        unique = "probe_id, model_id, baseline_type"
        data_cols = [n for n in CALIBRATION_BASELINE_COLUMN_NAMES if n not in ("probe_id", "model_id", "baseline_type")]
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in data_cols)
        return (
            f"INSERT INTO calibration_baselines ({cols}) VALUES ({phs}) "
            f"ON CONFLICT ({unique}) DO UPDATE SET {set_clause}"
        )

    @staticmethod
    def _prompt_values(p: CalibrationPrompt) -> tuple:
        return (
            p.probe_id,
            p.dimension,
            p.position_percent,
            p.context_length,
            p.seed,
            p.full_text,
            p.actual_token_count,
            p.target_position_tokens,
            p.filler_ids_before,
            p.filler_ids_after,
            p.probe_hash,
            p.content_hash,
            p.generated_at,
        )

    @staticmethod
    def _baseline_values(b: CalibrationBaseline) -> tuple:
        return (
            b.probe_id,
            b.dimension,
            b.model_id,
            b.baseline_type,
            b.score,
            b.score_method,
            b.justification,
            b.raw_response,
            b.raw_test_response,
            b.error,
            b.timestamp,
        )
