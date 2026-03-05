"""Read-only database queries returning pandas DataFrames."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


class QueryManager:
    """Read-only access to an APEX results database.

    SQLite: opens a fresh read-only connection per query so WAL commits
    from a running probe session are immediately visible.
    PostgreSQL: keeps a persistent connection (MVCC = no stale reads).
    """

    def __init__(self, dsn_or_path: str | Path) -> None:
        self._dsn = str(dsn_or_path)
        self._is_postgres = self._dsn.startswith(("postgresql://", "postgres://"))
        self._ph = "%s" if self._is_postgres else "?"
        if self._is_postgres:
            import psycopg

            self._pg_conn = psycopg.connect(self._dsn)
            self._ensure_schema()
        else:
            self._db_path = str(Path(self._dsn).resolve())
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure the database schema is up-to-date (adds missing columns)."""
        from apex.db import get_backend

        try:
            backend = get_backend(self._dsn)
            backend.create_schema()
            backend.close()
        except Exception:
            pass  # Best-effort; read-only queries will still work for existing columns

    def _connect(self):
        if self._is_postgres:
            return self._pg_conn
        return sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)

    def _release(self, conn) -> None:
        if not self._is_postgres:
            conn.close()

    def _query_df(self, sql: str, conn, params=None) -> pd.DataFrame:
        """Execute *sql* and return a DataFrame.

        Uses ``pd.read_sql_query`` for SQLite (officially supported) and a
        manual cursor path for PostgreSQL (pandas only guarantees sqlite3 for
        raw DB-API connections).
        """
        if self._is_postgres:
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, params or ())
                    cols = [desc[0] for desc in cur.description]
                    data = cur.fetchall()
                return pd.DataFrame(data, columns=cols)
            except Exception:
                conn.rollback()
                raise
        return pd.read_sql_query(sql, conn, params=params)

    def close(self) -> None:
        if self._is_postgres:
            self._pg_conn.close()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_models(self) -> list[dict]:
        """Distinct models with metadata."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT DISTINCT model_id, model_architecture, model_parameters,
                          quantization, max_context_window
                   FROM probe_results ORDER BY model_id"""
            ).fetchall()
            return [
                dict(
                    model_id=r[0],
                    architecture=r[1],
                    parameters=r[2],
                    quantization=r[3],
                    max_context_window=r[4],
                )
                for r in rows
            ]
        finally:
            self._release(conn)

    def get_dimensions(self, model_id: str | None = None) -> list[str]:
        conn = self._connect()
        try:
            if model_id:
                rows = conn.execute(
                    f"SELECT DISTINCT dimension FROM probe_results WHERE model_id = {self._ph} ORDER BY dimension",
                    (model_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT DISTINCT dimension FROM probe_results ORDER BY dimension"
                ).fetchall()
            return [r[0] for r in rows]
        finally:
            self._release(conn)

    def get_context_lengths(self, model_id: str | None = None) -> list[int]:
        conn = self._connect()
        try:
            if model_id:
                rows = conn.execute(
                    f"SELECT DISTINCT context_length FROM probe_results WHERE model_id = {self._ph} ORDER BY context_length",
                    (model_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT DISTINCT context_length FROM probe_results ORDER BY context_length"
                ).fetchall()
            return [r[0] for r in rows]
        finally:
            self._release(conn)

    def get_probe_ids(self, model_id: str | None = None, dimension: str | None = None) -> list[str]:
        conn = self._connect()
        try:
            conditions, params = [], []
            if model_id:
                conditions.append(f"model_id = {self._ph}")
                params.append(model_id)
            if dimension:
                conditions.append(f"dimension = {self._ph}")
                params.append(dimension)
            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            rows = conn.execute(
                f"SELECT DISTINCT probe_id FROM probe_results{where} ORDER BY probe_id",
                params,
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            self._release(conn)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_run_summary(self, run_uuid: str | None = None) -> pd.DataFrame:
        """One row per model with result count, timestamp range, refusal count."""
        conn = self._connect()
        try:
            where = ""
            params: tuple = ()
            if run_uuid:
                where = f" WHERE run_uuid = {self._ph}"
                params = (run_uuid,)
            return self._query_df(
                f"""SELECT model_id, model_architecture, model_parameters, quantization,
                          COUNT(*) AS result_count,
                          SUM(CASE WHEN refused = 1 THEN 1 ELSE 0 END) AS refused_count,
                          SUM(CASE WHEN score IS NULL THEN 1 ELSE 0 END) AS null_score_count,
                          MIN(timestamp) AS first_timestamp,
                          MAX(timestamp) AS last_timestamp
                   FROM probe_results{where}
                   GROUP BY model_id, model_architecture, model_parameters, quantization
                   ORDER BY model_id""",
                conn,
                params=params,
            )
        finally:
            self._release(conn)

    def get_run_configs(self) -> pd.DataFrame:
        """Per-run_uuid config summary extracted from probe_results."""
        conn = self._connect()
        try:
            return self._query_df(
                """SELECT run_uuid, model_id, model_architecture, model_parameters,
                          quantization, max_context_window, filler_type, temperature,
                          COUNT(*) AS result_count,
                          COUNT(DISTINCT probe_id) AS distinct_probes,
                          COUNT(DISTINCT dimension) AS distinct_dimensions,
                          COUNT(DISTINCT context_length) AS distinct_ctx_lengths,
                          COUNT(DISTINCT target_position_percent) AS distinct_positions,
                          MAX(run_number) AS max_run_number,
                          SUM(CASE WHEN refused = 1 THEN 1 ELSE 0 END) AS refused_count,
                          SUM(CASE WHEN score IS NULL THEN 1 ELSE 0 END) AS null_score_count,
                          AVG(score) AS mean_score,
                          MIN(timestamp) AS first_timestamp,
                          MAX(timestamp) AS last_timestamp
                   FROM probe_results
                   WHERE run_uuid IS NOT NULL
                   GROUP BY run_uuid, model_id, model_architecture, model_parameters,
                            quantization, max_context_window, filler_type, temperature
                   ORDER BY first_timestamp DESC""",
                conn,
            )
        except Exception:
            return pd.DataFrame()
        finally:
            self._release(conn)

    def get_run_context_lengths(self, run_uuid: str) -> list[int]:
        """Distinct context lengths used in a specific run."""
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT DISTINCT context_length FROM probe_results WHERE run_uuid = {self._ph} ORDER BY context_length",
                (run_uuid,),
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []
        finally:
            self._release(conn)

    def get_run_dimension_breakdown(self, run_uuid: str) -> pd.DataFrame:
        """Per-dimension counts and mean scores for a specific run."""
        conn = self._connect()
        try:
            return self._query_df(
                f"""SELECT dimension,
                          COUNT(*) AS count,
                          AVG(score) AS mean_score,
                          MIN(score) AS min_score,
                          MAX(score) AS max_score,
                          SUM(CASE WHEN refused = 1 THEN 1 ELSE 0 END) AS refused
                   FROM probe_results
                   WHERE run_uuid = {self._ph}
                   GROUP BY dimension
                   ORDER BY dimension""",
                conn,
                params=(run_uuid,),
            )
        except Exception:
            return pd.DataFrame(columns=["dimension", "count", "mean_score", "min_score", "max_score", "refused"])
        finally:
            self._release(conn)

    def get_dimension_breakdown(self, model_id: str) -> pd.DataFrame:
        """Per-dimension counts and mean scores for a model."""
        conn = self._connect()
        try:
            return self._query_df(
                f"""SELECT dimension,
                          COUNT(*) AS count,
                          AVG(score) AS mean_score,
                          MIN(score) AS min_score,
                          MAX(score) AS max_score,
                          SUM(CASE WHEN refused = 1 THEN 1 ELSE 0 END) AS refused
                   FROM probe_results
                   WHERE model_id = {self._ph}
                   GROUP BY dimension
                   ORDER BY dimension""",
                conn,
                params=(model_id,),
            )
        finally:
            self._release(conn)

    # ------------------------------------------------------------------
    # Curve data
    # ------------------------------------------------------------------

    def get_curve_data(
        self,
        model_id: str,
        context_length: int | None = None,
        dimension: str | None = None,
    ) -> pd.DataFrame:
        """Raw score rows for curve plotting."""
        conn = self._connect()
        try:
            conditions = [f"model_id = {self._ph}"]
            params: list = [model_id]
            if context_length is not None:
                conditions.append(f"context_length = {self._ph}")
                params.append(context_length)
            if dimension is not None:
                conditions.append(f"dimension = {self._ph}")
                params.append(dimension)
            where = " WHERE " + " AND ".join(conditions)
            return self._query_df(
                f"""SELECT probe_id, dimension,
                           target_position_percent * 100 AS target_position_percent,
                           score, context_length, content_type, run_number, refused,
                           run_uuid
                    FROM probe_results{where}
                    ORDER BY target_position_percent""",
                conn,
                params=params,
            )
        finally:
            self._release(conn)

    def get_run_uuids_for_model(
        self,
        model_id: str,
        context_length: int | None = None,
    ) -> pd.DataFrame:
        """Run UUIDs for a model with metadata: (run_uuid, filler_type, result_count, first_ts, last_ts)."""
        conn = self._connect()
        try:
            conditions = [f"model_id = {self._ph}", "run_uuid IS NOT NULL"]
            params: list = [model_id]
            if context_length is not None:
                conditions.append(f"context_length = {self._ph}")
                params.append(context_length)
            where = " WHERE " + " AND ".join(conditions)
            return self._query_df(
                f"""SELECT run_uuid, filler_type, COUNT(*) AS result_count,
                           MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts
                    FROM probe_results{where}
                    GROUP BY run_uuid, filler_type
                    ORDER BY first_ts""",
                conn,
                params=params,
            )
        except Exception:
            return pd.DataFrame(columns=["run_uuid", "filler_type", "result_count", "first_ts", "last_ts"])
        finally:
            self._release(conn)

    def get_cross_model_data(
        self,
        model_ids: list[str],
        dimension: str | None = None,
        context_length: int | None = None,
    ) -> pd.DataFrame:
        """Curve data across multiple models."""
        conn = self._connect()
        try:
            placeholders = ",".join(self._ph for _ in model_ids)
            conditions = [f"model_id IN ({placeholders})"]
            params: list = list(model_ids)
            if dimension:
                conditions.append(f"dimension = {self._ph}")
                params.append(dimension)
            if context_length is not None:
                conditions.append(f"context_length = {self._ph}")
                params.append(context_length)
            where = " WHERE " + " AND ".join(conditions)
            return self._query_df(
                f"""SELECT model_id, probe_id, dimension,
                           target_position_percent * 100 AS target_position_percent,
                           score, context_length, run_number, refused
                    FROM probe_results{where}
                    ORDER BY model_id, target_position_percent""",
                conn,
                params=params,
            )
        finally:
            self._release(conn)

    # ------------------------------------------------------------------
    # Probe detail
    # ------------------------------------------------------------------

    def get_probe_detail(self, probe_id: str, model_id: str | None = None) -> pd.DataFrame:
        conn = self._connect()
        try:
            conditions = [f"probe_id = {self._ph}"]
            params: list = [probe_id]
            if model_id:
                conditions.append(f"model_id = {self._ph}")
                params.append(model_id)
            where = " WHERE " + " AND ".join(conditions)
            return self._query_df(
                f"""SELECT model_id, probe_id, dimension,
                           target_position_percent * 100 AS target_position_percent,
                           score, context_length, run_number, score_method,
                           raw_response, raw_test_response, refused,
                           content_type, probe_content
                    FROM probe_results{where}
                    ORDER BY target_position_percent""",
                conn,
                params=params,
            )
        finally:
            self._release(conn)

    def get_probe_metadata(self, probe_id: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute(
                f"""SELECT probe_id, dimension, content_type, probe_content, score_method
                   FROM probe_results WHERE probe_id = {self._ph} LIMIT 1""",
                (probe_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(
                probe_id=row[0],
                dimension=row[1],
                content_type=row[2],
                probe_content=row[3],
                score_method=row[4],
            )
        finally:
            self._release(conn)

    # ------------------------------------------------------------------
    # Run monitoring
    # ------------------------------------------------------------------

    def get_run_progress(self) -> pd.DataFrame:
        """Per-model completed count with distinct probes/positions/ctx_lengths for total estimate."""
        conn = self._connect()
        try:
            return self._query_df(
                """SELECT model_id,
                          COUNT(*) AS completed,
                          COUNT(DISTINCT probe_id) AS distinct_probes,
                          COUNT(DISTINCT target_position_percent) AS distinct_positions,
                          COUNT(DISTINCT context_length) AS distinct_ctx_lengths,
                          MIN(timestamp) AS first_ts,
                          MAX(timestamp) AS last_ts
                   FROM probe_results
                   GROUP BY model_id
                   ORDER BY model_id""",
                conn,
            )
        finally:
            self._release(conn)

    def get_recent_results(self, limit: int = 20) -> pd.DataFrame:
        """Most recent results ordered by timestamp DESC."""
        conn = self._connect()
        try:
            return self._query_df(
                f"""SELECT timestamp, model_id, probe_id, dimension,
                          target_position_percent * 100 AS target_position_percent,
                          context_length, score, score_method, refused
                   FROM probe_results
                   ORDER BY timestamp DESC
                   LIMIT {self._ph}""",
                conn,
                params=(limit,),
            )
        finally:
            self._release(conn)

    def get_score_method_breakdown(self) -> pd.DataFrame:
        """Per-model count by score_method."""
        conn = self._connect()
        try:
            return self._query_df(
                """SELECT model_id, score_method, COUNT(*) AS count
                   FROM probe_results
                   GROUP BY model_id, score_method
                   ORDER BY model_id, score_method""",
                conn,
            )
        finally:
            self._release(conn)

    def get_dimension_progress(self) -> pd.DataFrame:
        """Per-(model, dimension) completion counts."""
        conn = self._connect()
        try:
            return self._query_df(
                """SELECT model_id, dimension, COUNT(*) AS completed,
                          COUNT(DISTINCT probe_id) AS distinct_probes,
                          COUNT(DISTINCT target_position_percent) AS distinct_positions,
                          AVG(score) AS mean_score
                   FROM probe_results
                   GROUP BY model_id, dimension
                   ORDER BY model_id, dimension""",
                conn,
            )
        finally:
            self._release(conn)

    def get_recent_errors(self, limit: int = 10, run_uuid: str | None = None) -> pd.DataFrame:
        """Recent refused or null-score results with response snippet."""
        conn = self._connect()
        try:
            conditions = ["(refused = 1 OR score IS NULL)"]
            params: list = []
            if run_uuid:
                conditions.append(f"run_uuid = {self._ph}")
                params.append(run_uuid)
            where = " WHERE " + " AND ".join(conditions)
            params.append(limit)
            return self._query_df(
                f"""SELECT timestamp, model_id, probe_id, dimension,
                          target_position_percent * 100 AS target_position_percent,
                          context_length, score, refused,
                          SUBSTR(raw_response, 1, 200) AS response_snippet
                   FROM probe_results{where}
                   ORDER BY timestamp DESC
                   LIMIT {self._ph}""",
                conn,
                params=tuple(params),
            )
        finally:
            self._release(conn)

    # ------------------------------------------------------------------
    # Run UUID management
    # ------------------------------------------------------------------

    def get_latest_run_uuid(self) -> str | None:
        """Return the run_uuid of the most recent result, or None."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT run_uuid FROM probe_results ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None
        finally:
            self._release(conn)

    def get_run_uuids(self) -> pd.DataFrame:
        """Distinct run UUIDs with model, count, timestamp range."""
        conn = self._connect()
        try:
            return self._query_df(
                """SELECT run_uuid, model_id, COUNT(*) AS count,
                          MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts
                   FROM probe_results
                   WHERE run_uuid IS NOT NULL
                   GROUP BY run_uuid, model_id
                   ORDER BY last_ts DESC""",
                conn,
            )
        except Exception:
            # Column may not exist yet in databases that haven't been migrated
            return pd.DataFrame(columns=["run_uuid", "model_id", "count", "first_ts", "last_ts"])
        finally:
            self._release(conn)

    def delete_by_run_uuid(self, run_uuid: str) -> int:
        """Delete all results for a run UUID. Returns row count."""
        if self._is_postgres:
            try:
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM probe_results WHERE run_uuid = %s",
                        (run_uuid,),
                    )
                    count = cur.rowcount
                self._pg_conn.commit()
                return count
            except Exception:
                self._pg_conn.rollback()
                raise
        else:
            # Open a writable connection for SQLite
            conn = sqlite3.connect(self._db_path)
            try:
                cursor = conn.execute(
                    "DELETE FROM probe_results WHERE run_uuid = ?",
                    (run_uuid,),
                )
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

    def delete_by_model(self, model_id: str) -> int:
        """Delete all results for a model. Returns row count."""
        if self._is_postgres:
            try:
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM probe_results WHERE model_id = %s",
                        (model_id,),
                    )
                    count = cur.rowcount
                self._pg_conn.commit()
                return count
            except Exception:
                self._pg_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self._db_path)
            try:
                cursor = conn.execute(
                    "DELETE FROM probe_results WHERE model_id = ?",
                    (model_id,),
                )
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Server launch history
    # ------------------------------------------------------------------

    def record_launch(
        self,
        launch_id: str,
        node: str,
        model_path: str,
        port: int,
        requested_ctx_per_slot: int,
        parallel: int,
        total_ctx: int,
        gpu_layers: int,
        threads: int | None,
        flash_attn: bool,
        llama_server_bin: str,
        pid: int,
        status: str,
        launched_at: str,
        notes: str | None = None,
    ) -> None:
        """INSERT a server launch record."""
        from apex.db import LAUNCH_COLUMN_NAMES

        values = (
            launch_id, node, model_path, port, requested_ctx_per_slot,
            parallel, total_ctx, gpu_layers, threads, int(flash_attn),
            llama_server_bin, pid, status, None, None, None, None,
            launched_at, notes,
        )
        cols = ", ".join(LAUNCH_COLUMN_NAMES)
        phs = ", ".join(self._ph for _ in LAUNCH_COLUMN_NAMES)
        sql = f"INSERT INTO server_launches ({cols}) VALUES ({phs})"

        if self._is_postgres:
            try:
                with self._pg_conn.cursor() as cur:
                    cur.execute(sql, values)
                self._pg_conn.commit()
            except Exception:
                self._pg_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(sql, values)
                conn.commit()
            finally:
                conn.close()

    def update_launch_actual(
        self,
        launch_id: str,
        actual_ctx_per_slot: int | None = None,
        model_id_reported: str | None = None,
        n_params: int | None = None,
        n_ctx_train: int | None = None,
        notes: str | None = None,
    ) -> None:
        """UPDATE a launch record with post-startup values."""
        sql = (
            f"UPDATE server_launches SET "
            f"actual_ctx_per_slot = {self._ph}, "
            f"model_id_reported = {self._ph}, "
            f"n_params = {self._ph}, "
            f"n_ctx_train = {self._ph}, "
            f"notes = {self._ph}, "
            f"status = {self._ph} "
            f"WHERE launch_id = {self._ph}"
        )
        status = "running"
        values = (actual_ctx_per_slot, model_id_reported, n_params, n_ctx_train, notes, status, launch_id)

        if self._is_postgres:
            try:
                with self._pg_conn.cursor() as cur:
                    cur.execute(sql, values)
                self._pg_conn.commit()
            except Exception:
                self._pg_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(sql, values)
                conn.commit()
            finally:
                conn.close()

    def get_launch_history(self, limit: int = 20) -> pd.DataFrame:
        """SELECT recent server launches, newest first."""
        conn = self._connect()
        try:
            return self._query_df(
                f"SELECT * FROM server_launches ORDER BY launched_at DESC LIMIT {self._ph}",
                conn,
                params=(limit,),
            )
        except Exception:
            return pd.DataFrame()
        finally:
            self._release(conn)

    def get_launch_by_id(self, launch_id: str) -> dict | None:
        """Return a single launch record as dict, or None."""
        conn = self._connect()
        try:
            df = self._query_df(
                f"SELECT * FROM server_launches WHERE launch_id = {self._ph}",
                conn,
                params=(launch_id,),
            )
            if df.empty:
                return None
            return df.iloc[0].to_dict()
        except Exception:
            return None
        finally:
            self._release(conn)

    # ------------------------------------------------------------------
    # Aggregation (in pandas, not SQL)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def has_calibration_tables(self) -> bool:
        """Check whether calibration_baselines and calibration_prompts tables exist."""
        conn = self._connect()
        try:
            if self._is_postgres:
                df = self._query_df(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name IN ('calibration_baselines', 'calibration_prompts')",
                    conn,
                )
            else:
                df = self._query_df(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('calibration_baselines', 'calibration_prompts')",
                    conn,
                )
            return len(df) == 2
        except Exception:
            return False
        finally:
            self._release(conn)

    def get_calibration_status(self) -> dict:
        """Return {prompt_count, baseline_df} summarising calibration state."""
        conn = self._connect()
        try:
            prompt_df = self._query_df("SELECT COUNT(*) AS cnt FROM calibration_prompts", conn)
            prompt_count = int(prompt_df.iloc[0]["cnt"]) if not prompt_df.empty else 0
            baseline_df = self._query_df(
                "SELECT baseline_type, model_id, COUNT(*) AS count "
                "FROM calibration_baselines GROUP BY baseline_type, model_id "
                "ORDER BY model_id, baseline_type",
                conn,
            )
            return {"prompt_count": prompt_count, "baseline_df": baseline_df}
        except Exception:
            return {"prompt_count": 0, "baseline_df": pd.DataFrame(columns=["baseline_type", "model_id", "count"])}
        finally:
            self._release(conn)

    def get_baseline_models(self) -> list[str]:
        """Distinct model_ids that have calibration baselines recorded."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT model_id FROM calibration_baselines ORDER BY model_id"
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []
        finally:
            self._release(conn)

    def get_calibrated_models(self) -> list[str]:
        """Distinct model_ids that have calibration baselines OR calibrated run data."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT model_id FROM calibration_baselines "
                "UNION "
                "SELECT DISTINCT model_id FROM probe_results WHERE filler_type = 'calibrated' "
                "ORDER BY model_id"
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []
        finally:
            self._release(conn)

    def get_baselines_overview(self, model_id: str) -> pd.DataFrame:
        """Baseline rows for a model: probe_id, dimension, baseline_type, score, score_method."""
        conn = self._connect()
        try:
            return self._query_df(
                f"SELECT probe_id, dimension, baseline_type, score, score_method "
                f"FROM calibration_baselines WHERE model_id = {self._ph} "
                f"ORDER BY dimension, probe_id, baseline_type",
                conn,
                params=(model_id,),
            )
        except Exception:
            return pd.DataFrame(columns=["probe_id", "dimension", "baseline_type", "score", "score_method"])
        finally:
            self._release(conn)

    def get_calibrated_curve_data(
        self,
        model_id: str,
        context_length: int | None = None,
        dimension: str | None = None,
    ) -> pd.DataFrame:
        """Curve data filtered to filler_type = 'calibrated'."""
        conn = self._connect()
        try:
            conditions = [f"model_id = {self._ph}", "filler_type = 'calibrated'"]
            params: list = [model_id]
            if context_length is not None:
                conditions.append(f"context_length = {self._ph}")
                params.append(context_length)
            if dimension is not None:
                conditions.append(f"dimension = {self._ph}")
                params.append(dimension)
            where = " WHERE " + " AND ".join(conditions)
            return self._query_df(
                f"""SELECT probe_id, dimension,
                           target_position_percent * 100 AS target_position_percent,
                           score, context_length, content_type, run_number, refused
                    FROM probe_results{where}
                    ORDER BY target_position_percent""",
                conn,
                params=params,
            )
        except Exception:
            return pd.DataFrame(columns=[
                "probe_id", "dimension", "target_position_percent",
                "score", "context_length", "content_type", "run_number", "refused",
            ])
        finally:
            self._release(conn)

    def get_dynamic_curve_data(
        self,
        model_id: str,
        context_length: int | None = None,
        dimension: str | None = None,
    ) -> pd.DataFrame:
        """Curve data filtered to filler_type != 'calibrated'."""
        conn = self._connect()
        try:
            conditions = [f"model_id = {self._ph}", "filler_type != 'calibrated'"]
            params: list = [model_id]
            if context_length is not None:
                conditions.append(f"context_length = {self._ph}")
                params.append(context_length)
            if dimension is not None:
                conditions.append(f"dimension = {self._ph}")
                params.append(dimension)
            where = " WHERE " + " AND ".join(conditions)
            return self._query_df(
                f"""SELECT probe_id, dimension,
                           target_position_percent * 100 AS target_position_percent,
                           score, context_length, content_type, run_number, refused
                    FROM probe_results{where}
                    ORDER BY target_position_percent""",
                conn,
                params=params,
            )
        except Exception:
            return pd.DataFrame(columns=[
                "probe_id", "dimension", "target_position_percent",
                "score", "context_length", "content_type", "run_number", "refused",
            ])
        finally:
            self._release(conn)

    @staticmethod
    def normalize_by_baselines(
        raw_df: pd.DataFrame,
        baselines_df: pd.DataFrame,
        baseline_type: str = "anchored",
    ) -> pd.DataFrame:
        """Normalize raw scores by baseline and aggregate to (dimension, position) → mean/std/CI.

        Returns same shape as ``aggregate_curve`` output with 'normalized' replacing 'score'.
        """
        if raw_df.empty or baselines_df.empty:
            return pd.DataFrame(
                columns=["dimension", "target_position_percent", "mean", "std", "count", "ci_lower", "ci_upper"]
            )

        bl = baselines_df[baselines_df["baseline_type"] == baseline_type][["probe_id", "score"]].rename(
            columns={"score": "baseline_score"}
        )
        if bl.empty:
            return pd.DataFrame(
                columns=["dimension", "target_position_percent", "mean", "std", "count", "ci_lower", "ci_upper"]
            )

        scored = raw_df[raw_df["score"].notna() & (raw_df["refused"] == 0)].copy()
        if scored.empty:
            return pd.DataFrame(
                columns=["dimension", "target_position_percent", "mean", "std", "count", "ci_lower", "ci_upper"]
            )

        merged = scored.merge(bl, on="probe_id", how="inner")
        if merged.empty:
            return pd.DataFrame(
                columns=["dimension", "target_position_percent", "mean", "std", "count", "ci_lower", "ci_upper"]
            )

        merged["normalized"] = merged["score"] / merged["baseline_score"].replace(0, float("nan"))

        agg = (
            merged.groupby(["dimension", "target_position_percent"])["normalized"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["std"] = agg["std"].fillna(0)
        agg["ci_lower"] = agg["mean"] - 1.96 * agg["std"] / agg["count"] ** 0.5
        agg["ci_upper"] = agg["mean"] + 1.96 * agg["std"] / agg["count"] ** 0.5
        return agg

    # ------------------------------------------------------------------
    # Aggregation (in pandas, not SQL)
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_curve(df: pd.DataFrame, group_col: str = "dimension") -> pd.DataFrame:
        """Aggregate raw scores into mean/std/CI per (group, position).

        Returns DataFrame with columns: [group_col, target_position_percent,
        mean, std, count, ci_lower, ci_upper].
        """
        if df.empty:
            return pd.DataFrame(
                columns=[group_col, "target_position_percent", "mean", "std", "count", "ci_lower", "ci_upper"]
            )

        scored = df[df["score"].notna() & (df["refused"] == 0)].copy()
        if scored.empty:
            return pd.DataFrame(
                columns=[group_col, "target_position_percent", "mean", "std", "count", "ci_lower", "ci_upper"]
            )

        agg = (
            scored.groupby([group_col, "target_position_percent"])["score"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["std"] = agg["std"].fillna(0)
        # 95% CI = mean +/- 1.96 * std / sqrt(n)
        agg["ci_lower"] = (agg["mean"] - 1.96 * agg["std"] / agg["count"] ** 0.5).clip(0, 1)
        agg["ci_upper"] = (agg["mean"] + 1.96 * agg["std"] / agg["count"] ** 0.5).clip(0, 1)
        return agg

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_dimension_correlations(agg_df: pd.DataFrame) -> dict[str, float]:
        """Pearson r between each pair of dimensions at shared positions."""
        dims = sorted(agg_df["dimension"].unique()) if "dimension" in agg_df.columns else []
        if len(dims) < 2:
            return {}

        pivoted = agg_df.pivot_table(
            index="target_position_percent", columns="dimension", values="mean"
        ).dropna()

        correlations = {}
        for i, d1 in enumerate(dims):
            for d2 in dims[i + 1 :]:
                if d1 in pivoted.columns and d2 in pivoted.columns:
                    r = pivoted[d1].corr(pivoted[d2])
                    correlations[f"{d1} vs {d2}"] = round(r, 3)
        return correlations

    @staticmethod
    def find_sweet_spots(agg_df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """Positions where all dimensions score above threshold."""
        if "dimension" not in agg_df.columns or agg_df.empty:
            return pd.DataFrame(columns=["target_position_percent"])

        pivoted = agg_df.pivot_table(
            index="target_position_percent", columns="dimension", values="mean"
        ).dropna()

        if pivoted.empty:
            return pd.DataFrame(columns=["target_position_percent"])

        mask = (pivoted >= threshold).all(axis=1)
        return pivoted[mask].reset_index()[["target_position_percent"]]
