"""Database backend abstraction for SQLite and PostgreSQL."""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Column definitions for probe_results (name, type_template).
# {float} is replaced with REAL (SQLite) or DOUBLE PRECISION (PostgreSQL).
_COLUMNS = [
    ("model_id", "TEXT NOT NULL"),
    ("model_architecture", "TEXT NOT NULL"),
    ("model_parameters", "TEXT NOT NULL"),
    ("quantization", "TEXT NOT NULL"),
    ("max_context_window", "INTEGER NOT NULL"),
    ("context_length", "INTEGER NOT NULL"),
    ("context_fill_ratio", "{float} NOT NULL"),
    ("target_position", "INTEGER NOT NULL"),
    ("target_position_percent", "{float} NOT NULL"),
    ("dimension", "TEXT NOT NULL"),
    ("content_type", "TEXT NOT NULL"),
    ("probe_id", "TEXT NOT NULL"),
    ("probe_content", "TEXT NOT NULL"),
    ("filler_type", "TEXT NOT NULL"),
    ("test_query_id", "TEXT NOT NULL"),
    ("temperature", "{float} NOT NULL"),
    ("run_number", "INTEGER NOT NULL"),
    ("total_runs", "INTEGER NOT NULL"),
    ("score", "{float}"),
    ("score_method", "TEXT NOT NULL"),
    ("raw_response", "TEXT NOT NULL"),
    ("raw_test_response", "TEXT NOT NULL"),
    ("evaluator_model_id", "TEXT"),
    ("evaluator_justification", "TEXT"),
    ("latency_ms", "INTEGER NOT NULL"),
    ("timestamp", "TEXT NOT NULL"),
    ("library_version", "TEXT NOT NULL"),
    ("framework_version", "TEXT NOT NULL"),
    ("refused", "INTEGER NOT NULL DEFAULT 0"),
    ("run_uuid", "TEXT"),
]

COLUMN_NAMES = [name for name, _ in _COLUMNS]

UNIQUE_KEY = ("model_id", "probe_id", "target_position_percent", "context_length", "run_number")

# Non-key columns — updated on upsert conflict.
_DATA_COLUMNS = [name for name in COLUMN_NAMES if name not in UNIQUE_KEY]

# Column definitions for server_launches table.
_LAUNCH_COLUMNS = [
    ("launch_id", "TEXT NOT NULL"),
    ("node", "TEXT NOT NULL"),
    ("model_path", "TEXT NOT NULL"),
    ("port", "INTEGER NOT NULL"),
    ("requested_ctx_per_slot", "INTEGER NOT NULL"),
    ("parallel", "INTEGER NOT NULL"),
    ("total_ctx", "INTEGER NOT NULL"),
    ("gpu_layers", "INTEGER NOT NULL"),
    ("threads", "INTEGER"),
    ("flash_attn", "INTEGER NOT NULL"),
    ("llama_server_bin", "TEXT NOT NULL"),
    ("pid", "INTEGER NOT NULL"),
    ("status", "TEXT NOT NULL"),
    ("actual_ctx_per_slot", "INTEGER"),
    ("model_id_reported", "TEXT"),
    ("n_params", "INTEGER"),
    ("n_ctx_train", "INTEGER"),
    ("launched_at", "TEXT NOT NULL"),
    ("notes", "TEXT"),
]

LAUNCH_COLUMN_NAMES = [name for name, _ in _LAUNCH_COLUMNS]

# Column definitions for calibration_prompts table.
_CALIBRATION_PROMPT_COLUMNS = [
    ("probe_id", "TEXT NOT NULL"),
    ("dimension", "TEXT NOT NULL"),
    ("position_percent", "{float} NOT NULL"),
    ("context_length", "INTEGER NOT NULL"),
    ("seed", "BIGINT NOT NULL"),
    ("full_text", "TEXT NOT NULL"),
    ("actual_token_count", "INTEGER NOT NULL"),
    ("target_position_tokens", "INTEGER NOT NULL"),
    ("filler_ids_before", "TEXT NOT NULL"),
    ("filler_ids_after", "TEXT NOT NULL"),
    ("probe_hash", "TEXT NOT NULL"),
    ("content_hash", "TEXT NOT NULL"),
    ("generated_at", "TEXT NOT NULL"),
]
CALIBRATION_PROMPT_COLUMN_NAMES = [n for n, _ in _CALIBRATION_PROMPT_COLUMNS]

# Column definitions for calibration_baselines table.
_CALIBRATION_BASELINE_COLUMNS = [
    ("probe_id", "TEXT NOT NULL"),
    ("dimension", "TEXT NOT NULL"),
    ("model_id", "TEXT NOT NULL"),
    ("baseline_type", "TEXT NOT NULL"),  # "bare" or "anchored"
    ("score", "{float}"),
    ("score_method", "TEXT NOT NULL"),
    ("justification", "TEXT"),
    ("raw_response", "TEXT"),
    ("raw_test_response", "TEXT"),
    ("error", "TEXT"),
    ("timestamp", "TEXT NOT NULL"),
]
CALIBRATION_BASELINE_COLUMN_NAMES = [n for n, _ in _CALIBRATION_BASELINE_COLUMNS]


class DatabaseBackend(ABC):
    """Abstract database backend."""

    @property
    @abstractmethod
    def placeholder(self) -> str:
        """Parameter placeholder (``?`` for SQLite, ``%s`` for PostgreSQL)."""
        ...

    @property
    @abstractmethod
    def connection(self) -> Any:
        """Return the underlying DB-API 2.0 connection."""
        ...

    @abstractmethod
    def create_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        ...

    @abstractmethod
    def upsert_sql(self) -> str:
        """Return the full upsert SQL for probe_results."""
        ...

    def ensure_columns(self) -> None:
        """Add any missing columns to an existing table (safe schema migration)."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the connection."""
        ...


class SqliteBackend(DatabaseBackend):
    """SQLite backend with WAL mode."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")

    @property
    def placeholder(self) -> str:
        return "?"

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def create_schema(self) -> None:
        col_defs = ",\n    ".join(
            f"{name} {typedef.replace('{float}', 'REAL')}"
            for name, typedef in _COLUMNS
        )
        unique = ", ".join(UNIQUE_KEY)
        self._conn.execute(f"""
CREATE TABLE IF NOT EXISTS probe_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {col_defs},
    UNIQUE({unique})
)""")
        self._conn.commit()
        self.ensure_columns()
        self._conn.executescript("""
CREATE INDEX IF NOT EXISTS idx_model_probe
    ON probe_results(model_id, probe_id, target_position_percent, context_length);

CREATE INDEX IF NOT EXISTS idx_model_dimension
    ON probe_results(model_id, dimension, context_length);

CREATE INDEX IF NOT EXISTS idx_run_uuid
    ON probe_results(run_uuid);
""")
        self._conn.commit()

        # Server launches table
        launch_col_defs = ",\n    ".join(
            f"{name} {typedef}" for name, typedef in _LAUNCH_COLUMNS
        )
        self._conn.execute(f"""
CREATE TABLE IF NOT EXISTS server_launches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {launch_col_defs},
    UNIQUE(launch_id)
)""")
        self._conn.commit()
        self.ensure_launch_columns()
        self._conn.execute("""
CREATE INDEX IF NOT EXISTS idx_launch_timestamp
    ON server_launches(launched_at)""")
        self._conn.commit()

    def ensure_columns(self) -> None:
        cursor = self._conn.execute("PRAGMA table_info(probe_results)")
        existing = {row[1] for row in cursor.fetchall()}
        for name, typedef in _COLUMNS:
            if name not in existing:
                col_type = typedef.replace("{float}", "REAL")
                self._conn.execute(
                    f"ALTER TABLE probe_results ADD COLUMN {name} {col_type}"
                )
        self._conn.commit()

    def ensure_launch_columns(self) -> None:
        cursor = self._conn.execute("PRAGMA table_info(server_launches)")
        existing = {row[1] for row in cursor.fetchall()}
        for name, typedef in _LAUNCH_COLUMNS:
            if name not in existing:
                self._conn.execute(
                    f"ALTER TABLE server_launches ADD COLUMN {name} {typedef}"
                )
        self._conn.commit()

    def upsert_sql(self) -> str:
        cols = ", ".join(COLUMN_NAMES)
        phs = ", ".join("?" for _ in COLUMN_NAMES)
        return f"INSERT OR REPLACE INTO probe_results ({cols}) VALUES ({phs})"

    def close(self) -> None:
        self._conn.close()


class PostgresBackend(DatabaseBackend):
    """PostgreSQL backend using psycopg v3."""

    def __init__(self, dsn: str) -> None:
        import psycopg

        self._conn = psycopg.connect(dsn, autocommit=False)

    @property
    def placeholder(self) -> str:
        return "%s"

    @property
    def connection(self):
        return self._conn

    def create_schema(self) -> None:
        col_defs = ",\n    ".join(
            f"{name} {typedef.replace('{float}', 'DOUBLE PRECISION')}"
            for name, typedef in _COLUMNS
        )
        unique = ", ".join(UNIQUE_KEY)
        with self._conn.cursor() as cur:
            cur.execute(f"""
CREATE TABLE IF NOT EXISTS probe_results (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    {col_defs},
    UNIQUE({unique})
)""")
        self._conn.commit()
        self.ensure_columns()
        with self._conn.cursor() as cur:
            cur.execute("""
CREATE INDEX IF NOT EXISTS idx_model_probe
    ON probe_results(model_id, probe_id, target_position_percent, context_length)""")
            cur.execute("""
CREATE INDEX IF NOT EXISTS idx_model_dimension
    ON probe_results(model_id, dimension, context_length)""")
            cur.execute("""
CREATE INDEX IF NOT EXISTS idx_run_uuid
    ON probe_results(run_uuid)""")
        self._conn.commit()

        # Server launches table
        launch_col_defs = ",\n    ".join(
            f"{name} {typedef}" for name, typedef in _LAUNCH_COLUMNS
        )
        with self._conn.cursor() as cur:
            cur.execute(f"""
CREATE TABLE IF NOT EXISTS server_launches (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    {launch_col_defs},
    UNIQUE(launch_id)
)""")
        self._conn.commit()
        self.ensure_launch_columns()
        with self._conn.cursor() as cur:
            cur.execute("""
CREATE INDEX IF NOT EXISTS idx_launch_timestamp
    ON server_launches(launched_at)""")
        self._conn.commit()

    def ensure_columns(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'probe_results'"
            )
            existing = {row[0] for row in cur.fetchall()}
        for name, typedef in _COLUMNS:
            if name not in existing:
                col_type = typedef.replace("{float}", "DOUBLE PRECISION")
                with self._conn.cursor() as cur:
                    cur.execute(
                        f"ALTER TABLE probe_results ADD COLUMN {name} {col_type}"
                    )
        self._conn.commit()

    def ensure_launch_columns(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'server_launches'"
            )
            existing = {row[0] for row in cur.fetchall()}
        for name, typedef in _LAUNCH_COLUMNS:
            if name not in existing:
                with self._conn.cursor() as cur:
                    cur.execute(
                        f"ALTER TABLE server_launches ADD COLUMN {name} {typedef}"
                    )
        self._conn.commit()

    def upsert_sql(self) -> str:
        cols = ", ".join(COLUMN_NAMES)
        phs = ", ".join("%s" for _ in COLUMN_NAMES)
        unique = ", ".join(UNIQUE_KEY)
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in _DATA_COLUMNS)
        return (
            f"INSERT INTO probe_results ({cols}) VALUES ({phs}) "
            f"ON CONFLICT ({unique}) DO UPDATE SET {set_clause}"
        )

    def close(self) -> None:
        self._conn.close()


def create_calibration_schema(backend: DatabaseBackend) -> None:
    """Create calibration tables and indexes (SQLite or PostgreSQL)."""
    is_sqlite = isinstance(backend, SqliteBackend)
    float_type = "REAL" if is_sqlite else "DOUBLE PRECISION"
    id_col = "id INTEGER PRIMARY KEY AUTOINCREMENT" if is_sqlite else "id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY"

    # --- calibration_prompts ---
    prompt_col_defs = ",\n    ".join(
        f"{name} {typedef.replace('{float}', float_type)}"
        for name, typedef in _CALIBRATION_PROMPT_COLUMNS
    )
    prompt_ddl = f"""
CREATE TABLE IF NOT EXISTS calibration_prompts (
    {id_col},
    {prompt_col_defs},
    UNIQUE(probe_id, position_percent, context_length)
)"""

    # --- calibration_baselines ---
    baseline_col_defs = ",\n    ".join(
        f"{name} {typedef.replace('{float}', float_type)}"
        for name, typedef in _CALIBRATION_BASELINE_COLUMNS
    )
    baseline_ddl = f"""
CREATE TABLE IF NOT EXISTS calibration_baselines (
    {id_col},
    {baseline_col_defs},
    UNIQUE(probe_id, model_id, baseline_type)
)"""

    if is_sqlite:
        conn = backend.connection
        conn.execute(prompt_ddl)
        conn.execute(baseline_ddl)
        conn.executescript("""
CREATE INDEX IF NOT EXISTS idx_cal_prompt_probe
    ON calibration_prompts(probe_id);
CREATE INDEX IF NOT EXISTS idx_cal_prompt_ctx
    ON calibration_prompts(context_length);
CREATE INDEX IF NOT EXISTS idx_cal_baseline_model
    ON calibration_baselines(model_id);
CREATE INDEX IF NOT EXISTS idx_cal_baseline_probe
    ON calibration_baselines(probe_id);
""")
        conn.commit()
    else:
        conn = backend.connection
        with conn.cursor() as cur:
            cur.execute(prompt_ddl)
            cur.execute(baseline_ddl)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cal_prompt_probe ON calibration_prompts(probe_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cal_prompt_ctx ON calibration_prompts(context_length)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cal_baseline_model ON calibration_baselines(model_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cal_baseline_probe ON calibration_baselines(probe_id)")
        conn.commit()


def get_backend(dsn_or_path: str) -> DatabaseBackend:
    """Factory: return the appropriate backend based on the DSN prefix."""
    if dsn_or_path.startswith(("postgresql://", "postgres://")):
        return PostgresBackend(dsn_or_path)
    return SqliteBackend(dsn_or_path)
