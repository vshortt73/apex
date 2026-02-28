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
]

COLUMN_NAMES = [name for name, _ in _COLUMNS]

UNIQUE_KEY = ("model_id", "probe_id", "target_position_percent", "context_length", "run_number")

# Non-key columns — updated on upsert conflict.
_DATA_COLUMNS = [name for name in COLUMN_NAMES if name not in UNIQUE_KEY]


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
        self._conn.executescript(f"""
CREATE TABLE IF NOT EXISTS probe_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {col_defs},
    UNIQUE({unique})
);

CREATE INDEX IF NOT EXISTS idx_model_probe
    ON probe_results(model_id, probe_id, target_position_percent, context_length);

CREATE INDEX IF NOT EXISTS idx_model_dimension
    ON probe_results(model_id, dimension, context_length);
""")
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
            cur.execute("""
CREATE INDEX IF NOT EXISTS idx_model_probe
    ON probe_results(model_id, probe_id, target_position_percent, context_length)""")
            cur.execute("""
CREATE INDEX IF NOT EXISTS idx_model_dimension
    ON probe_results(model_id, dimension, context_length)""")
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


def get_backend(dsn_or_path: str) -> DatabaseBackend:
    """Factory: return the appropriate backend based on the DSN prefix."""
    if dsn_or_path.startswith(("postgresql://", "postgres://")):
        return PostgresBackend(dsn_or_path)
    return SqliteBackend(dsn_or_path)
