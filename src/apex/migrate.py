"""One-time SQLite → PostgreSQL migration."""

from __future__ import annotations

import sqlite3

from apex.db import COLUMN_NAMES, UNIQUE_KEY, get_backend


def migrate(sqlite_path: str, pg_dsn: str) -> tuple[int, int, int]:
    """Migrate all probe_results from SQLite to PostgreSQL.

    Returns ``(rows_read, rows_inserted, rows_skipped)``.
    The insert uses ``ON CONFLICT DO NOTHING`` so the migration is idempotent.
    """
    # --- Source: SQLite ---
    src = sqlite3.connect(sqlite_path)
    cols_select = ", ".join(COLUMN_NAMES)
    rows = src.execute(
        f"SELECT {cols_select} FROM probe_results ORDER BY id"
    ).fetchall()
    src.close()

    # --- Destination: PostgreSQL ---
    backend = get_backend(pg_dsn)
    backend.create_schema()
    conn = backend.connection

    cols = ", ".join(COLUMN_NAMES)
    phs = ", ".join("%s" for _ in COLUMN_NAMES)
    unique = ", ".join(UNIQUE_KEY)
    insert_sql = (
        f"INSERT INTO probe_results ({cols}) VALUES ({phs}) "
        f"ON CONFLICT ({unique}) DO NOTHING"
    )

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM probe_results")
        count_before = cur.fetchone()[0]

    with conn.cursor() as cur:
        cur.executemany(insert_sql, rows)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM probe_results")
        count_after = cur.fetchone()[0]

    backend.close()

    rows_read = len(rows)
    rows_inserted = count_after - count_before
    return rows_read, rows_inserted, rows_read - rows_inserted
