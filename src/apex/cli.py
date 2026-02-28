"""CLI interface: run, status, export, validate, migrate."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from apex import __version__


def _resolve_dsn(args_db: str) -> str:
    """Resolve database DSN: APEX_DATABASE_URL env var > CLI argument."""
    return os.environ.get("APEX_DATABASE_URL", args_db)


def cmd_run(args: argparse.Namespace) -> None:
    from apex.config import load_config
    from apex.libraries import ProbeLibrary
    from apex.runner import ProbeRunner
    from apex.storage import ResultStore

    config = load_config(args.config)
    library = ProbeLibrary(config.data_dir)
    dsn = config.database_dsn
    store = ResultStore(dsn)

    print(f"APEX v{__version__} — Starting probe run")
    print(f"  Models: {len(config.models)}")
    print(f"  Probes: {len(library.probes)}")
    print(f"  Positions: {config.positions}")
    print(f"  Context lengths: {config.context_lengths}")
    print(f"  Repetitions: {config.repetitions}")
    print(f"  Database: {dsn}")
    print()

    runner = ProbeRunner(config, library, store)
    try:
        runner.run()
    finally:
        store.close()

    print("\nRun complete.")


def cmd_status(args: argparse.Namespace) -> None:
    from apex.storage import ResultStore

    dsn = _resolve_dsn(args.db)
    store = ResultStore(dsn)
    total = store.count_results()
    print(f"Database: {dsn}")
    print(f"Total results: {total}")

    if args.model:
        model_count = store.count_results(model_id=args.model)
        print(f"Results for {args.model}: {model_count}")

    store.close()


def cmd_export(args: argparse.Namespace) -> None:
    from apex.storage import ResultStore

    dsn = _resolve_dsn(args.db)
    store = ResultStore(dsn)
    filters = {}
    if args.model:
        filters["model_id"] = args.model
    if args.dimension:
        filters["dimension"] = args.dimension

    count = store.export_json(args.output, **filters)
    store.close()

    print(f"Exported {count} results to {args.output}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    try:
        from apex.dashboard import create_app
    except ImportError:
        print(
            "Dashboard dependencies not installed.\n"
            "Install with: pip install apex[dashboard]",
            file=sys.stderr,
        )
        sys.exit(1)

    dsn = _resolve_dsn(args.db) if args.db else os.environ.get("APEX_DATABASE_URL", "")
    if not dsn:
        # Fall back to dashboard config file
        from apex.dashboard.config import DashboardConfig
        cfg = DashboardConfig.load("configs/dashboard.yaml")
        dsn = cfg.resolve_database_url()
    app = create_app(dsn)
    print(f"APEX Dashboard — http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=args.debug)


def cmd_rescore(args: argparse.Namespace) -> None:
    from apex.libraries import ProbeLibrary
    from apex.scoring.base import ScoringDispatcher
    from apex.storage import ResultStore

    dsn = _resolve_dsn(args.db)
    store = ResultStore(dsn)
    library = ProbeLibrary(args.data_dir)
    dispatcher = ScoringDispatcher(evaluator_adapter=None)

    # Fetch all results that use programmatic or exact_match scoring
    rows = store.query_results()
    rescoreable = [
        r for r in rows
        if r["score_method"] in ("programmatic", "exact_match")
    ]

    print(f"APEX rescore — {dsn}")
    print(f"  Total results: {len(rows)}")
    print(f"  Rescoreable (programmatic + exact_match): {len(rescoreable)}")
    print()

    changed = 0
    total_delta = 0.0

    for row in rescoreable:
        probe = library.probes.get(row["probe_id"])
        if not probe:
            continue
        query = library.get_query_for_probe(row["probe_id"])
        if not query:
            continue

        raw_response = row["raw_test_response"]
        old_score = row["score"]

        new_score, _eval_model, justification = dispatcher.score(probe, query, raw_response)

        if new_score is None:
            continue

        if old_score != new_score:
            delta = abs((new_score or 0.0) - (old_score or 0.0))
            total_delta += delta
            changed += 1

            if args.dry_run:
                print(f"  {row['probe_id']} pos={row['target_position_percent']:.0f}% "
                      f"ctx={row['context_length']}: {old_score} -> {new_score} ({justification})")
            else:
                store.update_score(row["id"], new_score, justification)

    mean_delta = total_delta / changed if changed else 0.0
    action = "would change" if args.dry_run else "changed"
    print(f"\nRescored: {len(rescoreable)}, {action}: {changed}, mean delta: {mean_delta:.4f}")

    store.close()


def cmd_migrate(args: argparse.Namespace) -> None:
    from apex.migrate import migrate

    pg_dsn = os.environ.get("APEX_DATABASE_URL")
    if not pg_dsn:
        print(
            "Error: APEX_DATABASE_URL environment variable not set.\n"
            "Set it to the target PostgreSQL DSN, e.g.:\n"
            "  export APEX_DATABASE_URL=\"postgresql://apexuser:pass@localhost:5432/apex\"",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Migrating: {args.sqlite_file} → PostgreSQL")
    rows_read, rows_inserted, rows_skipped = migrate(args.sqlite_file, pg_dsn)
    print(f"  Rows read:     {rows_read}")
    print(f"  Rows inserted: {rows_inserted}")
    print(f"  Rows skipped:  {rows_skipped}")


def cmd_validate(args: argparse.Namespace) -> None:
    from apex.config import load_config
    from apex.libraries import ProbeLibrary

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Config validation FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Config: {args.config} — OK")
    print(f"  Models: {len(config.models)}")
    for m in config.models:
        print(f"    - {m.name} ({m.backend}, {m.model_name})")
    print(f"  Positions: {config.positions}")
    print(f"  Context lengths: {config.context_lengths}")
    print(f"  Repetitions: {config.repetitions}")

    # Validate library
    try:
        library = ProbeLibrary(config.data_dir)
        print(f"  Library version: {library.library_version}")
        print(f"  Fillers loaded: {len(library.fillers)}")
        print(f"  Probes loaded: {len(library.probes)}")
        print(f"  Queries loaded: {len(library.queries)}")
    except Exception as e:
        print(f"  Library load warning: {e}")

    # Estimate work
    n_probes = len(library.probes) if library.probes else 0
    total = (
        n_probes
        * len(config.positions)
        * len(config.context_lengths)
        * config.repetitions
        * len(config.models)
    )
    print(f"  Estimated probe executions: {total}")

    print("\nValidation passed.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="apex",
        description="APEX — Attention Profiling and Empirical Cross-model Optimization",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Execute a probe run")
    p_run.add_argument("config", help="Path to YAML config file")
    p_run.set_defaults(func=cmd_run)

    # status
    p_status = subparsers.add_parser("status", help="Show run status")
    p_status.add_argument("db", help="Path to results database or PostgreSQL DSN")
    p_status.add_argument("--model", help="Filter by model ID")
    p_status.set_defaults(func=cmd_status)

    # export
    p_export = subparsers.add_parser("export", help="Export results to JSON")
    p_export.add_argument("db", help="Path to results database or PostgreSQL DSN")
    p_export.add_argument("-o", "--output", default="export.json", help="Output file")
    p_export.add_argument("--model", help="Filter by model ID")
    p_export.add_argument("--dimension", help="Filter by dimension")
    p_export.set_defaults(func=cmd_export)

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch interactive results dashboard")
    p_dash.add_argument("db", nargs="?", default="", help="Path to results database or PostgreSQL DSN (default: from config)")
    p_dash.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    p_dash.add_argument("--port", type=int, default=8050, help="Port (default: 8050)")
    p_dash.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    p_dash.set_defaults(func=cmd_dashboard)

    # rescore
    p_rescore = subparsers.add_parser("rescore", help="Rescore existing results with updated scoring logic")
    p_rescore.add_argument("db", help="Path to results database or PostgreSQL DSN")
    p_rescore.add_argument("--data-dir", default="data", help="Path to data directory (default: data)")
    p_rescore.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    p_rescore.set_defaults(func=cmd_rescore)

    # migrate
    p_migrate = subparsers.add_parser("migrate", help="Migrate SQLite results to PostgreSQL")
    p_migrate.add_argument("sqlite_file", help="Path to source SQLite database")
    p_migrate.set_defaults(func=cmd_migrate)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate config and libraries")
    p_validate.add_argument("config", help="Path to YAML config file")
    p_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args.func(args)
