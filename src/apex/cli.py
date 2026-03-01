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

    # Build evaluator adapter if args provided
    evaluator_adapter = None
    if getattr(args, "evaluator_backend", None) and getattr(args, "evaluator_model", None):
        from apex.models.base import get_adapter
        evaluator_adapter = get_adapter(
            backend=args.evaluator_backend,
            model_name=args.evaluator_model,
            base_url=getattr(args, "evaluator_url", None),
        )

    dispatcher = ScoringDispatcher(evaluator_adapter=evaluator_adapter)

    # Determine which score methods to rescore
    score_method_filter = getattr(args, "score_method", "all")
    if score_method_filter == "all":
        allowed_methods = {"programmatic", "exact_match", "evaluator"}
    elif score_method_filter == "evaluator":
        allowed_methods = {"evaluator"}
    elif score_method_filter == "programmatic":
        allowed_methods = {"programmatic"}
    elif score_method_filter == "exact_match":
        allowed_methods = {"exact_match"}
    else:
        allowed_methods = {"programmatic", "exact_match", "evaluator"}

    # Without an evaluator adapter, skip evaluator method
    if evaluator_adapter is None:
        allowed_methods.discard("evaluator")

    # Fetch results with optional model filter
    model_filter = getattr(args, "model", None)
    rows = store.query_results(model_id=model_filter)

    # Filter by score method
    rescoreable = [r for r in rows if r["score_method"] in allowed_methods]

    # Filter null-only if requested
    null_only = getattr(args, "null_only", False)
    if null_only:
        rescoreable = [r for r in rescoreable if r["score"] is None]

    print(f"APEX rescore — {dsn}")
    print(f"  Total results: {len(rows)}")
    print(f"  Rescoreable: {len(rescoreable)}")
    if model_filter:
        print(f"  Model filter: {model_filter}")
    print(f"  Methods: {', '.join(sorted(allowed_methods))}")
    if null_only:
        print(f"  NULL-only: yes")
    if evaluator_adapter:
        print(f"  Evaluator: {args.evaluator_backend}/{args.evaluator_model}")
    print()

    changed = 0
    total_delta = 0.0
    counts_by_method: dict[str, int] = {}

    for row in rescoreable:
        probe = library.probes.get(row["probe_id"])
        if not probe:
            continue
        query = library.get_query_for_probe(row["probe_id"])
        if not query:
            continue

        raw_response = row["raw_test_response"]
        old_score = row["score"]

        new_score, eval_model_id, justification = dispatcher.score(probe, query, raw_response)

        if new_score is None:
            continue

        if old_score != new_score:
            delta = abs((new_score or 0.0) - (old_score or 0.0))
            total_delta += delta
            changed += 1
            method = row["score_method"]
            counts_by_method[method] = counts_by_method.get(method, 0) + 1

            if args.dry_run:
                print(f"  [{method}] {row['probe_id']} pos={row['target_position_percent']:.0f}% "
                      f"ctx={row['context_length']}: {old_score} -> {new_score} ({justification})")
            else:
                store.update_score(
                    row["id"], new_score, justification,
                    evaluator_model_id=eval_model_id if row["score_method"] == "evaluator" else None,
                )

    mean_delta = total_delta / changed if changed else 0.0
    action = "would change" if args.dry_run else "changed"
    print(f"\nRescored: {len(rescoreable)}, {action}: {changed}, mean delta: {mean_delta:.4f}")
    if counts_by_method:
        for method, count in sorted(counts_by_method.items()):
            print(f"  {method}: {count}")

    store.close()


def cmd_delete(args: argparse.Namespace) -> None:
    from apex.storage import ResultStore

    dsn = _resolve_dsn(args.db)
    store = ResultStore(dsn)

    # --list-runs: show available run UUIDs and exit
    if args.list_runs:
        runs = store.get_run_uuids()
        if not runs:
            print("No run UUIDs found in database.")
        else:
            print(f"{'Run UUID':<38} {'Model':<40} {'Count':>6}  {'First':>24}  {'Last':>24}")
            print("-" * 140)
            for r in runs:
                print(
                    f"{r['run_uuid']:<38} {r['model_id']:<40} {r['count']:>6}  "
                    f"{r['first_ts']:>24}  {r['last_ts']:>24}"
                )
        store.close()
        return

    # Determine deletion mode
    if args.run_uuid:
        label = f"run_uuid={args.run_uuid}"
        count_sql_fn = lambda: len(store.query_results())  # noqa: E731
        # Preview count via a query
        cursor = store._conn.execute(
            f"SELECT COUNT(*) FROM probe_results WHERE run_uuid = {store._ph}",
            (args.run_uuid,),
        )
        count = cursor.fetchone()[0]
        delete_fn = lambda: store.delete_by_run_uuid(args.run_uuid)  # noqa: E731
    elif args.model:
        filters: dict = {"model_id": args.model}
        if args.dimension:
            filters["dimension"] = args.dimension
        if args.probe_id:
            filters["probe_id"] = args.probe_id
        if args.context_length:
            filters["context_length"] = args.context_length

        parts = [f"model={args.model}"]
        if args.dimension:
            parts.append(f"dimension={args.dimension}")
        if args.probe_id:
            parts.append(f"probe_id={args.probe_id}")
        if args.context_length:
            parts.append(f"context_length={args.context_length}")
        label = ", ".join(parts)

        # Preview count
        conditions = []
        params: list = []
        for key, val in filters.items():
            conditions.append(f"{key} = {store._ph}")
            params.append(val)
        where = " WHERE " + " AND ".join(conditions)
        cursor = store._conn.execute(
            f"SELECT COUNT(*) FROM probe_results{where}", params
        )
        count = cursor.fetchone()[0]

        if len(filters) == 1 and "model_id" in filters:
            delete_fn = lambda: store.delete_by_model(args.model)  # noqa: E731
        else:
            delete_fn = lambda: store.delete_by_filters(**filters)  # noqa: E731
    else:
        print("Error: specify --run-uuid or --model to select results to delete.", file=sys.stderr)
        store.close()
        sys.exit(1)

    if count == 0:
        print(f"No results found matching {label}.")
        store.close()
        return

    if args.dry_run:
        print(f"Dry run: {count} result(s) would be deleted ({label}).")
        store.close()
        return

    # Confirm
    if not args.yes:
        answer = input(f"Delete {count} result(s) matching {label}? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            store.close()
            return

    deleted = delete_fn()
    print(f"Deleted {deleted} result(s) ({label}).")
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
    p_rescore.add_argument("--evaluator-backend", help="Evaluator backend (e.g. llamacpp, openai)")
    p_rescore.add_argument("--evaluator-model", help="Evaluator model name (e.g. Qwen_Qwen3-30B-A3B-Q4_K_M)")
    p_rescore.add_argument("--evaluator-url", help="Evaluator base URL (e.g. http://node2:8080)")
    p_rescore.add_argument("--model", help="Filter by model_id")
    p_rescore.add_argument(
        "--score-method", default="all",
        choices=["all", "evaluator", "programmatic", "exact_match"],
        help="Which score methods to rescore (default: all)",
    )
    p_rescore.add_argument("--null-only", action="store_true", help="Only rescore results where score IS NULL")
    p_rescore.set_defaults(func=cmd_rescore)

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete results from database")
    p_delete.add_argument("db", help="Path to results database or PostgreSQL DSN")
    p_delete.add_argument("--run-uuid", help="Delete all results from a specific run UUID")
    p_delete.add_argument("--model", help="Delete all results for a model")
    p_delete.add_argument("--dimension", help="Narrow deletion to a dimension (use with --model)")
    p_delete.add_argument("--probe-id", help="Narrow deletion to a specific probe")
    p_delete.add_argument("--context-length", type=int, help="Narrow deletion to a specific context length")
    p_delete.add_argument("--dry-run", action="store_true", help="Preview count without deleting")
    p_delete.add_argument("--list-runs", action="store_true", help="List distinct run UUIDs with summary info")
    p_delete.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    p_delete.set_defaults(func=cmd_delete)

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
