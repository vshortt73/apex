"""CLI interface: run, status, export, validate, migrate."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

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
    if args.workers is not None:
        config.workers = args.workers
    if args.calibrated:
        config.use_calibration = True
    library = ProbeLibrary(config.data_dir)
    dsn = config.database_dsn
    store = ResultStore(dsn)

    mode = "calibrated (frozen prompts)" if config.use_calibration else "dynamic (assembled)"
    print(f"APEX v{__version__} — Starting probe run")
    print(f"  Mode: {mode}")
    print(f"  Models: {len(config.models)}")
    print(f"  Probes: {len(library.probes)}")
    print(f"  Positions: {config.positions}")
    print(f"  Context lengths: {config.context_lengths}")
    print(f"  Repetitions: {config.repetitions}")
    print(f"  Workers: {config.workers}")
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
        _config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "dashboard.yaml"
        cfg = DashboardConfig.load(_config_path)
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


def cmd_calibrate_generate(args: argparse.Namespace) -> None:
    from apex.calibration import CALIBRATION_CONTEXT_LENGTHS, CALIBRATION_POSITIONS, CalibrationGenerator
    from apex.calibration_store import CalibrationStore
    from apex.libraries import ProbeLibrary

    dsn = _resolve_dsn(args.db)
    library = ProbeLibrary(args.data_dir)
    store = CalibrationStore(dsn)

    positions = CALIBRATION_POSITIONS
    context_lengths = args.context_lengths or CALIBRATION_CONTEXT_LENGTHS

    if args.force:
        deleted = store.delete_prompts()
        if deleted:
            print(f"Deleted {deleted} existing prompts.")

    generator = CalibrationGenerator(library, tokenizer_spec=args.tokenizer)
    prompts = generator.generate(
        positions=positions,
        context_lengths=context_lengths,
        seed=args.seed,
    )
    store.write_prompts(prompts)

    print(f"APEX calibrate generate — {dsn}")
    print(f"  Probes: {len(library.probes)}")
    print(f"  Positions: {len(positions)}")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Generated: {len(prompts)} frozen prompts")
    print(f"  Stored: {store.count_prompts()} total")
    store.close()


def cmd_calibrate_validate(args: argparse.Namespace) -> None:
    from apex.calibration import CalibrationValidator
    from apex.calibration_store import CalibrationStore
    from apex.libraries import ProbeLibrary

    dsn = _resolve_dsn(args.db)
    library = ProbeLibrary(args.data_dir)
    store = CalibrationStore(dsn)

    prompts = store.get_prompts()
    if not prompts:
        print("No calibration prompts found in database.")
        store.close()
        sys.exit(1)

    validator = CalibrationValidator(library, tokenizer_spec=args.tokenizer)
    results = validator.validate(prompts)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print(f"APEX calibrate validate — {dsn}")
    print(f"  Total prompts: {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if args.verbose or failed > 0:
        for r in results:
            if not r.passed:
                print(f"  FAIL {r.probe_id} pos={r.position_percent:.0%} ctx={r.context_length}")
                for msg in r.messages:
                    print(f"    - {msg}")

    store.close()
    if failed > 0:
        sys.exit(1)


def cmd_calibrate_baseline(args: argparse.Namespace) -> None:
    from apex.calibration import BaselineRunner
    from apex.calibration_store import CalibrationStore
    from apex.config import load_config
    from apex.libraries import ProbeLibrary
    from apex.models.base import get_adapter
    from apex.scoring.base import ScoringDispatcher

    config = load_config(args.config)
    dsn = _resolve_dsn(args.db) if args.db else config.database_dsn
    library = ProbeLibrary(args.data_dir or config.data_dir)
    store = CalibrationStore(dsn)

    # Pick model — first from --model flag, else first in config
    model_cfg = None
    if args.model:
        for m in config.models:
            if m.name == args.model:
                model_cfg = m
                break
        if model_cfg is None:
            print(f"Error: model '{args.model}' not found in config.", file=sys.stderr)
            store.close()
            sys.exit(1)
    else:
        model_cfg = config.models[0]

    adapter = get_adapter(
        backend=model_cfg.backend,
        model_name=model_cfg.model_name,
        base_url=model_cfg.base_url,
        api_key=model_cfg.api_key,
        temperature=0.0,
        model_info_overrides={
            "model_id": model_cfg.name,
            "architecture": model_cfg.architecture,
            "parameters": model_cfg.parameters,
            "quantization": model_cfg.quantization,
            "max_context_window": model_cfg.max_context_window,
            "tokenizer": model_cfg.tokenizer,
        },
    )

    evaluator_adapter = None
    if config.evaluator_models:
        ev_cfg = config.evaluator_models[0]
        evaluator_adapter = get_adapter(
            backend=ev_cfg.backend,
            model_name=ev_cfg.model_name,
            base_url=ev_cfg.base_url,
            api_key=ev_cfg.api_key,
            temperature=0.0,
            model_info_overrides={"model_id": ev_cfg.name},
        )

    dispatcher = ScoringDispatcher(evaluator_adapter)
    baseline_type = args.type

    if args.force:
        deleted = store.delete_baselines(
            model_id=model_cfg.name, baseline_type=baseline_type,
        )
        if deleted:
            print(f"Deleted {deleted} existing {baseline_type} baselines for {model_cfg.name}.")

    probe_ids = args.probe_ids if args.probe_ids else None

    runner = BaselineRunner(library, dispatcher, adapter, store)
    baselines = runner.run_baselines(baseline_type=baseline_type, probe_ids=probe_ids)

    scored = sum(1 for b in baselines if b.score is not None)
    errors = sum(1 for b in baselines if b.error is not None)
    print(f"APEX calibrate baseline — {dsn}")
    print(f"  Model: {model_cfg.name}")
    print(f"  Type: {baseline_type}")
    print(f"  Baselines run: {len(baselines)}")
    print(f"  Scored: {scored}")
    if errors:
        print(f"  Errors: {errors}")
    print(f"  Total stored: {store.count_baselines(model_id=model_cfg.name)}")
    store.close()


def cmd_calibrate_export(args: argparse.Namespace) -> None:
    from apex.calibration_store import CalibrationStore

    dsn = _resolve_dsn(args.db)
    store = CalibrationStore(dsn)

    counts = store.export_json(
        output_path=args.output,
        model_id=getattr(args, "model", None),
        dimension=getattr(args, "dimension", None),
        baseline_type=getattr(args, "type", None),
    )
    store.close()

    print(f"Exported calibration data to {args.output}")
    print(f"  Prompts:   {counts['prompts']}")
    print(f"  Baselines: {counts['baselines']}")


def cmd_calibrate_import(args: argparse.Namespace) -> None:
    from apex.calibration_store import CalibrationStore

    dsn = _resolve_dsn(args.db)

    try:
        store = CalibrationStore(dsn)
        counts = store.import_json(args.input)
        store.close()
    except FileNotFoundError:
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Imported calibration data from {args.input}")
    print(f"  Prompts:   {counts['prompts']}")
    print(f"  Baselines: {counts['baselines']}")


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
    p_run.add_argument("--workers", type=int, default=None, help="Number of parallel workers (overrides config)")
    p_run.add_argument("--calibrated", action="store_true", help="Use frozen calibration prompts instead of assembling fresh")
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

    # calibrate (nested subcommands)
    p_cal = subparsers.add_parser("calibrate", help="Calibration subsystem: generate, validate, baseline, export, import")
    cal_sub = p_cal.add_subparsers(dest="cal_command", required=True)

    # calibrate generate
    p_cal_gen = cal_sub.add_parser("generate", help="Generate frozen calibration prompts")
    p_cal_gen.add_argument("--db", default="results.db", help="Database path or PostgreSQL DSN")
    p_cal_gen.add_argument("--data-dir", default="data", help="Path to data directory")
    p_cal_gen.add_argument("--tokenizer", default="approximate", help="Tokenizer spec")
    p_cal_gen.add_argument("--seed", type=int, default=42, help="RNG seed")
    p_cal_gen.add_argument("--context-lengths", type=int, nargs="+", default=None, help="Context lengths")
    p_cal_gen.add_argument("--force", action="store_true", help="Delete existing prompts before generating")
    p_cal_gen.set_defaults(func=cmd_calibrate_generate)

    # calibrate validate
    p_cal_val = cal_sub.add_parser("validate", help="Validate frozen calibration prompts")
    p_cal_val.add_argument("--db", default="results.db", help="Database path or PostgreSQL DSN")
    p_cal_val.add_argument("--data-dir", default="data", help="Path to data directory")
    p_cal_val.add_argument("--tokenizer", default="approximate", help="Tokenizer spec")
    p_cal_val.add_argument("--verbose", action="store_true", help="Show detailed results")
    p_cal_val.set_defaults(func=cmd_calibrate_validate)

    # calibrate baseline
    p_cal_base = cal_sub.add_parser("baseline", help="Run calibration baselines")
    p_cal_base.add_argument("config", help="Path to YAML config file")
    p_cal_base.add_argument("--db", default="", help="Database path or PostgreSQL DSN (default: from config)")
    p_cal_base.add_argument("--data-dir", default="", help="Path to data directory (default: from config)")
    p_cal_base.add_argument("--model", help="Model name from config")
    p_cal_base.add_argument("--type", default="bare", choices=["bare", "anchored"], help="Baseline type")
    p_cal_base.add_argument("--probe-ids", nargs="+", help="Specific probe IDs to baseline")
    p_cal_base.add_argument("--force", action="store_true", help="Delete existing baselines before running")
    p_cal_base.set_defaults(func=cmd_calibrate_baseline)

    # calibrate export
    p_cal_exp = cal_sub.add_parser("export", help="Export calibration data to JSON")
    p_cal_exp.add_argument("--db", default="results.db", help="Database path or PostgreSQL DSN")
    p_cal_exp.add_argument("-o", "--output", default="calibration.json", help="Output file (default: calibration.json)")
    p_cal_exp.add_argument("--model", help="Filter baselines by model ID")
    p_cal_exp.add_argument("--dimension", help="Filter by dimension")
    p_cal_exp.add_argument("--type", choices=["bare", "anchored"], help="Filter baselines by type")
    p_cal_exp.set_defaults(func=cmd_calibrate_export)

    # calibrate import
    p_cal_imp = cal_sub.add_parser("import", help="Import calibration data from JSON")
    p_cal_imp.add_argument("input", help="Path to calibration JSON file")
    p_cal_imp.add_argument("--db", default="results.db", help="Target database path or PostgreSQL DSN")
    p_cal_imp.set_defaults(func=cmd_calibrate_import)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate config and libraries")
    p_validate.add_argument("config", help="Path to YAML config file")
    p_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args.func(args)
