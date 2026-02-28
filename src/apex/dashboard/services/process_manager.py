"""Manage APEX run processes — start, stop, and track probe runs."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml

# Project root derived from package location: src/apex/dashboard/services/ → project root
_PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# Fallback default base_urls per backend (used if no config provided)
_DEFAULT_BACKEND_URLS = {
    "llamacpp": "http://localhost:8080",
    "ollama": "http://localhost:11434",
    "sglang": "http://localhost:30000",
}

# Cloud backends that need API keys instead of a running server
_CLOUD_BACKENDS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


@dataclass
class PreflightResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class RunInfo:
    run_id: str
    pid: int
    config_path: str
    config_summary: str
    start_time: datetime
    status: Literal["running", "stopped", "finished", "crashed"] = "running"
    error_output: str = ""


class ProcessManager:
    """Singleton-style manager for APEX run subprocesses."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        backend_defaults: dict[str, str] | None = None,
    ) -> None:
        self._project_root = Path(project_root).resolve() if project_root else _PACKAGE_PROJECT_ROOT
        self._configs_dir = self._project_root / "configs"
        self._configs_dir.mkdir(exist_ok=True)
        self._venv_python = self._project_root / ".venv" / "bin" / "python"
        self._backend_defaults = backend_defaults or dict(_DEFAULT_BACKEND_URLS)
        self._runs: dict[str, RunInfo] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._stderr_files: dict[str, str] = {}
        self._rediscover()

    def _rediscover(self) -> None:
        """On startup, find any running `apex run` processes."""
        try:
            import psutil
        except ImportError:
            return

        for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
            try:
                cmdline = proc.info["cmdline"] or []
                if "-m" in cmdline and "apex" in cmdline and "run" in cmdline:
                    pid = proc.info["pid"]
                    config_path = ""
                    try:
                        run_idx = cmdline.index("run")
                        if run_idx + 1 < len(cmdline):
                            config_path = cmdline[run_idx + 1]
                    except (ValueError, IndexError):
                        pass

                    run_id = f"discovered-{pid}"
                    summary = _summarize_config(config_path) if config_path else "Discovered run"
                    self._runs[run_id] = RunInfo(
                        run_id=run_id,
                        pid=pid,
                        config_path=config_path,
                        config_summary=summary,
                        start_time=datetime.fromtimestamp(proc.info["create_time"]),
                        status="running",
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def preflight_check(self, config_dict: dict) -> PreflightResult:
        """Validate everything needed for a successful run before spawning."""
        errors = []
        warnings = []

        # --- Models ---
        models = config_dict.get("models", [])
        if not models:
            errors.append("No models configured")

        for m in models:
            name = m.get("name", m.get("model_name", "?"))
            backend = m.get("backend", "")

            if backend in _CLOUD_BACKENDS:
                # Check API key
                env_var = _CLOUD_BACKENDS[backend]
                api_key = m.get("api_key") or os.environ.get(env_var)
                if not api_key:
                    errors.append(
                        f"Model '{name}' ({backend}): "
                        f"No API key — set ${env_var} or add api_key to model config"
                    )
            elif backend in self._backend_defaults:
                # Check server health
                base_url = m.get("base_url") or self._backend_defaults[backend]
                status = _check_server_health(base_url, backend)
                if status != "ok":
                    errors.append(
                        f"Model '{name}' ({backend}): "
                        f"Server not reachable at {base_url} — {status}"
                    )

        # --- Evaluator models ---
        eval_models = config_dict.get("evaluator_models", [])
        for m in eval_models:
            name = m.get("name", m.get("model_name", "?"))
            backend = m.get("backend", "")
            if backend in _CLOUD_BACKENDS:
                env_var = _CLOUD_BACKENDS[backend]
                api_key = m.get("api_key") or os.environ.get(env_var)
                if not api_key:
                    errors.append(
                        f"Evaluator '{name}' ({backend}): No API key — set ${env_var}"
                    )
            elif backend in self._backend_defaults:
                base_url = m.get("base_url") or self._backend_defaults[backend]
                status = _check_server_health(base_url, backend)
                if status != "ok":
                    errors.append(
                        f"Evaluator '{name}' ({backend}): "
                        f"Server not reachable at {base_url} — {status}"
                    )

        # --- Data files ---
        data_dir = Path(self._project_root / config_dict.get("data", {}).get("directory", "data"))
        filler_dir = data_dir / "filler"
        probes_dir = data_dir / "probes"

        if not filler_dir.is_dir() or not any(filler_dir.glob("*.json")):
            errors.append(f"No filler data found in {filler_dir}")
        if not probes_dir.is_dir() or not any(probes_dir.glob("*.json")):
            errors.append(f"No probe data found in {probes_dir}")

        # --- Database ---
        db_url = config_dict.get("database", {}).get("url")
        if db_url and db_url.startswith(("postgresql://", "postgres://")):
            db_status = _check_postgres(db_url)
            if db_status != "ok":
                errors.append(f"Database not reachable: {db_status}")

        # --- Positions and context ---
        positions = config_dict.get("positions", [])
        if not positions:
            errors.append("No positions configured")

        ctx_lengths = config_dict.get("context_lengths", [])
        if not ctx_lengths:
            errors.append("No context lengths configured")

        # --- Context vs model max ---
        for m in models:
            max_ctx = m.get("max_context_window", 4096)
            for cl in ctx_lengths:
                if cl > max_ctx:
                    warnings.append(
                        f"Context length {cl} exceeds max_context_window {max_ctx} "
                        f"for model '{m.get('name', '?')}'"
                    )

        return PreflightResult(
            ok=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def start_run(self, config_dict: dict) -> tuple[str, RunInfo]:
        """Write config YAML, spawn subprocess, verify it survives startup.

        Returns (run_id, RunInfo). Check info.status — may be 'crashed' if
        the process died during the startup grace period.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        config_path = self._configs_dir / f"{run_id}.yaml"
        config_path.write_text(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))

        python = str(self._venv_python) if self._venv_python.exists() else "python"

        # Capture stderr to a temp file so we can report why it died
        stderr_path = self._configs_dir / f"{run_id}.stderr"
        stderr_file = open(stderr_path, "w")

        proc = subprocess.Popen(
            [python, "-m", "apex", "run", str(config_path)],
            cwd=str(self._project_root),
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
            start_new_session=True,
        )

        summary = _summarize_config_dict(config_dict)
        info = RunInfo(
            run_id=run_id,
            pid=proc.pid,
            config_path=str(config_path),
            config_summary=summary,
            start_time=datetime.now(),
            status="running",
        )
        self._runs[run_id] = info
        self._processes[run_id] = proc
        self._stderr_files[run_id] = str(stderr_path)

        # Grace period: wait briefly and check if process is still alive
        time.sleep(2)
        exit_code = proc.poll()
        if exit_code is not None:
            # Process already died
            stderr_file.close()
            error_text = ""
            try:
                error_text = Path(stderr_path).read_text().strip()
            except OSError:
                pass
            info.status = "crashed"
            info.error_output = error_text
        else:
            stderr_file.close()

        return run_id, info

    def stop_run(self, run_id: str) -> bool:
        """Send SIGTERM to a run. Returns True if the signal was sent."""
        info = self._runs.get(run_id)
        if not info or info.status not in ("running",):
            return False

        try:
            os.kill(info.pid, signal.SIGTERM)
        except ProcessLookupError:
            info.status = "finished"
            return False

        # Wait up to 5s then SIGKILL
        proc = self._processes.get(run_id)
        if proc:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.kill(info.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        info.status = "stopped"
        return True

    def get_runs(self) -> list[RunInfo]:
        """Return all tracked runs, refreshing status of running ones."""
        for run_id, info in self._runs.items():
            if info.status == "running":
                if not _pid_alive(info.pid):
                    # Process died — read stderr for clues
                    stderr_path = self._stderr_files.get(run_id)
                    error_text = ""
                    if stderr_path:
                        try:
                            error_text = Path(stderr_path).read_text().strip()
                        except OSError:
                            pass
                    info.status = "crashed" if error_text else "finished"
                    info.error_output = error_text
        return sorted(self._runs.values(), key=lambda r: r.start_time, reverse=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _check_server_health(base_url: str, backend: str) -> str:
    """Quick connectivity check for a model server. Returns 'ok' or error description."""
    import httpx

    if backend == "llamacpp":
        url = f"{base_url}/health"
    elif backend == "ollama":
        url = f"{base_url}/api/tags"
    elif backend == "sglang":
        url = f"{base_url}/health"
    else:
        url = f"{base_url}/health"

    try:
        resp = httpx.get(url, timeout=5)
        if resp.status_code < 500:
            return "ok"
        return f"Server returned HTTP {resp.status_code}"
    except httpx.ConnectError:
        return "Connection refused — is the server running?"
    except httpx.ConnectTimeout:
        return "Connection timed out"
    except Exception as e:
        return str(e)


def _check_postgres(dsn: str) -> str:
    """Quick PostgreSQL connectivity check."""
    try:
        import psycopg
        conn = psycopg.connect(dsn, connect_timeout=5)
        conn.close()
        return "ok"
    except Exception as e:
        return str(e)


def _summarize_config(config_path: str) -> str:
    """Read a YAML config and produce a one-line summary."""
    try:
        data = yaml.safe_load(Path(config_path).read_text())
        return _summarize_config_dict(data)
    except Exception:
        return config_path


def _summarize_config_dict(data: dict) -> str:
    models = data.get("models", [])
    model_names = [m.get("name", m.get("model_name", "?")) for m in models]
    n_ctx = len(data.get("context_lengths", []))
    n_pos = len(data.get("positions", []))
    return f"{', '.join(model_names[:3])} | {n_pos} pos | {n_ctx} ctx"
