"""Dashboard configuration — YAML-persisted settings with auto-detection."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# --- Dataclass hierarchy ---

@dataclass
class InfraConfig:
    llama_server_bin: str = ""
    models_dir: str = ""


@dataclass
class NodeConfig:
    name: str = "node1"
    host: str = "local"
    label: str = "Node 1 (local)"
    enabled: bool = True


@dataclass
class DatabaseConfig:
    url: str = "postgresql://apex:apex@localhost:5432/apex"


@dataclass
class BackendDefaults:
    llamacpp: str = "http://localhost:8080"
    ollama: str = "http://localhost:11434"
    sglang: str = "http://localhost:30000"

    def as_dict(self) -> dict[str, str]:
        return {
            "llamacpp": self.llamacpp,
            "ollama": self.ollama,
            "sglang": self.sglang,
        }


@dataclass
class ServerDefaults:
    port: int = 8080
    ctx_size: int = 8192
    gpu_layers: int = 999
    parallel: int = 1
    flash_attn: bool = True
    threads: int = 0
    reasoning_format: str = "none"
    reasoning_budget: int = -1


@dataclass
class RunDefaults:
    seed: int = 42
    temperature: float = 0.0
    repetitions: int = 1
    filler_type: str = "neutral"
    max_tokens: int = 512


@dataclass
class DashboardConfig:
    infra: InfraConfig = field(default_factory=InfraConfig)
    nodes: list[NodeConfig] = field(default_factory=lambda: [
        NodeConfig(name="node1", host="local", label="Node 1 (local)", enabled=True),
    ])
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    backend_defaults: BackendDefaults = field(default_factory=BackendDefaults)
    server_defaults: ServerDefaults = field(default_factory=ServerDefaults)
    run_defaults: RunDefaults = field(default_factory=RunDefaults)

    def resolve_database_url(self) -> str:
        """Resolve database URL: env var > config file > built-in default."""
        env_url = os.environ.get("APEX_DATABASE_URL")
        if env_url:
            return env_url
        return self.database.url

    @staticmethod
    def load(path: str | Path) -> DashboardConfig:
        """Load config from YAML file. Returns defaults if file doesn't exist."""
        p = Path(path)
        if not p.is_file():
            return DashboardConfig()

        try:
            data = yaml.safe_load(p.read_text()) or {}
        except Exception:
            return DashboardConfig()

        return _from_dict(data)

    def save(self, path: str | Path) -> None:
        """Write config to YAML file, creating parent directories if needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = _to_dict(self)
        p.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


# --- Serialization helpers ---

def _from_dict(data: dict) -> DashboardConfig:
    """Build DashboardConfig from a raw YAML dict."""
    infra_data = data.get("infra", {})
    infra = InfraConfig(
        llama_server_bin=infra_data.get("llama_server_bin", ""),
        models_dir=infra_data.get("models_dir", ""),
    )

    nodes = []
    for nd in data.get("nodes", []):
        nodes.append(NodeConfig(
            name=nd.get("name", "node1"),
            host=nd.get("host", "local"),
            label=nd.get("label", ""),
            enabled=nd.get("enabled", True),
        ))
    if not nodes:
        nodes = [NodeConfig(name="node1", host="local", label="Node 1 (local)", enabled=True)]

    db_data = data.get("database", {})
    database = DatabaseConfig(url=db_data.get("url", "postgresql://apex:apex@localhost:5432/apex"))

    bd = data.get("backend_defaults", {})
    backend_defaults = BackendDefaults(
        llamacpp=bd.get("llamacpp", "http://localhost:8080"),
        ollama=bd.get("ollama", "http://localhost:11434"),
        sglang=bd.get("sglang", "http://localhost:30000"),
    )

    sd = data.get("server_defaults", {})
    server_defaults = ServerDefaults(
        port=sd.get("port", 8080),
        ctx_size=sd.get("ctx_size", 8192),
        gpu_layers=sd.get("gpu_layers", 999),
        parallel=sd.get("parallel", 1),
        flash_attn=sd.get("flash_attn", True),
        threads=sd.get("threads", 0),
        reasoning_format=sd.get("reasoning_format", "none"),
        reasoning_budget=sd.get("reasoning_budget", -1),
    )

    rd = data.get("run_defaults", {})
    run_defaults = RunDefaults(
        seed=rd.get("seed", 42),
        temperature=rd.get("temperature", 0.0),
        repetitions=rd.get("repetitions", 1),
        filler_type=rd.get("filler_type", "neutral"),
        max_tokens=rd.get("max_tokens", 512),
    )

    return DashboardConfig(
        infra=infra,
        nodes=nodes,
        database=database,
        backend_defaults=backend_defaults,
        server_defaults=server_defaults,
        run_defaults=run_defaults,
    )


def _to_dict(cfg: DashboardConfig) -> dict:
    """Convert DashboardConfig to a plain dict for YAML serialization."""
    return {
        "infra": {
            "llama_server_bin": cfg.infra.llama_server_bin,
            "models_dir": cfg.infra.models_dir,
        },
        "nodes": [
            {
                "name": n.name,
                "host": n.host,
                "label": n.label,
                "enabled": n.enabled,
            }
            for n in cfg.nodes
        ],
        "database": {
            "url": cfg.database.url,
        },
        "backend_defaults": {
            "llamacpp": cfg.backend_defaults.llamacpp,
            "ollama": cfg.backend_defaults.ollama,
            "sglang": cfg.backend_defaults.sglang,
        },
        "server_defaults": {
            "port": cfg.server_defaults.port,
            "ctx_size": cfg.server_defaults.ctx_size,
            "gpu_layers": cfg.server_defaults.gpu_layers,
            "parallel": cfg.server_defaults.parallel,
            "flash_attn": cfg.server_defaults.flash_attn,
            "threads": cfg.server_defaults.threads,
            "reasoning_format": cfg.server_defaults.reasoning_format,
            "reasoning_budget": cfg.server_defaults.reasoning_budget,
        },
        "run_defaults": {
            "seed": cfg.run_defaults.seed,
            "temperature": cfg.run_defaults.temperature,
            "repetitions": cfg.run_defaults.repetitions,
            "filler_type": cfg.run_defaults.filler_type,
            "max_tokens": cfg.run_defaults.max_tokens,
        },
    }


# --- Auto-detection (on-demand, never at startup) ---

@dataclass
class DetectedValue:
    field: str
    value: str
    source: str


def auto_detect() -> list[DetectedValue]:
    """Run all detection heuristics. Returns list of detected values."""
    results: list[DetectedValue] = []

    llama = _detect_llama_server()
    if llama:
        results.append(DetectedValue("infra.llama_server_bin", llama[0], llama[1]))

    models = _detect_models_dir()
    if models:
        results.append(DetectedValue("infra.models_dir", models[0], models[1]))

    db = _detect_database()
    if db:
        results.append(DetectedValue("database.url", db[0], db[1]))

    gpu = _detect_gpu()
    if gpu:
        results.append(DetectedValue("gpu.name", gpu[0], gpu[1]))

    return results


def _detect_llama_server() -> tuple[str, str] | None:
    """Search common paths for llama-server binary."""
    candidates = [
        "/programs/llama.cpp/build/bin/llama-server",
        str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"),
        "/usr/local/bin/llama-server",
    ]

    for path in candidates:
        if Path(path).is_file() and os.access(path, os.X_OK):
            return path, f"Found at {path}"

    which_result = shutil.which("llama-server")
    if which_result:
        return which_result, f"Found via PATH: {which_result}"

    return None


def _detect_models_dir() -> tuple[str, str] | None:
    """Search common paths for directories containing .gguf files."""
    candidates = [
        "/models/llm_models/",
        "/models/",
        str(Path.home() / "models"),
    ]

    for path in candidates:
        d = Path(path)
        if d.is_dir():
            gguf_count = sum(1 for _ in d.rglob("*.gguf"))
            if gguf_count > 0:
                return path, f"Found {gguf_count} .gguf file(s) in {path}"

    return None


def _detect_database() -> tuple[str, str] | None:
    """Try connecting to default PostgreSQL."""
    url = "postgresql://apex:apex@localhost:5432/apex"
    try:
        import psycopg
        conn = psycopg.connect(url, connect_timeout=3)
        conn.close()
        return url, "Connected to default PostgreSQL (apex@localhost:5432/apex)"
    except Exception:
        return None


def _detect_gpu() -> tuple[str, str] | None:
    """Query nvidia-smi for GPU name."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            name = result.stdout.strip().splitlines()[0].strip()
            return name, f"Detected via nvidia-smi"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None
