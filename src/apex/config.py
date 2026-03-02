"""YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from apex.types import FillerType


@dataclass
class ModelConfig:
    name: str
    backend: str
    model_name: str
    tokenizer: str = "approximate"
    max_context_window: int = 4096
    architecture: str = "unknown"
    parameters: str = "unknown"
    quantization: str = "none"
    base_url: str | None = None
    api_key: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RunConfig:
    seed: int = 42
    temperature: float = 0.0
    repetitions: int = 1
    filler_type: str = "neutral"
    data_dir: str = "data"
    output_db: str = "results.db"
    database_url: str | None = None
    positions: list[float] = field(
        default_factory=lambda: [
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
        ]
    )
    context_lengths: list[int] = field(default_factory=lambda: [4096])
    workers: int = 1
    use_calibration: bool = False
    probe_select: str | list[str] = "all"
    models: list[ModelConfig] = field(default_factory=list)
    evaluator_models: list[ModelConfig] = field(default_factory=list)

    @property
    def filler_type_enum(self) -> FillerType:
        return FillerType(self.filler_type)

    @property
    def database_dsn(self) -> str:
        """Resolve database DSN: APEX_DATABASE_URL env > database_url > output_db."""
        import os

        return os.environ.get("APEX_DATABASE_URL") or self.database_url or self.output_db


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")

    run = data.get("run", {})
    data_sec = data.get("data", {})
    probes_sec = data.get("probes", {})
    database_sec = data.get("database", {})

    config = RunConfig(
        seed=run.get("seed", 42),
        temperature=run.get("temperature", 0.0),
        repetitions=run.get("repetitions", 1),
        filler_type=run.get("filler_type", "neutral"),
        data_dir=data_sec.get("directory", "data"),
        output_db=data_sec.get("output_db", "results.db"),
        database_url=database_sec.get("url"),
        positions=data.get("positions", [
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
        ]),
        context_lengths=data.get("context_lengths", [4096]),
        workers=run.get("workers", 1),
        use_calibration=run.get("use_calibration", False),
        probe_select=probes_sec.get("select", "all"),
    )

    for m in data.get("models", []):
        config.models.append(ModelConfig.from_dict(m))

    for m in data.get("evaluator_models", []):
        config.evaluator_models.append(ModelConfig.from_dict(m))

    validate_config(config)
    return config


def validate_config(config: RunConfig) -> list[str]:
    """Validate config, return list of warnings. Raises ValueError on errors."""
    errors = []
    warnings = []

    if not config.models:
        errors.append("No models configured")

    for m in config.models:
        if m.backend not in ("ollama", "llamacpp", "sglang", "openai", "anthropic", "google"):
            errors.append(f"Unknown backend '{m.backend}' for model '{m.name}'")

    if not config.positions:
        errors.append("No positions configured")
    else:
        for p in config.positions:
            if not 0.0 < p < 1.0:
                errors.append(f"Position {p} must be between 0 and 1 exclusive")

    if not config.context_lengths:
        errors.append("No context lengths configured")
    else:
        for cl in config.context_lengths:
            if cl < 512:
                warnings.append(f"Context length {cl} is very small")

    if config.repetitions < 1:
        errors.append("Repetitions must be >= 1")

    if config.workers < 1:
        errors.append("Workers must be >= 1")

    if config.temperature < 0.0:
        errors.append("Temperature must be >= 0")

    try:
        FillerType(config.filler_type)
    except ValueError:
        errors.append(f"Unknown filler type '{config.filler_type}'")

    if errors:
        raise ValueError("Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return warnings
