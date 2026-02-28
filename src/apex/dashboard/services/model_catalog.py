"""Scan filesystem for .gguf model files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelFile:
    path: str
    filename: str
    size_gb: float
    parent_dir: str  # architecture hint from directory name


_cache: list[ModelFile] | None = None
_cache_base: str | None = None


def scan_models(base_dir: str = "") -> list[ModelFile]:
    """Walk directory tree for .gguf files. Cached after first call per base_dir."""
    global _cache, _cache_base
    if _cache is not None and _cache_base == base_dir:
        return _cache

    root = Path(base_dir)
    if not root.is_dir():
        return []

    results = []
    for p in sorted(root.rglob("*.gguf")):
        try:
            size_gb = round(p.stat().st_size / (1024**3), 2)
        except OSError:
            size_gb = 0.0
        results.append(ModelFile(
            path=str(p),
            filename=p.name,
            size_gb=size_gb,
            parent_dir=p.parent.name,
        ))

    _cache = results
    _cache_base = base_dir
    return results


def invalidate_cache() -> None:
    global _cache, _cache_base
    _cache = None
    _cache_base = None
