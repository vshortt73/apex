"""APEX Dashboard — interactive visualization of probe results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dash import Dash


def create_app(db_path: str) -> Dash:
    """Factory: build and return the configured Dash application."""
    from apex.dashboard.app import build_app

    return build_app(db_path)
