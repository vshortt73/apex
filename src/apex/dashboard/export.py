"""PNG/SVG/CSV export helpers and reusable Dash components."""

from __future__ import annotations

import io
import base64
from typing import TYPE_CHECKING

from dash import html, dcc, Input, Output, State, callback_context

if TYPE_CHECKING:
    import plotly.graph_objects as go
    import pandas as pd


def figure_to_png_bytes(fig: go.Figure, width: int = 1200, height: int = 600, scale: int = 2) -> bytes:
    return fig.to_image(format="png", width=width, height=height, scale=scale, engine="kaleido")


def figure_to_svg_bytes(fig: go.Figure, width: int = 1200, height: int = 600) -> bytes:
    return fig.to_image(format="svg", width=width, height=height, engine="kaleido")


def dataframe_to_csv_string(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def make_export_buttons(view_id: str) -> html.Div:
    """Reusable export button group for a view."""
    return html.Div(
        [
            html.Button("PNG", id=f"{view_id}-export-png", n_clicks=0,
                         style={"marginRight": "8px", "padding": "4px 12px", "cursor": "pointer"}),
            html.Button("SVG", id=f"{view_id}-export-svg", n_clicks=0,
                         style={"marginRight": "8px", "padding": "4px 12px", "cursor": "pointer"}),
            html.Button("CSV", id=f"{view_id}-export-csv", n_clicks=0,
                         style={"padding": "4px 12px", "cursor": "pointer"}),
            dcc.Download(id=f"{view_id}-download"),
        ],
        style={"display": "flex", "alignItems": "center", "marginTop": "8px"},
    )


def register_export_callbacks(app, view_id: str, get_figure_fn, get_dataframe_fn):
    """Register PNG/SVG/CSV download callbacks for a view.

    Parameters
    ----------
    app : Dash
        The Dash application.
    view_id : str
        Prefix used in component IDs.
    get_figure_fn : callable
        Returns the current plotly Figure (or None).
    get_dataframe_fn : callable
        Returns the current pandas DataFrame (or None).
    """

    @app.callback(
        Output(f"{view_id}-download", "data"),
        Input(f"{view_id}-export-png", "n_clicks"),
        Input(f"{view_id}-export-svg", "n_clicks"),
        Input(f"{view_id}-export-csv", "n_clicks"),
        prevent_initial_call=True,
    )
    def _export(png_clicks, svg_clicks, csv_clicks):
        triggered = callback_context.triggered_id
        if triggered == f"{view_id}-export-png":
            fig = get_figure_fn()
            if fig is None:
                return None
            img_bytes = figure_to_png_bytes(fig)
            encoded = base64.b64encode(img_bytes).decode()
            return dict(
                content=encoded,
                filename=f"{view_id}.png",
                base64=True,
                type="image/png",
            )
        elif triggered == f"{view_id}-export-svg":
            fig = get_figure_fn()
            if fig is None:
                return None
            svg_bytes = figure_to_svg_bytes(fig)
            return dict(
                content=svg_bytes.decode("utf-8"),
                filename=f"{view_id}.svg",
                type="image/svg+xml",
            )
        elif triggered == f"{view_id}-export-csv":
            df = get_dataframe_fn()
            if df is None:
                return None
            return dict(
                content=dataframe_to_csv_string(df),
                filename=f"{view_id}.csv",
                type="text/csv",
            )
        return None
