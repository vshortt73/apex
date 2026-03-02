"""Dash application: tab layout, interval refresh, and view registration."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse, urlunparse

from dash import Dash, html, dcc

from apex.dashboard.config import DashboardConfig
from apex.dashboard.queries import QueryManager
from apex.dashboard.styles import BG_PAGE, BG_CARD, BORDER_COLOR, TEXT_PRIMARY, TEXT_MUTED

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "dashboard.yaml"


def _sanitize_dsn(dsn: str) -> str:
    """Strip password from a PostgreSQL DSN for client-side display."""
    if not dsn.startswith(("postgresql://", "postgres://")):
        return dsn
    parsed = urlparse(dsn)
    if parsed.password:
        netloc = f"{parsed.username}:***@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"
        return urlunparse(parsed._replace(netloc=netloc))
    return dsn

_ASSETS_DIR = str(Path(__file__).parent / "assets")
from apex.dashboard.views import (
    run_monitor,
    run_control,
    infrastructure,
    summary,
    curve_explorer,
    dimension_comparison,
    context_scaling,
    cross_model,
    probe_detail,
    calibration,
    settings,
)
from apex.dashboard.services.process_manager import ProcessManager


def build_app(db_path: str) -> Dash:
    """Create and configure the Dash application."""
    dashboard_config = DashboardConfig.load(_CONFIG_PATH)
    qm = QueryManager(db_path)
    pm = ProcessManager(backend_defaults=dashboard_config.backend_defaults.as_dict())

    app = Dash(
        __name__,
        title="APEX Console",
        suppress_callback_exceptions=True,
        assets_folder=_ASSETS_DIR,
    )

    # Dark theme tab styling
    tab_style = {
        "backgroundColor": BG_PAGE,
        "color": TEXT_MUTED,
        "border": f"1px solid {BORDER_COLOR}",
        "borderBottom": "none",
        "padding": "8px 16px",
    }
    tab_selected_style = {
        "backgroundColor": BG_CARD,
        "color": TEXT_PRIMARY,
        "border": f"1px solid {BORDER_COLOR}",
        "borderBottom": f"2px solid {TEXT_PRIMARY}",
        "padding": "8px 16px",
    }

    app.layout = html.Div([
        # Header
        html.Div([
            html.H2("APEX Console", style={"margin": "0", "flex": "1", "color": TEXT_PRIMARY}),
            html.Div([
                html.Button(
                    "Pause", id="refresh-toggle", n_clicks=0,
                    style={
                        "padding": "4px 12px", "cursor": "pointer", "marginRight": "8px",
                        "backgroundColor": BG_CARD, "color": TEXT_PRIMARY,
                        "border": f"1px solid {BORDER_COLOR}", "borderRadius": "4px",
                    },
                ),
                html.Span(id="refresh-status", style={"color": TEXT_MUTED, "fontSize": "12px"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "display": "flex", "alignItems": "center",
            "padding": "16px 24px", "borderBottom": f"1px solid {BORDER_COLOR}",
            "backgroundColor": BG_CARD,
        }),

        # Live refresh interval (30s)
        dcc.Interval(id="refresh-interval", interval=30_000, n_intervals=0),

        # Store for db path (credentials stripped for client safety)
        dcc.Store(id="db-path", data=_sanitize_dsn(db_path)),

        # Config version counter — incremented by Settings save to notify other tabs
        dcc.Store(id="config-version", data=0),

        # Tabs
        dcc.Tabs(
            id="main-tabs",
            value="monitor",
            children=[
                dcc.Tab(label="Run Monitor", value="monitor", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Run Control", value="runctl", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Infrastructure", value="infra", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Summary", value="summary", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Curve Explorer", value="curve", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Dimension Comparison", value="dimcmp", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Context Scaling", value="ctxscale", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Cross-Model", value="xmodel", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Probe Detail", value="probe", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Calibration", value="calibration", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Settings", value="settings", style=tab_style, selected_style=tab_selected_style),
            ],
            style={"padding": "0 24px"},
        ),

        # Tab content
        html.Div(id="tab-content", style={"padding": "16px 24px"}),
    ], style={
        "fontFamily": "Inter, Helvetica, Arial, sans-serif",
        "backgroundColor": BG_PAGE,
        "color": TEXT_PRIMARY,
        "minHeight": "100vh",
    })

    # -- Register tab switching --
    from dash import Input, Output

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab(tab):
        if tab == "monitor":
            return run_monitor.layout()
        elif tab == "runctl":
            return run_control.layout(dashboard_config)
        elif tab == "infra":
            return infrastructure.layout(dashboard_config)
        elif tab == "summary":
            return summary.layout()
        elif tab == "curve":
            return curve_explorer.layout()
        elif tab == "dimcmp":
            return dimension_comparison.layout()
        elif tab == "ctxscale":
            return context_scaling.layout()
        elif tab == "xmodel":
            return cross_model.layout()
        elif tab == "probe":
            return probe_detail.layout()
        elif tab == "calibration":
            return calibration.layout()
        elif tab == "settings":
            return settings.layout(dashboard_config)
        return html.P("Unknown tab")

    # -- Refresh toggle --
    @app.callback(
        Output("refresh-interval", "disabled"),
        Output("refresh-toggle", "children"),
        Output("refresh-status", "children"),
        Input("refresh-toggle", "n_clicks"),
    )
    def toggle_refresh(n_clicks):
        paused = n_clicks % 2 == 1
        return paused, ("Resume" if paused else "Pause"), ("Paused" if paused else "Auto-refresh: 30s")

    # -- Register all view callbacks --
    run_monitor.register_callbacks(app, qm, pm)
    run_control.register_callbacks(app, qm, pm, dashboard_config)
    infrastructure.register_callbacks(app, qm, dashboard_config)
    summary.register_callbacks(app, qm)
    curve_explorer.register_callbacks(app, qm)
    dimension_comparison.register_callbacks(app, qm)
    context_scaling.register_callbacks(app, qm)
    cross_model.register_callbacks(app, qm)
    probe_detail.register_callbacks(app, qm)
    calibration.register_callbacks(app, qm)
    settings.register_callbacks(app, dashboard_config)

    return app
