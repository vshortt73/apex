"""View 1: Attention Curve Explorer — primary visualization."""

from __future__ import annotations

from dash import html, dcc, Input, Output
import plotly.graph_objects as go

from apex.dashboard.styles import (
    CARD_STYLE,
    CONTROLS_STYLE,
    GUIDE_STYLE,
    LABEL_STYLE,
    DIMENSION_COLORS,
    DIMENSION_LABELS,
    FIGURE_LAYOUT,
    PLOTLY_CONFIG,
    hex_to_rgba,
)
from apex.dashboard.export import make_export_buttons, register_export_callbacks

# Module-level state for export
_current_figure = None
_current_dataframe = None


def layout() -> html.Div:
    return html.Div([
        html.H3("Attention Curve Explorer"),
        html.Details(
            [
                html.Summary("How to read this", style={"cursor": "pointer", "fontWeight": "600"}),
                html.P(
                    "X-axis: where the probe was placed in the context (0% = start, 100% = end). "
                    "Y-axis: score (0\u20131). Shaded band: 95% confidence interval. A flat line means "
                    "uniform attention; a U-shape means primacy/recency bias; a downward slope means "
                    "the model forgets information placed later in the context.",
                    style={"margin": "8px 0 0 0"},
                ),
            ],
            style=GUIDE_STYLE,
        ),
        html.Div(
            [
                html.Div([
                    html.Label("Model", style=LABEL_STYLE),
                    dcc.Dropdown(id="curve-model", clearable=False),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Context Length", style=LABEL_STYLE),
                    dcc.Dropdown(id="curve-context-length", clearable=False),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Dimensions", style=LABEL_STYLE),
                    dcc.Checklist(
                        id="curve-dimensions",
                        options=[],
                        value=[],
                        inline=True,
                        style={"fontSize": "13px"},
                    ),
                ], style={"flex": "2", "marginRight": "12px"}),
                html.Div([
                    dcc.Checklist(
                        id="curve-options",
                        options=[
                            {"label": " Show CI bands", "value": "ci"},
                            {"label": " Show scatter", "value": "scatter"},
                        ],
                        value=["ci"],
                        style={"fontSize": "13px"},
                    ),
                ], style={"flex": "1"}),
            ],
            style={**CONTROLS_STYLE, "display": "flex", "alignItems": "flex-end"},
        ),
        html.Div([
            dcc.Graph(id="curve-graph", config=PLOTLY_CONFIG, style={"height": "500px"}),
            make_export_buttons("curve"),
        ], style=CARD_STYLE),
    ])


def register_callbacks(app, qm):
    # Populate model dropdown
    @app.callback(
        Output("curve-model", "options"),
        Output("curve-model", "value"),
        Input("refresh-interval", "n_intervals"),
    )
    def update_models(_n):
        models = qm.get_models()
        opts = [{"label": m["model_id"], "value": m["model_id"]} for m in models]
        value = opts[0]["value"] if opts else None
        return opts, value

    # Cascade: model → context lengths
    @app.callback(
        Output("curve-context-length", "options"),
        Output("curve-context-length", "value"),
        Input("curve-model", "value"),
    )
    def update_context_lengths(model_id):
        if not model_id:
            return [], None
        cls = qm.get_context_lengths(model_id)
        opts = [{"label": str(c), "value": c} for c in cls]
        value = opts[0]["value"] if opts else None
        return opts, value

    # Cascade: model → dimensions
    @app.callback(
        Output("curve-dimensions", "options"),
        Output("curve-dimensions", "value"),
        Input("curve-model", "value"),
    )
    def update_dimensions(model_id):
        if not model_id:
            return [], []
        dims = qm.get_dimensions(model_id)
        opts = [{"label": DIMENSION_LABELS.get(d, d), "value": d} for d in dims]
        values = dims  # all selected by default
        return opts, values

    # Build the figure
    @app.callback(
        Output("curve-graph", "figure"),
        Input("curve-model", "value"),
        Input("curve-context-length", "value"),
        Input("curve-dimensions", "value"),
        Input("curve-options", "value"),
    )
    def update_figure(model_id, context_length, dimensions, options):
        global _current_figure, _current_dataframe

        fig = go.Figure()
        fig.update_layout(**FIGURE_LAYOUT)

        if not model_id or not context_length or not dimensions:
            fig.update_layout(title="Select model, context length, and dimensions")
            _current_figure = fig
            _current_dataframe = None
            return fig

        show_ci = "ci" in (options or [])
        show_scatter = "scatter" in (options or [])

        raw_df = qm.get_curve_data(model_id, context_length)
        raw_df = raw_df[raw_df["dimension"].isin(dimensions)]
        _current_dataframe = raw_df

        agg = qm.aggregate_curve(raw_df, "dimension")

        for dim in dimensions:
            color = DIMENSION_COLORS.get(dim, "#666")
            label = DIMENSION_LABELS.get(dim, dim)
            dim_data = agg[agg["dimension"] == dim].sort_values("target_position_percent")

            if dim_data.empty:
                continue

            # CI band
            if show_ci and len(dim_data) > 1:
                fig.add_trace(go.Scatter(
                    x=dim_data["target_position_percent"],
                    y=dim_data["ci_upper"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=dim_data["target_position_percent"],
                    y=dim_data["ci_lower"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=hex_to_rgba(color),
                    showlegend=False,
                    hoverinfo="skip",
                ))

            # Mean line
            fig.add_trace(go.Scatter(
                x=dim_data["target_position_percent"],
                y=dim_data["mean"],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=5),
            ))

            # Scatter overlay
            if show_scatter:
                scatter_data = raw_df[
                    (raw_df["dimension"] == dim) & raw_df["score"].notna() & (raw_df["refused"] == 0)
                ]
                fig.add_trace(go.Scatter(
                    x=scatter_data["target_position_percent"],
                    y=scatter_data["score"],
                    mode="markers",
                    name=f"{label} (individual)",
                    marker=dict(color=color, size=4, opacity=0.3),
                    showlegend=False,
                ))

        fig.update_layout(title=f"{model_id} — {context_length} tokens")
        _current_figure = fig
        return fig

    register_export_callbacks(
        app, "curve",
        lambda: _current_figure,
        lambda: _current_dataframe,
    )
