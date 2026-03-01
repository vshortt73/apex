"""View 4: Cross-Model Comparison — one dimension+context length, multiple models."""

from __future__ import annotations

from dash import html, dcc, Input, Output, State, no_update
import plotly.graph_objects as go

from apex.dashboard.styles import (
    CARD_STYLE,
    CONTROLS_STYLE,
    GUIDE_STYLE,
    LABEL_STYLE,
    DIMENSION_LABELS,
    MODEL_COLOR_CYCLE,
    FIGURE_LAYOUT,
    PLOTLY_CONFIG,
    hex_to_rgba,
)
from apex.dashboard.export import make_export_buttons, register_export_callbacks

_current_figure = None
_current_dataframe = None


def layout() -> html.Div:
    return html.Div([
        html.H3("Cross-Model Comparison"),
        html.Details(
            [
                html.Summary("How to read this", style={"cursor": "pointer", "fontWeight": "600"}),
                html.P(
                    "Same dimension and context length, multiple models overlaid. Higher curves = better "
                    "retention at that position. Shape differences reveal architectural attention patterns "
                    "\u2014 dense transformers often show U-shaped curves while MoE/SSM architectures may "
                    "behave differently.",
                    style={"margin": "8px 0 0 0"},
                ),
            ],
            style=GUIDE_STYLE,
        ),
        html.Div(
            [
                html.Div([
                    html.Label("Models", style=LABEL_STYLE),
                    dcc.Dropdown(id="xmodel-models", multi=True),
                ], style={"flex": "2", "marginRight": "12px"}),
                html.Div([
                    html.Label("Dimension", style=LABEL_STYLE),
                    dcc.Dropdown(id="xmodel-dimension", clearable=False),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Context Length", style=LABEL_STYLE),
                    dcc.Dropdown(id="xmodel-context-length", clearable=False),
                ], style={"flex": "1"}),
            ],
            style={**CONTROLS_STYLE, "display": "flex", "alignItems": "flex-end"},
        ),
        html.Div([
            dcc.Graph(id="xmodel-graph", config=PLOTLY_CONFIG, style={"height": "500px"}),
            make_export_buttons("xmodel"),
        ], style=CARD_STYLE),
    ])


def register_callbacks(app, qm):
    @app.callback(
        Output("xmodel-models", "options"),
        Output("xmodel-models", "value"),
        Input("refresh-interval", "n_intervals"),
        State("xmodel-models", "value"),
    )
    def update_models(_n, current):
        models = qm.get_models()
        opts = [{"label": m["model_id"], "value": m["model_id"]} for m in models]
        ids = {m["model_id"] for m in models}
        if current and all(v in ids for v in current):
            return opts, no_update
        return opts, [m["model_id"] for m in models]

    @app.callback(
        Output("xmodel-dimension", "options"),
        Output("xmodel-dimension", "value"),
        Input("xmodel-models", "value"),
    )
    def update_dimensions(model_ids):
        if not model_ids:
            return [], None
        dims = qm.get_dimensions()
        opts = [{"label": DIMENSION_LABELS.get(d, d), "value": d} for d in dims]
        value = opts[0]["value"] if opts else None
        return opts, value

    @app.callback(
        Output("xmodel-context-length", "options"),
        Output("xmodel-context-length", "value"),
        Input("xmodel-models", "value"),
    )
    def update_context_lengths(model_ids):
        if not model_ids:
            return [], None
        cls = qm.get_context_lengths()
        opts = [{"label": str(c), "value": c} for c in cls]
        value = opts[0]["value"] if opts else None
        return opts, value

    @app.callback(
        Output("xmodel-graph", "figure"),
        Input("xmodel-models", "value"),
        Input("xmodel-dimension", "value"),
        Input("xmodel-context-length", "value"),
    )
    def update_figure(model_ids, dimension, context_length):
        global _current_figure, _current_dataframe

        fig = go.Figure()
        fig.update_layout(**FIGURE_LAYOUT)

        if not model_ids or not dimension or not context_length:
            fig.update_layout(title="Select models, dimension, and context length")
            _current_figure = fig
            _current_dataframe = None
            return fig

        raw_df = qm.get_cross_model_data(model_ids, dimension, context_length)
        _current_dataframe = raw_df

        agg = qm.aggregate_curve(raw_df, "model_id")

        for i, mid in enumerate(sorted(agg["model_id"].unique())):
            color = MODEL_COLOR_CYCLE[i % len(MODEL_COLOR_CYCLE)]
            model_data = agg[agg["model_id"] == mid].sort_values("target_position_percent")

            if model_data.empty:
                continue

            # CI band
            fig.add_trace(go.Scatter(
                x=model_data["target_position_percent"],
                y=model_data["ci_upper"],
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=model_data["target_position_percent"],
                y=model_data["ci_lower"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=hex_to_rgba(color),
                showlegend=False, hoverinfo="skip",
            ))

            fig.add_trace(go.Scatter(
                x=model_data["target_position_percent"],
                y=model_data["mean"],
                mode="lines+markers",
                name=mid,
                line=dict(color=color, width=2.5),
                marker=dict(size=5),
            ))

        dim_label = DIMENSION_LABELS.get(dimension, dimension)
        fig.update_layout(title=f"Cross-Model — {dim_label} @ {context_length} tokens")
        _current_figure = fig
        return fig

    register_export_callbacks(
        app, "xmodel",
        lambda: _current_figure,
        lambda: _current_dataframe,
    )
