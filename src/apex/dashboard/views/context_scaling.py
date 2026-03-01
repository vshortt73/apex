"""View 3: Context Length Scaling — one dimension, multiple context lengths."""

from __future__ import annotations

from dash import html, dcc, Input, Output, State, no_update
import plotly.graph_objects as go

from apex.dashboard.styles import (
    CARD_STYLE,
    CONTROLS_STYLE,
    GUIDE_STYLE,
    LABEL_STYLE,
    DIMENSION_LABELS,
    CONTEXT_LENGTH_SCALE,
    FIGURE_LAYOUT,
    PLOTLY_CONFIG,
)
from apex.dashboard.export import make_export_buttons, register_export_callbacks

_current_figure = None
_current_dataframe = None


def layout() -> html.Div:
    return html.Div([
        html.H3("Context Length Scaling"),
        html.Details(
            [
                html.Summary("How to read this", style={"cursor": "pointer", "fontWeight": "600"}),
                html.P(
                    "Same dimension at different context lengths. Curves shifting down at longer contexts "
                    "indicate degraded attention \u2014 the model struggles to maintain performance as the "
                    "window grows. Compare shapes to identify the model\u2019s effective context ceiling.",
                    style={"margin": "8px 0 0 0"},
                ),
            ],
            style=GUIDE_STYLE,
        ),
        html.Div(
            [
                html.Div([
                    html.Label("Model", style=LABEL_STYLE),
                    dcc.Dropdown(id="ctxscale-model", clearable=False),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Dimension", style=LABEL_STYLE),
                    dcc.Dropdown(id="ctxscale-dimension", clearable=False),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Context Lengths", style=LABEL_STYLE),
                    dcc.Checklist(
                        id="ctxscale-lengths",
                        options=[],
                        value=[],
                        inline=True,
                        style={"fontSize": "13px"},
                    ),
                ], style={"flex": "2"}),
            ],
            style={**CONTROLS_STYLE, "display": "flex", "alignItems": "flex-end"},
        ),
        html.Div([
            dcc.Graph(id="ctxscale-graph", config=PLOTLY_CONFIG, style={"height": "500px"}),
            make_export_buttons("ctxscale"),
        ], style=CARD_STYLE),
    ])


def register_callbacks(app, qm):
    @app.callback(
        Output("ctxscale-model", "options"),
        Output("ctxscale-model", "value"),
        Input("refresh-interval", "n_intervals"),
        State("ctxscale-model", "value"),
    )
    def update_models(_n, current):
        models = qm.get_models()
        opts = [{"label": m["model_id"], "value": m["model_id"]} for m in models]
        ids = {m["model_id"] for m in models}
        if current and current in ids:
            return opts, no_update
        return opts, (opts[0]["value"] if opts else None)

    @app.callback(
        Output("ctxscale-dimension", "options"),
        Output("ctxscale-dimension", "value"),
        Input("ctxscale-model", "value"),
    )
    def update_dimensions(model_id):
        if not model_id:
            return [], None
        dims = qm.get_dimensions(model_id)
        opts = [{"label": DIMENSION_LABELS.get(d, d), "value": d} for d in dims]
        value = opts[0]["value"] if opts else None
        return opts, value

    @app.callback(
        Output("ctxscale-lengths", "options"),
        Output("ctxscale-lengths", "value"),
        Input("ctxscale-model", "value"),
    )
    def update_lengths(model_id):
        if not model_id:
            return [], []
        cls = qm.get_context_lengths(model_id)
        opts = [{"label": str(c), "value": c} for c in cls]
        return opts, cls  # all selected by default

    @app.callback(
        Output("ctxscale-graph", "figure"),
        Input("ctxscale-model", "value"),
        Input("ctxscale-dimension", "value"),
        Input("ctxscale-lengths", "value"),
    )
    def update_figure(model_id, dimension, lengths):
        global _current_figure, _current_dataframe

        fig = go.Figure()
        fig.update_layout(**FIGURE_LAYOUT)

        if not model_id or not dimension or not lengths:
            fig.update_layout(title="Select model, dimension, and context lengths")
            _current_figure = fig
            _current_dataframe = None
            return fig

        raw_df = qm.get_curve_data(model_id, dimension=dimension)
        raw_df = raw_df[raw_df["context_length"].isin(lengths)]
        _current_dataframe = raw_df

        agg = qm.aggregate_curve(raw_df, "context_length")
        sorted_lengths = sorted(agg["context_length"].unique())

        for i, cl in enumerate(sorted_lengths):
            color_idx = min(i, len(CONTEXT_LENGTH_SCALE) - 1)
            color = CONTEXT_LENGTH_SCALE[color_idx]
            cl_data = agg[agg["context_length"] == cl].sort_values("target_position_percent")

            if cl_data.empty:
                continue

            fig.add_trace(go.Scatter(
                x=cl_data["target_position_percent"],
                y=cl_data["mean"],
                mode="lines+markers",
                name=f"{cl} tokens",
                line=dict(color=color, width=2.5),
                marker=dict(size=5),
            ))

        dim_label = DIMENSION_LABELS.get(dimension, dimension)
        fig.update_layout(title=f"Context Scaling — {model_id} / {dim_label}")
        _current_figure = fig
        return fig

    register_export_callbacks(
        app, "ctxscale",
        lambda: _current_figure,
        lambda: _current_dataframe,
    )
