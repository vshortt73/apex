"""View 2: Dimension Comparison — overlay all dimensions with correlation analysis."""

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

_current_figure = None
_current_dataframe = None


def layout() -> html.Div:
    return html.Div([
        html.H3("Dimension Comparison"),
        html.Details(
            [
                html.Summary("How to read this", style={"cursor": "pointer", "fontWeight": "600"}),
                html.P(
                    "Overlays all three dimensions on one chart. Correlation values (bottom) show how "
                    "similarly the model treats each dimension across positions. Diverging curves reveal "
                    "dimension-specific attention patterns \u2014 e.g., factual recall may hold steady while "
                    "application drops off.",
                    style={"margin": "8px 0 0 0"},
                ),
            ],
            style=GUIDE_STYLE,
        ),
        html.Div(
            [
                html.Div([
                    html.Label("Model", style=LABEL_STYLE),
                    dcc.Dropdown(id="dimcmp-model", clearable=False),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Context Length", style=LABEL_STYLE),
                    dcc.Dropdown(id="dimcmp-context-length", clearable=False),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Sweet Spot Threshold", style=LABEL_STYLE),
                    dcc.Slider(
                        id="dimcmp-threshold",
                        min=0.5, max=1.0, step=0.05, value=0.7,
                        marks={v: f"{v:.1f}" for v in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
                    ),
                ], style={"flex": "2"}),
            ],
            style={**CONTROLS_STYLE, "display": "flex", "alignItems": "flex-end"},
        ),
        html.Div([
            dcc.Graph(id="dimcmp-graph", config=PLOTLY_CONFIG, style={"height": "500px"}),
            make_export_buttons("dimcmp"),
        ], style=CARD_STYLE),
        html.Div(id="dimcmp-analysis", style=CARD_STYLE),
    ])


def register_callbacks(app, qm):
    @app.callback(
        Output("dimcmp-model", "options"),
        Output("dimcmp-model", "value"),
        Input("refresh-interval", "n_intervals"),
    )
    def update_models(_n):
        models = qm.get_models()
        opts = [{"label": m["model_id"], "value": m["model_id"]} for m in models]
        value = opts[0]["value"] if opts else None
        return opts, value

    @app.callback(
        Output("dimcmp-context-length", "options"),
        Output("dimcmp-context-length", "value"),
        Input("dimcmp-model", "value"),
    )
    def update_context_lengths(model_id):
        if not model_id:
            return [], None
        cls = qm.get_context_lengths(model_id)
        opts = [{"label": str(c), "value": c} for c in cls]
        value = opts[0]["value"] if opts else None
        return opts, value

    @app.callback(
        Output("dimcmp-graph", "figure"),
        Output("dimcmp-analysis", "children"),
        Input("dimcmp-model", "value"),
        Input("dimcmp-context-length", "value"),
        Input("dimcmp-threshold", "value"),
    )
    def update_figure(model_id, context_length, threshold):
        global _current_figure, _current_dataframe

        fig = go.Figure()
        fig.update_layout(**FIGURE_LAYOUT)

        if not model_id or not context_length:
            fig.update_layout(title="Select model and context length")
            _current_figure = fig
            _current_dataframe = None
            return fig, ""

        raw_df = qm.get_curve_data(model_id, context_length)
        _current_dataframe = raw_df
        agg = qm.aggregate_curve(raw_df, "dimension")

        dims = sorted(agg["dimension"].unique()) if not agg.empty else []

        for dim in dims:
            color = DIMENSION_COLORS.get(dim, "#666")
            label = DIMENSION_LABELS.get(dim, dim)
            dim_data = agg[agg["dimension"] == dim].sort_values("target_position_percent")

            if dim_data.empty:
                continue

            # CI band
            fig.add_trace(go.Scatter(
                x=dim_data["target_position_percent"],
                y=dim_data["ci_upper"],
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=dim_data["target_position_percent"],
                y=dim_data["ci_lower"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=hex_to_rgba(color),
                showlegend=False, hoverinfo="skip",
            ))

            fig.add_trace(go.Scatter(
                x=dim_data["target_position_percent"],
                y=dim_data["mean"],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=5),
            ))

        # Threshold line
        fig.add_hline(y=threshold, line_dash="dash", line_color="#8888aa",
                      annotation_text=f"Threshold: {threshold}")

        # Sweet spot highlighting
        sweet = qm.find_sweet_spots(agg, threshold)
        if not sweet.empty:
            positions = sorted(sweet["target_position_percent"].tolist())
            # Find contiguous ranges
            ranges = _find_ranges(positions)
            for start, end in ranges:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor="rgba(0,158,115,0.1)",
                    line_width=0,
                    annotation_text="sweet spot" if start == ranges[0][0] else None,
                    annotation_position="top left",
                )

        fig.update_layout(title=f"Dimension Comparison — {model_id} @ {context_length} tokens")
        _current_figure = fig

        # Analysis panel
        analysis = []
        corrs = qm.compute_dimension_correlations(agg)
        if corrs:
            corr_items = [html.Li(f"{k}: r = {v}") for k, v in corrs.items()]
            analysis.append(html.Div([
                html.H5("Dimension Correlations (Pearson r)"),
                html.Ul(corr_items),
            ]))

        if not sweet.empty:
            pos_list = ", ".join(f"{p}%" for p in sorted(sweet["target_position_percent"].tolist()))
            analysis.append(html.Div([
                html.H5(f"Sweet Spots (all dims >= {threshold})"),
                html.P(pos_list),
            ]))
        else:
            analysis.append(html.P(f"No positions where all dimensions score >= {threshold}"))

        return fig, analysis

    register_export_callbacks(
        app, "dimcmp",
        lambda: _current_figure,
        lambda: _current_dataframe,
    )


def _find_ranges(positions: list[float]) -> list[tuple[float, float]]:
    """Group positions into contiguous ranges (within 15% gap)."""
    if not positions:
        return []
    ranges = []
    start = positions[0]
    prev = positions[0]
    for p in positions[1:]:
        if p - prev > 15:
            ranges.append((start, prev))
            start = p
        prev = p
    ranges.append((start, prev))
    return ranges
