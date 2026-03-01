"""View 5: Probe Detail — heatmap overview + click-to-drill response viewer."""

from __future__ import annotations

from dash import html, dcc, Input, Output, State, no_update
import plotly.graph_objects as go
import pandas as pd

from apex.dashboard.styles import (
    CARD_STYLE,
    CONTROLS_STYLE,
    GUIDE_STYLE,
    LABEL_STYLE,
    DIMENSION_COLORS,
    DIMENSION_LABELS,
    FIGURE_LAYOUT,
    PLOTLY_CONFIG,
    PRE_STYLE,
    BG_CARD,
    BG_PLOT,
    BORDER_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    hex_to_rgba,
)


def layout() -> html.Div:
    return html.Div([
        html.H3("Probe Detail"),
        html.Details(
            [
                html.Summary("How to read this", style={"cursor": "pointer", "fontWeight": "600"}),
                html.P(
                    "Heatmap: rows are probes, columns are positions, color intensity = score. "
                    "Click any cell to see the model\u2019s raw response below. Use this to diagnose "
                    "why specific probes score low at certain positions \u2014 the raw text often reveals "
                    "whether the model forgot, hallucinated, or refused.",
                    style={"margin": "8px 0 0 0"},
                ),
            ],
            style=GUIDE_STYLE,
        ),

        # Controls for heatmap
        html.Div(
            [
                html.Div([
                    html.Label("Model", style=LABEL_STYLE),
                    dcc.Dropdown(id="probe-model", clearable=False),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Context Length", style=LABEL_STYLE),
                    dcc.Dropdown(id="probe-context-length", clearable=False),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Dimension", style=LABEL_STYLE),
                    dcc.Dropdown(id="probe-dimension", clearable=False),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Sort by", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="probe-sort",
                        options=[
                            {"label": "Mean score (worst first)", "value": "mean_asc"},
                            {"label": "Mean score (best first)", "value": "mean_desc"},
                            {"label": "Variance (most variable first)", "value": "var_desc"},
                            {"label": "Probe ID", "value": "id"},
                        ],
                        value="var_desc",
                        clearable=False,
                    ),
                ], style={"flex": "1"}),
            ],
            style={**CONTROLS_STYLE, "display": "flex", "alignItems": "flex-end"},
        ),

        # Heatmap
        html.Div([
            dcc.Graph(id="probe-heatmap", config=PLOTLY_CONFIG, style={"height": "600px"}),
            html.P(
                "Click any cell to inspect the probe response below.",
                style={"color": "#999", "fontSize": "12px", "marginTop": "4px"},
            ),
        ], style=CARD_STYLE),

        # Drill-down: metadata + response viewer
        html.Div(id="probe-drill-down", style=CARD_STYLE),
    ])


def register_callbacks(app, qm):
    @app.callback(
        Output("probe-model", "options"),
        Output("probe-model", "value"),
        Input("refresh-interval", "n_intervals"),
        State("probe-model", "value"),
    )
    def update_models(_n, current):
        models = qm.get_models()
        opts = [{"label": m["model_id"], "value": m["model_id"]} for m in models]
        ids = {m["model_id"] for m in models}
        if current and current in ids:
            return opts, no_update
        return opts, (opts[0]["value"] if opts else None)

    @app.callback(
        Output("probe-context-length", "options"),
        Output("probe-context-length", "value"),
        Input("probe-model", "value"),
    )
    def update_context_lengths(model_id):
        if not model_id:
            return [], None
        cls = qm.get_context_lengths(model_id)
        opts = [{"label": str(c), "value": c} for c in cls]
        value = opts[0]["value"] if opts else None
        return opts, value

    @app.callback(
        Output("probe-dimension", "options"),
        Output("probe-dimension", "value"),
        Input("probe-model", "value"),
    )
    def update_dimensions(model_id):
        if not model_id:
            return [], None
        dims = qm.get_dimensions(model_id)
        opts = [{"label": DIMENSION_LABELS.get(d, d), "value": d} for d in dims]
        value = opts[0]["value"] if opts else None
        return opts, value

    @app.callback(
        Output("probe-heatmap", "figure"),
        Input("probe-model", "value"),
        Input("probe-context-length", "value"),
        Input("probe-dimension", "value"),
        Input("probe-sort", "value"),
    )
    def update_heatmap(model_id, context_length, dimension, sort_by):
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color=TEXT_PRIMARY),
            paper_bgcolor=BG_CARD,
            plot_bgcolor=BG_PLOT,
            margin=dict(l=120, r=30, t=60, b=50),
        )

        if not model_id or not context_length or not dimension:
            fig.update_layout(title="Select model, context length, and dimension")
            return fig

        raw_df = qm.get_curve_data(model_id, context_length, dimension)
        if raw_df.empty:
            fig.update_layout(title="No data for this combination")
            return fig

        # Pivot: probes × positions
        scored = raw_df[raw_df["score"].notna()].copy()
        if scored.empty:
            fig.update_layout(title="No scored results")
            return fig

        pivot = scored.pivot_table(
            index="probe_id",
            columns="target_position_percent",
            values="score",
            aggfunc="mean",
        )

        # Sort probes
        if sort_by == "mean_asc":
            order = pivot.mean(axis=1).sort_values().index
        elif sort_by == "mean_desc":
            order = pivot.mean(axis=1).sort_values(ascending=False).index
        elif sort_by == "var_desc":
            order = pivot.std(axis=1).fillna(0).sort_values(ascending=False).index
        else:
            order = sorted(pivot.index)

        pivot = pivot.reindex(order)
        positions = sorted(pivot.columns)
        probe_ids = list(pivot.index)

        # Build heatmap
        z = pivot[positions].values
        x_labels = [f"{p:.0f}%" for p in positions]

        fig.add_trace(go.Heatmap(
            z=z,
            x=x_labels,
            y=probe_ids,
            colorscale=[
                [0.0, "#d73027"],
                [0.5, "#fee08b"],
                [1.0, "#1a9850"],
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(title="Score", tickvals=[0, 0.25, 0.5, 0.75, 1.0]),
            hovertemplate=(
                "Probe: %{y}<br>"
                "Position: %{x}<br>"
                "Score: %{z:.2f}<br>"
                "<extra></extra>"
            ),
            # Store raw position values for click handling
            customdata=[[p for p in positions] for _ in probe_ids],
        ))

        dim_label = DIMENSION_LABELS.get(dimension, dimension)
        fig.update_layout(
            title=f"Probe Heatmap — {model_id} / {dim_label} / {context_length} tokens",
            xaxis_title="Probe Position (% of context)",
            yaxis=dict(
                title="",
                autorange="reversed",
                tickfont=dict(size=10),
            ),
        )

        return fig

    @app.callback(
        Output("probe-drill-down", "children"),
        Input("probe-heatmap", "clickData"),
        State("probe-model", "value"),
        State("probe-context-length", "value"),
    )
    def drill_down(click_data, model_id, context_length):
        if not click_data or not model_id:
            return html.P("Click a cell in the heatmap to inspect the probe response.", style={"color": TEXT_MUTED})

        point = click_data["points"][0]
        probe_id = point["y"]
        # Get raw position from customdata
        col_idx = point["pointNumber"][1] if "pointNumber" in point else 0
        position = point.get("customdata", [None])
        if isinstance(position, list) and col_idx < len(position):
            position = position[col_idx]
        score_val = point.get("z")

        # Get metadata
        meta = qm.get_probe_metadata(probe_id)
        if meta is None:
            return html.P(f"No metadata found for {probe_id}.")

        dim_label = DIMENSION_LABELS.get(meta["dimension"], meta["dimension"])
        dim_color = DIMENSION_COLORS.get(meta["dimension"], "#666")

        # Find the matching row for the response
        df = qm.get_probe_detail(probe_id, model_id)
        if position is not None:
            matches = df[
                (df["target_position_percent"].round(1) == round(position, 1))
                & (df["context_length"] == context_length)
            ]
        else:
            matches = pd.DataFrame()

        # Build the drill-down view
        children = [
            html.H4(probe_id, style={"marginTop": "0", "marginBottom": "8px"}),
            html.Div([
                html.Span(
                    dim_label,
                    style={
                        "backgroundColor": hex_to_rgba(dim_color, 0.25),
                        "color": dim_color,
                        "padding": "2px 8px",
                        "borderRadius": "4px",
                        "fontWeight": "600",
                        "marginRight": "12px",
                    },
                ),
                html.Span(f"Type: {meta['content_type']}", style={"marginRight": "12px", "color": TEXT_SECONDARY}),
                html.Span(f"Scoring: {meta['score_method']}", style={"marginRight": "12px", "color": TEXT_SECONDARY}),
                html.Span(
                    f"Score: {score_val:.2f}" if score_val is not None else "Score: N/A",
                    style={"fontWeight": "600"},
                ),
            ], style={"marginBottom": "12px"}),

            # Probe content
            html.Details([
                html.Summary("Probe Content (the embedded instruction/fact)"),
                html.Pre(
                    meta["probe_content"],
                    style={**PRE_STYLE, "maxHeight": "150px"},
                ),
            ], style={"marginBottom": "12px"}),
        ]

        if not matches.empty:
            row = matches.iloc[0]
            children.append(
                html.Div([
                    html.Div([
                        html.H6(f"Turn 1 Response (position {position:.0f}%)"),
                        html.Pre(
                            row["raw_response"],
                            style=PRE_STYLE,
                        ),
                    ], style={"flex": "1", "marginRight": "8px"}),
                    html.Div([
                        html.H6("Turn 2 Response (to test query)"),
                        html.Pre(
                            row["raw_test_response"],
                            style=PRE_STYLE,
                        ),
                    ], style={"flex": "1"}),
                ], style={"display": "flex"})
            )
        else:
            children.append(html.P("Response data not available for this cell.", style={"color": TEXT_MUTED}))

        return html.Div(children)
