"""View 1: Attention Curve Explorer — primary visualization."""

from __future__ import annotations

from dash import html, dcc, Input, Output, State, no_update
import plotly.graph_objects as go

from apex.dashboard.styles import (
    CARD_STYLE,
    CONTROLS_STYLE,
    GUIDE_STYLE,
    LABEL_STYLE,
    DIMENSION_COLORS,
    DIMENSION_LABELS,
    MODEL_COLOR_CYCLE,
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
        # Row 1: Model, Context Length, View Mode
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
                    html.Label("View Mode", style=LABEL_STYLE),
                    dcc.RadioItems(
                        id="curve-view-mode",
                        options=[
                            {"label": " Aggregated", "value": "aggregated"},
                            {"label": " Per Run", "value": "per_run"},
                        ],
                        value="aggregated",
                        inline=True,
                        style={"fontSize": "13px"},
                    ),
                ], style={"flex": "1"}),
            ],
            style={**CONTROLS_STYLE, "display": "flex", "alignItems": "flex-end"},
        ),
        # Row 2: Aggregated controls (visible in aggregated mode)
        html.Div(
            id="curve-agg-controls",
            children=[
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
        # Row 3: Per-run controls (hidden by default)
        html.Div(
            id="curve-perrun-controls",
            children=[
                html.Div([
                    html.Label("Dimension", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="curve-dimension-single",
                        clearable=False,
                    ),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Runs", style=LABEL_STYLE),
                    dcc.Checklist(
                        id="curve-run-selector",
                        options=[],
                        value=[],
                        inline=False,
                        style={"fontSize": "12px", "maxHeight": "120px", "overflowY": "auto"},
                    ),
                ], style={"flex": "2", "marginRight": "12px"}),
                html.Div([
                    dcc.Checklist(
                        id="curve-per-run-options",
                        options=[
                            {"label": " Aggregate ref", "value": "ref"},
                            {"label": " Show CI bands", "value": "ci"},
                        ],
                        value=["ref"],
                        style={"fontSize": "13px"},
                    ),
                ], style={"flex": "1"}),
            ],
            style={**CONTROLS_STYLE, "display": "none", "alignItems": "flex-end"},
        ),
        html.Div([
            dcc.Graph(id="curve-graph", config=PLOTLY_CONFIG, style={"height": "500px"}),
            make_export_buttons("curve"),
        ], style=CARD_STYLE),
    ])


def register_callbacks(app, qm):
    # Populate model dropdown (preserve user selection across refreshes)
    @app.callback(
        Output("curve-model", "options"),
        Output("curve-model", "value"),
        Input("refresh-interval", "n_intervals"),
        State("curve-model", "value"),
    )
    def update_models(_n, current):
        models = qm.get_models()
        opts = [{"label": m["model_id"], "value": m["model_id"]} for m in models]
        ids = {m["model_id"] for m in models}
        if current and current in ids:
            return opts, no_update
        return opts, (opts[0]["value"] if opts else None)

    # Cascade: model -> context lengths
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

    # Cascade: model -> dimensions (multi-select for aggregated mode)
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

    # Cascade: model -> single dimension dropdown (for per-run mode)
    @app.callback(
        Output("curve-dimension-single", "options"),
        Output("curve-dimension-single", "value"),
        Input("curve-model", "value"),
    )
    def update_single_dimension(model_id):
        if not model_id:
            return [], None
        dims = qm.get_dimensions(model_id)
        opts = [{"label": DIMENSION_LABELS.get(d, d), "value": d} for d in dims]
        return opts, (dims[0] if dims else None)

    # Toggle visibility of aggregated vs per-run controls
    @app.callback(
        Output("curve-agg-controls", "style"),
        Output("curve-perrun-controls", "style"),
        Input("curve-view-mode", "value"),
    )
    def toggle_view_controls(view_mode):
        agg_base = {**CONTROLS_STYLE, "alignItems": "flex-end"}
        perrun_base = {**CONTROLS_STYLE, "alignItems": "flex-end"}
        if view_mode == "per_run":
            return {**agg_base, "display": "none"}, {**perrun_base, "display": "flex"}
        return {**agg_base, "display": "flex"}, {**perrun_base, "display": "none"}

    # Populate run selector when model/context/view-mode change
    @app.callback(
        Output("curve-run-selector", "options"),
        Output("curve-run-selector", "value"),
        Input("curve-model", "value"),
        Input("curve-context-length", "value"),
        Input("curve-view-mode", "value"),
    )
    def update_run_selector(model_id, context_length, view_mode):
        if view_mode != "per_run" or not model_id:
            return [], []
        runs_df = qm.get_run_uuids_for_model(model_id, context_length)
        if runs_df.empty:
            return [], []
        opts = []
        for _, row in runs_df.iterrows():
            uuid_short = str(row["run_uuid"])[:8]
            filler = row.get("filler_type", "")
            ts = str(row.get("first_ts", ""))[:10]
            n = row.get("result_count", 0)
            label = f"{uuid_short} {filler} {ts} ({n} results)"
            opts.append({"label": label, "value": row["run_uuid"]})
        values = [o["value"] for o in opts]  # all selected by default
        return opts, values

    # Build the figure
    @app.callback(
        Output("curve-graph", "figure"),
        Input("curve-model", "value"),
        Input("curve-context-length", "value"),
        Input("curve-dimensions", "value"),
        Input("curve-options", "value"),
        Input("curve-view-mode", "value"),
        Input("curve-dimension-single", "value"),
        Input("curve-run-selector", "value"),
        Input("curve-per-run-options", "value"),
    )
    def update_figure(
        model_id, context_length, dimensions, options,
        view_mode, dimension_single, selected_runs, per_run_options,
    ):
        global _current_figure, _current_dataframe

        fig = go.Figure()
        fig.update_layout(**FIGURE_LAYOUT)

        if not model_id or not context_length:
            fig.update_layout(title="Select model and context length")
            _current_figure = fig
            _current_dataframe = None
            return fig

        if view_mode == "per_run":
            fig = _build_per_run_figure(
                qm, fig, model_id, context_length,
                dimension_single, selected_runs, per_run_options,
            )
        else:
            fig = _build_aggregated_figure(
                qm, fig, model_id, context_length, dimensions, options,
            )

        _current_figure = fig
        return fig

    register_export_callbacks(
        app, "curve",
        lambda: _current_figure,
        lambda: _current_dataframe,
    )


def _build_aggregated_figure(qm, fig, model_id, context_length, dimensions, options):
    """Original aggregated view — one curve per dimension."""
    global _current_dataframe

    if not dimensions:
        fig.update_layout(title="Select at least one dimension")
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

    fig.update_layout(title=f"{model_id} \u2014 {context_length} tokens")
    return fig


def _build_per_run_figure(qm, fig, model_id, context_length, dimension, selected_runs, options):
    """Per-run view — one curve per run_uuid for a single dimension."""
    global _current_dataframe

    if not dimension:
        fig.update_layout(title="Select a dimension")
        _current_dataframe = None
        return fig

    raw_df = qm.get_curve_data(model_id, context_length, dimension)
    _current_dataframe = raw_df

    if raw_df.empty or "run_uuid" not in raw_df.columns:
        fig.update_layout(title="No data available")
        return fig

    # Filter to rows with run_uuid tracking
    uuid_df = raw_df[raw_df["run_uuid"].notna()]
    if uuid_df.empty:
        fig.update_layout(title="No runs with UUID tracking")
        return fig

    options = options or []
    show_ref = "ref" in options
    show_ci = "ci" in options

    # Aggregate reference line (all data for this dimension, regardless of run selection)
    if show_ref:
        agg_all = qm.aggregate_curve(raw_df, "dimension")
        ref_data = agg_all[agg_all["dimension"] == dimension].sort_values("target_position_percent")
        if not ref_data.empty:
            if show_ci and len(ref_data) > 1:
                fig.add_trace(go.Scatter(
                    x=ref_data["target_position_percent"],
                    y=ref_data["ci_upper"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=ref_data["target_position_percent"],
                    y=ref_data["ci_lower"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(136,136,136,0.15)",
                    showlegend=False,
                    hoverinfo="skip",
                ))
            fig.add_trace(go.Scatter(
                x=ref_data["target_position_percent"],
                y=ref_data["mean"],
                mode="lines",
                name="Aggregate",
                line=dict(color="#888", width=3, dash="dash"),
            ))

    # Per-run lines
    if not selected_runs:
        fig.update_layout(title=f"{model_id} \u2014 {dimension} @ {context_length} tokens (no runs selected)")
        return fig

    n_runs = 0
    for i, run_uuid in enumerate(selected_runs):
        run_data = uuid_df[uuid_df["run_uuid"] == run_uuid]
        if run_data.empty:
            continue
        run_agg = qm.aggregate_curve(run_data, "run_uuid")
        run_agg = run_agg.sort_values("target_position_percent")
        if run_agg.empty:
            continue

        color = MODEL_COLOR_CYCLE[i % len(MODEL_COLOR_CYCLE)]
        uuid_short = str(run_uuid)[:8]

        fig.add_trace(go.Scatter(
            x=run_agg["target_position_percent"],
            y=run_agg["mean"],
            mode="lines+markers",
            name=uuid_short,
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ))
        n_runs += 1

    dim_label = DIMENSION_LABELS.get(dimension, dimension)
    fig.update_layout(
        title=f"{model_id} \u2014 {dim_label} @ {context_length} tokens ({n_runs} run{'s' if n_runs != 1 else ''})",
    )
    return fig
