"""View: Calibration — baseline scores, normalized curves, and calibrated vs dynamic comparison."""

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
    FIGURE_LAYOUT,
    PLOTLY_CONFIG,
    TEXT_MUTED,
    TEXT_SECONDARY,
    hex_to_rgba,
    WONG,
)
from apex.dashboard.export import make_export_buttons, register_export_callbacks

# Module-level state for export (one set per chart section)
_baseline_figure = None
_baseline_dataframe = None
_norm_figure = None
_norm_dataframe = None
_cmp_figure = None
_cmp_dataframe = None


def layout() -> html.Div:
    return html.Div([
        html.H3("Calibration"),
        html.Details(
            [
                html.Summary("How to read this", style={"cursor": "pointer", "fontWeight": "600"}),
                html.P([
                    html.Strong("Bare baseline"), " — probe sent alone with no filler context; measures the model's "
                    "inherent ability to answer each probe. ",
                    html.Strong("Anchored baseline"), " — probe sent with filler at a fixed neutral position; "
                    "measures performance under context load without position variation. ",
                    html.Strong("Filler factor"), " — anchored / bare; isolates the effect of filler content. ",
                    html.Strong("Position factor"), " — calibrated-run score / anchored baseline; isolates the "
                    "effect of probe position within the context window.",
                ], style={"margin": "8px 0 0 0"}),
            ],
            style=GUIDE_STYLE,
        ),

        # Shared controls bar
        html.Div(
            [
                html.Div([
                    html.Label("Model", style=LABEL_STYLE),
                    dcc.Dropdown(id="cal-model", clearable=False),
                ], style={"flex": "1", "marginRight": "12px"}),
                html.Div([
                    html.Label("Context Length", style=LABEL_STYLE),
                    dcc.Dropdown(id="cal-context-length", clearable=False),
                ], style={"flex": "1"}),
            ],
            style={**CONTROLS_STYLE, "display": "flex", "alignItems": "flex-end"},
        ),

        # Section A: Status summary
        html.Div(id="cal-status", style=CARD_STYLE),

        # Section B: Baseline bar chart
        html.Div([
            html.H4("Baseline Scores by Probe", style={"marginTop": "0"}),
            html.Div([
                html.Label("Dimensions", style=LABEL_STYLE),
                dcc.Checklist(
                    id="cal-baseline-dims",
                    options=[],
                    value=[],
                    inline=True,
                    style={"fontSize": "13px"},
                ),
            ], style={"marginBottom": "8px"}),
            dcc.Graph(id="cal-baseline-chart", config=PLOTLY_CONFIG, style={"height": "450px"}),
            make_export_buttons("cal-baseline"),
        ], style=CARD_STYLE),

        # Section C: Normalized attention curves
        html.Div([
            html.H4("Normalized Attention Curves (Position Factor)", style={"marginTop": "0"}),
            html.Div([
                html.Div([
                    html.Label("Dimensions", style=LABEL_STYLE),
                    dcc.Checklist(
                        id="cal-norm-dims",
                        options=[],
                        value=[],
                        inline=True,
                        style={"fontSize": "13px"},
                    ),
                ], style={"flex": "2", "marginRight": "12px"}),
                html.Div([
                    dcc.Checklist(
                        id="cal-norm-options",
                        options=[
                            {"label": " Show CI bands", "value": "ci"},
                            {"label": " Baseline ref (y=1)", "value": "ref"},
                        ],
                        value=["ci", "ref"],
                        style={"fontSize": "13px"},
                    ),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "8px"}),
            dcc.Graph(id="cal-norm-graph", config=PLOTLY_CONFIG, style={"height": "500px"}),
            make_export_buttons("cal-norm"),
        ], style=CARD_STYLE),

        # Section D: Calibrated vs Dynamic comparison
        html.Div([
            html.H4("Calibrated vs Dynamic Runs", style={"marginTop": "0"}),
            html.Div([
                html.Label("Dimension", style=LABEL_STYLE),
                dcc.Dropdown(id="cal-cmp-dimension", clearable=False,
                             style={"maxWidth": "300px"}),
            ], style={"marginBottom": "8px"}),
            dcc.Graph(id="cal-cmp-graph", config=PLOTLY_CONFIG, style={"height": "500px"}),
            make_export_buttons("cal-cmp"),
        ], style=CARD_STYLE),
    ])


def register_callbacks(app, qm):
    # 0. Status panel
    @app.callback(
        Output("cal-status", "children"),
        Input("refresh-interval", "n_intervals"),
    )
    def update_status(_n):
        if not qm.has_calibration_tables():
            return html.Div([
                html.P(
                    "No calibration data available.",
                    style={"fontWeight": "600", "marginBottom": "4px"},
                ),
                html.P(
                    "Run `apex calibrate generate` then `apex calibrate baseline <config>` to set up.",
                    style={"color": TEXT_MUTED, "margin": "0"},
                ),
            ])
        status = qm.get_calibration_status()
        prompt_count = status["prompt_count"]
        bl_df = status["baseline_df"]

        # Count total probes and baselines per model/type
        rows = []
        rows.append(html.Div([
            html.Span("Frozen prompts: ", style={"color": TEXT_SECONDARY}),
            html.Span(f"{prompt_count}", style={"fontWeight": "600"}),
        ], style={"marginBottom": "4px"}))

        if not bl_df.empty:
            for _, row in bl_df.iterrows():
                rows.append(html.Div([
                    html.Span(f"{row['model_id']}", style={"fontWeight": "600", "marginRight": "8px"}),
                    html.Span(f"{row['baseline_type']}: ", style={"color": TEXT_SECONDARY}),
                    html.Span(f"{int(row['count'])} baselines", style={"marginRight": "16px"}),
                ]))
        else:
            rows.append(html.Div(
                "No baselines recorded yet. Run `apex calibrate baseline <config> --type bare` "
                "and `--type anchored` to generate.",
                style={"color": TEXT_MUTED},
            ))

        # Check for calibrated runs
        models = qm.get_calibrated_models()
        has_cal_runs = False
        if models:
            cal_df = qm.get_calibrated_curve_data(models[0])
            has_cal_runs = not cal_df.empty

        if not has_cal_runs and not bl_df.empty:
            rows.append(html.Div([
                html.Span("Next step: ", style={"color": TEXT_SECONDARY, "marginTop": "8px"}),
                html.Span(
                    "Launch a run with \"Use calibrated prompts\" enabled in Run Control.",
                    style={"fontWeight": "600"},
                ),
            ], style={"marginTop": "8px"}))

        return html.Div(rows)

    # 1. Model dropdown
    @app.callback(
        Output("cal-model", "options"),
        Output("cal-model", "value"),
        Input("refresh-interval", "n_intervals"),
        State("cal-model", "value"),
    )
    def update_models(_n, current):
        models = qm.get_calibrated_models()
        opts = [{"label": m, "value": m} for m in models]
        if current and current in models:
            return opts, no_update
        return opts, (opts[0]["value"] if opts else None)

    # 2. Context length cascade
    @app.callback(
        Output("cal-context-length", "options"),
        Output("cal-context-length", "value"),
        Input("cal-model", "value"),
    )
    def update_context_lengths(model_id):
        if not model_id:
            return [], None
        cls = qm.get_context_lengths(model_id)
        opts = [{"label": str(c), "value": c} for c in cls]
        return opts, (opts[0]["value"] if opts else None)

    # 3. Dimension cascades (all three selectors from one callback)
    @app.callback(
        Output("cal-baseline-dims", "options"),
        Output("cal-baseline-dims", "value"),
        Output("cal-norm-dims", "options"),
        Output("cal-norm-dims", "value"),
        Output("cal-cmp-dimension", "options"),
        Output("cal-cmp-dimension", "value"),
        Input("cal-model", "value"),
    )
    def update_dimensions(model_id):
        if not model_id:
            return [], [], [], [], [], None
        dims = qm.get_dimensions(model_id)
        checklist_opts = [{"label": DIMENSION_LABELS.get(d, d), "value": d} for d in dims]
        dropdown_opts = [{"label": DIMENSION_LABELS.get(d, d), "value": d} for d in dims]
        return (
            checklist_opts, dims,          # baseline dims
            checklist_opts, dims,          # norm dims
            dropdown_opts, dims[0] if dims else None,  # cmp dimension
        )

    # 4. Baseline bar chart
    @app.callback(
        Output("cal-baseline-chart", "figure"),
        Input("cal-model", "value"),
        Input("cal-baseline-dims", "value"),
    )
    def update_baseline_chart(model_id, dimensions):
        global _baseline_figure, _baseline_dataframe

        fig = go.Figure()
        fig.update_layout(**FIGURE_LAYOUT)
        fig.update_layout(
            xaxis=dict(title="Probe", ticksuffix="", range=None, dtick=None),
            yaxis=dict(title="Score", range=[-0.05, 1.05]),
            barmode="group",
        )

        if not model_id or not dimensions:
            fig.update_layout(title="Select a model and dimensions")
            _baseline_figure = fig
            _baseline_dataframe = None
            return fig

        baselines = qm.get_baselines_overview(model_id)
        if baselines.empty:
            fig.update_layout(title="No baselines available for this model")
            _baseline_figure = fig
            _baseline_dataframe = None
            return fig

        baselines = baselines[baselines["dimension"].isin(dimensions)]
        _baseline_dataframe = baselines

        # Compute filler_factor = anchored / bare per probe
        pivoted = baselines.pivot_table(index=["probe_id", "dimension"], columns="baseline_type", values="score").reset_index()
        if "bare" in pivoted.columns and "anchored" in pivoted.columns:
            pivoted["filler_factor"] = pivoted["anchored"] / pivoted["bare"].replace(0, float("nan"))
        else:
            pivoted["filler_factor"] = None

        colors = {"bare": WONG["blue"], "anchored": WONG["orange"], "filler_factor": WONG["green"]}
        labels = {"bare": "Bare", "anchored": "Anchored", "filler_factor": "Filler Factor"}

        for bt in ["bare", "anchored", "filler_factor"]:
            if bt not in pivoted.columns:
                continue
            subset = pivoted.dropna(subset=[bt])
            if subset.empty:
                continue
            probe_labels = [f"{r['probe_id'][:20]} ({DIMENSION_LABELS.get(r['dimension'], r['dimension'])[:3]})"
                            for _, r in subset.iterrows()]
            fig.add_trace(go.Bar(
                x=probe_labels,
                y=subset[bt],
                name=labels[bt],
                marker_color=colors[bt],
            ))

        fig.update_layout(title=f"{model_id} — Baseline Scores")
        _baseline_figure = fig
        return fig

    # 5. Normalized curves (position factor)
    @app.callback(
        Output("cal-norm-graph", "figure"),
        Input("cal-model", "value"),
        Input("cal-context-length", "value"),
        Input("cal-norm-dims", "value"),
        Input("cal-norm-options", "value"),
    )
    def update_norm_curves(model_id, context_length, dimensions, options):
        global _norm_figure, _norm_dataframe

        fig = go.Figure()
        norm_layout = {**FIGURE_LAYOUT}
        norm_layout["yaxis"] = dict(
            title="Position Factor (score / anchored baseline)",
            gridcolor="#2e2e4a",
            zerolinecolor="#3a3a5c",
        )
        fig.update_layout(**norm_layout)

        if not model_id or not context_length or not dimensions:
            fig.update_layout(title="Select model, context length, and dimensions")
            _norm_figure = fig
            _norm_dataframe = None
            return fig

        options = options or []
        show_ci = "ci" in options
        show_ref = "ref" in options

        raw_df = qm.get_calibrated_curve_data(model_id, context_length)
        baselines_df = qm.get_baselines_overview(model_id)
        raw_df = raw_df[raw_df["dimension"].isin(dimensions)]
        _norm_dataframe = raw_df

        agg = qm.normalize_by_baselines(raw_df, baselines_df, "anchored")

        if agg.empty:
            fig.update_layout(title="No calibrated run data for normalization")
            _norm_figure = fig
            return fig

        # Reference line at y=1.0
        if show_ref:
            fig.add_hline(y=1.0, line_dash="dot", line_color=TEXT_MUTED,
                          annotation_text="baseline", annotation_position="top right")

        for dim in dimensions:
            color = DIMENSION_COLORS.get(dim, "#666")
            label = DIMENSION_LABELS.get(dim, dim)
            dim_data = agg[agg["dimension"] == dim].sort_values("target_position_percent")
            if dim_data.empty:
                continue

            if show_ci and len(dim_data) > 1:
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

        fig.update_layout(title=f"{model_id} — Position Factor at {context_length} tokens")
        _norm_figure = fig
        return fig

    # 6. Calibrated vs Dynamic comparison
    @app.callback(
        Output("cal-cmp-graph", "figure"),
        Input("cal-model", "value"),
        Input("cal-context-length", "value"),
        Input("cal-cmp-dimension", "value"),
    )
    def update_cmp_chart(model_id, context_length, dimension):
        global _cmp_figure, _cmp_dataframe

        fig = go.Figure()
        fig.update_layout(**FIGURE_LAYOUT)

        if not model_id or not context_length or not dimension:
            fig.update_layout(title="Select model, context length, and dimension")
            _cmp_figure = fig
            _cmp_dataframe = None
            return fig

        cal_df = qm.get_calibrated_curve_data(model_id, context_length, dimension)
        dyn_df = qm.get_dynamic_curve_data(model_id, context_length, dimension)
        _cmp_dataframe = cal_df if not cal_df.empty else dyn_df

        color_cal = WONG["blue"]
        color_dyn = WONG["orange"]

        # Dynamic curve
        if not dyn_df.empty:
            dyn_agg = qm.aggregate_curve(dyn_df, "dimension")
            dyn_dim = dyn_agg[dyn_agg["dimension"] == dimension].sort_values("target_position_percent")
            if not dyn_dim.empty:
                fig.add_trace(go.Scatter(
                    x=dyn_dim["target_position_percent"],
                    y=dyn_dim["ci_upper"],
                    mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=dyn_dim["target_position_percent"],
                    y=dyn_dim["ci_lower"],
                    mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=hex_to_rgba(color_dyn, 0.15),
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=dyn_dim["target_position_percent"],
                    y=dyn_dim["mean"],
                    mode="lines+markers",
                    name="Dynamic",
                    line=dict(color=color_dyn, width=2.5, dash="dash"),
                    marker=dict(size=5),
                ))

        # Calibrated curve
        if not cal_df.empty:
            cal_agg = qm.aggregate_curve(cal_df, "dimension")
            cal_dim = cal_agg[cal_agg["dimension"] == dimension].sort_values("target_position_percent")
            if not cal_dim.empty:
                fig.add_trace(go.Scatter(
                    x=cal_dim["target_position_percent"],
                    y=cal_dim["ci_upper"],
                    mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=cal_dim["target_position_percent"],
                    y=cal_dim["ci_lower"],
                    mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=hex_to_rgba(color_cal, 0.15),
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=cal_dim["target_position_percent"],
                    y=cal_dim["mean"],
                    mode="lines+markers",
                    name="Calibrated",
                    line=dict(color=color_cal, width=2.5),
                    marker=dict(size=5),
                ))
        elif not dyn_df.empty:
            fig.add_annotation(
                text="No calibrated runs available — showing dynamic only",
                xref="paper", yref="paper", x=0.5, y=0.95,
                showarrow=False, font=dict(color=TEXT_MUTED, size=12),
            )

        dim_label = DIMENSION_LABELS.get(dimension, dimension)
        fig.update_layout(title=f"{model_id} — {dim_label} at {context_length} tokens")
        _cmp_figure = fig
        return fig

    # Export callbacks for each section
    register_export_callbacks(app, "cal-baseline", lambda: _baseline_figure, lambda: _baseline_dataframe)
    register_export_callbacks(app, "cal-norm", lambda: _norm_figure, lambda: _norm_dataframe)
    register_export_callbacks(app, "cal-cmp", lambda: _cmp_figure, lambda: _cmp_dataframe)
