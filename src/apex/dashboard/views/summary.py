"""View 6: Run Summary — per-run cards with config details and dimension breakdown."""

from __future__ import annotations

from dash import html, dcc, Input, Output

from apex.dashboard.styles import (
    CARD_STYLE, GUIDE_STYLE, BG_CONTROLS, BORDER_COLOR,
    DIMENSION_COLORS, DIMENSION_LABELS,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, WONG,
)

_LABEL_STYLE = {"fontSize": "12px", "color": TEXT_SECONDARY, "minWidth": "100px", "display": "inline-block"}
_VALUE_STYLE = {"fontSize": "13px"}


def layout() -> html.Div:
    return html.Div(
        [
            html.H3("Run Summary"),
            html.Details(
                [
                    html.Summary("How to read this", style={"cursor": "pointer", "fontWeight": "600"}),
                    html.P(
                        "Each card represents one run (identified by UUID). Shows the run configuration, "
                        "result counts, refusal rates, and per-dimension score distributions. "
                        "Scores range 0\u20131 (higher = better retention).",
                        style={"margin": "8px 0 0 0"},
                    ),
                ],
                style=GUIDE_STYLE,
            ),
            html.Div(id="summary-content"),
        ]
    )


def _config_row(label, value, value_style=None):
    return html.Div([
        html.Span(label, style=_LABEL_STYLE),
        html.Span(value, style={**_VALUE_STYLE, **(value_style or {})}),
    ], style={"marginBottom": "2px"})


def register_callbacks(app, qm):
    @app.callback(
        Output("summary-content", "children"),
        Input("refresh-interval", "n_intervals"),
    )
    def update_summary(_n):
        run_configs = qm.get_run_configs()

        # Fall back to the old model-level summary for legacy data without run_uuids
        legacy_summary = qm.get_run_summary()

        if run_configs.empty and legacy_summary.empty:
            return html.P("No results yet. Start a probe run to see data here.")

        cards = []

        # Per-run cards (newest first)
        if not run_configs.empty:
            for _, row in run_configs.iterrows():
                run_uuid = row["run_uuid"]
                model_id = row["model_id"]
                filler_type = row.get("filler_type", "neutral") or "neutral"
                is_calibrated = filler_type == "calibrated"

                # Config section
                ctx_lengths = qm.get_run_context_lengths(run_uuid)
                ctx_str = ", ".join(f"{c:,}" for c in ctx_lengths)

                mode = "Calibrated (frozen)" if is_calibrated else "Dynamic (assembled)"
                mode_style = {"color": WONG["blue"], "fontWeight": "600"} if is_calibrated else {"fontWeight": "600"}

                config_rows = [
                    _config_row("Mode:", mode, mode_style),
                    _config_row("Filler:", filler_type),
                    _config_row("Context:", ctx_str or "N/A"),
                    _config_row("Positions:", str(int(row["distinct_positions"]))),
                    _config_row("Temperature:", str(row["temperature"])),
                    _config_row("Repetitions:", str(int(row["max_run_number"])) if row["max_run_number"] else "1"),
                    _config_row("Probes:", f"{int(row['distinct_probes'])} across {int(row['distinct_dimensions'])} dimensions"),
                ]

                # Dimension breakdown
                breakdown = qm.get_run_dimension_breakdown(run_uuid)
                dim_table_rows = []
                for _, dr in breakdown.iterrows():
                    dim = dr["dimension"]
                    label = DIMENSION_LABELS.get(dim, dim)
                    color = DIMENSION_COLORS.get(dim, "#666")
                    mean_score = f"{dr['mean_score']:.3f}" if dr["mean_score"] is not None else "N/A"
                    _td_center = {"textAlign": "center", "padding": "4px 12px"}
                    dim_table_rows.append(
                        html.Tr([
                            html.Td(
                                html.Span(label, style={"color": color, "fontWeight": "600"}),
                                style={"textAlign": "left", "padding": "4px 12px"},
                            ),
                            html.Td(str(int(dr["count"])), style=_td_center),
                            html.Td(mean_score, style=_td_center),
                            html.Td(str(int(dr["refused"])) if dr["refused"] else "0", style=_td_center),
                        ])
                    )

                # Result counts
                result_parts = [
                    html.Span(f"{int(row['result_count'])} results", style={"fontWeight": "600", "marginRight": "16px"}),
                ]
                if row["refused_count"] and int(row["refused_count"]) > 0:
                    result_parts.append(html.Span(
                        f"{int(row['refused_count'])} refused",
                        style={"color": "#ff6b6b", "marginRight": "16px"},
                    ))
                if row["null_score_count"] and int(row["null_score_count"]) > 0:
                    result_parts.append(html.Span(
                        f"{int(row['null_score_count'])} unscored",
                        style={"color": TEXT_MUTED},
                    ))

                card = html.Div([
                    # Header: model name + run UUID
                    html.Div([
                        html.H4(model_id, style={"marginTop": "0", "marginBottom": "0", "flex": "1"}),
                        html.Span(
                            run_uuid[:8],
                            title=run_uuid,
                            style={"color": TEXT_MUTED, "fontSize": "11px", "fontFamily": "monospace"},
                        ),
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),

                    # Model spec
                    html.Div([
                        html.Span(f"Arch: {row['model_architecture']}", style={"marginRight": "16px"}),
                        html.Span(f"Params: {row['model_parameters']}", style={"marginRight": "16px"}),
                        html.Span(f"Quant: {row['quantization']}"),
                    ], style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginBottom": "10px"}),

                    # Config details
                    html.Div(
                        config_rows,
                        style={
                            "backgroundColor": BG_CONTROLS,
                            "borderRadius": "6px",
                            "padding": "10px 14px",
                            "marginBottom": "12px",
                            "border": f"1px solid {BORDER_COLOR}",
                        },
                    ),

                    # Result counts
                    html.Div(result_parts, style={"marginBottom": "12px"}),

                    # Dimension table
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Dimension", style={"textAlign": "left", "padding": "4px 12px"}),
                            html.Th("Count", style={"textAlign": "center", "padding": "4px 12px"}),
                            html.Th("Mean Score", style={"textAlign": "center", "padding": "4px 12px"}),
                            html.Th("Refused", style={"textAlign": "center", "padding": "4px 12px"}),
                        ])),
                        html.Tbody(dim_table_rows),
                    ], style={"fontSize": "13px", "borderCollapse": "collapse"}) if dim_table_rows else None,

                    html.Div(
                        f"Time range: {row['first_timestamp']} — {row['last_timestamp']}",
                        style={"color": TEXT_MUTED, "fontSize": "11px", "marginTop": "12px"},
                    ),
                ], style=CARD_STYLE)
                cards.append(card)

        # Legacy cards for results without run_uuid (if any)
        if not legacy_summary.empty:
            # Check if there are results with NULL run_uuid
            has_legacy = False
            for _, row in legacy_summary.iterrows():
                model_id = row["model_id"]
                # If all results for this model are already covered by run_uuid cards, skip
                if not run_configs.empty and model_id in run_configs["model_id"].values:
                    run_total = int(run_configs[run_configs["model_id"] == model_id]["result_count"].sum())
                    if run_total >= int(row["result_count"]):
                        continue
                has_legacy = True
                breakdown = qm.get_dimension_breakdown(model_id)
                dim_rows = []
                for _, dr in breakdown.iterrows():
                    dim = dr["dimension"]
                    label = DIMENSION_LABELS.get(dim, dim)
                    color = DIMENSION_COLORS.get(dim, "#666")
                    mean_score = f"{dr['mean_score']:.3f}" if dr["mean_score"] is not None else "N/A"
                    _td_center = {"textAlign": "center", "padding": "4px 12px"}
                    dim_rows.append(
                        html.Tr([
                            html.Td(
                                html.Span(label, style={"color": color, "fontWeight": "600"}),
                                style={"textAlign": "left", "padding": "4px 12px"},
                            ),
                            html.Td(str(int(dr["count"])), style=_td_center),
                            html.Td(mean_score, style=_td_center),
                            html.Td(str(int(dr["refused"])) if dr["refused"] else "0", style=_td_center),
                        ])
                    )

                card = html.Div([
                    html.H4(model_id, style={"marginTop": "0", "marginBottom": "8px"}),
                    html.Div([
                        html.Span(f"Arch: {row['model_architecture']}", style={"marginRight": "16px"}),
                        html.Span(f"Params: {row['model_parameters']}", style={"marginRight": "16px"}),
                        html.Span(f"Quant: {row['quantization']}"),
                    ], style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginBottom": "12px"}),
                    html.Div([
                        html.Span(f"{int(row['result_count'])} results", style={"fontWeight": "600", "marginRight": "16px"}),
                        html.Span(f"{int(row['refused_count'])} refused", style={"color": "#ff6b6b", "marginRight": "16px"}) if row["refused_count"] > 0 else None,
                        html.Span(f"{int(row['null_score_count'])} unscored", style={"color": TEXT_MUTED}) if row["null_score_count"] > 0 else None,
                    ], style={"marginBottom": "12px"}),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Dimension", style={"textAlign": "left", "padding": "4px 12px"}),
                            html.Th("Count", style={"textAlign": "center", "padding": "4px 12px"}),
                            html.Th("Mean Score", style={"textAlign": "center", "padding": "4px 12px"}),
                            html.Th("Refused", style={"textAlign": "center", "padding": "4px 12px"}),
                        ])),
                        html.Tbody(dim_rows),
                    ], style={"fontSize": "13px", "borderCollapse": "collapse"}),
                    html.Div(
                        f"Time range: {row['first_timestamp']} — {row['last_timestamp']}",
                        style={"color": TEXT_MUTED, "fontSize": "11px", "marginTop": "12px"},
                    ),
                ], style=CARD_STYLE)
                cards.append(card)

        return cards
