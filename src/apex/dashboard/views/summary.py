"""View 6: Run Summary — landing page with model cards and dimension breakdown."""

from __future__ import annotations

from dash import html, dcc, Input, Output
import plotly.graph_objects as go

from apex.dashboard.styles import CARD_STYLE, GUIDE_STYLE, DIMENSION_COLORS, DIMENSION_LABELS, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED


def layout() -> html.Div:
    return html.Div(
        [
            html.H3("Run Summary"),
            html.Details(
                [
                    html.Summary("How to read this", style={"cursor": "pointer", "fontWeight": "600"}),
                    html.P(
                        "Model cards show result counts, refusal rates, and per-dimension score "
                        "distributions. Scores range 0\u20131 (higher = better retention). Refusals are "
                        "cases where the model declined to respond.",
                        style={"margin": "8px 0 0 0"},
                    ),
                ],
                style=GUIDE_STYLE,
            ),
            html.Div(id="summary-content"),
        ]
    )


def register_callbacks(app, qm):
    @app.callback(
        Output("summary-content", "children"),
        Input("refresh-interval", "n_intervals"),
    )
    def update_summary(_n):
        summary = qm.get_run_summary()
        if summary.empty:
            return html.P("No results yet. Start a probe run to see data here.")

        cards = []
        for _, row in summary.iterrows():
            model_id = row["model_id"]
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
                            html.Span(
                                label,
                                style={"color": color, "fontWeight": "600"},
                            ),
                            style={"textAlign": "left", "padding": "4px 12px"},
                        ),
                        html.Td(str(int(dr["count"])), style=_td_center),
                        html.Td(mean_score, style=_td_center),
                        html.Td(str(int(dr["refused"])) if dr["refused"] else "0", style=_td_center),
                    ])
                )

            card = html.Div(
                [
                    html.H4(model_id, style={"marginTop": "0", "marginBottom": "8px"}),
                    html.Div(
                        [
                            html.Span(f"Arch: {row['model_architecture']}", style={"marginRight": "16px"}),
                            html.Span(f"Params: {row['model_parameters']}", style={"marginRight": "16px"}),
                            html.Span(f"Quant: {row['quantization']}"),
                        ],
                        style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginBottom": "12px"},
                    ),
                    html.Div(
                        [
                            html.Span(f"{int(row['result_count'])} results", style={"fontWeight": "600", "marginRight": "16px"}),
                            html.Span(f"{int(row['refused_count'])} refused", style={"color": "#ff6b6b", "marginRight": "16px"}) if row["refused_count"] > 0 else None,
                            html.Span(f"{int(row['null_score_count'])} unscored", style={"color": TEXT_MUTED}) if row["null_score_count"] > 0 else None,
                        ],
                        style={"marginBottom": "12px"},
                    ),
                    html.Table(
                        [
                            html.Thead(html.Tr([
                                html.Th("Dimension", style={"textAlign": "left", "padding": "4px 12px"}),
                                html.Th("Count", style={"textAlign": "center", "padding": "4px 12px"}),
                                html.Th("Mean Score", style={"textAlign": "center", "padding": "4px 12px"}),
                                html.Th("Refused", style={"textAlign": "center", "padding": "4px 12px"}),
                            ])),
                            html.Tbody(dim_rows),
                        ],
                        style={"fontSize": "13px", "borderCollapse": "collapse"},
                    ),
                    html.Div(
                        f"Time range: {row['first_timestamp']} — {row['last_timestamp']}",
                        style={"color": TEXT_MUTED, "fontSize": "11px", "marginTop": "12px"},
                    ),
                ],
                style=CARD_STYLE,
            )
            cards.append(card)

        return cards
