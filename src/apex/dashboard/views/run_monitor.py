"""Run Monitor — live mission-control view of active probe runs."""

from __future__ import annotations

from dash import html, dcc, Input, Output

from apex.dashboard.styles import (
    CARD_STYLE, GUIDE_STYLE, BG_CARD, BG_CONTROLS, BG_PLOT,
    BORDER_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, PRE_STYLE,
    DIMENSION_COLORS, DIMENSION_LABELS, WONG,
)


def layout() -> html.Div:
    return html.Div([
        html.H3("Run Monitor"),
        html.Details([
            html.Summary("How to read this", style={"cursor": "pointer", "fontWeight": "600"}),
            html.P(
                "Live view of active APEX runs. Progress bars show completed vs estimated total "
                "executions. The activity feed shows the 20 most recent results. Refreshes every 10 seconds.",
                style={"margin": "8px 0 0 0"},
            ),
        ], style=GUIDE_STYLE),

        # 10-second refresh for this tab
        dcc.Interval(id="monitor-interval", interval=10_000, n_intervals=0),

        # Progress cards
        html.Div(id="monitor-progress-cards"),

        # Two-column layout: dimension progress + score method breakdown
        html.Div([
            html.Div(id="monitor-dimension-progress", style={"flex": "1", "minWidth": "0"}),
            html.Div(id="monitor-score-methods", style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # Activity feed + error tracker side by side
        html.Div([
            html.Div(id="monitor-activity-feed", style={"flex": "2", "minWidth": "0"}),
            html.Div(id="monitor-error-tracker", style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "16px"}),
    ])


def register_callbacks(app, qm):
    @app.callback(
        Output("monitor-progress-cards", "children"),
        Input("monitor-interval", "n_intervals"),
    )
    def update_progress(_n):
        df = qm.get_run_progress()
        if df.empty:
            return html.P(
                "No results yet. Start a probe run to see progress here.",
                style={"color": TEXT_MUTED, "padding": "20px"},
            )

        cards = []
        for _, row in df.iterrows():
            model_id = row["model_id"]
            completed = int(row["completed"])
            n_probes = int(row["distinct_probes"])
            n_pos = int(row["distinct_positions"])
            n_ctx = int(row["distinct_ctx_lengths"])
            estimated = n_probes * n_pos * n_ctx
            # Avoid div by zero; estimated could be exact count if all combos done
            pct = min(100, round(completed / max(estimated, 1) * 100, 1))

            # Rate calculation
            first_ts = str(row["first_ts"]) if row["first_ts"] else ""
            last_ts = str(row["last_ts"]) if row["last_ts"] else ""
            rate_text = ""
            try:
                from datetime import datetime
                fmt_candidates = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"]
                t0 = t1 = None
                for fmt in fmt_candidates:
                    try:
                        t0 = datetime.strptime(first_ts[:26], fmt)
                        break
                    except ValueError:
                        continue
                for fmt in fmt_candidates:
                    try:
                        t1 = datetime.strptime(last_ts[:26], fmt)
                        break
                    except ValueError:
                        continue
                if t0 and t1:
                    elapsed_min = max((t1 - t0).total_seconds() / 60, 0.01)
                    rate = completed / elapsed_min
                    rate_text = f"{rate:.1f} probes/min"
                    if estimated > completed:
                        remaining = (estimated - completed) / rate
                        if remaining < 60:
                            rate_text += f" | ETA: {remaining:.0f}m"
                        else:
                            rate_text += f" | ETA: {remaining / 60:.1f}h"
            except Exception:
                pass

            bar_color = WONG["green"] if pct >= 100 else WONG["blue"]

            card = html.Div([
                html.Div([
                    html.Span(model_id, style={"fontWeight": "700", "fontSize": "15px"}),
                    html.Span(
                        f"{completed:,} / ~{estimated:,}",
                        style={"color": TEXT_SECONDARY, "fontSize": "13px", "marginLeft": "auto"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),

                # Progress bar
                html.Div([
                    html.Div(style={
                        "width": f"{pct}%",
                        "height": "100%",
                        "backgroundColor": bar_color,
                        "borderRadius": "4px",
                        "transition": "width 0.5s ease",
                    }),
                ], style={
                    "height": "20px",
                    "backgroundColor": BG_PLOT,
                    "borderRadius": "4px",
                    "overflow": "hidden",
                    "marginBottom": "6px",
                }),

                html.Div([
                    html.Span(f"{pct}%", style={"fontWeight": "600", "marginRight": "16px"}),
                    html.Span(rate_text, style={"color": TEXT_SECONDARY, "fontSize": "12px"}),
                ], style={"fontSize": "13px"}),
            ], style=CARD_STYLE)
            cards.append(card)

        return cards

    @app.callback(
        Output("monitor-dimension-progress", "children"),
        Input("monitor-interval", "n_intervals"),
    )
    def update_dimension_progress(_n):
        df = qm.get_dimension_progress()
        if df.empty:
            return html.Div()

        rows = []
        for _, row in df.iterrows():
            model_id = row["model_id"]
            dim = row["dimension"]
            completed = int(row["completed"])
            n_probes = int(row["distinct_probes"])
            n_pos = int(row["distinct_positions"])
            estimated = max(n_probes * n_pos, 1)
            pct = min(100, round(completed / estimated * 100, 1))
            color = DIMENSION_COLORS.get(dim, WONG["white"])
            label = DIMENSION_LABELS.get(dim, dim)
            mean_score = row["mean_score"]
            score_text = f"{mean_score:.3f}" if mean_score is not None else "N/A"

            rows.append(html.Div([
                html.Div([
                    html.Span(model_id, style={"fontSize": "12px", "color": TEXT_SECONDARY, "marginRight": "8px"}),
                    html.Span(label, style={"fontWeight": "600", "color": color, "fontSize": "13px"}),
                    html.Span(
                        f"x\u0304={score_text}",
                        style={"marginLeft": "auto", "fontSize": "12px", "color": TEXT_SECONDARY},
                    ),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
                html.Div([
                    html.Div(style={
                        "width": f"{pct}%", "height": "100%",
                        "backgroundColor": color, "borderRadius": "3px",
                    }),
                ], style={
                    "height": "8px", "backgroundColor": BG_PLOT,
                    "borderRadius": "3px", "marginBottom": "2px",
                }),
                html.Span(
                    f"{completed}/{estimated} ({pct}%)",
                    style={"fontSize": "11px", "color": TEXT_MUTED},
                ),
            ], style={"marginBottom": "10px"}))

        return html.Div([
            html.H4("Dimension Progress", style={"marginTop": "0", "marginBottom": "12px"}),
            *rows,
        ], style=CARD_STYLE)

    @app.callback(
        Output("monitor-score-methods", "children"),
        Input("monitor-interval", "n_intervals"),
    )
    def update_score_methods(_n):
        df = qm.get_score_method_breakdown()
        if df.empty:
            return html.Div()

        METHOD_COLORS = {
            "exact_match": WONG["blue"],
            "programmatic": WONG["green"],
            "semantic": WONG["yellow"],
            "evaluator": WONG["vermillion"],
        }

        model_groups = {}
        for _, row in df.iterrows():
            mid = row["model_id"]
            if mid not in model_groups:
                model_groups[mid] = []
            model_groups[mid].append((row["score_method"], int(row["count"])))

        items = []
        for model_id, methods in model_groups.items():
            total = sum(c for _, c in methods)
            method_bars = []
            for method, count in methods:
                color = METHOD_COLORS.get(method, WONG["white"])
                pct = count / max(total, 1) * 100
                method_bars.append(html.Div([
                    html.Span(
                        f"{method}",
                        style={"color": color, "fontWeight": "600", "fontSize": "12px", "minWidth": "100px"},
                    ),
                    html.Div([
                        html.Div(style={
                            "width": f"{pct}%", "height": "100%",
                            "backgroundColor": color, "borderRadius": "3px",
                        }),
                    ], style={
                        "height": "8px", "backgroundColor": BG_PLOT,
                        "borderRadius": "3px", "flex": "1", "margin": "0 8px",
                    }),
                    html.Span(f"{count:,}", style={"fontSize": "12px", "color": TEXT_SECONDARY, "minWidth": "50px", "textAlign": "right"}),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}))

            items.append(html.Div([
                html.Div(model_id, style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "6px"}),
                *method_bars,
            ], style={"marginBottom": "12px"}))

        return html.Div([
            html.H4("Score Methods", style={"marginTop": "0", "marginBottom": "12px"}),
            *items,
        ], style=CARD_STYLE)

    @app.callback(
        Output("monitor-activity-feed", "children"),
        Input("monitor-interval", "n_intervals"),
    )
    def update_activity_feed(_n):
        df = qm.get_recent_results(limit=20)
        if df.empty:
            return html.Div()

        _th = {"padding": "6px 10px", "textAlign": "left", "borderBottom": f"1px solid {BORDER_COLOR}", "fontSize": "12px", "fontWeight": "600"}
        _td = {"padding": "5px 10px", "borderBottom": f"1px solid {BORDER_COLOR}", "fontSize": "12px"}

        rows = []
        for _, r in df.iterrows():
            score = r["score"]
            score_str = f"{score:.3f}" if score is not None else "null"
            refused = r["refused"]
            dim = r["dimension"]
            color = DIMENSION_COLORS.get(dim, TEXT_PRIMARY)

            score_style = dict(_td)
            if refused:
                score_style["color"] = "#ff6b6b"
                score_str = "REFUSED"
            elif score is None:
                score_style["color"] = TEXT_MUTED
            elif score >= 0.7:
                score_style["color"] = WONG["green"]
            elif score < 0.3:
                score_style["color"] = WONG["vermillion"]

            pos = r["target_position_percent"]
            pos_str = f"{pos:.1f}%" if pos is not None else "?"

            rows.append(html.Tr([
                html.Td(str(r["timestamp"])[-8:], style=_td),
                html.Td(r["probe_id"], style={**_td, "maxWidth": "140px", "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
                html.Td(pos_str, style=_td),
                html.Td(DIMENSION_LABELS.get(dim, dim), style={**_td, "color": color}),
                html.Td(score_str, style=score_style),
                html.Td(r["score_method"] or "", style={**_td, "color": TEXT_MUTED}),
            ]))

        return html.Div([
            html.H4("Activity Feed", style={"marginTop": "0", "marginBottom": "12px"}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Time", style=_th),
                    html.Th("Probe", style=_th),
                    html.Th("Position", style=_th),
                    html.Th("Dimension", style=_th),
                    html.Th("Score", style=_th),
                    html.Th("Method", style=_th),
                ])),
                html.Tbody(rows),
            ], style={"width": "100%", "borderCollapse": "collapse"}),
        ], style=CARD_STYLE)

    @app.callback(
        Output("monitor-error-tracker", "children"),
        Input("monitor-interval", "n_intervals"),
    )
    def update_error_tracker(_n):
        df = qm.get_recent_errors(limit=10)

        # Also get summary for counts
        summary = qm.get_run_summary()
        refused_total = int(summary["refused_count"].sum()) if not summary.empty else 0
        null_total = int(summary["null_score_count"].sum()) if not summary.empty else 0

        header_items = []
        if refused_total:
            header_items.append(html.Span(
                f"{refused_total} refused",
                style={"color": "#ff6b6b", "fontWeight": "600", "marginRight": "16px"},
            ))
        if null_total:
            header_items.append(html.Span(
                f"{null_total} unscored",
                style={"color": WONG["yellow"], "fontWeight": "600"},
            ))
        if not header_items:
            header_items.append(html.Span("No errors", style={"color": WONG["green"]}))

        error_items = []
        if not df.empty:
            for _, r in df.iterrows():
                is_refused = r["refused"]
                label = "REFUSED" if is_refused else "NULL SCORE"
                label_color = "#ff6b6b" if is_refused else WONG["yellow"]
                snippet = r.get("response_snippet", "") or ""

                error_items.append(html.Div([
                    html.Div([
                        html.Span(label, style={"color": label_color, "fontWeight": "600", "fontSize": "11px", "marginRight": "8px"}),
                        html.Span(r["probe_id"], style={"fontSize": "11px", "color": TEXT_SECONDARY}),
                    ], style={"marginBottom": "4px"}),
                    html.Div(
                        snippet[:150] + ("..." if len(snippet) > 150 else ""),
                        style={"fontSize": "11px", "color": TEXT_MUTED, "whiteSpace": "pre-wrap", "maxHeight": "40px", "overflow": "hidden"},
                    ) if snippet else None,
                ], style={"marginBottom": "10px", "paddingBottom": "8px", "borderBottom": f"1px solid {BORDER_COLOR}"}))

        return html.Div([
            html.H4("Error Tracker", style={"marginTop": "0", "marginBottom": "8px"}),
            html.Div(header_items, style={"marginBottom": "12px"}),
            *error_items,
        ], style=CARD_STYLE)
