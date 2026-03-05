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

        # System status bar (GPU / VRAM / CPU / RAM)
        html.Div(id="monitor-system-bar"),

        # Active run configurations
        html.Div(id="monitor-run-configs"),

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


def register_callbacks(app, qm, pm=None):
    @app.callback(
        Output("monitor-run-configs", "children"),
        Input("monitor-interval", "n_intervals"),
    )
    def update_run_configs(_n):
        if pm is None:
            return html.Div()

        import yaml
        from pathlib import Path

        active = [r for r in pm.get_runs() if r.status == "running"]
        if not active:
            return html.Div()

        cards = []
        for run_info in active:
            # Read the full config YAML
            config = {}
            if run_info.config_path:
                try:
                    config = yaml.safe_load(Path(run_info.config_path).read_text()) or {}
                except Exception:
                    pass

            run_sec = config.get("run", {})
            models = config.get("models", [])
            evaluators = config.get("evaluator_models", [])
            positions = config.get("positions", [])
            ctx_lengths = config.get("context_lengths", [])

            # Model names
            model_names = [m.get("name", "?") for m in models]
            eval_names = [m.get("name", "?") for m in evaluators]

            # Format context lengths
            ctx_str = ", ".join(f"{c:,}" for c in sorted(ctx_lengths))

            # Run mode
            calibrated = run_sec.get("use_calibration", False)
            mode = "Calibrated (frozen)" if calibrated else "Dynamic (assembled)"

            _label = {"fontSize": "12px", "color": TEXT_SECONDARY, "minWidth": "110px", "display": "inline-block"}
            _value = {"fontSize": "13px", "fontWeight": "600"}

            def _row(label, value, value_style=None):
                return html.Div([
                    html.Span(label, style=_label),
                    html.Span(value, style={**_value, **(value_style or {})}),
                ], style={"marginBottom": "4px"})

            rows = [
                _row("Mode:", mode, {"color": WONG["blue"]} if calibrated else {}),
                _row("Model:", ", ".join(model_names)),
            ]

            if models:
                m = models[0]
                details = []
                if m.get("architecture", "unknown") != "unknown":
                    details.append(m["architecture"])
                if m.get("parameters"):
                    details.append(m["parameters"])
                if m.get("quantization"):
                    details.append(m["quantization"])
                if details:
                    rows.append(_row("Spec:", " / ".join(details)))
                rows.append(_row("Backend:", f"{m.get('backend', '?')} @ {m.get('base_url', '?')}"))

            if eval_names:
                rows.append(_row("Evaluator:", ", ".join(eval_names)))

            rows.extend([
                _row("Positions:", f"{len(positions)} ({positions[0]}–{positions[-1]})" if positions else "none"),
                _row("Context:", ctx_str or "none"),
                _row("Filler:", run_sec.get("filler_type", "neutral")),
                _row("Workers:", str(run_sec.get("workers", 1))),
                _row("Seed:", str(run_sec.get("seed", 42))),
                _row("Temperature:", str(run_sec.get("temperature", 0.0))),
                _row("Repetitions:", str(run_sec.get("repetitions", 1))),
            ])

            # Elapsed time
            elapsed_text = ""
            try:
                from datetime import datetime
                elapsed = datetime.now() - run_info.start_time
                hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
                minutes = remainder // 60
                if hours > 0:
                    elapsed_text = f"{hours}h {minutes}m"
                else:
                    elapsed_text = f"{minutes}m"
            except Exception:
                pass

            header_parts = [
                html.Span("Active Run", style={"fontWeight": "700", "fontSize": "15px"}),
                html.Span(f"PID {run_info.pid}", style={"color": TEXT_MUTED, "fontSize": "12px", "marginLeft": "12px"}),
            ]
            if elapsed_text:
                header_parts.append(html.Span(
                    f"running {elapsed_text}",
                    style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginLeft": "auto"},
                ))

            card = html.Div([
                html.Div(header_parts, style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
                *rows,
            ], style=CARD_STYLE)
            cards.append(card)

        return cards

    @app.callback(
        Output("monitor-system-bar", "children"),
        Input("monitor-interval", "n_intervals"),
    )
    def update_system_bar(_n):
        from apex.dashboard.services.infra import get_gpu_stats, get_system_stats

        gpu_list = get_gpu_stats()
        sys = get_system_stats()

        def _temp_color(temp_c: int | None) -> str:
            if temp_c is None:
                return TEXT_MUTED
            if temp_c < 70:
                return WONG["green"]
            if temp_c < 85:
                return WONG["orange"]
            return WONG["vermillion"]

        def _usage_color(pct: float) -> str:
            if pct < 70:
                return WONG["green"]
            if pct < 90:
                return WONG["orange"]
            return WONG["vermillion"]

        def _mini_bar(pct: float, color: str) -> html.Div:
            return html.Div([
                html.Div(style={
                    "width": f"{min(pct, 100):.0f}%", "height": "100%",
                    "backgroundColor": color, "borderRadius": "3px",
                }),
            ], style={
                "height": "8px", "width": "80px", "backgroundColor": BG_PLOT,
                "borderRadius": "3px", "display": "inline-block", "verticalAlign": "middle",
            })

        _sep = {"borderLeft": f"1px solid {BORDER_COLOR}", "height": "28px", "margin": "0 16px"}
        items = []

        # GPU section — aggregate for compact bar
        if gpu_list:
            gpu0 = gpu_list[0]
            n_gpus = len(gpu_list)
            gpu_label = gpu0.name if n_gpus == 1 else f"{gpu0.name} (x{n_gpus})"
            max_util = max(g.utilization_pct for g in gpu_list)
            max_temp = max(g.temperature_c for g in gpu_list)
            total_vram_used = sum(g.vram_used_mb for g in gpu_list)
            total_vram_total = sum(g.vram_total_mb for g in gpu_list)

            items.extend([
                html.Span(gpu_label, style={"fontWeight": "700", "fontSize": "13px", "marginRight": "8px"}),
                html.Span(
                    f"{max_util}%",
                    style={"fontSize": "13px", "marginRight": "6px"},
                ),
                html.Span(
                    f"{max_temp}\u00b0C",
                    style={"color": _temp_color(max_temp), "fontSize": "13px"},
                ),
                html.Div(style=_sep),
                # VRAM
                html.Span("VRAM ", style={"fontSize": "12px", "color": TEXT_SECONDARY, "marginRight": "6px"}),
            ])
            vram_pct = total_vram_used / max(total_vram_total, 1) * 100
            vram_gb_used = total_vram_used / 1024
            vram_gb_total = total_vram_total / 1024
            items.extend([
                _mini_bar(vram_pct, _usage_color(vram_pct)),
                html.Span(
                    f" {vram_gb_used:.1f} / {vram_gb_total:.1f} GB",
                    style={"fontSize": "12px", "color": TEXT_SECONDARY, "marginLeft": "6px"},
                ),
                html.Div(style=_sep),
            ])
        else:
            items.extend([
                html.Span("GPU ", style={"fontSize": "12px", "color": TEXT_SECONDARY, "marginRight": "4px"}),
                html.Span("N/A", style={"fontSize": "13px", "color": TEXT_MUTED}),
                html.Div(style=_sep),
            ])

        # CPU temp
        items.append(html.Span("CPU ", style={"fontSize": "12px", "color": TEXT_SECONDARY, "marginRight": "4px"}))
        if sys.cpu_temp_c is not None:
            items.append(html.Span(
                f"{sys.cpu_temp_c}\u00b0C",
                style={"fontSize": "13px", "color": _temp_color(sys.cpu_temp_c)},
            ))
        else:
            items.append(html.Span("N/A", style={"fontSize": "13px", "color": TEXT_MUTED}))

        items.append(html.Div(style=_sep))

        # RAM
        items.append(html.Span("RAM ", style={"fontSize": "12px", "color": TEXT_SECONDARY, "marginRight": "6px"}))
        if sys.ram_total_gb > 0:
            ram_pct = sys.ram_used_gb / sys.ram_total_gb * 100
            items.extend([
                _mini_bar(ram_pct, _usage_color(ram_pct)),
                html.Span(
                    f" {sys.ram_used_gb:.1f} / {sys.ram_total_gb:.1f} GB",
                    style={"fontSize": "12px", "color": TEXT_SECONDARY, "marginLeft": "6px"},
                ),
            ])
        else:
            items.append(html.Span("N/A", style={"fontSize": "13px", "color": TEXT_MUTED}))

        return html.Div(
            items,
            style={
                **CARD_STYLE,
                "display": "flex",
                "alignItems": "center",
                "padding": "10px 20px",
                "marginBottom": "16px",
            },
        )

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
                            rate_text += f" | ETE: {remaining:.0f}m"
                        else:
                            rate_text += f" | ETE: {remaining / 60:.1f}h"
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
        latest_uuid = qm.get_latest_run_uuid()
        df = qm.get_recent_errors(limit=10, run_uuid=latest_uuid)

        # Also get summary for counts — scoped to current run
        summary = qm.get_run_summary(run_uuid=latest_uuid)
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
                        html.Span(r["model_id"], style={"fontSize": "11px", "fontWeight": "600", "marginRight": "8px"}),
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
