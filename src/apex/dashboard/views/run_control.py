"""Run Control — configure and launch APEX runs from the UI."""

from __future__ import annotations

import json

from dash import html, dcc, Input, Output, State, callback_context, no_update, ALL

from apex.dashboard.styles import (
    CARD_STYLE, CONTROLS_STYLE, GUIDE_STYLE, LABEL_STYLE, BG_CARD, BG_CONTROLS, BG_PLOT,
    BORDER_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, WONG,
)


# Position presets
POSITIONS_QUICK = [0.1, 0.5, 0.9]
POSITIONS_STANDARD = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
POSITIONS_DENSE = [
    0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98,
]

COMMON_CTX_LENGTHS = [2048, 4096, 8192, 16384, 32768]

_INPUT_STYLE = {
    "width": "100%", "padding": "6px", "backgroundColor": BG_PLOT,
    "color": TEXT_PRIMARY, "border": f"1px solid {BORDER_COLOR}", "borderRadius": "4px",
}

_BTN_STYLE = {
    "padding": "6px 14px", "cursor": "pointer", "border": "none",
    "borderRadius": "4px", "fontSize": "12px", "fontWeight": "600",
}


def layout(dashboard_config=None) -> html.Div:
    from apex.dashboard.config import DashboardConfig
    cfg = dashboard_config or DashboardConfig()
    return html.Div([
        html.H3("Run Control"),
        html.Details([
            html.Summary("How to read this", style={"cursor": "pointer", "fontWeight": "600"}),
            html.P(
                "Configure and launch APEX probe runs. Add models, select probes and positions, "
                "then launch. The run will appear in Run Monitor.",
                style={"margin": "8px 0 0 0"},
            ),
        ], style=GUIDE_STYLE),

        dcc.Interval(id="runctl-interval", interval=5_000, n_intervals=0),

        # Store for added models (list of model config dicts)
        dcc.Store(id="runctl-models-store", data=[]),
        dcc.Store(id="runctl-evaluator-store", data=[]),

        # === Model Selection ===
        html.Div([
            html.H4("Models", style={"marginTop": "0", "marginBottom": "12px"}),

            html.Div([
                html.Div([
                    html.Label("Model File (.gguf)", style=LABEL_STYLE),
                    dcc.Dropdown(id="runctl-model-file", options=[], placeholder="Select model...", style={"width": "100%"}),
                ], style={"flex": "3", "minWidth": "300px"}),
                html.Div([
                    html.Label("Backend", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="runctl-backend",
                        options=[
                            {"label": "llama.cpp", "value": "llamacpp"},
                            {"label": "Ollama", "value": "ollama"},
                            {"label": "SGLang", "value": "sglang"},
                            {"label": "OpenAI", "value": "openai"},
                            {"label": "Anthropic", "value": "anthropic"},
                            {"label": "Google", "value": "google"},
                        ],
                        value="llamacpp",
                        clearable=False,
                        style={"width": "100%"},
                    ),
                ], style={"flex": "1", "minWidth": "130px"}),
            ], style={"display": "flex", "gap": "12px", "marginBottom": "12px", "flexWrap": "wrap"}),

            html.Div([
                html.Div([
                    html.Label("Display Name", style=LABEL_STYLE),
                    dcc.Input(id="runctl-model-name", type="text", placeholder="auto from filename", style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Architecture", style=LABEL_STYLE),
                    dcc.Input(id="runctl-architecture", type="text", placeholder="e.g. llama, qwen2", style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Parameters", style=LABEL_STYLE),
                    dcc.Input(id="runctl-parameters", type="text", placeholder="e.g. 7B, 70B", style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Quantization", style=LABEL_STYLE),
                    dcc.Input(id="runctl-quantization", type="text", placeholder="e.g. Q4_K_M", style=_INPUT_STYLE),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "12px", "marginBottom": "12px", "flexWrap": "wrap"}),

            html.Div([
                html.Div([
                    html.Label("Base URL", style=LABEL_STYLE),
                    dcc.Input(id="runctl-base-url", type="text", placeholder="http://localhost:8080", style=_INPUT_STYLE),
                ], style={"flex": "2"}),
                html.Div([
                    html.Label("Max Context Window", style=LABEL_STYLE),
                    dcc.Input(id="runctl-max-ctx", type="number", value=8192, min=512, style=_INPUT_STYLE),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "12px", "marginBottom": "12px", "flexWrap": "wrap"}),

            html.Button(
                "Add Model", id="runctl-add-model-btn", n_clicks=0,
                style={**_BTN_STYLE, "backgroundColor": WONG["blue"], "color": "#000"},
            ),

            # Added models display
            html.Div(id="runctl-models-display", style={"marginTop": "12px"}),
        ], style=CARD_STYLE),

        # === Evaluator Model (optional) ===
        html.Div([
            html.H4("Evaluator Model (Optional)", style={"marginTop": "0", "marginBottom": "12px"}),
            html.P(
                "For probes using 'evaluator' score method. Leave empty to skip evaluator-scored probes.",
                style={"fontSize": "12px", "color": TEXT_MUTED, "marginBottom": "12px"},
            ),
            html.Div([
                html.Div([
                    html.Label("Backend", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="runctl-eval-backend",
                        options=[
                            {"label": "OpenAI", "value": "openai"},
                            {"label": "Anthropic", "value": "anthropic"},
                            {"label": "Google", "value": "google"},
                            {"label": "llama.cpp", "value": "llamacpp"},
                        ],
                        placeholder="Select backend...",
                        style={"width": "100%"},
                    ),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Model Name", style=LABEL_STYLE),
                    dcc.Input(id="runctl-eval-model-name", type="text", placeholder="e.g. gpt-4o-mini", style=_INPUT_STYLE),
                ], style={"flex": "2"}),
                html.Div([
                    html.Label("Base URL (if applicable)", style=LABEL_STYLE),
                    dcc.Input(id="runctl-eval-base-url", type="text", placeholder="http://localhost:8080", style=_INPUT_STYLE),
                ], style={"flex": "2"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Button(
                "Set Evaluator", id="runctl-set-eval-btn", n_clicks=0,
                style={**_BTN_STYLE, "backgroundColor": WONG["purple"], "color": "#000"},
            ),
            html.Div(id="runctl-evaluator-display", style={"marginTop": "8px"}),
        ], style=CARD_STYLE),

        # === Probe Selection ===
        html.Div([
            html.H4("Probe Selection", style={"marginTop": "0", "marginBottom": "12px"}),
            dcc.RadioItems(
                id="runctl-probe-mode",
                options=[
                    {"label": "All probes", "value": "all"},
                    {"label": "By dimension", "value": "dimension"},
                    {"label": "Specific probes", "value": "specific"},
                ],
                value="all",
                inline=True,
                style={"marginBottom": "12px"},
            ),
            html.Div(id="runctl-probe-detail", style={"marginBottom": "8px"}),
        ], style=CARD_STYLE),

        # === Positions ===
        html.Div([
            html.H4("Positions", style={"marginTop": "0", "marginBottom": "12px"}),
            html.Div([
                html.Button(
                    "Quick (3)", id="runctl-pos-quick", n_clicks=0,
                    style={**_BTN_STYLE, "backgroundColor": BG_CONTROLS, "color": TEXT_PRIMARY, "border": f"1px solid {BORDER_COLOR}", "marginRight": "8px"},
                ),
                html.Button(
                    "Standard (13)", id="runctl-pos-standard", n_clicks=0,
                    style={**_BTN_STYLE, "backgroundColor": WONG["blue"], "color": "#000", "marginRight": "8px"},
                ),
                html.Button(
                    "Dense (22)", id="runctl-pos-dense", n_clicks=0,
                    style={**_BTN_STYLE, "backgroundColor": BG_CONTROLS, "color": TEXT_PRIMARY, "border": f"1px solid {BORDER_COLOR}", "marginRight": "8px"},
                ),
            ], style={"marginBottom": "12px"}),
            html.Label("Custom positions (comma-separated decimals 0-1)", style={**LABEL_STYLE, "fontSize": "12px"}),
            dcc.Input(
                id="runctl-positions-input",
                type="text",
                value=", ".join(str(p) for p in POSITIONS_STANDARD),
                style={**_INPUT_STYLE, "width": "100%"},
            ),
        ], style=CARD_STYLE),

        # === Context Lengths ===
        html.Div([
            html.H4("Context Lengths", style={"marginTop": "0", "marginBottom": "12px"}),
            dcc.Checklist(
                id="runctl-ctx-lengths",
                options=[{"label": f" {cl:,}", "value": cl} for cl in COMMON_CTX_LENGTHS],
                value=[4096],
                inline=True,
                style={"marginBottom": "12px"},
            ),
            html.Label("Custom (comma-separated integers)", style={**LABEL_STYLE, "fontSize": "12px"}),
            dcc.Input(
                id="runctl-ctx-custom",
                type="text",
                placeholder="e.g. 6144, 12288",
                style=_INPUT_STYLE,
            ),
        ], style=CARD_STYLE),

        # === Run Parameters ===
        html.Div([
            html.H4("Run Parameters", style={"marginTop": "0", "marginBottom": "12px"}),
            html.Div([
                html.Div([
                    html.Label("Seed", style=LABEL_STYLE),
                    dcc.Input(id="runctl-seed", type="number", value=cfg.run_defaults.seed, style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Temperature", style=LABEL_STYLE),
                    dcc.Input(id="runctl-temperature", type="number", value=cfg.run_defaults.temperature, step=0.1, min=0, style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Repetitions", style=LABEL_STYLE),
                    dcc.Input(id="runctl-repetitions", type="number", value=cfg.run_defaults.repetitions, min=1, style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Filler Type", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="runctl-filler-type",
                        options=[
                            {"label": "Neutral", "value": "neutral"},
                            {"label": "Adversarial", "value": "adversarial"},
                            {"label": "Topical", "value": "topical"},
                        ],
                        value=cfg.run_defaults.filler_type,
                        clearable=False,
                        style={"width": "100%"},
                    ),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=CARD_STYLE),

        # === Work Estimate + Launch ===
        html.Div([
            html.Div(id="runctl-work-estimate", style={"marginBottom": "16px", "fontSize": "15px"}),
            html.Div([
                html.Button(
                    "Launch Run", id="runctl-launch-btn", n_clicks=0,
                    style={
                        "padding": "10px 32px", "cursor": "pointer",
                        "backgroundColor": WONG["green"], "color": "#000",
                        "border": "none", "borderRadius": "6px",
                        "fontWeight": "700", "fontSize": "16px", "marginRight": "16px",
                    },
                ),
                html.Button(
                    "Stop Run", id="runctl-stop-btn", n_clicks=0,
                    style={
                        "padding": "10px 24px", "cursor": "pointer",
                        "backgroundColor": WONG["vermillion"], "color": "#fff",
                        "border": "none", "borderRadius": "6px",
                        "fontWeight": "700", "fontSize": "14px",
                    },
                ),
            ]),
            html.Div(id="runctl-launch-feedback", style={"marginTop": "12px"}),
        ], style=CARD_STYLE),

        # === Active Runs ===
        html.Div(id="runctl-active-runs"),
    ])


def register_callbacks(app, qm, process_manager, dashboard_config=None):
    from apex.dashboard.config import DashboardConfig
    from apex.dashboard.services import model_catalog

    cfg = dashboard_config or DashboardConfig()

    # Populate model file dropdown
    @app.callback(
        Output("runctl-model-file", "options"),
        Input("runctl-interval", "n_intervals"),
    )
    def populate_model_files(_n):
        models = model_catalog.scan_models(cfg.infra.models_dir)
        return [
            {"label": f"{m.filename} ({m.size_gb} GB)", "value": m.path}
            for m in models
        ]

    # Add model to store
    @app.callback(
        Output("runctl-models-store", "data"),
        Input("runctl-add-model-btn", "n_clicks"),
        State("runctl-models-store", "data"),
        State("runctl-model-file", "value"),
        State("runctl-backend", "value"),
        State("runctl-model-name", "value"),
        State("runctl-architecture", "value"),
        State("runctl-parameters", "value"),
        State("runctl-quantization", "value"),
        State("runctl-base-url", "value"),
        State("runctl-max-ctx", "value"),
        prevent_initial_call=True,
    )
    def add_model(n_clicks, current_models, file_path, backend, name, arch, params, quant, base_url, max_ctx):
        if not file_path and not name:
            return current_models

        # Auto-derive name from filename
        if not name and file_path:
            name = file_path.split("/")[-1].replace(".gguf", "")

        model = {
            "name": name or "unnamed",
            "backend": backend or "llamacpp",
            "model_name": name or "unnamed",
            "architecture": arch or "unknown",
            "parameters": params or "unknown",
            "quantization": quant or "none",
            "base_url": base_url or None,
            "max_context_window": int(max_ctx or 8192),
        }
        return current_models + [model]

    # Display added models
    @app.callback(
        Output("runctl-models-display", "children"),
        Input("runctl-models-store", "data"),
    )
    def display_models(models):
        if not models:
            return html.Span("No models added yet.", style={"color": TEXT_MUTED, "fontSize": "13px"})

        chips = []
        for i, m in enumerate(models):
            chips.append(html.Div([
                html.Span(
                    f"{m['name']} ({m['backend']})",
                    style={"fontWeight": "600", "marginRight": "8px"},
                ),
                html.Span(
                    f"{m['architecture']} / {m['parameters']} / {m['quantization']}",
                    style={"fontSize": "12px", "color": TEXT_SECONDARY, "marginRight": "8px"},
                ),
                html.Button(
                    "\u00d7", id={"type": "runctl-remove-model", "index": i},
                    n_clicks=0,
                    style={
                        "padding": "2px 8px", "cursor": "pointer", "fontSize": "14px",
                        "backgroundColor": "transparent", "color": WONG["vermillion"],
                        "border": f"1px solid {WONG['vermillion']}", "borderRadius": "4px",
                    },
                ),
            ], style={
                "display": "flex", "alignItems": "center", "padding": "6px 12px",
                "backgroundColor": BG_CONTROLS, "borderRadius": "6px",
                "border": f"1px solid {BORDER_COLOR}", "marginBottom": "6px",
            }))

        return html.Div(chips)

    # Remove model from store
    @app.callback(
        Output("runctl-models-store", "data", allow_duplicate=True),
        Input({"type": "runctl-remove-model", "index": ALL}, "n_clicks"),
        State("runctl-models-store", "data"),
        prevent_initial_call=True,
    )
    def remove_model(n_clicks_list, current_models):
        from dash import ctx
        if not ctx.triggered_id or not any(n_clicks_list):
            return no_update
        idx = ctx.triggered_id.get("index", -1)
        if 0 <= idx < len(current_models):
            return current_models[:idx] + current_models[idx + 1:]
        return current_models

    # Set evaluator
    @app.callback(
        Output("runctl-evaluator-store", "data"),
        Output("runctl-evaluator-display", "children"),
        Input("runctl-set-eval-btn", "n_clicks"),
        State("runctl-eval-backend", "value"),
        State("runctl-eval-model-name", "value"),
        State("runctl-eval-base-url", "value"),
        prevent_initial_call=True,
    )
    def set_evaluator(n_clicks, backend, model_name, base_url):
        if not backend or not model_name:
            return [], html.Span("Evaluator cleared.", style={"color": TEXT_MUTED, "fontSize": "13px"})

        evaluator = [{
            "name": model_name,
            "backend": backend,
            "model_name": model_name,
            "base_url": base_url or None,
        }]
        display = html.Span(
            f"Evaluator: {model_name} ({backend})",
            style={"color": WONG["purple"], "fontWeight": "600", "fontSize": "13px"},
        )
        return evaluator, display

    # Probe selection detail
    @app.callback(
        Output("runctl-probe-detail", "children"),
        Input("runctl-probe-mode", "value"),
    )
    def probe_detail(mode):
        if mode == "all":
            return html.Span("All available probes will be run.", style={"color": TEXT_SECONDARY, "fontSize": "13px"})
        elif mode == "dimension":
            return dcc.Checklist(
                id="runctl-probe-dimensions",
                options=[
                    {"label": " Factual Recall", "value": "factual_recall"},
                    {"label": " Application", "value": "application"},
                    {"label": " Salience", "value": "salience"},
                ],
                value=["factual_recall", "application", "salience"],
                inline=True,
            )
        elif mode == "specific":
            probe_ids = qm.get_probe_ids()
            return dcc.Dropdown(
                id="runctl-probe-ids",
                options=[{"label": pid, "value": pid} for pid in probe_ids],
                multi=True,
                placeholder="Select specific probes...",
            )
        return html.Div()

    # Position presets
    @app.callback(
        Output("runctl-positions-input", "value"),
        Input("runctl-pos-quick", "n_clicks"),
        Input("runctl-pos-standard", "n_clicks"),
        Input("runctl-pos-dense", "n_clicks"),
        prevent_initial_call=True,
    )
    def set_positions(quick, standard, dense):
        from dash import ctx
        tid = ctx.triggered_id
        if tid == "runctl-pos-quick":
            return ", ".join(str(p) for p in POSITIONS_QUICK)
        elif tid == "runctl-pos-standard":
            return ", ".join(str(p) for p in POSITIONS_STANDARD)
        elif tid == "runctl-pos-dense":
            return ", ".join(str(p) for p in POSITIONS_DENSE)
        return no_update

    # Work estimate
    @app.callback(
        Output("runctl-work-estimate", "children"),
        Input("runctl-models-store", "data"),
        Input("runctl-positions-input", "value"),
        Input("runctl-ctx-lengths", "value"),
        Input("runctl-ctx-custom", "value"),
        Input("runctl-repetitions", "value"),
        Input("runctl-probe-mode", "value"),
    )
    def work_estimate(models, positions_str, ctx_checked, ctx_custom, reps, probe_mode):
        n_models = len(models) if models else 0

        # Parse positions
        positions = _parse_positions(positions_str)
        n_pos = len(positions)

        # Parse context lengths
        ctx_lengths = list(ctx_checked or [])
        if ctx_custom:
            for part in ctx_custom.split(","):
                part = part.strip()
                if part.isdigit():
                    val = int(part)
                    if val not in ctx_lengths:
                        ctx_lengths.append(val)
        n_ctx = len(ctx_lengths)

        reps = int(reps or 1)

        # Probe count estimate (rough)
        n_probes = "all"
        if probe_mode == "all":
            n_probes = "all"
        elif probe_mode == "dimension":
            n_probes = "dimension subset"
        elif probe_mode == "specific":
            n_probes = "selected"

        total = n_models * n_pos * n_ctx * reps
        per_model = n_pos * n_ctx * reps

        if n_models == 0:
            return html.Span("Add at least one model to see work estimate.", style={"color": TEXT_MUTED})

        return html.Div([
            html.Span(f"{n_models} model(s)", style={"fontWeight": "700", "marginRight": "4px"}),
            html.Span(" \u00d7 ", style={"color": TEXT_MUTED}),
            html.Span(f"{n_probes} probes", style={"fontWeight": "700", "marginRight": "4px"}),
            html.Span(" \u00d7 ", style={"color": TEXT_MUTED}),
            html.Span(f"{n_pos} positions", style={"fontWeight": "700", "marginRight": "4px"}),
            html.Span(" \u00d7 ", style={"color": TEXT_MUTED}),
            html.Span(f"{n_ctx} context lengths", style={"fontWeight": "700", "marginRight": "4px"}),
            html.Span(" \u00d7 ", style={"color": TEXT_MUTED}),
            html.Span(f"{reps} reps", style={"fontWeight": "700"}),
            html.Span(
                f" = {per_model:,} executions/model \u00d7 probes",
                style={"color": TEXT_SECONDARY, "marginLeft": "8px"},
            ),
        ])

    # Launch run
    @app.callback(
        Output("runctl-launch-feedback", "children"),
        Input("runctl-launch-btn", "n_clicks"),
        State("runctl-models-store", "data"),
        State("runctl-evaluator-store", "data"),
        State("runctl-probe-mode", "value"),
        State("runctl-positions-input", "value"),
        State("runctl-ctx-lengths", "value"),
        State("runctl-ctx-custom", "value"),
        State("runctl-seed", "value"),
        State("runctl-temperature", "value"),
        State("runctl-repetitions", "value"),
        State("runctl-filler-type", "value"),
        prevent_initial_call=True,
    )
    def launch_run(n_clicks, models, evaluator_models, probe_mode, positions_str,
                   ctx_checked, ctx_custom, seed, temperature, repetitions, filler_type):
        if not models:
            return html.Span("Add at least one model first.", style={"color": WONG["orange"]})

        positions = _parse_positions(positions_str)
        if not positions:
            return html.Span("No valid positions specified.", style={"color": WONG["orange"]})

        ctx_lengths = list(ctx_checked or [])
        if ctx_custom:
            for part in ctx_custom.split(","):
                part = part.strip()
                if part.isdigit():
                    val = int(part)
                    if val not in ctx_lengths:
                        ctx_lengths.append(val)
        if not ctx_lengths:
            return html.Span("Select at least one context length.", style={"color": WONG["orange"]})

        # Build probe_select
        probe_select = "all"
        if probe_mode == "all":
            probe_select = "all"

        # Build config dict matching YAML structure
        config = {
            "run": {
                "seed": int(seed or 42),
                "temperature": float(temperature or 0.0),
                "repetitions": int(repetitions or 1),
                "filler_type": filler_type or "neutral",
            },
            "data": {"directory": "data"},
            "database": {"url": cfg.resolve_database_url()},
            "positions": positions,
            "context_lengths": sorted(ctx_lengths),
            "probes": {"select": probe_select},
            "models": models,
        }
        if evaluator_models:
            config["evaluator_models"] = evaluator_models

        # --- Pre-flight checks ---
        preflight = process_manager.preflight_check(config)

        if not preflight.ok:
            items = []
            for err in preflight.errors:
                items.append(html.Div([
                    html.Span("\u2717 ", style={"color": WONG["vermillion"], "fontWeight": "700"}),
                    html.Span(err),
                ], style={"marginBottom": "6px", "fontSize": "13px"}))
            for warn in preflight.warnings:
                items.append(html.Div([
                    html.Span("\u26a0 ", style={"color": WONG["orange"], "fontWeight": "700"}),
                    html.Span(warn, style={"color": TEXT_SECONDARY}),
                ], style={"marginBottom": "4px", "fontSize": "12px"}))

            return html.Div([
                html.Div(
                    "Pre-flight check failed — fix these before launching:",
                    style={"fontWeight": "700", "color": WONG["vermillion"], "marginBottom": "10px"},
                ),
                *items,
            ], style={
                "backgroundColor": BG_CONTROLS, "padding": "12px 16px",
                "borderRadius": "6px", "border": f"1px solid {WONG['vermillion']}",
            })

        # --- Warnings only (still launch) ---
        warning_items = []
        for warn in preflight.warnings:
            warning_items.append(html.Div([
                html.Span("\u26a0 ", style={"color": WONG["orange"]}),
                html.Span(warn, style={"color": TEXT_SECONDARY, "fontSize": "12px"}),
            ], style={"marginBottom": "2px"}))

        # --- Launch ---
        try:
            run_id, info = process_manager.start_run(config)
        except Exception as e:
            return html.Span(f"Launch failed: {e}", style={"color": WONG["vermillion"]})

        if info.status == "crashed":
            # Process died during startup
            error_lines = info.error_output or "No output captured"
            # Trim to last meaningful lines
            lines = error_lines.strip().splitlines()
            display_lines = lines[-15:] if len(lines) > 15 else lines

            return html.Div([
                html.Div(
                    f"Run {run_id} crashed immediately after launch (PID {info.pid}):",
                    style={"fontWeight": "700", "color": WONG["vermillion"], "marginBottom": "8px"},
                ),
                html.Pre(
                    "\n".join(display_lines),
                    style={
                        "whiteSpace": "pre-wrap", "fontSize": "12px",
                        "backgroundColor": "#1a1a30", "color": "#d0d0e0",
                        "padding": "10px", "borderRadius": "4px",
                        "border": f"1px solid {BORDER_COLOR}",
                        "maxHeight": "300px", "overflowY": "auto",
                    },
                ),
            ])

        # Success
        result = [
            html.Span(
                f"Launched: {run_id} (PID {info.pid})",
                style={"color": WONG["green"], "fontWeight": "600"},
            ),
        ]
        if warning_items:
            result.append(html.Div(warning_items, style={"marginTop": "8px"}))

        return html.Div(result)

    # Stop run
    @app.callback(
        Output("runctl-launch-feedback", "children", allow_duplicate=True),
        Input("runctl-stop-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def stop_latest_run(n_clicks):
        runs = process_manager.get_runs()
        active = [r for r in runs if r.status == "running"]
        if not active:
            return html.Span("No active run to stop.", style={"color": TEXT_MUTED})

        # Stop the most recent active run
        run = active[0]
        ok = process_manager.stop_run(run.run_id)
        if ok:
            return html.Span(f"Stopped {run.run_id} (PID {run.pid})", style={"color": WONG["orange"]})
        return html.Span(f"Failed to stop {run.run_id}", style={"color": WONG["vermillion"]})

    # Active runs display
    @app.callback(
        Output("runctl-active-runs", "children"),
        Input("runctl-interval", "n_intervals"),
    )
    def display_active_runs(_n):
        runs = process_manager.get_runs()
        if not runs:
            return html.Div()

        STATUS_COLORS = {
            "running": WONG["green"],
            "stopped": WONG["orange"],
            "finished": WONG["blue"],
            "crashed": WONG["vermillion"],
        }

        items = []
        for r in runs[:10]:
            color = STATUS_COLORS.get(r.status, TEXT_MUTED)
            row_children = [
                html.Div([
                    html.Span("\u25cf ", style={"color": color}),
                    html.Span(r.run_id, style={"fontWeight": "600", "marginRight": "12px"}),
                    html.Span(f"PID {r.pid}", style={"color": TEXT_MUTED, "fontSize": "12px", "marginRight": "12px"}),
                    html.Span(r.status, style={"color": color, "fontSize": "12px", "fontWeight": "600"}),
                ]),
                html.Div([
                    html.Span(r.config_summary, style={"fontSize": "12px", "color": TEXT_SECONDARY}),
                    html.Span(
                        f" | Started: {r.start_time.strftime('%H:%M:%S')}",
                        style={"fontSize": "12px", "color": TEXT_MUTED},
                    ),
                ], style={"marginTop": "4px"}),
            ]
            if r.status == "crashed" and r.error_output:
                # Show last few lines of error
                err_lines = r.error_output.strip().splitlines()[-5:]
                row_children.append(html.Pre(
                    "\n".join(err_lines),
                    style={
                        "whiteSpace": "pre-wrap", "fontSize": "11px",
                        "backgroundColor": "#1a1a30", "color": "#d0d0e0",
                        "padding": "6px", "borderRadius": "4px",
                        "border": f"1px solid {BORDER_COLOR}",
                        "marginTop": "6px", "maxHeight": "100px", "overflowY": "auto",
                    },
                ))
            items.append(html.Div(
                row_children,
                style={"marginBottom": "10px", "paddingBottom": "8px", "borderBottom": f"1px solid {BORDER_COLOR}"},
            ))

        return html.Div([
            html.H4("Run History", style={"marginTop": "0", "marginBottom": "12px"}),
            *items,
        ], style=CARD_STYLE)

    # Sync form defaults when settings are saved
    @app.callback(
        Output("runctl-seed", "value"),
        Output("runctl-temperature", "value"),
        Output("runctl-repetitions", "value"),
        Output("runctl-filler-type", "value"),
        Input("config-version", "data"),
        prevent_initial_call=True,
    )
    def sync_run_defaults(_version):
        return (
            cfg.run_defaults.seed,
            cfg.run_defaults.temperature,
            cfg.run_defaults.repetitions,
            cfg.run_defaults.filler_type,
        )


def _parse_positions(positions_str: str | None) -> list[float]:
    """Parse comma-separated position values."""
    if not positions_str:
        return []
    positions = []
    for part in positions_str.split(","):
        part = part.strip()
        try:
            val = float(part)
            if 0.0 < val < 1.0:
                positions.append(val)
        except ValueError:
            continue
    return sorted(set(positions))
