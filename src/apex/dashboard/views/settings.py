"""Settings — dashboard configuration with persistence, validation, and auto-detection."""

from __future__ import annotations

from pathlib import Path

from dash import html, dcc, Input, Output, State, no_update

from apex.dashboard.styles import (
    CARD_STYLE, CONTROLS_STYLE, LABEL_STYLE, BG_CARD, BG_CONTROLS, BG_PLOT,
    BORDER_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, WONG,
)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / "configs" / "dashboard.yaml"

_INPUT_STYLE = {
    "width": "100%", "padding": "6px", "backgroundColor": BG_PLOT,
    "color": TEXT_PRIMARY, "border": f"1px solid {BORDER_COLOR}", "borderRadius": "4px",
}

_BTN_STYLE = {
    "padding": "6px 14px", "cursor": "pointer", "border": "none",
    "borderRadius": "4px", "fontSize": "12px", "fontWeight": "600",
}

_TEST_BTN_STYLE = {
    **_BTN_STYLE,
    "backgroundColor": BG_CONTROLS,
    "color": TEXT_PRIMARY,
    "border": f"1px solid {BORDER_COLOR}",
    "marginLeft": "8px",
    "padding": "6px 12px",
}

_STATUS_OK = {"color": WONG["green"], "marginLeft": "8px", "fontSize": "13px", "fontWeight": "600"}
_STATUS_ERR = {"color": WONG["vermillion"], "marginLeft": "8px", "fontSize": "13px"}


def layout(dashboard_config=None) -> html.Div:
    from apex.dashboard.config import DashboardConfig
    cfg = dashboard_config or DashboardConfig()

    # Build node rows from config (skip first local node — always present)
    remote_nodes = [n for n in cfg.nodes if n.host != "local"]

    return html.Div([
        html.H3("Settings"),

        # === Infrastructure Paths ===
        html.Div([
            html.H4("Infrastructure Paths", style={"marginTop": "0", "marginBottom": "12px"}),

            # llama-server binary
            html.Div([
                html.Label("llama-server Binary", style=LABEL_STYLE),
                html.Div([
                    dcc.Input(
                        id="settings-llama-bin", type="text",
                        value=cfg.infra.llama_server_bin,
                        placeholder="/usr/local/bin/llama-server",
                        style={**_INPUT_STYLE, "flex": "1"},
                    ),
                    html.Button("Test", id="settings-test-llama-btn", n_clicks=0, style=_TEST_BTN_STYLE),
                    html.Span(id="settings-llama-status"),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
            ], style={"marginBottom": "12px"}),

            # Models directory
            html.Div([
                html.Label("Models Directory", style=LABEL_STYLE),
                html.Div([
                    dcc.Input(
                        id="settings-models-dir", type="text",
                        value=cfg.infra.models_dir,
                        placeholder="/path/to/models/",
                        style={**_INPUT_STYLE, "flex": "1"},
                    ),
                    html.Button("Test", id="settings-test-models-btn", n_clicks=0, style=_TEST_BTN_STYLE),
                    html.Span(id="settings-models-status"),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
            ]),
        ], style=CARD_STYLE),

        # === Nodes ===
        html.Div([
            html.H4("Nodes", style={"marginTop": "0", "marginBottom": "12px"}),

            # Local node (always present, not editable)
            html.Div([
                html.Span("Node 1 (local)", style={"fontWeight": "600", "marginRight": "12px"}),
                html.Span("always enabled", style={"color": TEXT_MUTED, "fontSize": "12px"}),
            ], style={"marginBottom": "12px"}),

            # Remote node
            html.Div([
                html.Label("Remote Node Host", style=LABEL_STYLE),
                html.Div([
                    dcc.Input(
                        id="settings-node2-host", type="text",
                        value=remote_nodes[0].host if remote_nodes else "",
                        placeholder="e.g. 192.168.1.101",
                        style={**_INPUT_STYLE, "flex": "1"},
                    ),
                    dcc.Input(
                        id="settings-node2-label", type="text",
                        value=remote_nodes[0].label if remote_nodes else "Node 2 (remote)",
                        placeholder="Display label",
                        style={**_INPUT_STYLE, "flex": "1"},
                    ),
                    dcc.Checklist(
                        id="settings-node2-enabled",
                        options=[{"label": " Enabled", "value": "on"}],
                        value=["on"] if (remote_nodes and remote_nodes[0].enabled) else [],
                        style={"marginLeft": "8px"},
                    ),
                    html.Button("Test SSH", id="settings-test-ssh-btn", n_clicks=0, style=_TEST_BTN_STYLE),
                    html.Span(id="settings-ssh-status"),
                ], style={"display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap"}),
            ]),
        ], style=CARD_STYLE),

        # === Database ===
        html.Div([
            html.H4("Database", style={"marginTop": "0", "marginBottom": "12px"}),
            html.Div([
                dcc.Input(
                    id="settings-db-url", type="text",
                    value=cfg.database.url,
                    placeholder="postgresql://user:pass@host:5432/dbname",
                    style={**_INPUT_STYLE, "flex": "1"},
                ),
                html.Button("Test Connection", id="settings-test-db-btn", n_clicks=0, style=_TEST_BTN_STYLE),
                html.Span(id="settings-db-status"),
            ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
            html.P(
                "APEX_DATABASE_URL env var overrides this setting at runtime.",
                style={"fontSize": "11px", "color": TEXT_MUTED, "marginTop": "6px", "marginBottom": "0"},
            ),
        ], style=CARD_STYLE),

        # === Backend Defaults ===
        html.Div([
            html.H4("Backend Defaults", style={"marginTop": "0", "marginBottom": "12px"}),
            html.Div([
                html.Div([
                    html.Label("llama.cpp URL", style=LABEL_STYLE),
                    dcc.Input(
                        id="settings-backend-llamacpp", type="text",
                        value=cfg.backend_defaults.llamacpp,
                        style=_INPUT_STYLE,
                    ),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Ollama URL", style=LABEL_STYLE),
                    dcc.Input(
                        id="settings-backend-ollama", type="text",
                        value=cfg.backend_defaults.ollama,
                        style=_INPUT_STYLE,
                    ),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("SGLang URL", style=LABEL_STYLE),
                    dcc.Input(
                        id="settings-backend-sglang", type="text",
                        value=cfg.backend_defaults.sglang,
                        style=_INPUT_STYLE,
                    ),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=CARD_STYLE),

        # === Server Defaults ===
        html.Div([
            html.H4("Server Defaults", style={"marginTop": "0", "marginBottom": "12px"}),
            html.Div([
                html.Div([
                    html.Label("Port", style=LABEL_STYLE),
                    dcc.Input(id="settings-srv-port", type="number", value=cfg.server_defaults.port, style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Context Size", style=LABEL_STYLE),
                    dcc.Input(id="settings-srv-ctx-size", type="number", value=cfg.server_defaults.ctx_size, style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("GPU Layers", style=LABEL_STYLE),
                    dcc.Input(id="settings-srv-gpu-layers", type="number", value=cfg.server_defaults.gpu_layers, style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Parallel", style=LABEL_STYLE),
                    dcc.Input(id="settings-srv-parallel", type="number", value=cfg.server_defaults.parallel, style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Threads", style=LABEL_STYLE),
                    dcc.Input(id="settings-srv-threads", type="number", value=cfg.server_defaults.threads, style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Flash Attn", style=LABEL_STYLE),
                    dcc.Checklist(
                        id="settings-srv-flash-attn",
                        options=[{"label": " On", "value": "on"}],
                        value=["on"] if cfg.server_defaults.flash_attn else [],
                    ),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=CARD_STYLE),

        # === Run Defaults ===
        html.Div([
            html.H4("Run Defaults", style={"marginTop": "0", "marginBottom": "12px"}),
            html.Div([
                html.Div([
                    html.Label("Seed", style=LABEL_STYLE),
                    dcc.Input(id="settings-run-seed", type="number", value=cfg.run_defaults.seed, style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Temperature", style=LABEL_STYLE),
                    dcc.Input(id="settings-run-temperature", type="number", value=cfg.run_defaults.temperature, step=0.1, style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Repetitions", style=LABEL_STYLE),
                    dcc.Input(id="settings-run-repetitions", type="number", value=cfg.run_defaults.repetitions, min=1, style=_INPUT_STYLE),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Filler Type", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="settings-run-filler-type",
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

        # === Actions ===
        html.Div([
            html.Div([
                html.Button(
                    "Auto-Detect", id="settings-autodetect-btn", n_clicks=0,
                    style={**_BTN_STYLE, "backgroundColor": WONG["blue"], "color": "#000", "marginRight": "12px",
                           "padding": "8px 20px", "fontSize": "14px"},
                ),
                html.Button(
                    "Save", id="settings-save-btn", n_clicks=0,
                    style={**_BTN_STYLE, "backgroundColor": WONG["green"], "color": "#000",
                           "padding": "8px 24px", "fontSize": "14px"},
                ),
            ], style={"marginBottom": "12px"}),
            html.Div(id="settings-feedback", style={"fontSize": "13px"}),
        ], style=CARD_STYLE),
    ])


def register_callbacks(app, dashboard_config=None):
    from apex.dashboard.config import (
        DashboardConfig, InfraConfig, NodeConfig, DatabaseConfig,
        BackendDefaults, ServerDefaults, RunDefaults, auto_detect,
    )
    from apex.dashboard.services import infra

    # Test llama-server binary
    @app.callback(
        Output("settings-llama-status", "children"),
        Input("settings-test-llama-btn", "n_clicks"),
        State("settings-llama-bin", "value"),
        prevent_initial_call=True,
    )
    def test_llama_bin(n_clicks, path):
        if not path:
            return html.Span("No path specified", style=_STATUS_ERR)

        p = Path(path)
        if p.is_dir():
            return html.Span(
                "Path is a directory \u2014 provide the full path to the llama-server binary",
                style=_STATUS_ERR,
            )
        if not p.is_file():
            return html.Span("File not found", style=_STATUS_ERR)

        import os
        if not os.access(path, os.X_OK):
            return html.Span("Not executable", style=_STATUS_ERR)

        return html.Span("OK", style=_STATUS_OK)

    # Test models directory
    @app.callback(
        Output("settings-models-status", "children"),
        Input("settings-test-models-btn", "n_clicks"),
        State("settings-models-dir", "value"),
        prevent_initial_call=True,
    )
    def test_models_dir(n_clicks, path):
        if not path:
            return html.Span("No path specified", style=_STATUS_ERR)

        p = Path(path)
        if not p.is_dir():
            return html.Span("Directory not found", style=_STATUS_ERR)

        gguf_count = sum(1 for _ in p.rglob("*.gguf"))
        if gguf_count == 0:
            return html.Span("No .gguf files found", style=_STATUS_ERR)

        return html.Span(f"OK ({gguf_count} models)", style=_STATUS_OK)

    # Test SSH connection
    @app.callback(
        Output("settings-ssh-status", "children"),
        Input("settings-test-ssh-btn", "n_clicks"),
        State("settings-node2-host", "value"),
        prevent_initial_call=True,
    )
    def test_ssh(n_clicks, host):
        if not host:
            return html.Span("No host specified", style=_STATUS_ERR)

        ok = infra.check_node_reachable(host)
        if ok:
            return html.Span("OK", style=_STATUS_OK)
        return html.Span("Unreachable", style=_STATUS_ERR)

    # Test database connection
    @app.callback(
        Output("settings-db-status", "children"),
        Input("settings-test-db-btn", "n_clicks"),
        State("settings-db-url", "value"),
        prevent_initial_call=True,
    )
    def test_database(n_clicks, url):
        if not url:
            return html.Span("No URL specified", style=_STATUS_ERR)

        if url.startswith(("postgresql://", "postgres://")):
            try:
                import psycopg
                conn = psycopg.connect(url, connect_timeout=5, autocommit=True)
                conn.execute("SELECT 1")
                conn.close()
                return html.Span("OK", style=_STATUS_OK)
            except Exception as e:
                msg = str(e).split("\n")[0][:80]
                return html.Span(f"Failed: {msg}", style=_STATUS_ERR)
        else:
            # SQLite path
            import sqlite3
            try:
                p = Path(url)
                if not p.is_file():
                    return html.Span(f"File not found: {url}", style=_STATUS_ERR)
                conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
                conn.execute("SELECT name FROM sqlite_master LIMIT 1")
                conn.close()
                return html.Span("OK (SQLite)", style=_STATUS_OK)
            except Exception as e:
                msg = str(e).split("\n")[0][:80]
                return html.Span(f"Failed: {msg}", style=_STATUS_ERR)

    # Auto-detect
    @app.callback(
        Output("settings-llama-bin", "value"),
        Output("settings-models-dir", "value"),
        Output("settings-db-url", "value"),
        Output("settings-feedback", "children", allow_duplicate=True),
        Input("settings-autodetect-btn", "n_clicks"),
        State("settings-llama-bin", "value"),
        State("settings-models-dir", "value"),
        State("settings-db-url", "value"),
        prevent_initial_call=True,
    )
    def run_autodetect(n_clicks, cur_llama, cur_models, cur_db):
        results = auto_detect()

        llama_val = cur_llama
        models_val = cur_models
        db_val = cur_db

        report = []
        for r in results:
            if r.field == "infra.llama_server_bin":
                llama_val = r.value
                report.append(f"llama-server: {r.source}")
            elif r.field == "infra.models_dir":
                models_val = r.value
                report.append(f"Models dir: {r.source}")
            elif r.field == "database.url":
                db_val = r.value
                report.append(f"Database: {r.source}")
            elif r.field == "gpu.name":
                report.append(f"GPU: {r.value}")

        if report:
            feedback = html.Div([
                html.Span("Auto-detect results: ", style={"fontWeight": "600", "color": WONG["blue"]}),
                html.Ul([html.Li(r, style={"fontSize": "12px", "color": TEXT_SECONDARY}) for r in report],
                        style={"margin": "4px 0 0 16px", "padding": "0"}),
            ])
        else:
            feedback = html.Span(
                "Nothing detected. Configure paths manually.",
                style={"color": WONG["orange"]},
            )

        return llama_val, models_val, db_val, feedback

    # Save config
    @app.callback(
        Output("settings-feedback", "children"),
        Output("config-version", "data"),
        Input("settings-save-btn", "n_clicks"),
        State("config-version", "data"),
        State("settings-llama-bin", "value"),
        State("settings-models-dir", "value"),
        State("settings-node2-host", "value"),
        State("settings-node2-label", "value"),
        State("settings-node2-enabled", "value"),
        State("settings-db-url", "value"),
        State("settings-backend-llamacpp", "value"),
        State("settings-backend-ollama", "value"),
        State("settings-backend-sglang", "value"),
        State("settings-srv-port", "value"),
        State("settings-srv-ctx-size", "value"),
        State("settings-srv-gpu-layers", "value"),
        State("settings-srv-parallel", "value"),
        State("settings-srv-threads", "value"),
        State("settings-srv-flash-attn", "value"),
        State("settings-run-seed", "value"),
        State("settings-run-temperature", "value"),
        State("settings-run-repetitions", "value"),
        State("settings-run-filler-type", "value"),
        prevent_initial_call=True,
    )
    def save_config(
        n_clicks, current_version,
        llama_bin, models_dir,
        node2_host, node2_label, node2_enabled,
        db_url,
        be_llamacpp, be_ollama, be_sglang,
        srv_port, srv_ctx, srv_gpu, srv_parallel, srv_threads, srv_flash,
        run_seed, run_temp, run_reps, run_filler,
    ):
        try:
            existing = DashboardConfig.load(_CONFIG_PATH)
        except Exception:
            existing = DashboardConfig()

        nodes = [
            NodeConfig(name="node1", host="local", label="Node 1 (local)", enabled=True),
        ]
        if node2_host:
            nodes.append(NodeConfig(
                name="node2",
                host=node2_host,
                label=node2_label or "Node 2 (remote)",
                enabled=bool(node2_enabled and "on" in node2_enabled),
            ))

        existing.infra = InfraConfig(
            llama_server_bin=llama_bin or existing.infra.llama_server_bin,
            models_dir=models_dir or existing.infra.models_dir,
        )
        existing.nodes = nodes
        existing.database = DatabaseConfig(url=db_url or existing.database.url)
        existing.backend_defaults = BackendDefaults(
            llamacpp=be_llamacpp or existing.backend_defaults.llamacpp,
            ollama=be_ollama or existing.backend_defaults.ollama,
            sglang=be_sglang or existing.backend_defaults.sglang,
        )
        existing.server_defaults = ServerDefaults(
            port=int(srv_port or existing.server_defaults.port),
            ctx_size=int(srv_ctx or existing.server_defaults.ctx_size),
            gpu_layers=int(srv_gpu or existing.server_defaults.gpu_layers),
            parallel=int(srv_parallel or existing.server_defaults.parallel),
            flash_attn=bool(srv_flash and "on" in srv_flash),
            threads=int(srv_threads or existing.server_defaults.threads),
        )
        existing.run_defaults = RunDefaults(
            seed=int(run_seed or existing.run_defaults.seed),
            temperature=float(run_temp if run_temp is not None else existing.run_defaults.temperature),
            repetitions=int(run_reps or existing.run_defaults.repetitions),
            filler_type=run_filler or existing.run_defaults.filler_type,
        )

        try:
            existing.save(_CONFIG_PATH)

            # Update the in-memory config so other tabs pick up changes on next render
            if dashboard_config is not None:
                dashboard_config.infra = existing.infra
                dashboard_config.nodes = existing.nodes
                dashboard_config.database = existing.database
                dashboard_config.backend_defaults = existing.backend_defaults
                dashboard_config.server_defaults = existing.server_defaults
                dashboard_config.run_defaults = existing.run_defaults

            return (
                html.Span(
                    f"Saved to {_CONFIG_PATH}",
                    style={"color": WONG["green"], "fontWeight": "600"},
                ),
                (current_version or 0) + 1,
            )
        except Exception as e:
            return (
                html.Span(
                    f"Save failed: {e}",
                    style={"color": WONG["vermillion"]},
                ),
                current_version or 0,
            )
