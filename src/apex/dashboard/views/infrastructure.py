"""Infrastructure — manage llama-server instances and monitor GPU."""

from __future__ import annotations

from dash import html, dcc, Input, Output, State, callback_context, no_update, ALL

from apex.dashboard.styles import (
    CARD_STYLE, CONTROLS_STYLE, GUIDE_STYLE, LABEL_STYLE, BG_CARD, BG_CONTROLS, BG_PLOT,
    BORDER_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, WONG,
)


def layout(dashboard_config=None) -> html.Div:
    from apex.dashboard.config import DashboardConfig
    cfg = dashboard_config or DashboardConfig()
    return html.Div([
        html.H3("Infrastructure"),
        html.Details([
            html.Summary("How to read this", style={"cursor": "pointer", "fontWeight": "600"}),
            html.P(
                "Manage llama-server instances and monitor GPU resources. "
                "Server health is checked via /health endpoint. GPU stats from nvidia-smi.",
                style={"margin": "8px 0 0 0"},
            ),
        ], style=GUIDE_STYLE),

        dcc.Interval(id="infra-interval", interval=10_000, n_intervals=0),

        # GPU Monitor
        html.Div(id="infra-gpu-stats"),

        # Server status cards
        html.Div(id="infra-server-cards"),

        # Launch server form
        html.Div([
            html.H4("Launch Server", style={"marginTop": "0", "marginBottom": "12px"}),
            html.Div([
                # Row 1: Node + Model + Port
                html.Div([
                    html.Div([
                        html.Label("Node", style=LABEL_STYLE),
                        dcc.Dropdown(
                            id="infra-node-select",
                            options=[
                                {"label": n.label or n.name, "value": n.name}
                                for n in cfg.nodes if n.enabled
                            ],
                            value=cfg.nodes[0].name if cfg.nodes else "node1",
                            clearable=False,
                            style={"width": "100%"},
                        ),
                    ], style={"flex": "1", "minWidth": "150px"}),
                    html.Div([
                        html.Label("Model File", style=LABEL_STYLE),
                        dcc.Dropdown(
                            id="infra-model-select",
                            options=[],
                            placeholder="Select a model...",
                            style={"width": "100%"},
                        ),
                    ], style={"flex": "3", "minWidth": "300px"}),
                    html.Div([
                        html.Label("Port", style=LABEL_STYLE),
                        dcc.Input(
                            id="infra-port", type="number", value=cfg.server_defaults.port,
                            min=1024, max=65535,
                            style={"width": "100%", "padding": "6px", "backgroundColor": BG_PLOT,
                                   "color": TEXT_PRIMARY, "border": f"1px solid {BORDER_COLOR}",
                                   "borderRadius": "4px"},
                        ),
                    ], style={"flex": "1", "minWidth": "100px"}),
                ], style={"display": "flex", "gap": "12px", "marginBottom": "12px", "flexWrap": "wrap"}),

                # Row 2: Context size, GPU layers, Parallel, Threads
                html.Div([
                    html.Div([
                        html.Label("Context Size", style=LABEL_STYLE),
                        dcc.Input(
                            id="infra-ctx-size", type="number", value=cfg.server_defaults.ctx_size,
                            min=512, max=131072, step=512,
                            style={"width": "100%", "padding": "6px", "backgroundColor": BG_PLOT,
                                   "color": TEXT_PRIMARY, "border": f"1px solid {BORDER_COLOR}",
                                   "borderRadius": "4px"},
                        ),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("GPU Layers", style=LABEL_STYLE),
                        dcc.Input(
                            id="infra-gpu-layers", type="number", value=cfg.server_defaults.gpu_layers,
                            min=0, max=999,
                            style={"width": "100%", "padding": "6px", "backgroundColor": BG_PLOT,
                                   "color": TEXT_PRIMARY, "border": f"1px solid {BORDER_COLOR}",
                                   "borderRadius": "4px"},
                        ),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Parallel Slots", style=LABEL_STYLE),
                        dcc.Input(
                            id="infra-parallel", type="number", value=cfg.server_defaults.parallel,
                            min=1, max=16,
                            style={"width": "100%", "padding": "6px", "backgroundColor": BG_PLOT,
                                   "color": TEXT_PRIMARY, "border": f"1px solid {BORDER_COLOR}",
                                   "borderRadius": "4px"},
                        ),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Threads", style=LABEL_STYLE),
                        dcc.Input(
                            id="infra-threads", type="number",
                            value=cfg.server_defaults.threads or 0,
                            min=0, max=128, placeholder="auto",
                            style={"width": "100%", "padding": "6px", "backgroundColor": BG_PLOT,
                                   "color": TEXT_PRIMARY, "border": f"1px solid {BORDER_COLOR}",
                                   "borderRadius": "4px"},
                        ),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Flash Attention", style=LABEL_STYLE),
                        dcc.Checklist(
                            id="infra-flash-attn",
                            options=[{"label": " Enabled", "value": "on"}],
                            value=["on"] if cfg.server_defaults.flash_attn else [],
                            style={"paddingTop": "4px"},
                        ),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "12px", "marginBottom": "16px", "flexWrap": "wrap"}),

                html.Button(
                    "Start Server", id="infra-start-btn", n_clicks=0,
                    style={
                        "padding": "8px 24px", "cursor": "pointer",
                        "backgroundColor": WONG["green"], "color": "#000",
                        "border": "none", "borderRadius": "4px",
                        "fontWeight": "700", "fontSize": "14px",
                    },
                ),
                html.Div(id="infra-start-feedback", style={"marginTop": "8px", "fontSize": "13px"}),
            ]),
        ], style=CARD_STYLE),

        # Stop server area
        html.Div(id="infra-stop-feedback", style={"marginTop": "8px", "fontSize": "13px"}),

        # Node health
        html.Div(id="infra-node-health"),
    ])


def register_callbacks(app, qm, dashboard_config=None):
    from apex.dashboard.config import DashboardConfig
    from apex.dashboard.services import infra, model_catalog

    cfg = dashboard_config or DashboardConfig()

    # Build a lookup for node host by node name
    _node_hosts = {n.name: n.host for n in cfg.nodes}

    def _resolve_host(node_name: str) -> str:
        """Get SSH host for a node name. Returns 'local' for local nodes."""
        return _node_hosts.get(node_name, node_name)

    # Populate model dropdown on load
    @app.callback(
        Output("infra-model-select", "options"),
        Input("infra-interval", "n_intervals"),
    )
    def populate_models(_n):
        # Only scan once (cached)
        models = model_catalog.scan_models(cfg.infra.models_dir)
        return [
            {"label": f"{m.filename} ({m.size_gb} GB) — {m.parent_dir}", "value": m.path}
            for m in models
        ]

    # GPU stats
    @app.callback(
        Output("infra-gpu-stats", "children"),
        Input("infra-interval", "n_intervals"),
    )
    def update_gpu_stats(_n):
        stats = infra.get_gpu_stats("local")
        if not stats:
            return html.Div(
                "GPU stats unavailable (nvidia-smi not found or failed)",
                style={**CARD_STYLE, "color": TEXT_MUTED},
            )

        vram_pct = round(stats.vram_used_mb / max(stats.vram_total_mb, 1) * 100, 1)
        vram_color = WONG["green"] if vram_pct < 70 else (WONG["orange"] if vram_pct < 90 else WONG["vermillion"])
        temp_color = WONG["green"] if stats.temperature_c < 70 else (WONG["orange"] if stats.temperature_c < 85 else WONG["vermillion"])

        proc_items = []
        for p in stats.processes:
            proc_items.append(html.Div(
                f"PID {p['pid']}: {p['name']} — {p['vram_mb']} MB",
                style={"fontSize": "12px", "color": TEXT_SECONDARY, "marginLeft": "16px"},
            ))

        return html.Div([
            html.H4("GPU Status — Node 1", style={"marginTop": "0", "marginBottom": "12px"}),
            html.Div([
                # GPU name and utilization
                html.Div([
                    html.Span(stats.name, style={"fontWeight": "700", "marginRight": "16px"}),
                    html.Span(
                        f"{stats.utilization_pct}% util",
                        style={"color": WONG["blue"], "marginRight": "16px"},
                    ),
                    html.Span(
                        f"{stats.temperature_c}\u00b0C",
                        style={"color": temp_color, "marginRight": "16px"},
                    ),
                ], style={"marginBottom": "10px", "fontSize": "14px"}),

                # VRAM bar
                html.Div([
                    html.Span("VRAM: ", style={"fontSize": "13px", "marginRight": "8px"}),
                    html.Div([
                        html.Div(style={
                            "width": f"{vram_pct}%", "height": "100%",
                            "backgroundColor": vram_color, "borderRadius": "4px",
                        }),
                    ], style={
                        "height": "16px", "backgroundColor": BG_PLOT,
                        "borderRadius": "4px", "flex": "1",
                    }),
                    html.Span(
                        f"{stats.vram_used_mb}/{stats.vram_total_mb} MB ({vram_pct}%)",
                        style={"fontSize": "12px", "color": TEXT_SECONDARY, "marginLeft": "8px", "whiteSpace": "nowrap"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),

                # GPU processes
                *proc_items,
            ]),
        ], style=CARD_STYLE)

    # Server status cards
    @app.callback(
        Output("infra-server-cards", "children"),
        Input("infra-interval", "n_intervals"),
    )
    def update_server_cards(_n):
        all_servers = []
        for node in cfg.nodes:
            if not node.enabled:
                continue
            if node.host == "local":
                all_servers.extend(infra.get_running_servers("local"))
            else:
                all_servers.extend(infra.get_running_servers(
                    node.name, remote_host=node.host,
                ))

        if not all_servers:
            return html.Div(
                "No llama-server processes detected.",
                style={**CARD_STYLE, "color": TEXT_MUTED},
            )

        cards = []
        for srv in all_servers:
            # Health check
            host = "localhost" if _resolve_host(srv.node) == "local" else _resolve_host(srv.node)
            hc = infra.health_check(f"http://{host}:{srv.port}")
            srv.status = hc["status"]

            status_color = WONG["green"] if srv.status in ("ok", "healthy") else WONG["vermillion"]
            model_name = srv.model_path.split("/")[-1] if "/" in srv.model_path else srv.model_path

            cards.append(html.Div([
                html.Div([
                    html.Span(f"\u25cf", style={"color": status_color, "marginRight": "8px", "fontSize": "16px"}),
                    html.Span(f"{srv.node}:{srv.port}", style={"fontWeight": "700", "marginRight": "16px"}),
                    html.Span(f"PID {srv.pid}", style={"color": TEXT_MUTED, "fontSize": "12px"}),
                ], style={"marginBottom": "6px"}),
                html.Div([
                    html.Span(f"Model: {model_name}", style={"fontSize": "13px", "color": TEXT_SECONDARY}),
                    html.Span(
                        f" | Status: {srv.status} | Slots: {hc.get('slots', '?')}",
                        style={"fontSize": "13px", "color": TEXT_SECONDARY},
                    ),
                ], style={"marginBottom": "8px"}),
                html.Button(
                    "Stop", id={"type": "infra-stop-btn", "index": f"{srv.node}:{srv.pid}"},
                    n_clicks=0,
                    style={
                        "padding": "4px 16px", "cursor": "pointer",
                        "backgroundColor": WONG["vermillion"], "color": "#fff",
                        "border": "none", "borderRadius": "4px", "fontSize": "12px",
                    },
                ),
            ], style={**CARD_STYLE, "padding": "12px 16px"}))

        return html.Div([
            html.H4("Running Servers", style={"marginTop": "0", "marginBottom": "12px"}),
            *cards,
        ])

    # Stop server
    @app.callback(
        Output("infra-stop-feedback", "children"),
        Input({"type": "infra-stop-btn", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def stop_server(n_clicks_list):
        from dash import ctx
        if not ctx.triggered_id or not any(n_clicks_list):
            return no_update

        index = ctx.triggered_id.get("index", "")
        parts = index.split(":")
        if len(parts) != 2:
            return "Invalid server identifier."

        node, pid_str = parts
        try:
            pid = int(pid_str)
        except ValueError:
            return "Invalid PID."

        node_host = _resolve_host(node)
        node_arg = "local" if node_host == "local" else node
        ok = infra.stop_server(pid, node_arg, remote_host=node_host if node_host != "local" else "")
        if ok:
            return html.Span(f"Sent stop signal to PID {pid} on {node}.", style={"color": WONG["green"]})
        return html.Span(f"Failed to stop PID {pid}.", style={"color": WONG["vermillion"]})

    # Start server
    @app.callback(
        Output("infra-start-feedback", "children"),
        Input("infra-start-btn", "n_clicks"),
        State("infra-node-select", "value"),
        State("infra-model-select", "value"),
        State("infra-port", "value"),
        State("infra-ctx-size", "value"),
        State("infra-gpu-layers", "value"),
        State("infra-parallel", "value"),
        State("infra-threads", "value"),
        State("infra-flash-attn", "value"),
        prevent_initial_call=True,
    )
    def start_server(n_clicks, node, model_path, port, ctx_size, gpu_layers, parallel, threads, flash_attn):
        if not model_path:
            return html.Span("Select a model file first.", style={"color": WONG["orange"]})
        if not port:
            return html.Span("Port is required.", style={"color": WONG["orange"]})

        result = infra.start_server(
            node=node,
            model_path=model_path,
            port=int(port),
            ctx_size=int(ctx_size or cfg.server_defaults.ctx_size),
            flash_attn=bool(flash_attn and "on" in flash_attn),
            threads=int(threads) if threads else None,
            parallel=int(parallel or cfg.server_defaults.parallel),
            gpu_layers=int(gpu_layers or cfg.server_defaults.gpu_layers),
            llama_server_bin=cfg.infra.llama_server_bin,
            remote_host=_resolve_host(node) if _resolve_host(node) != "local" else "",
        )

        if not result:
            return html.Span("Failed to start server.", style={"color": WONG["vermillion"]})

        if result.status.startswith("crashed"):
            err_detail = result.status[len("crashed"):].lstrip(": ") or "process exited immediately"
            return html.Div([
                html.Span(
                    f"Server died on startup (PID {result.pid}): ",
                    style={"color": WONG["vermillion"], "fontWeight": "600"},
                ),
                html.Span(err_detail, style={"color": TEXT_SECONDARY, "fontSize": "12px"}),
            ])

        return html.Span(
            f"Started llama-server on {result.node}:{result.port} (PID {result.pid})",
            style={"color": WONG["green"]},
        )

    # Node health
    @app.callback(
        Output("infra-node-health", "children"),
        Input("infra-interval", "n_intervals"),
    )
    def update_node_health(_n):
        rows = []
        for node in cfg.nodes:
            if not node.enabled:
                continue
            if node.host == "local":
                color = WONG["green"]
                status_text = "running"
            else:
                reachable = infra.check_node_reachable(node.host)
                color = WONG["green"] if reachable else WONG["vermillion"]
                status_text = "reachable" if reachable else "unreachable"

            rows.append(html.Div([
                html.Span("\u25cf", style={"color": color, "marginRight": "6px"}),
                html.Span(f"{node.label or node.name}: ", style={"fontWeight": "600", "marginRight": "4px"}),
                html.Span(status_text, style={"color": color}),
            ], style={"marginBottom": "6px"}))

        return html.Div([
            html.H4("Node Health", style={"marginTop": "0", "marginBottom": "12px"}),
            *rows,
        ], style=CARD_STYLE)

    # Sync form defaults when settings are saved
    @app.callback(
        Output("infra-port", "value"),
        Output("infra-ctx-size", "value"),
        Output("infra-gpu-layers", "value"),
        Output("infra-parallel", "value"),
        Output("infra-threads", "value"),
        Output("infra-flash-attn", "value"),
        Input("config-version", "data"),
        prevent_initial_call=True,
    )
    def sync_server_defaults(_version):
        # Invalidate model cache so next scan uses updated models_dir
        model_catalog.invalidate_cache()
        sd = cfg.server_defaults
        return (
            sd.port,
            sd.ctx_size,
            sd.gpu_layers,
            sd.parallel,
            sd.threads or 0,
            ["on"] if sd.flash_attn else [],
        )
