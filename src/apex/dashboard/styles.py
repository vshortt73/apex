"""Wong colorblind-safe palette, Plotly templates, and style constants (dark theme)."""

from __future__ import annotations

# Wong colorblind-safe palette (Nature Methods 2011)
# Brightened slightly for dark background legibility
WONG = {
    "blue": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "yellow": "#F0E442",
    "sky": "#87CEEB",
    "vermillion": "#E8601C",
    "purple": "#CC79A7",
    "white": "#e0e0e0",
}

# Dimension colors — fixed assignment
DIMENSION_COLORS = {
    "factual_recall": WONG["blue"],
    "application": WONG["orange"],
    "salience": WONG["green"],
}

DIMENSION_LABELS = {
    "factual_recall": "Factual Recall",
    "application": "Application",
    "salience": "Salience",
}

# Model color cycle
MODEL_COLOR_CYCLE = [
    WONG["blue"],
    WONG["orange"],
    WONG["green"],
    WONG["vermillion"],
    WONG["sky"],
    WONG["purple"],
    WONG["yellow"],
]

# Sequential scales for context length overlays (dim to bright on dark bg)
CONTEXT_LENGTH_SCALE = [
    "#4a7298",
    "#5a9bd4",
    "#7ab8e0",
    "#9dd1f0",
    "#c6e6f7",
]

# Dark theme colors
BG_PAGE = "#1a1a2e"
BG_CARD = "#252540"
BG_CONTROLS = "#2a2a48"
BG_PLOT = "#1e1e36"
BORDER_COLOR = "#3a3a5c"
TEXT_PRIMARY = "#e0e0e0"
TEXT_SECONDARY = "#a0a0b8"
TEXT_MUTED = "#70708a"

# Shared Plotly layout template
FIGURE_LAYOUT = dict(
    template="plotly_dark",
    font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color=TEXT_PRIMARY),
    title_font_size=16,
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_PLOT,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color=TEXT_PRIMARY),
    ),
    xaxis=dict(
        title="Probe Position (% of context)",
        ticksuffix="%",
        range=[0, 100],
        dtick=10,
        gridcolor="#2e2e4a",
        zerolinecolor="#3a3a5c",
    ),
    yaxis=dict(
        title="Score",
        range=[-0.05, 1.05],
        gridcolor="#2e2e4a",
        zerolinecolor="#3a3a5c",
    ),
    margin=dict(l=60, r=30, t=60, b=50),
    hovermode="x unified",
)

# Modebar config for all figures
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "svg", "scale": 2},
}

# App-level CSS
CARD_STYLE = {
    "backgroundColor": BG_CARD,
    "borderRadius": "8px",
    "padding": "20px",
    "marginBottom": "16px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
    "border": f"1px solid {BORDER_COLOR}",
    "color": TEXT_PRIMARY,
}

CONTROLS_STYLE = {
    "backgroundColor": BG_CONTROLS,
    "borderRadius": "8px",
    "padding": "16px",
    "marginBottom": "16px",
    "border": f"1px solid {BORDER_COLOR}",
    "color": TEXT_PRIMARY,
}

def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert hex color to rgba() string for Plotly fill transparency."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


GUIDE_STYLE = {
    "backgroundColor": BG_CONTROLS,
    "border": f"1px solid {BORDER_COLOR}",
    "borderRadius": "6px",
    "padding": "10px 14px",
    "marginBottom": "16px",
    "color": TEXT_SECONDARY,
    "fontSize": "13px",
    "lineHeight": "1.5",
}

LABEL_STYLE = {
    "fontWeight": "600",
    "fontSize": "13px",
    "marginBottom": "4px",
    "color": TEXT_PRIMARY,
}

# Pre-block style (response viewer, probe content)
PRE_STYLE = {
    "whiteSpace": "pre-wrap",
    "fontSize": "12px",
    "backgroundColor": "#1a1a30",
    "color": "#d0d0e0",
    "padding": "8px",
    "borderRadius": "4px",
    "border": f"1px solid {BORDER_COLOR}",
    "maxHeight": "300px",
    "overflowY": "auto",
}
