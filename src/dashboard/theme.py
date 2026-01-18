"""
Polymarket-Style Theme for Dash Dashboard
=========================================
Dark theme inspired by Polymarket.
Colors: Black background, blue accents, green/red for signals.
"""

# ===========================================
# COLOR PALETTE - Polymarket Style
# ===========================================

COLORS = {
    # Background colors
    "bg_primary": "#0d0d0d",        # Near black
    "bg_secondary": "#1a1a2e",      # Dark blue-gray
    "bg_card": "#0f0f1a",           # Card background
    "bg_hover": "#252540",          # Hover state
    
    # Text colors
    "text_primary": "#ffffff",      # White
    "text_secondary": "#a0a0b0",    # Gray-blue
    "text_muted": "#666680",        # Muted gray-blue
    
    # Accent colors (Polymarket blue)
    "accent_primary": "#4a90d9",    # Polymarket blue
    "accent_secondary": "#6ba3e0",  # Lighter blue
    "accent_highlight": "#8cb8e8",  # Highlight blue
    
    # Signal colors
    "buy": "#00d26a",               # Green - Buy signal
    "sell": "#ff4757",              # Red - Sell signal
    "hold": "#4a90d9",              # Blue - Hold
    "neutral": "#a0a0b0",           # Gray - Neutral
    
    # Status colors
    "success": "#00d26a",           # Success green
    "warning": "#ffa502",           # Warning amber
    "error": "#ff4757",             # Error red
    "info": "#4a90d9",              # Info blue
    
    # Chart colors
    "chart_line": "#4a90d9",        # Primary line
    "chart_area": "rgba(74, 144, 217, 0.2)",  # Area fill
    "chart_grid": "#333355",        # Grid lines
    "chart_bid": "#00d26a",         # Bid depth
    "chart_ask": "#ff4757",         # Ask depth
    
    # Confidence colors
    "conf_high": "#00d26a",
    "conf_medium": "#ffa502",
    "conf_low": "#ff4444",
    
    # Border colors
    "border": "#0f3460",            # Default border
    "border_light": "#1a1a2e",      # Light border
}

# ===========================================
# TYPOGRAPHY
# ===========================================

FONTS = {
    "primary": "'Roboto Mono', 'Consolas', 'Monaco', monospace",
    "secondary": "'Inter', 'Segoe UI', 'Arial', sans-serif",
    "heading": "'Inter', 'Segoe UI', 'Arial', sans-serif",
}

# ===========================================
# COMPONENT STYLES
# ===========================================

# Card style
CARD_STYLE = {
    "backgroundColor": COLORS["bg_card"],
    "borderRadius": "8px",
    "padding": "20px",
    "marginBottom": "20px",
    "border": f"1px solid {COLORS['bg_hover']}",
    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.3)",
}

# Header style
HEADER_STYLE = {
    "backgroundColor": COLORS["bg_secondary"],
    "padding": "15px 30px",
    "borderBottom": f"2px solid {COLORS['accent_primary']}",
    "marginBottom": "20px",
}

# Table styles
TABLE_HEADER_STYLE = {
    "backgroundColor": COLORS["bg_secondary"],
    "color": COLORS["accent_primary"],
    "fontWeight": "bold",
    "textAlign": "left",
    "padding": "12px 15px",
    "borderBottom": f"2px solid {COLORS['accent_primary']}",
    "fontFamily": FONTS["primary"],
    "fontSize": "12px",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
}

TABLE_CELL_STYLE = {
    "backgroundColor": COLORS["bg_card"],
    "color": COLORS["text_primary"],
    "padding": "12px 15px",
    "borderBottom": f"1px solid {COLORS['bg_hover']}",
    "fontFamily": FONTS["primary"],
    "fontSize": "13px",
}

TABLE_ROW_HOVER = {
    "backgroundColor": COLORS["bg_hover"],
}

# Button styles
BUTTON_PRIMARY = {
    "backgroundColor": COLORS["accent_primary"],
    "color": COLORS["text_primary"],
    "border": "none",
    "borderRadius": "4px",
    "padding": "10px 20px",
    "fontWeight": "bold",
    "cursor": "pointer",
    "fontFamily": FONTS["secondary"],
    "fontSize": "14px",
    "textTransform": "uppercase",
    "letterSpacing": "1px",
}

BUTTON_SECONDARY = {
    "backgroundColor": "transparent",
    "color": COLORS["accent_primary"],
    "border": f"1px solid {COLORS['accent_primary']}",
    "borderRadius": "4px",
    "padding": "10px 20px",
    "fontWeight": "bold",
    "cursor": "pointer",
    "fontFamily": FONTS["secondary"],
}

# Metric box style
METRIC_BOX_STYLE = {
    "backgroundColor": COLORS["bg_card"],
    "borderRadius": "8px",
    "padding": "20px",
    "textAlign": "center",
    "border": f"1px solid {COLORS['bg_hover']}",
    "minWidth": "150px",
}

METRIC_VALUE_STYLE = {
    "fontSize": "28px",
    "fontWeight": "bold",
    "color": COLORS["accent_primary"],
    "fontFamily": FONTS["primary"],
    "marginBottom": "5px",
}

METRIC_LABEL_STYLE = {
    "fontSize": "12px",
    "color": COLORS["text_secondary"],
    "textTransform": "uppercase",
    "letterSpacing": "1px",
    "fontFamily": FONTS["secondary"],
}

# ===========================================
# PLOTLY CHART TEMPLATE
# ===========================================

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": COLORS["bg_card"],
        "plot_bgcolor": COLORS["bg_primary"],
        "font": {
            "family": FONTS["primary"],
            "color": COLORS["text_primary"],
            "size": 12
        },
        "title": {
            "font": {
                "family": FONTS["heading"],
                "size": 16,
                "color": COLORS["text_primary"]
            }
        },
        "xaxis": {
            "gridcolor": COLORS["chart_grid"],
            "linecolor": COLORS["chart_grid"],
            "tickfont": {"color": COLORS["text_secondary"]},
            "title": {"font": {"color": COLORS["text_secondary"]}}
        },
        "yaxis": {
            "gridcolor": COLORS["chart_grid"],
            "linecolor": COLORS["chart_grid"],
            "tickfont": {"color": COLORS["text_secondary"]},
            "title": {"font": {"color": COLORS["text_secondary"]}}
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": COLORS["text_secondary"]}
        },
        "colorway": [
            COLORS["accent_primary"],
            COLORS["buy"],
            COLORS["sell"],
            COLORS["neutral"],
            COLORS["accent_secondary"],
        ]
    }
}

# ===========================================
# DASH DATA TABLE STYLE
# ===========================================

DATA_TABLE_STYLE = {
    "style_header": {
        "backgroundColor": COLORS["bg_secondary"],
        "color": COLORS["accent_primary"],
        "fontWeight": "bold",
        "textAlign": "left",
        "padding": "12px",
        "border": "none",
        "borderBottom": f"2px solid {COLORS['accent_primary']}",
        "fontFamily": FONTS["primary"],
        "fontSize": "11px",
        "textTransform": "uppercase",
    },
    "style_cell": {
        "backgroundColor": COLORS["bg_card"],
        "color": COLORS["text_primary"],
        "padding": "12px",
        "border": "none",
        "borderBottom": f"1px solid {COLORS['bg_hover']}",
        "fontFamily": FONTS["primary"],
        "fontSize": "12px",
        "textAlign": "left",
        "maxWidth": "200px",
        "overflow": "hidden",
        "textOverflow": "ellipsis",
    },
    "style_data": {
        "backgroundColor": COLORS["bg_card"],
        "color": COLORS["text_primary"],
    },
    "style_data_conditional": [
        {
            "if": {"row_index": "odd"},
            "backgroundColor": COLORS["bg_secondary"],
        },
        {
            "if": {"state": "selected"},
            "backgroundColor": COLORS["bg_hover"],
            "border": f"1px solid {COLORS['accent_primary']}",
        },
        {
            "if": {"column_id": "action", "filter_query": "{action} = BUY"},
            "color": COLORS["buy"],
            "fontWeight": "bold",
        },
        {
            "if": {"column_id": "action", "filter_query": "{action} = SELL"},
            "color": COLORS["sell"],
            "fontWeight": "bold",
        },
        {
            "if": {"column_id": "confidence", "filter_query": "{confidence} = high"},
            "color": COLORS["conf_high"],
        },
        {
            "if": {"column_id": "confidence", "filter_query": "{confidence} = medium"},
            "color": COLORS["conf_medium"],
        },
        {
            "if": {"column_id": "confidence", "filter_query": "{confidence} = low"},
            "color": COLORS["conf_low"],
        },
    ],
    "style_table": {
        "overflowX": "auto",
        "borderRadius": "8px",
        "border": f"1px solid {COLORS['bg_hover']}",
    },
    "style_filter": {
        "backgroundColor": COLORS["bg_secondary"],
        "color": COLORS["text_primary"],
        "border": f"1px solid {COLORS['bg_hover']}",
    },
}

# ===========================================
# CSS STYLESHEET
# ===========================================

EXTERNAL_STYLESHEETS = [
    # Google Fonts
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500;700&display=swap",
]

# Custom CSS to inject
CUSTOM_CSS = f"""
/* Global styles */
body {{
    background-color: {COLORS['bg_primary']};
    color: {COLORS['text_primary']};
    font-family: {FONTS['secondary']};
    margin: 0;
    padding: 0;
}}

/* Live indicator pulse animation */
@keyframes pulse {{
    0% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
    100% {{ opacity: 1; }}
}}

.live-indicator {{
    animation: pulse 2s ease-in-out infinite;
}}

/* Scrollbar styling */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}

::-webkit-scrollbar-track {{
    background: {COLORS['bg_secondary']};
}}

::-webkit-scrollbar-thumb {{
    background: {COLORS['accent_primary']};
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: {COLORS['accent_secondary']};
}}

/* Selection */
::selection {{
    background: {COLORS['accent_primary']};
    color: {COLORS['text_primary']};
}}

/* Links */
a {{
    color: {COLORS['accent_primary']};
    text-decoration: none;
}}

a:hover {{
    color: {COLORS['accent_secondary']};
}}

/* Buttons hover */
.btn-primary:hover {{
    background-color: {COLORS['accent_secondary']} !important;
}}

/* Loading spinner */
._dash-loading {{
    background-color: {COLORS['bg_primary']};
}}

/* Dropdown styling */
.Select-control {{
    background-color: {COLORS['bg_card']} !important;
    border-color: {COLORS['bg_hover']} !important;
}}

.Select-menu-outer {{
    background-color: {COLORS['bg_card']} !important;
    border-color: {COLORS['bg_hover']} !important;
}}

.Select-option {{
    background-color: {COLORS['bg_card']} !important;
    color: {COLORS['text_primary']} !important;
}}

.Select-option:hover {{
    background-color: {COLORS['bg_hover']} !important;
}}
"""


def get_signal_color(signal: str) -> str:
    """Get color for a trade signal."""
    signal = signal.upper() if signal else ""
    if signal == "BUY":
        return COLORS["buy"]
    elif signal == "SELL":
        return COLORS["sell"]
    elif signal == "HOLD":
        return COLORS["hold"]
    return COLORS["neutral"]


def get_confidence_color(confidence: str) -> str:
    """Get color for confidence level."""
    confidence = confidence.lower() if confidence else ""
    if confidence == "high":
        return COLORS["conf_high"]
    elif confidence == "medium":
        return COLORS["conf_medium"]
    return COLORS["conf_low"]


def get_bias_color(bias: str) -> str:
    """Get color for AI bias."""
    bias = bias.upper() if bias else ""
    if bias == "UNDERPRICED":
        return COLORS["buy"]
    elif bias == "OVERPRICED":
        return COLORS["sell"]
    return COLORS["neutral"]
