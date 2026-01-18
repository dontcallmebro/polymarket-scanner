"""
Polymarket Volatility Dashboard - Main Application
===================================================
Bloomberg-style dashboard for analyzing Polymarket volatility
and generating trade recommendations.

Run with: python -m src.dashboard.app
Or: python src/dashboard/app.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dash import Dash, html, dcc, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import dash_table
import pandas as pd
from datetime import datetime
import threading
import time
import logging

# Local imports
from src.dashboard.theme import (
    COLORS, FONTS, CARD_STYLE, DATA_TABLE_STYLE,
    EXTERNAL_STYLESHEETS, CUSTOM_CSS,
    get_signal_color, get_confidence_color
)
from src.dashboard.components import (
    create_header, create_summary_metrics, create_trade_card,
    create_volatility_chart, create_signals_distribution_chart,
    create_loading_spinner
)
from src.api_client import client
from src.database import db
from src.trade_recommender import trade_recommender
from src.ai_analyzer import ai_analyzer
from src.realtime_prices import price_stream
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================
# INITIALIZE DASH APP
# ===========================================

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        *EXTERNAL_STYLESHEETS
    ],
    suppress_callback_exceptions=True,
    title="Polymarket Scanner"
)

# Inject custom CSS via index_string
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
''' + CUSTOM_CSS + '''
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

server = app.server

# ===========================================
# LAYOUT
# ===========================================

app.layout = html.Div([
    
    # Header
    create_header(),
    
    # Main content
    html.Div([
        # Controls row
        html.Div([
            html.Div([
                html.Label("Signal Filter:", style={"color": COLORS["text_secondary"], "marginRight": "10px"}),
                dcc.Dropdown(
                    id="signal-filter",
                    options=[
                        {"label": "All Signals", "value": "ALL"},
                        {"label": "🟢 BUY Only", "value": "BUY"},
                        {"label": "🔴 SELL Only", "value": "SELL"},
                    ],
                    value="ALL",
                    clearable=False,
                    style={"width": "150px", "backgroundColor": COLORS["bg_card"]}
                )
            ], style={"display": "flex", "alignItems": "center"}),
            
            html.Div([
                html.Label("Min Confidence:", style={"color": COLORS["text_secondary"], "marginRight": "10px"}),
                dcc.Dropdown(
                    id="confidence-filter",
                    options=[
                        {"label": "Any", "value": "any"},
                        {"label": "Medium+", "value": "medium"},
                        {"label": "High Only", "value": "high"},
                    ],
                    value="any",
                    clearable=False,
                    style={"width": "130px"}
                )
            ], style={"display": "flex", "alignItems": "center"}),
            
            html.Div([
                html.Label("AI Analysis:", style={"color": COLORS["text_secondary"], "marginRight": "10px"}),
                dcc.Checklist(
                    id="ai-toggle",
                    options=[{"label": " Enable", "value": "enabled"}],
                    value=[],  # Disabled by default to avoid startup errors
                    style={"color": COLORS["text_primary"]}
                )
            ], style={"display": "flex", "alignItems": "center"}),
            
            html.Div([
                html.Label("Min Price:", style={"color": COLORS["text_secondary"], "marginRight": "10px"}),
                dcc.Input(
                    id="min-prob-filter",
                    type="number",
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.03,
                    placeholder="0.03",
                    style={"width": "70px", "backgroundColor": COLORS["bg_card"], "color": COLORS["text_primary"], "border": f"1px solid {COLORS['border']}", "borderRadius": "4px", "padding": "5px"}
                )
            ], style={"display": "flex", "alignItems": "center"}),
            
            html.Div([
                html.Label("Max Price:", style={"color": COLORS["text_secondary"], "marginRight": "10px"}),
                dcc.Input(
                    id="max-prob-filter",
                    type="number",
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.95,
                    placeholder="0.95",
                    style={"width": "70px", "backgroundColor": COLORS["bg_card"], "color": COLORS["text_primary"], "border": f"1px solid {COLORS['border']}", "borderRadius": "4px", "padding": "5px"}
                )
            ], style={"display": "flex", "alignItems": "center"}),
            
            html.Button(
                "🔄 Refresh Data",
                id="refresh-button",
                n_clicks=0,
                style={
                    "backgroundColor": COLORS["accent_primary"],
                    "color": COLORS["text_primary"],
                    "border": "none",
                    "borderRadius": "4px",
                    "padding": "10px 20px",
                    "fontWeight": "bold",
                    "cursor": "pointer",
                    "marginLeft": "auto"
                }
            )
        ], style={
            "display": "flex",
            "gap": "30px",
            "alignItems": "center",
            "marginBottom": "10px",
            "padding": "15px 20px",
            "backgroundColor": COLORS["bg_secondary"],
            "borderRadius": "8px"
        }),
        
        # Model Parameters Row - Granular Controls
        html.Div([
            html.Span("⚙️ Model Parameters:", style={"color": COLORS["accent_primary"], "fontWeight": "bold", "marginRight": "20px"}),
            
            html.Div([
                html.Label("Min Age (days):", style={"color": COLORS["text_secondary"], "marginRight": "8px", "fontSize": "12px"}),
                dcc.Input(
                    id="min-age-days",
                    type="number",
                    min=0,
                    max=365,
                    step=1,
                    value=14,
                    style={"width": "60px", "backgroundColor": COLORS["bg_card"], "color": COLORS["text_primary"], "border": f"1px solid {COLORS['border']}", "borderRadius": "4px", "padding": "5px", "fontSize": "12px"}
                )
            ], style={"display": "flex", "alignItems": "center"}),
            
            html.Div([
                html.Label("Min Maturity (days):", style={"color": COLORS["text_secondary"], "marginRight": "8px", "fontSize": "12px"}),
                dcc.Input(
                    id="min-maturity-days",
                    type="number",
                    min=0,
                    max=365,
                    step=1,
                    value=14,
                    style={"width": "60px", "backgroundColor": COLORS["bg_card"], "color": COLORS["text_primary"], "border": f"1px solid {COLORS['border']}", "borderRadius": "4px", "padding": "5px", "fontSize": "12px"}
                )
            ], style={"display": "flex", "alignItems": "center"}),
            
            html.Div([
                html.Label("Min Volume ($):", style={"color": COLORS["text_secondary"], "marginRight": "8px", "fontSize": "12px"}),
                dcc.Input(
                    id="min-volume",
                    type="number",
                    min=0,
                    max=1000000,
                    step=100,
                    value=1000,
                    style={"width": "80px", "backgroundColor": COLORS["bg_card"], "color": COLORS["text_primary"], "border": f"1px solid {COLORS['border']}", "borderRadius": "4px", "padding": "5px", "fontSize": "12px"}
                )
            ], style={"display": "flex", "alignItems": "center"}),
            
            html.Div([
                html.Label("Min R/R Ratio:", style={"color": COLORS["text_secondary"], "marginRight": "8px", "fontSize": "12px"}),
                dcc.Input(
                    id="min-rr-ratio",
                    type="number",
                    min=0,
                    max=10,
                    step=0.1,
                    value=0.5,
                    style={"width": "60px", "backgroundColor": COLORS["bg_card"], "color": COLORS["text_primary"], "border": f"1px solid {COLORS['border']}", "borderRadius": "4px", "padding": "5px", "fontSize": "12px"}
                )
            ], style={"display": "flex", "alignItems": "center"}),
            
            html.Div([
                html.Label("Results:", style={"color": COLORS["text_secondary"], "marginRight": "8px", "fontSize": "12px"}),
                dcc.Input(
                    id="num-results",
                    type="number",
                    min=10,
                    max=500,
                    step=10,
                    value=100,
                    style={"width": "60px", "backgroundColor": COLORS["bg_card"], "color": COLORS["text_primary"], "border": f"1px solid {COLORS['border']}", "borderRadius": "4px", "padding": "5px", "fontSize": "12px"}
                )
            ], style={"display": "flex", "alignItems": "center"}),
            
        ], style={
            "display": "flex",
            "gap": "25px",
            "alignItems": "center",
            "marginBottom": "10px",
            "padding": "12px 20px",
            "backgroundColor": COLORS["bg_card"],
            "borderRadius": "8px",
            "border": f"1px solid {COLORS['border']}"
        }),
        
        # Mini Terminal / Log Console
        html.Div([
            html.Div([
                html.Span("▶ ", style={"color": COLORS["success"]}),
                html.Span("CONSOLE", style={"color": COLORS["accent_primary"], "fontWeight": "bold", "fontSize": "11px"}),
                html.Span(" | ", style={"color": COLORS["text_muted"]}),
                html.Span(id="console-status", children="Ready", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
            ], style={"marginBottom": "8px"}),
            html.Div(
                id="console-output",
                children=[
                    html.Div("Waiting for data fetch...", style={"color": COLORS["text_muted"]})
                ],
                style={
                    "fontFamily": "'Consolas', 'Monaco', 'Courier New', monospace",
                    "fontSize": "11px",
                    "lineHeight": "1.4",
                    "maxHeight": "120px",
                    "overflowY": "auto",
                    "padding": "8px 10px",
                    "backgroundColor": "#000000",
                    "borderRadius": "4px",
                    "border": f"1px solid {COLORS['border']}"
                }
            )
        ], style={
            "marginBottom": "15px",
            "padding": "10px 15px",
            "backgroundColor": COLORS["bg_secondary"],
            "borderRadius": "8px",
            "border": f"1px solid {COLORS['border']}"
        }),
        
        # Summary metrics
        html.Div(id="summary-metrics"),
        
        # Loading indicator
        dcc.Loading(
            id="loading",
            type="circle",
            color=COLORS["accent_primary"],
            children=[
                # Main compact table - MOVED TO TOP
                html.Div([
                    html.H3([
                        html.Span("◆ ", style={"color": COLORS["accent_primary"]}),
                        html.Span(id="table-title", children="TOP VOLATILE MARKETS")
                    ], style={
                        "color": COLORS["text_primary"],
                        "marginBottom": "15px",
                        "fontSize": "16px",
                        "fontFamily": FONTS["heading"]
                    }),
                    html.P("Click on a row to see detailed analysis", style={
                        "color": COLORS["text_muted"],
                        "fontSize": "12px",
                        "marginBottom": "15px"
                    }),
                    html.Div(id="main-markets-table")
                ], style=CARD_STYLE),
                
                # Charts row (compact) - NOW BELOW TABLE
                html.Div([
                    # Signal distribution
                    html.Div([
                        html.H4("Signal Distribution", style={"color": COLORS["text_secondary"], "marginBottom": "10px", "fontSize": "14px"}),
                        html.Div(id="signals-chart")
                    ], style={**CARD_STYLE, "flex": "1", "minWidth": "300px"}),
                    
                    # Volatility chart
                    html.Div([
                        html.H4("Top Volatility", style={"color": COLORS["text_secondary"], "marginBottom": "10px", "fontSize": "14px"}),
                        html.Div(id="volatility-chart")
                    ], style={**CARD_STYLE, "flex": "2", "minWidth": "400px"}),
                ], style={
                    "display": "flex",
                    "gap": "20px",
                    "flexWrap": "wrap",
                    "marginTop": "20px"
                })
            ]
        ),
        
        # Detail Modal
        dbc.Modal([
            dbc.ModalHeader(
                dbc.ModalTitle(id="modal-title", style={"color": COLORS["text_primary"]}),
                close_button=True,
                style={"backgroundColor": COLORS["bg_secondary"], "borderBottom": f"1px solid {COLORS['border']}"}
            ),
            dbc.ModalBody(id="modal-body", style={"backgroundColor": COLORS["bg_card"]}),
            dbc.ModalFooter(
                html.Div([
                    html.A(
                        "Open on Polymarket →",
                        id="modal-polymarket-link",
                        href="#",
                        target="_blank",
                        style={
                            "color": COLORS["accent_primary"],
                            "textDecoration": "none",
                            "fontWeight": "bold"
                        }
                    )
                ]),
                style={"backgroundColor": COLORS["bg_secondary"], "borderTop": f"1px solid {COLORS['border']}"}
            )
        ], id="detail-modal", size="lg", is_open=False, centered=True),
        
        # Store for data
        dcc.Store(id="recommendations-store"),
        dcc.Store(id="selected-market-id"),
        dcc.Store(id="live-prices-store"),
        
        # Interval for full data refresh (every 5 minutes)
        dcc.Interval(
            id="auto-refresh",
            interval=5 * 60 * 1000,  # 5 minutes in milliseconds
            n_intervals=0
        ),
        
        # Interval for live price updates (every 10 seconds)
        dcc.Interval(
            id="live-price-refresh",
            interval=10 * 1000,  # 10 seconds
            n_intervals=0
        )
        
    ], style={
        "padding": "0 30px 30px 30px",
        "maxWidth": "1800px",
        "margin": "0 auto"
    })
    
], style={
    "backgroundColor": COLORS["bg_primary"],
    "minHeight": "100vh",
    "fontFamily": FONTS["secondary"]
})


# ===========================================
# CALLBACKS
# ===========================================

@callback(
    Output("current-time", "children"),
    Input("auto-refresh", "n_intervals")
)
def update_time(n):
    """Update the current time display."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


@callback(
    Output("recommendations-store", "data"),
    [Input("refresh-button", "n_clicks"),
     Input("auto-refresh", "n_intervals")],
    [State("signal-filter", "value"),
     State("confidence-filter", "value"),
     State("ai-toggle", "value"),
     State("min-prob-filter", "value"),
     State("max-prob-filter", "value"),
     State("min-age-days", "value"),
     State("min-maturity-days", "value"),
     State("min-volume", "value"),
     State("min-rr-ratio", "value"),
     State("num-results", "value")]
)
def fetch_recommendations(n_clicks, n_intervals, signal_filter, confidence_filter, ai_toggle, min_prob, max_prob, min_age_days, min_maturity_days, min_volume, min_rr_ratio, num_results):
    """Fetch trade recommendations from the API."""
    logger.info("Fetching recommendations...")
    
    total_fetched = 0
    logs = []
    try:
        # Determine filters
        sig_filter = None if signal_filter == "ALL" else signal_filter
        conf_filter = None if confidence_filter == "any" else confidence_filter
        include_ai = "enabled" in (ai_toggle or [])
        
        # Use user-provided values or defaults
        n_results = num_results if num_results else 100
        age_days = min_age_days if min_age_days is not None else 14
        maturity_days = min_maturity_days if min_maturity_days is not None else 14
        volume = min_volume if min_volume is not None else 1000
        rr_ratio = min_rr_ratio if min_rr_ratio is not None else 0.5
        # Default price range: 0.03 to 0.95
        price_min = min_prob if min_prob is not None else 0.03
        price_max = max_prob if max_prob is not None else 0.95
        
        logs.append(f"[INFO] Starting fetch with params: age≥{age_days}d, maturity≥{maturity_days}d, vol≥${volume}, R/R≥{rr_ratio}")
        logs.append(f"[INFO] Price filter: {price_min} - {price_max}")
        
        # Generate recommendations with user-controlled parameters
        recs = trade_recommender.generate_recommendations(
            n=n_results,
            include_ai=include_ai,
            signal_filter=sig_filter,
            min_confidence=conf_filter,
            min_rr_ratio=rr_ratio,
            min_price=price_min,
            max_price=price_max,
            min_age_days=age_days,
            min_maturity_days=maturity_days,
            min_volume=volume
        )
        
        # Get stats from recommender
        stats = trade_recommender.get_last_stats()
        
        logs.append(f"[API] Fetched {stats.get('total_fetched', 0)} markets from Polymarket API")
        logs.append(f"[FILTER] After criteria filter: {stats.get('after_criteria', 0)} markets")
        if stats.get('after_price_filter'):
            logs.append(f"[FILTER] After price filter: {stats.get('after_price_filter', 0)} markets")
        logs.append(f"[SCORE] Scored and ranked markets")
        
        # Convert to dict list
        data = trade_recommender.to_dict_list(recs)
        
        logs.append(f"[OK] Returning top {len(data)} recommendations")
        
        logger.info(f"Fetched {len(data)} recommendations")
        # Add metadata to first item for display
        if data:
            data[0]["_total_fetched"] = stats.get("total_fetched", 0)
            data[0]["_after_criteria"] = stats.get("after_criteria", 0)
            data[0]["_after_price_filter"] = stats.get("after_price_filter", 0)
            data[0]["_logs"] = logs
        return data
        
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}")
        return [{"_logs": [f"[ERROR] {str(e)}"]}]


@callback(
    Output("summary-metrics", "children"),
    Input("recommendations-store", "data")
)
def update_summary_metrics(data):
    """Update the summary metrics display."""
    if not data:
        return create_summary_metrics({
            "total_fetched": 0,
            "total_markets": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "avg_volatility": 0,
            "high_confidence": 0,
            "ai_analyzed": 0
        })
    
    buy_signals = len([d for d in data if d.get("action") == "BUY"])
    sell_signals = len([d for d in data if d.get("action") == "SELL"])
    high_conf = len([d for d in data if d.get("confidence") == "high"])
    ai_analyzed = len([d for d in data if d.get("ai_bias")])
    
    vol_scores = [d.get("vol_score", 0) for d in data if d.get("vol_score")]
    avg_vol = sum(vol_scores) / len(vol_scores) / 100 if vol_scores else 0
    
    # Extract stats from metadata
    total_fetched = data[0].get("_total_fetched", 0) if data else 0
    after_criteria = data[0].get("_after_criteria", 0) if data else 0
    after_price_filter = data[0].get("_after_price_filter", 0) if data else 0
    
    return create_summary_metrics({
        "total_fetched": total_fetched,
        "after_criteria": after_criteria,
        "after_price_filter": after_price_filter,
        "total_markets": len(data),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "avg_volatility": avg_vol,
        "high_confidence": high_conf,
        "ai_analyzed": ai_analyzed
    })


@callback(
    Output("table-title", "children"),
    Input("recommendations-store", "data")
)
def update_table_title(data):
    """Update the table title with current count."""
    count = len(data) if data else 0
    return f"TOP {count} VOLATILE MARKETS"


@callback(
    [Output("console-output", "children"),
     Output("console-status", "children")],
    Input("recommendations-store", "data")
)
def update_console(data):
    """Update the console with log messages."""
    if not data:
        return [html.Div("Waiting for data fetch...", style={"color": COLORS["text_muted"]})], "Ready"
    
    logs = data[0].get("_logs", []) if data else []
    
    if not logs:
        return [html.Div("No logs available", style={"color": COLORS["text_muted"]})], "Ready"
    
    # Create colored log lines
    log_elements = []
    for log in logs:
        if "[ERROR]" in log:
            color = COLORS["error"]
        elif "[OK]" in log:
            color = COLORS["success"]
        elif "[API]" in log:
            color = COLORS["accent_primary"]
        elif "[FILTER]" in log:
            color = COLORS["accent_secondary"]
        elif "[SCORE]" in log:
            color = "#9b59b6"  # Purple
        elif "[INFO]" in log:
            color = COLORS["text_secondary"]
        else:
            color = COLORS["text_muted"]
        
        log_elements.append(html.Div(log, style={"color": color}))
    
    # Add timestamp
    from datetime import datetime
    timestamp = datetime.utcnow().strftime("%H:%M:%S UTC")
    status = f"Last update: {timestamp}"
    
    return log_elements, status


@callback(
    Output("signals-chart", "children"),
    Input("recommendations-store", "data")
)
def update_signals_chart(data):
    """Update the signals distribution chart."""
    if not data:
        return html.Div("No data", style={"color": COLORS["text_muted"]})
    
    return create_signals_distribution_chart(data)


@callback(
    Output("volatility-chart", "children"),
    Input("recommendations-store", "data")
)
def update_volatility_chart(data):
    """Update the volatility chart."""
    if not data:
        return html.Div("No data", style={"color": COLORS["text_muted"]})
    
    return create_volatility_chart(data)


@callback(
    Output("main-markets-table", "children"),
    Input("recommendations-store", "data")
)
def update_main_table(data):
    """Update the main compact markets table."""
    if not data:
        return html.Div("No data available. Click Refresh to load data.", 
                       style={"color": COLORS["text_muted"], "padding": "20px"})
    
    # Filter out metadata fields that start with underscore
    clean_data = []
    for item in data:
        clean_item = {k: v for k, v in item.items() if not k.startswith('_')}
        clean_data.append(clean_item)
    
    # Prepare DataFrame
    df = pd.DataFrame(clean_data)
    
    # Add market_id as hidden column for row selection
    df["id"] = df["market_id"]
    
    # Format columns - Convert prices to probabilities
    df["signal_display"] = df["action"].apply(lambda x: f"🟢 {x}" if x == "BUY" else f"🔴 {x}" if x == "SELL" else x)
    df["entry_fmt"] = df["entry_price"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
    df["target_fmt"] = df["target_price"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
    df["stop_fmt"] = df["stop_loss"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
    df["current_fmt"] = df["current_price"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
    df["rr_fmt"] = df["risk_reward"].apply(lambda x: f"{x:.2f}x" if pd.notna(x) and x > 0 else "-")
    df["score_fmt"] = df["total_score"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "-")
    df["vol_fmt"] = df["vol_score"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "-")
    
    # Calculate annualized volatility from range_24h if available
    def calc_annualized_vol(row):
        if pd.notna(row.get('range_24h')) and isinstance(row['range_24h'], str) and '-' in row['range_24h']:
            try:
                parts = row['range_24h'].replace('$', '').split('-')
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                mid = (low + high) / 2
                if mid > 0:
                    daily_vol = (high - low) / mid
                    annualized_vol = daily_vol * (365 ** 0.5)
                    return f"{annualized_vol*100:.1f}%"
            except:
                pass
        return "-"
    
    df["annualized_vol_fmt"] = df.apply(calc_annualized_vol, axis=1)
    df["volume_24h_fmt"] = df["volume_24h"].apply(lambda x: f"${x/1000:.1f}K" if pd.notna(x) and x >= 1000 else f"${x:.0f}" if pd.notna(x) else "-")
    df["volume_7d_fmt"] = df["volume_7d"].apply(lambda x: f"${x/1000:.1f}K" if pd.notna(x) and x >= 1000 else f"${x:.0f}" if pd.notna(x) else "-")
    df["volume_total_fmt"] = df["volume"].apply(lambda x: f"${x/1000000:.2f}M" if pd.notna(x) and x >= 1000000 else f"${x/1000:.1f}K" if pd.notna(x) and x >= 1000 else f"${x:.0f}" if pd.notna(x) else "-")
    # Calculate daily average volume = volume_7d / 7
    df["daily_avg_vol_fmt"] = df.apply(
        lambda row: f"${row['volume_7d']/7/1000:.1f}K" if pd.notna(row.get('volume_7d')) and row['volume_7d']/7 >= 1000 
        else f"${row['volume_7d']/7:.0f}" if pd.notna(row.get('volume_7d')) 
        else "-",
        axis=1
    )
    # Vol Ratio = volume_24h / average_daily_volume where average = volume_7d / 7
    df["vol_ratio_fmt"] = df.apply(
        lambda row: f"{row['volume_24h'] / (row['volume_7d']/7):.2f}x" 
        if pd.notna(row.get('volume_24h')) and pd.notna(row.get('volume_7d')) and row['volume_7d'] > 0 and row['volume_24h'] > 0
        else "-", 
        axis=1
    )
    df["start_date_fmt"] = df["start_date"].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d") if pd.notna(x) else "-")
    df["end_date_fmt"] = df["end_date"].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d") if pd.notna(x) else "-")
    df["question_short"] = df["question"]
    df["conf_fmt"] = df["confidence"].apply(lambda x: "🔥" if x == "high" else "⚡" if x == "medium" else "•")
    df["ai_fmt"] = df["ai_bias"].apply(lambda x: x if pd.notna(x) else "-")
    # Create actual Polymarket links using slug - Testing /market/ format
    df["link_fmt"] = df.apply(
        lambda row: f"[🔗](https://polymarket.com/market/{row['slug']})" if pd.notna(row.get('slug')) and row.get('slug') else "-",
        axis=1
    )
    # Calculate Target Performance (yield from entry to target)
    try:
        df["target_perf_fmt"] = df.apply(
            lambda row: f"+{((row.get('target_price', 0) - row.get('entry_price', 0)) / row.get('entry_price', 1) * 100):.0f}%" 
            if pd.notna(row.get('entry_price')) and pd.notna(row.get('target_price')) and row.get('entry_price', 0) > 0
            else "-",
            axis=1
        )
    except Exception as e:
        logger.error(f"Error calculating target_perf_fmt: {e}")
        df["target_perf_fmt"] = "-"
    
    columns = [
        {"name": "#", "id": "rank"},
        {"name": "Signal", "id": "signal_display"},
        {"name": "", "id": "conf_fmt"},
        {"name": "Market", "id": "question_short"},
        {"name": "Link", "id": "link_fmt", "presentation": "markdown"},
        {"name": "Current", "id": "current_fmt"},
        {"name": "Entry", "id": "entry_fmt"},
        {"name": "Target", "id": "target_fmt"},
        {"name": "Perf", "id": "target_perf_fmt"},
        {"name": "Stop", "id": "stop_fmt"},
        {"name": "R/R", "id": "rr_fmt"},
        {"name": "Ann. Vol", "id": "annualized_vol_fmt"},
        {"name": "Vol 24h", "id": "volume_24h_fmt"},
        {"name": "Daily Avg", "id": "daily_avg_vol_fmt"},
        {"name": "Vol 7d", "id": "volume_7d_fmt"},
        {"name": "Vol Total", "id": "volume_total_fmt"},
        {"name": "Vol Ratio", "id": "vol_ratio_fmt"},
        {"name": "Start", "id": "start_date_fmt"},
        {"name": "End", "id": "end_date_fmt"},
        {"name": "Score", "id": "score_fmt"},
        {"name": "Vol", "id": "vol_fmt"},
    ]
    
    return dash_table.DataTable(
        id="markets-datatable",
        data=df.to_dict("records"),
        columns=columns,
        page_size=50,
        sort_action="native",
        filter_action="native",
        row_selectable=False,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": COLORS["bg_secondary"],
            "color": COLORS["text_primary"],
            "fontWeight": "bold",
            "fontSize": "13px",
            "padding": "12px 10px",
            "borderBottom": f"2px solid {COLORS['accent_primary']}",
            "textAlign": "left"
        },
        style_cell={
            "backgroundColor": COLORS["bg_card"],
            "color": COLORS["text_primary"],
            "fontSize": "13px",
            "padding": "10px",
            "borderBottom": f"1px solid {COLORS['border']}",
            "textAlign": "left",
            "whiteSpace": "nowrap",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "maxWidth": "300px",
            "cursor": "pointer"
        },
        style_cell_conditional=[
            {"if": {"column_id": "rank"}, "width": "40px", "textAlign": "center"},
            {"if": {"column_id": "signal_display"}, "width": "80px", "fontWeight": "bold"},
            {"if": {"column_id": "conf_fmt"}, "width": "30px", "textAlign": "center"},
            {"if": {"column_id": "question_short"}, "width": "500px", "maxWidth": "500px", "whiteSpace": "normal", "textOverflow": "clip"},
            {"if": {"column_id": "link_fmt"}, "width": "60px", "textAlign": "center"},
            {"if": {"column_id": "current_fmt"}, "width": "70px", "fontFamily": "monospace"},
            {"if": {"column_id": "entry_fmt"}, "width": "70px", "fontFamily": "monospace"},
            {"if": {"column_id": "target_fmt"}, "width": "70px", "fontFamily": "monospace", "color": COLORS["accent_primary"]},
            {"if": {"column_id": "stop_fmt"}, "width": "70px", "fontFamily": "monospace", "color": COLORS["sell"]},
            {"if": {"column_id": "rr_fmt"}, "width": "60px", "fontFamily": "monospace"},
            {"if": {"column_id": "annualized_vol_fmt"}, "width": "80px", "textAlign": "center", "fontFamily": "monospace", "fontWeight": "bold"},
            {"if": {"column_id": "volume_24h_fmt"}, "width": "80px", "textAlign": "right", "fontFamily": "monospace"},
            {"if": {"column_id": "volume_7d_fmt"}, "width": "80px", "textAlign": "right", "fontFamily": "monospace"},
            {"if": {"column_id": "volume_total_fmt"}, "width": "90px", "textAlign": "right", "fontFamily": "monospace"},
            {"if": {"column_id": "vol_ratio_fmt"}, "width": "70px", "textAlign": "center", "fontFamily": "monospace"},
            {"if": {"column_id": "start_date_fmt"}, "width": "90px", "textAlign": "center", "fontFamily": "monospace", "fontSize": "11px"},
            {"if": {"column_id": "end_date_fmt"}, "width": "90px", "textAlign": "center", "fontFamily": "monospace", "fontSize": "11px"},
            {"if": {"column_id": "score_fmt"}, "width": "50px", "textAlign": "center"},
            {"if": {"column_id": "vol_fmt"}, "width": "50px", "textAlign": "center"},
        ],
        style_data_conditional=[
            # High confidence highlight with orange border
            {
                "if": {"filter_query": "{confidence} = 'high'"},
                "borderLeft": f"3px solid {COLORS['accent_primary']}"
            },
            # Hover effect - dark gray background with white text
            {
                "if": {"state": "active"},
                "backgroundColor": COLORS["bg_hover"],
                "color": COLORS["text_primary"],
                "border": f"1px solid {COLORS['accent_primary']}"
            }
        ],
        style_filter={
            "backgroundColor": COLORS["bg_secondary"],
            "color": COLORS["text_primary"],
            "fontSize": "11px"
        }
    )


@callback(
    [Output("detail-modal", "is_open"),
     Output("modal-title", "children"),
     Output("modal-body", "children"),
     Output("modal-polymarket-link", "href")],
    [Input("markets-datatable", "active_cell")],
    [State("markets-datatable", "data"),
     State("detail-modal", "is_open")]
)
def toggle_modal(active_cell, table_data, is_open):
    """Show detail modal when clicking on a row (except Link column)."""
    if active_cell is None:
        return False, "", "", "#"
    
    row_idx = active_cell["row"]
    col_id = active_cell.get("column_id", "")
    row_data = table_data[row_idx]
    
    # If user clicked on Link column, open in new tab instead of modal
    if col_id == "link_fmt":
        slug = row_data.get("slug", "")
        if slug:
            import webbrowser
            webbrowser.open(f"https://polymarket.com/market/{slug}")
        return False, "", "", "#"
    
    market_id = row_data.get("market_id", "")
    question = row_data.get("question", "")
    
    # Helper function for metric card
    def metric_card(label, value, color=None, subtitle=None):
        return html.Div([
            html.Span(label, style={"color": COLORS["text_muted"], "fontSize": "11px", "display": "block"}),
            html.Div(value, style={"fontSize": "16px", "fontWeight": "bold", "fontFamily": "monospace", "color": color or COLORS["text_primary"]}),
            html.Span(subtitle, style={"color": COLORS["text_muted"], "fontSize": "10px"}) if subtitle else None
        ], style={"flex": "1", "textAlign": "center", "padding": "8px"})
    
    # Helper for section
    def section(title, icon, children):
        return html.Div([
            html.H5([icon, " ", title], style={"color": COLORS["accent_primary"], "marginBottom": "10px", "fontSize": "14px"}),
            html.Div(children, style={"backgroundColor": COLORS["bg_secondary"], "borderRadius": "8px", "padding": "12px"})
        ], style={"marginBottom": "15px"})
    
    # Calculate derived metrics
    entry_price = row_data.get('entry_price', 0) or 0
    target_price = row_data.get('target_price', 0) or 0
    stop_loss = row_data.get('stop_loss', 0) or 0
    current_price = row_data.get('current_price', 0) or 0
    
    profit_pct = ((target_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
    loss_pct = ((entry_price - stop_loss) / entry_price * 100) if entry_price > 0 else 0
    
    # Scoring breakdown
    vol_score = row_data.get('vol_score', 0) or 0
    liq_score = row_data.get('liq_score', 0) or 0
    total_score = row_data.get('total_score', 0) or 0
    
    # Volume metrics
    volume_24h = row_data.get('volume_24h', 0) or 0
    volume_7d = row_data.get('volume_7d', 0) or 0
    volume_total = row_data.get('volume', 0) or 0
    
    # Dates
    start_date = row_data.get('start_date', '')
    end_date = row_data.get('end_date', '')
    
    # Calculate days active and days remaining
    from datetime import datetime
    days_active = "-"
    days_remaining = "-"
    try:
        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            days_active = (datetime.now(start_dt.tzinfo) - start_dt).days
        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            days_remaining = max(0, (end_dt - datetime.now(end_dt.tzinfo)).days)
    except:
        pass
    
    # Build modal content
    modal_body = html.Div([
        # Signal Header
        html.Div([
            html.Div([
                html.Span(
                    f"{'🟢 BUY' if row_data.get('action') == 'BUY' else '🔴 SELL'}",
                    style={"fontSize": "28px", "fontWeight": "bold", 
                           "color": COLORS["buy"] if row_data.get("action") == "BUY" else COLORS["sell"]}
                ),
                html.Span(f"  Rank #{row_data.get('rank', '-')}", style={"color": COLORS["text_muted"], "marginLeft": "15px", "fontSize": "14px"})
            ]),
            html.Div([
                html.Span("Confidence: ", style={"color": COLORS["text_muted"]}),
                html.Span(
                    f"{'🔥🔥🔥' if row_data.get('confidence') == 'high' else '⚡⚡' if row_data.get('confidence') == 'medium' else '•'} {(row_data.get('confidence') or 'N/A').upper()}",
                    style={"color": COLORS["buy"] if row_data.get('confidence') == 'high' else COLORS["warning"] if row_data.get('confidence') == 'medium' else COLORS["text_muted"]}
                )
            ], style={"marginTop": "5px"})
        ], style={"marginBottom": "20px", "borderBottom": f"1px solid {COLORS['border']}", "paddingBottom": "15px"}),
        
        # Trade Levels Section
        section("Trade Levels", "📊", html.Div([
            html.Div([
                metric_card("Current Price", f"{current_price*100:.1f}%", COLORS["accent_primary"]),
                metric_card("Entry Price", f"{entry_price*100:.1f}%", COLORS["text_primary"]),
                metric_card("Target Price", f"{target_price*100:.1f}%", COLORS["buy"], f"+{profit_pct:.0f}%"),
                metric_card("Stop Loss", f"{stop_loss*100:.1f}%", COLORS["sell"], f"-{loss_pct:.0f}%"),
            ], style={"display": "flex", "gap": "10px"}),
            html.Div([
                metric_card("Risk/Reward", f"{row_data.get('risk_reward', 0):.2f}x", COLORS["accent_primary"]),
                metric_card("Potential Profit", f"+{profit_pct:.0f}%", COLORS["buy"]),
                metric_card("Potential Loss", f"-{loss_pct:.0f}%", COLORS["sell"]),
                metric_card("Range 24h", row_data.get('range_24h', '-'), COLORS["text_secondary"]),
            ], style={"display": "flex", "gap": "10px", "marginTop": "10px"})
        ])),
        
        # Scoring Breakdown Section
        section("Scoring Breakdown", "🎯", html.Div([
            html.Div([
                html.P("Le score total est calculé à partir de 4 composantes pondérées:", 
                       style={"color": COLORS["text_muted"], "fontSize": "11px", "marginBottom": "10px"}),
            ]),
            html.Div([
                metric_card("Total Score", f"{total_score:.0f}/100", COLORS["accent_primary"]),
                metric_card("Volatility (35%)", f"{vol_score:.0f}", COLORS["warning"], "Plus volatile = mieux"),
                metric_card("Liquidity (25%)", f"{liq_score:.0f}", COLORS["info"], "Volume élevé = mieux"),
                metric_card("Spread", f"{row_data.get('spread_bps', 0) or 0:.0f} bps" if row_data.get('spread_bps') else "N/A", COLORS["text_secondary"]),
            ], style={"display": "flex", "gap": "10px"}),
            html.Div([
                html.Div([
                    html.Div(style={
                        "height": "8px", 
                        "width": f"{min(100, total_score)}%", 
                        "backgroundColor": COLORS["buy"] if total_score >= 70 else COLORS["warning"] if total_score >= 50 else COLORS["sell"],
                        "borderRadius": "4px"
                    })
                ], style={"backgroundColor": COLORS["bg_card"], "borderRadius": "4px", "marginTop": "10px"})
            ])
        ])),
        
        # Volume & Liquidity Section
        section("Volume & Liquidity", "💰", html.Div([
            html.Div([
                metric_card("Volume 24h", f"${volume_24h:,.0f}", COLORS["accent_primary"]),
                metric_card("Volume 7d", f"${volume_7d:,.0f}", COLORS["text_primary"]),
                metric_card("Volume Total", f"${volume_total:,.0f}", COLORS["text_primary"]),
                metric_card("Daily Avg", f"${volume_total/max(1,days_active if isinstance(days_active, int) else 1):,.0f}", COLORS["text_secondary"]),
            ], style={"display": "flex", "gap": "10px"})
        ])),
        
        # Market Info Section
        section("Market Info", "📅", html.Div([
            html.Div([
                metric_card("Category", row_data.get('category', 'N/A'), COLORS["text_secondary"]),
                metric_card("Days Active", str(days_active), COLORS["text_primary"]),
                metric_card("Days to Resolution", str(days_remaining), COLORS["warning"] if isinstance(days_remaining, int) and days_remaining < 30 else COLORS["text_primary"]),
                metric_card("Market ID", market_id[:8] + "..." if len(market_id) > 8 else market_id, COLORS["text_muted"]),
            ], style={"display": "flex", "gap": "10px"}),
            html.Div([
                html.Div([
                    html.Span("Start: ", style={"color": COLORS["text_muted"], "fontSize": "11px"}),
                    html.Span(start_date[:10] if start_date else "N/A", style={"color": COLORS["text_secondary"], "fontSize": "11px"})
                ], style={"flex": "1"}),
                html.Div([
                    html.Span("End: ", style={"color": COLORS["text_muted"], "fontSize": "11px"}),
                    html.Span(end_date[:10] if end_date else "N/A", style={"color": COLORS["text_secondary"], "fontSize": "11px"})
                ], style={"flex": "1"})
            ], style={"display": "flex", "gap": "20px", "marginTop": "10px"})
        ])),
        
        # AI Analysis Section (if available)
        section("AI Analysis", "🤖", html.Div([
            html.Div([
                metric_card("AI Bias", 
                    row_data.get("ai_bias", "Not analyzed") or "Not analyzed",
                    COLORS["buy"] if row_data.get("ai_bias") == "UNDERPRICED" else COLORS["sell"] if row_data.get("ai_bias") == "OVERPRICED" else COLORS["text_muted"]
                ),
                metric_card("AI Probability",
                    f"{row_data.get('ai_prob', 0)*100:.0f}%" if row_data.get("ai_prob") else "N/A",
                    COLORS["text_primary"]
                ),
                metric_card("vs Current Price",
                    f"{(row_data.get('ai_prob', 0) - current_price)*100:+.0f}%" if row_data.get("ai_prob") else "N/A",
                    COLORS["buy"] if row_data.get('ai_prob') and row_data.get('ai_prob') > current_price else COLORS["sell"]
                ),
            ], style={"display": "flex", "gap": "10px"}),
            html.P(
                row_data.get("ai_summary", "") or "Enable AI analysis and refresh to get AI insights.",
                style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginTop": "10px", "fontStyle": "italic", "padding": "10px", "backgroundColor": COLORS["bg_card"], "borderRadius": "4px"}
            )
        ])),
        
        # Technical IDs (collapsed by default)
        html.Details([
            html.Summary("🔧 Technical Details", style={"color": COLORS["text_muted"], "cursor": "pointer", "fontSize": "12px"}),
            html.Div([
                html.Div([
                    html.Span("Token ID: ", style={"color": COLORS["text_muted"]}),
                    html.Code(row_data.get('token_id', 'N/A'), style={"fontSize": "10px", "color": COLORS["text_secondary"]})
                ]),
                html.Div([
                    html.Span("Condition ID: ", style={"color": COLORS["text_muted"]}),
                    html.Code(row_data.get('condition_id', 'N/A'), style={"fontSize": "10px", "color": COLORS["text_secondary"]})
                ]),
                html.Div([
                    html.Span("Slug: ", style={"color": COLORS["text_muted"]}),
                    html.Code(row_data.get('slug', 'N/A'), style={"fontSize": "10px", "color": COLORS["text_secondary"]})
                ]),
            ], style={"marginTop": "10px", "padding": "10px", "backgroundColor": COLORS["bg_secondary"], "borderRadius": "4px"})
        ], style={"marginTop": "15px"})
    ])
    
    # Use slug for proper Polymarket URL
    slug = row_data.get("slug", "")
    polymarket_url = f"https://polymarket.com/market/{slug}" if slug else "#"
    
    return True, question, modal_body, polymarket_url


# ===========================================
# LIVE PRICE UPDATES
# ===========================================

@callback(
    Output("live-prices-store", "data"),
    Input("live-price-refresh", "n_intervals"),
    State("recommendations-store", "data")
)
def fetch_live_prices(n_intervals, recommendations):
    """Fetch live prices for displayed markets every 10 seconds."""
    if not recommendations:
        return {}
    
    try:
        # Collect token IDs from recommendations
        token_ids = []
        market_to_token = {}
        
        for rec in recommendations[:50]:  # Limit to top 50 for performance
            token_id = rec.get("token_id")
            market_id = rec.get("market_id")
            if token_id and market_id:
                token_ids.append(token_id)
                market_to_token[token_id] = market_id
        
        if not token_ids:
            return {}
        
        # Batch fetch midpoints (20 at a time)
        live_prices = {}
        batch_size = 20
        
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i:i + batch_size]
            try:
                url = "https://clob.polymarket.com/midpoints"
                body = [{"token_id": tid} for tid in batch]
                response = requests.post(url, json=body, timeout=5)
                
                if response.status_code == 200:
                    results = response.json()
                    for j, result in enumerate(results):
                        if j < len(batch) and "mid" in result:
                            try:
                                token_id = batch[j]
                                market_id = market_to_token.get(token_id)
                                if market_id:
                                    live_prices[market_id] = float(result["mid"])
                            except (ValueError, TypeError):
                                pass
            except Exception as e:
                logger.debug(f"Error fetching batch prices: {e}")
                continue
        
        logger.debug(f"Updated {len(live_prices)} live prices")
        return live_prices
        
    except Exception as e:
        logger.error(f"Error in live price update: {e}")
        return {}


# ===========================================
# RUN SERVER
# ===========================================

def run_dashboard(debug: bool = True, port: int = 8050):
    """Run the dashboard server."""
    print(f"""
    ========================================================
    
       POLYMARKET SCANNER
       
       Dashboard running at: http://localhost:{port}
       
       Press Ctrl+C to stop
       
    ========================================================
    """)
    
    app.run(debug=debug, port=port, host="0.0.0.0")


if __name__ == "__main__":
    run_dashboard(debug=True, port=8050)
