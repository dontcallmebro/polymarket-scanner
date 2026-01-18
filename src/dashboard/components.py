"""
Polymarket Volatility Dashboard - Components
=============================================
Reusable UI components for the dashboard.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional

from .theme import (
    COLORS, FONTS, CARD_STYLE, METRIC_BOX_STYLE, 
    METRIC_VALUE_STYLE, METRIC_LABEL_STYLE,
    get_signal_color, get_confidence_color, get_bias_color,
    PLOTLY_TEMPLATE
)


def create_metric_card(title: str, value: str, subtitle: str = "", color: str = None):
    """Create a metric display card."""
    value_color = color or COLORS["accent_primary"]
    
    return html.Div([
        html.Div(value, style={**METRIC_VALUE_STYLE, "color": value_color}),
        html.Div(title, style=METRIC_LABEL_STYLE),
        html.Div(subtitle, style={"fontSize": "11px", "color": COLORS["text_muted"], "marginTop": "5px"}) if subtitle else None
    ], style=METRIC_BOX_STYLE)


def create_header():
    """Create the dashboard header."""
    return html.Div([
        html.Div([
            html.Div([
                html.Span("◆ ", style={"color": COLORS["accent_primary"], "fontSize": "24px"}),
                html.Span("POLYMARKET ", style={
                    "fontWeight": "bold",
                    "fontSize": "20px",
                    "color": COLORS["text_primary"],
                    "fontFamily": FONTS["heading"]
                }),
                html.Span("VOLATILITY SCANNER", style={
                    "fontWeight": "300",
                    "fontSize": "20px",
                    "color": COLORS["accent_primary"],
                    "fontFamily": FONTS["heading"]
                }),
            ], style={"display": "flex", "alignItems": "center"}),
            
            html.Div([
                html.Span("●", style={"color": COLORS["success"], "marginRight": "8px"}, className="live-indicator"),
                html.Span("LIVE", style={
                    "color": COLORS["success"],
                    "fontSize": "12px",
                    "fontWeight": "bold",
                    "marginRight": "20px"
                }),
                html.Span(id="current-time", style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "12px",
                    "fontFamily": FONTS["primary"]
                }),
                html.Span(" | Updates every 10s", style={
                    "color": COLORS["text_muted"],
                    "fontSize": "10px",
                    "marginLeft": "10px"
                })
            ], style={"display": "flex", "alignItems": "center"})
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "width": "100%"
        })
    ], style={
        "backgroundColor": COLORS["bg_secondary"],
        "padding": "15px 30px",
        "borderBottom": f"2px solid {COLORS['accent_primary']}",
        "marginBottom": "20px",
    })


def create_summary_metrics(data: Dict):
    """Create the summary metrics row with explanation."""
    
    # Explanation paragraph
    explanation = html.Div([
        html.P([
            html.Strong("📊 Comment fonctionne le modèle: "),
            "Le scanner récupère tous les marchés actifs de Polymarket via l'API (~24,000+). ",
            "Il filtre ensuite par critères: âge > 14 jours, maturité > 14 jours, volume > $1K. ",
            "Si des bornes de prix sont définies (Min/Max Price), seuls les marchés dans cette plage sont analysés. ",
            "Le modèle calcule un score de volatilité basé sur: range de prix 24h/7j, volume, liquidité et spread. ",
            "Les signaux BUY/SELL sont générés par mean reversion (achat quand le prix est dans les 30% bas de sa plage, vente dans les 30% hauts)."
        ], style={"color": COLORS["text_secondary"], "fontSize": "12px", "margin": "0 0 15px 0", "lineHeight": "1.5"})
    ])
    
    # Pipeline metrics
    pipeline_info = f"API -> {data.get('total_fetched', 0):,} -> Critères -> {data.get('after_criteria', 0):,} -> Prix -> {data.get('after_price_filter', 0):,} -> Top {data.get('total_markets', 0)}"
    
    return html.Div([
        explanation,
        html.Div([
            html.Span("Pipeline: ", style={"color": COLORS["text_muted"], "fontSize": "11px"}),
            html.Span(pipeline_info, style={"color": COLORS["accent_primary"], "fontSize": "11px", "fontFamily": "monospace"})
        ], style={"marginBottom": "10px"}),
        html.Div([
            create_metric_card(
                "API Fetched",
                f"{data.get('total_fetched', 0):,}",
                "Total markets",
                COLORS["text_muted"]
            ),
            create_metric_card(
                "After Criteria",
                f"{data.get('after_criteria', 0):,}",
                "Age/Maturity/Vol",
                COLORS["text_secondary"]
            ),
            create_metric_card(
                "In Price Range",
                f"{data.get('after_price_filter', 0):,}",
                "Min-Max filter",
                COLORS["accent_primary"]
            ),
            create_metric_card(
                "Displayed",
                str(data.get("total_markets", 0)),
                "Top scored"
            ),
            create_metric_card(
                "Buy Signals",
                str(data.get("buy_signals", 0)),
                "Opportunities",
                COLORS["buy"]
            ),
            create_metric_card(
                "Sell Signals",
                str(data.get("sell_signals", 0)),
                "Opportunities",
                COLORS["sell"]
            ),
            create_metric_card(
                "High Confidence",
                str(data.get("high_confidence", 0)),
                "Trades",
                COLORS["conf_high"]
            ),
        ], style={
            "display": "flex",
            "gap": "15px",
            "marginBottom": "10px",
            "overflowX": "auto",
            "padding": "5px"
        })
    ])


def create_trade_card(rec: Dict, rank: int):
    """Create a trade recommendation card."""
    action = rec.get("action", "HOLD")
    confidence = rec.get("confidence", "low")
    
    action_color = get_signal_color(action)
    conf_color = get_confidence_color(confidence)
    
    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.Span(f"#{rank}", style={
                    "backgroundColor": COLORS["accent_primary"],
                    "color": COLORS["text_primary"],
                    "padding": "4px 10px",
                    "borderRadius": "4px",
                    "fontWeight": "bold",
                    "marginRight": "10px"
                }),
                html.Span(action, style={
                    "color": action_color,
                    "fontWeight": "bold",
                    "fontSize": "16px",
                    "marginRight": "10px"
                }),
                html.Span(confidence.upper(), style={
                    "color": conf_color,
                    "fontSize": "12px",
                    "padding": "2px 8px",
                    "border": f"1px solid {conf_color}",
                    "borderRadius": "3px"
                }),
            ]),
            html.Span(rec.get("category", ""), style={
                "color": COLORS["text_muted"],
                "fontSize": "11px"
            })
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "marginBottom": "12px"
        }),
        
        # Question
        html.Div(rec.get("question", "")[:80] + "..." if len(rec.get("question", "")) > 80 else rec.get("question", ""), style={
            "color": COLORS["text_primary"],
            "fontSize": "14px",
            "marginBottom": "15px",
            "lineHeight": "1.4"
        }),
        
        # Prices row
        html.Div([
            html.Div([
                html.Div("Entry", style={"color": COLORS["text_muted"], "fontSize": "10px"}),
                html.Div(f"${rec.get('entry_price', 0):.3f}", style={"color": COLORS["text_primary"], "fontWeight": "bold"})
            ], style={"textAlign": "center"}),
            html.Div("→", style={"color": COLORS["accent_primary"], "fontSize": "20px"}),
            html.Div([
                html.Div("Target", style={"color": COLORS["text_muted"], "fontSize": "10px"}),
                html.Div(f"${rec.get('target_price', 0):.3f}", style={"color": COLORS["buy"], "fontWeight": "bold"})
            ], style={"textAlign": "center"}),
            html.Div([
                html.Div("Stop", style={"color": COLORS["text_muted"], "fontSize": "10px"}),
                html.Div(f"${rec.get('stop_loss', 0):.3f}", style={"color": COLORS["sell"], "fontWeight": "bold"})
            ], style={"textAlign": "center"}),
        ], style={
            "display": "flex",
            "justifyContent": "space-around",
            "alignItems": "center",
            "backgroundColor": COLORS["bg_primary"],
            "padding": "12px",
            "borderRadius": "6px",
            "marginBottom": "12px"
        }),
        
        # Metrics row
        html.Div([
            html.Div([
                html.Span("R/R: ", style={"color": COLORS["text_muted"]}),
                html.Span(f"{rec.get('risk_reward', 0):.2f}x", style={"color": COLORS["accent_primary"], "fontWeight": "bold"})
            ]),
            html.Div([
                html.Span("Score: ", style={"color": COLORS["text_muted"]}),
                html.Span(f"{rec.get('total_score', 0):.0f}", style={"color": COLORS["accent_secondary"]})
            ]),
            html.Div([
                html.Span("Vol: ", style={"color": COLORS["text_muted"]}),
                html.Span(f"${rec.get('volume_24h', 0):,.0f}", style={"color": COLORS["text_secondary"]})
            ]),
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "fontSize": "12px",
            "marginBottom": "12px"
        }),
        
        # AI Analysis (if available)
        html.Div([
            html.Div([
                html.Span("🤖 AI: ", style={"marginRight": "5px"}),
                html.Span(rec.get("ai_bias", "N/A"), style={
                    "color": get_bias_color(rec.get("ai_bias")),
                    "fontWeight": "bold"
                }),
                html.Span(f" ({rec.get('ai_prob', 0):.0%})" if rec.get('ai_prob') else "", style={
                    "color": COLORS["text_muted"]
                })
            ]),
            html.Div(rec.get("ai_summary", "")[:60] + "..." if rec.get("ai_summary") and len(rec.get("ai_summary", "")) > 60 else rec.get("ai_summary", ""), style={
                "color": COLORS["text_muted"],
                "fontSize": "11px",
                "marginTop": "4px"
            }) if rec.get("ai_summary") else None
        ], style={
            "backgroundColor": COLORS["bg_secondary"],
            "padding": "10px",
            "borderRadius": "4px",
            "fontSize": "12px"
        }) if rec.get("ai_bias") else None
        
    ], style={
        **CARD_STYLE,
        "border": f"1px solid {action_color}22",
        "borderLeft": f"4px solid {action_color}",
    })


def create_volatility_chart(data: List[Dict]):
    """Create volatility distribution chart."""
    if not data:
        return html.Div("No data available", style={"color": COLORS["text_muted"], "padding": "20px"})
    
    fig = go.Figure()
    
    # Extract volatility scores
    vol_scores = [d.get("vol_score", 0) for d in data if d.get("vol_score")]
    questions = [d.get("question", "")[:30] + "..." for d in data[:20]]
    
    fig.add_trace(go.Bar(
        x=questions[:20],
        y=vol_scores[:20],
        marker_color=COLORS["accent_primary"],
        marker_line_color=COLORS["accent_secondary"],
        marker_line_width=1,
        opacity=0.8
    ))
    
    fig.update_layout(
        title="Top 20 Markets by Volatility Score",
        paper_bgcolor=COLORS["bg_card"],
        plot_bgcolor=COLORS["bg_primary"],
        font={"family": FONTS["primary"], "color": COLORS["text_primary"]},
        xaxis={"tickangle": -45, "gridcolor": COLORS["chart_grid"]},
        yaxis={"gridcolor": COLORS["chart_grid"], "title": "Volatility Score"},
        margin={"l": 50, "r": 20, "t": 50, "b": 120},
        height=350
    )
    
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_orderbook_depth_chart(bids: List, asks: List):
    """Create order book depth visualization."""
    fig = go.Figure()
    
    if bids:
        bid_prices = [float(b.get("price", 0)) for b in bids]
        bid_sizes = [float(b.get("size", 0)) for b in bids]
        # Cumulative depth
        bid_cum = []
        total = 0
        for s in bid_sizes:
            total += s
            bid_cum.append(total)
        
        fig.add_trace(go.Scatter(
            x=bid_prices,
            y=bid_cum,
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.2)',
            line={"color": COLORS["chart_bid"], "width": 2},
            name="Bids"
        ))
    
    if asks:
        ask_prices = [float(a.get("price", 0)) for a in asks]
        ask_sizes = [float(a.get("size", 0)) for a in asks]
        ask_cum = []
        total = 0
        for s in ask_sizes:
            total += s
            ask_cum.append(total)
        
        fig.add_trace(go.Scatter(
            x=ask_prices,
            y=ask_cum,
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.2)',
            line={"color": COLORS["chart_ask"], "width": 2},
            name="Asks"
        ))
    
    fig.update_layout(
        title="Order Book Depth",
        paper_bgcolor=COLORS["bg_card"],
        plot_bgcolor=COLORS["bg_primary"],
        font={"family": FONTS["primary"], "color": COLORS["text_primary"]},
        xaxis={"title": "Price", "gridcolor": COLORS["chart_grid"]},
        yaxis={"title": "Cumulative Size", "gridcolor": COLORS["chart_grid"]},
        legend={"orientation": "h", "y": 1.1},
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        height=300
    )
    
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_signals_distribution_chart(data: List[Dict]):
    """Create pie chart of signal distribution."""
    if not data:
        return html.Div("No data", style={"color": COLORS["text_muted"]})
    
    buy_count = len([d for d in data if d.get("action") == "BUY"])
    sell_count = len([d for d in data if d.get("action") == "SELL"])
    hold_count = len([d for d in data if d.get("action") == "HOLD"])
    
    fig = go.Figure(data=[go.Pie(
        labels=["BUY", "SELL", "HOLD"],
        values=[buy_count, sell_count, hold_count],
        marker={"colors": [COLORS["buy"], COLORS["sell"], COLORS["hold"]]},
        hole=0.6,
        textinfo="label+value",
        textfont={"color": COLORS["text_primary"]}
    )])
    
    fig.update_layout(
        paper_bgcolor=COLORS["bg_card"],
        plot_bgcolor=COLORS["bg_primary"],
        font={"family": FONTS["primary"], "color": COLORS["text_primary"]},
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=200,
        annotations=[{
            "text": "Signals",
            "showarrow": False,
            "font": {"size": 14, "color": COLORS["text_secondary"]}
        }]
    )
    
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def create_loading_spinner():
    """Create a loading spinner component."""
    return html.Div([
        html.Div(className="spinner", style={
            "width": "50px",
            "height": "50px",
            "border": f"4px solid {COLORS['bg_hover']}",
            "borderTop": f"4px solid {COLORS['accent_primary']}",
            "borderRadius": "50%",
            "animation": "spin 1s linear infinite"
        }),
        html.Div("Loading data...", style={
            "color": COLORS["text_secondary"],
            "marginTop": "15px"
        })
    ], style={
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "justifyContent": "center",
        "padding": "50px"
    })
