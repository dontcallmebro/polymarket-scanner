"""
Polymarket Volatility Analyzer - Main Entry Point
==================================================
Run this script to start the dashboard or execute data collection.

Usage:
    python main.py dashboard     - Start the web dashboard
    python main.py collect       - Run one-time data collection
    python main.py analyze       - Generate trade recommendations (console)
    python main.py scheduler     - Start continuous data collection
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def collect_data():
    """Collect current market data and store in database."""
    from src.api_client import client
    from src.database import db
    from src.orderbook_analyzer import orderbook_analyzer
    
    logger.info("Starting data collection...")
    
    # Fetch all markets
    markets = client.get_all_markets(active=True, min_volume=500)
    logger.info(f"Fetched {len(markets)} markets")
    
    # Filter by criteria
    filtered = client.filter_markets_by_criteria(
        markets,
        min_age_days=14,
        min_maturity_days=14,
        min_volume=500
    )
    logger.info(f"Filtered to {len(filtered)} eligible markets")
    
    # Store markets in database
    db.upsert_markets(markets)
    
    # Collect price and orderbook data for filtered markets
    collected = 0
    for market in filtered[:200]:  # Limit to top 200 to avoid rate limits
        market_id = market.get("id")
        token_ids = client.parse_clob_token_ids(market)
        
        if not token_ids:
            continue
        
        token_id = token_ids[0]
        
        # Get current prices
        prices = client.parse_outcome_prices(market)
        if prices:
            price = prices[0]
            
            # Get order book
            orderbook = client.get_order_book(token_id)
            
            if orderbook:
                bids = orderbook.get("bids", [])
                asks = orderbook.get("asks", [])
                
                best_bid = float(bids[0]["price"]) if bids else None
                best_ask = float(asks[0]["price"]) if asks else None
                mid = (best_bid + best_ask) / 2 if best_bid and best_ask else price
                
                # Store price history
                db.insert_price(
                    market_id=market_id,
                    token_id=token_id,
                    price=price,
                    bid=best_bid,
                    ask=best_ask,
                    mid=mid,
                    volume_24h=float(market.get("volume24hr", 0) or 0)
                )
                
                # Store orderbook snapshot
                db.insert_orderbook_snapshot(market_id, token_id, orderbook)
                
                collected += 1
    
    logger.info(f"Collected data for {collected} markets")
    return collected


def run_scheduler():
    """Run continuous data collection on a schedule."""
    import schedule
    import time
    
    logger.info("Starting data collection scheduler...")
    logger.info("Collecting data every 15 minutes")
    
    # Run immediately
    collect_data()
    
    # Schedule regular collection
    schedule.every(15).minutes.do(collect_data)
    
    while True:
        schedule.run_pending()
        time.sleep(60)


def analyze_and_print():
    """Generate and print trade recommendations to console."""
    from src.trade_recommender import trade_recommender
    
    logger.info("Generating trade recommendations...")
    
    recs = trade_recommender.generate_recommendations(
        n=10,
        include_ai=True,
        min_rr_ratio=1.0
    )
    
    output = trade_recommender.format_recommendations_table(recs)
    print(output)
    
    return recs


def run_dashboard(port: int = 8050, debug: bool = True):
    """Start the web dashboard."""
    from src.dashboard.app import run_dashboard as start_dash
    start_dash(debug=debug, port=port)


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Volatility Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py dashboard      Start web dashboard on port 8050
  python main.py dashboard -p 8080  Start on custom port
  python main.py collect        Run one-time data collection
  python main.py analyze        Print trade recommendations to console
  python main.py scheduler      Start continuous data collection
        """
    )
    
    parser.add_argument(
        "command",
        choices=["dashboard", "collect", "analyze", "scheduler"],
        help="Command to run"
    )
    
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8050,
        help="Port for dashboard (default: 8050)"
    )
    
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug mode for dashboard"
    )
    
    args = parser.parse_args()
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   ◆ POLYMARKET VOLATILITY ANALYZER                       ║
    ║                                                          ║
    ║   Command: {args.command:<44} ║
    ║   Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'):<47} ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    if args.command == "dashboard":
        run_dashboard(port=args.port, debug=not args.no_debug)
    
    elif args.command == "collect":
        collect_data()
    
    elif args.command == "analyze":
        analyze_and_print()
    
    elif args.command == "scheduler":
        run_scheduler()


if __name__ == "__main__":
    main()
