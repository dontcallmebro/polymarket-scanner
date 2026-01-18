"""
Order Book Analyzer
===================
Analyzes order book depth and liquidity:
- Bid/Ask spread
- Depth at various price levels
- Imbalance ratio (buying vs selling pressure)
- Liquidity scoring
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import json

from .database import db
from .api_client import client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrderBookMetrics:
    """Order book analysis metrics."""
    market_id: str
    token_id: str
    
    # Best prices
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    mid_price: Optional[float] = None
    
    # Spread
    spread: Optional[float] = None
    spread_bps: Optional[float] = None  # Spread in basis points
    
    # Depth (total size on each side)
    bid_depth_total: Optional[float] = None
    ask_depth_total: Optional[float] = None
    
    # Depth at levels (size within X% of best price)
    bid_depth_1pct: Optional[float] = None
    ask_depth_1pct: Optional[float] = None
    bid_depth_5pct: Optional[float] = None
    ask_depth_5pct: Optional[float] = None
    
    # Number of levels
    bid_levels: int = 0
    ask_levels: int = 0
    
    # Imbalance ratio (0.5 = balanced, >0.5 = more bids, <0.5 = more asks)
    imbalance_ratio: Optional[float] = None
    
    # Volume-weighted average prices
    vwap_bid: Optional[float] = None
    vwap_ask: Optional[float] = None
    
    # Liquidity score (composite metric)
    liquidity_score: Optional[float] = None
    
    # Raw orderbook (for reference)
    bids: List[Dict] = None
    asks: List[Dict] = None


class OrderBookAnalyzer:
    """Analyzes order book depth and liquidity."""
    
    def __init__(self):
        self.db = db
        self.client = client
    
    def fetch_and_analyze(self, market_id: str, token_id: str) -> Optional[OrderBookMetrics]:
        """
        Fetch current order book from API and analyze it.
        Also stores snapshot in database.
        """
        # Fetch from API
        orderbook = self.client.get_order_book(token_id)
        
        if not orderbook:
            logger.debug(f"No orderbook data for {token_id}")
            return None
        
        # Store snapshot
        self.db.insert_orderbook_snapshot(market_id, token_id, orderbook)
        
        # Analyze
        return self.analyze_orderbook(market_id, token_id, orderbook)
    
    def analyze_orderbook(self, market_id: str, token_id: str, 
                          orderbook: Dict) -> Optional[OrderBookMetrics]:
        """
        Analyze an order book and compute metrics.
        """
        metrics = OrderBookMetrics(
            market_id=market_id,
            token_id=token_id,
            bids=[],
            asks=[]
        )
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids and not asks:
            return None
        
        # Parse bids and asks
        parsed_bids = []
        for b in bids:
            try:
                parsed_bids.append({
                    "price": float(b.get("price", 0)),
                    "size": float(b.get("size", 0))
                })
            except (ValueError, TypeError):
                continue
        
        parsed_asks = []
        for a in asks:
            try:
                parsed_asks.append({
                    "price": float(a.get("price", 0)),
                    "size": float(a.get("size", 0))
                })
            except (ValueError, TypeError):
                continue
        
        # Sort: bids descending (highest first), asks ascending (lowest first)
        parsed_bids.sort(key=lambda x: x["price"], reverse=True)
        parsed_asks.sort(key=lambda x: x["price"])
        
        metrics.bids = parsed_bids
        metrics.asks = parsed_asks
        metrics.bid_levels = len(parsed_bids)
        metrics.ask_levels = len(parsed_asks)
        
        # Best prices
        if parsed_bids:
            metrics.best_bid = parsed_bids[0]["price"]
        if parsed_asks:
            metrics.best_ask = parsed_asks[0]["price"]
        
        # Mid price and spread
        if metrics.best_bid and metrics.best_ask:
            metrics.mid_price = (metrics.best_bid + metrics.best_ask) / 2
            metrics.spread = metrics.best_ask - metrics.best_bid
            if metrics.mid_price > 0:
                metrics.spread_bps = (metrics.spread / metrics.mid_price) * 10000
        
        # Total depth
        metrics.bid_depth_total = sum(b["size"] for b in parsed_bids)
        metrics.ask_depth_total = sum(a["size"] for a in parsed_asks)
        
        # Depth at price levels (within X% of best)
        if metrics.best_bid:
            metrics.bid_depth_1pct = self._depth_within_pct(parsed_bids, metrics.best_bid, 0.01, is_bid=True)
            metrics.bid_depth_5pct = self._depth_within_pct(parsed_bids, metrics.best_bid, 0.05, is_bid=True)
        
        if metrics.best_ask:
            metrics.ask_depth_1pct = self._depth_within_pct(parsed_asks, metrics.best_ask, 0.01, is_bid=False)
            metrics.ask_depth_5pct = self._depth_within_pct(parsed_asks, metrics.best_ask, 0.05, is_bid=False)
        
        # Imbalance ratio
        total_depth = (metrics.bid_depth_total or 0) + (metrics.ask_depth_total or 0)
        if total_depth > 0:
            metrics.imbalance_ratio = (metrics.bid_depth_total or 0) / total_depth
        
        # VWAP
        metrics.vwap_bid = self._calculate_vwap(parsed_bids)
        metrics.vwap_ask = self._calculate_vwap(parsed_asks)
        
        # Liquidity score
        metrics.liquidity_score = self._calculate_liquidity_score(metrics)
        
        return metrics
    
    def _depth_within_pct(self, orders: List[Dict], best_price: float, 
                          pct: float, is_bid: bool) -> float:
        """
        Calculate total depth within X% of the best price.
        For bids: within (best_bid * (1 - pct)) to best_bid
        For asks: within best_ask to (best_ask * (1 + pct))
        """
        if is_bid:
            threshold = best_price * (1 - pct)
            return sum(o["size"] for o in orders if o["price"] >= threshold)
        else:
            threshold = best_price * (1 + pct)
            return sum(o["size"] for o in orders if o["price"] <= threshold)
    
    def _calculate_vwap(self, orders: List[Dict]) -> Optional[float]:
        """Calculate volume-weighted average price."""
        if not orders:
            return None
        
        total_value = sum(o["price"] * o["size"] for o in orders)
        total_size = sum(o["size"] for o in orders)
        
        if total_size == 0:
            return None
        
        return total_value / total_size
    
    def _calculate_liquidity_score(self, metrics: OrderBookMetrics) -> Optional[float]:
        """
        Calculate a composite liquidity score (0-100).
        
        Factors:
        - Tighter spread = better
        - More depth = better
        - More levels = better
        - Balanced imbalance = better
        """
        if metrics.spread is None:
            return None
        
        score = 0
        max_score = 0
        
        # Spread score (30 points max)
        # <1% spread = full points, >10% = 0 points
        max_score += 30
        if metrics.spread_bps is not None:
            spread_pct = metrics.spread_bps / 100
            if spread_pct <= 1:
                score += 30
            elif spread_pct < 10:
                score += 30 * (1 - (spread_pct - 1) / 9)
        
        # Depth score (30 points max)
        max_score += 30
        total_depth = (metrics.bid_depth_total or 0) + (metrics.ask_depth_total or 0)
        # $10k+ depth = full points
        if total_depth >= 10000:
            score += 30
        elif total_depth > 0:
            score += 30 * min(1, total_depth / 10000)
        
        # Levels score (20 points max)
        max_score += 20
        total_levels = metrics.bid_levels + metrics.ask_levels
        # 20+ levels = full points
        if total_levels >= 20:
            score += 20
        else:
            score += total_levels
        
        # Balance score (20 points max)
        max_score += 20
        if metrics.imbalance_ratio is not None:
            # Closer to 0.5 = better
            balance = 1 - abs(metrics.imbalance_ratio - 0.5) * 2
            score += 20 * balance
        
        return (score / max_score) * 100 if max_score > 0 else None
    
    def analyze_markets(self, markets: List[Dict]) -> List[OrderBookMetrics]:
        """
        Analyze order books for multiple markets.
        """
        results = []
        
        for market in markets:
            market_id = market.get("id")
            token_ids = self.client.parse_clob_token_ids(market)
            
            if not token_ids:
                continue
            
            # Analyze first token (YES outcome)
            token_id = token_ids[0]
            metrics = self.fetch_and_analyze(market_id, token_id)
            
            if metrics:
                results.append(metrics)
        
        # Sort by liquidity score (descending)
        results.sort(
            key=lambda m: m.liquidity_score if m.liquidity_score else 0,
            reverse=True
        )
        
        logger.info(f"Analyzed orderbooks for {len(results)} markets")
        return results
    
    def get_depth_summary(self, metrics: OrderBookMetrics) -> Dict:
        """
        Get a summary of order book depth for display.
        """
        return {
            "best_bid": f"${metrics.best_bid:.3f}" if metrics.best_bid else "N/A",
            "best_ask": f"${metrics.best_ask:.3f}" if metrics.best_ask else "N/A",
            "spread": f"{metrics.spread_bps:.1f} bps" if metrics.spread_bps else "N/A",
            "bid_depth": f"${metrics.bid_depth_total:,.0f}" if metrics.bid_depth_total else "N/A",
            "ask_depth": f"${metrics.ask_depth_total:,.0f}" if metrics.ask_depth_total else "N/A",
            "imbalance": f"{metrics.imbalance_ratio:.1%}" if metrics.imbalance_ratio else "N/A",
            "liquidity_score": f"{metrics.liquidity_score:.1f}/100" if metrics.liquidity_score else "N/A"
        }


# Singleton instance
orderbook_analyzer = OrderBookAnalyzer()


if __name__ == "__main__":
    print("Testing Order Book Analyzer...")
    
    # Fetch a sample market
    markets = client.get_markets(limit=1)
    
    if markets:
        market = markets[0]
        token_ids = client.parse_clob_token_ids(market)
        
        if token_ids:
            metrics = orderbook_analyzer.fetch_and_analyze(market["id"], token_ids[0])
            
            if metrics:
                print(f"\nMarket: {market.get('question', 'N/A')[:60]}")
                summary = orderbook_analyzer.get_depth_summary(metrics)
                for k, v in summary.items():
                    print(f"  {k}: {v}")
