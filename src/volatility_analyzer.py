"""
Volatility Analyzer
===================
Calculates price volatility across multiple timeframes:
- 4 hours
- 12 hours
- 24 hours
- 7 days
- Since market creation (all-time)

Also tracks price ranges (low/high) for each timeframe.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .database import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VolatilityMetrics:
    """Volatility metrics for a market."""
    market_id: str
    token_id: str
    
    # Volatility (standard deviation of returns)
    volatility_4h: Optional[float] = None
    volatility_12h: Optional[float] = None
    volatility_24h: Optional[float] = None
    volatility_7d: Optional[float] = None
    volatility_all: Optional[float] = None
    
    # Price ranges
    low_4h: Optional[float] = None
    high_4h: Optional[float] = None
    low_12h: Optional[float] = None
    high_12h: Optional[float] = None
    low_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_7d: Optional[float] = None
    high_7d: Optional[float] = None
    low_all: Optional[float] = None
    high_all: Optional[float] = None
    
    # Current price info
    current_price: Optional[float] = None
    price_change_24h: Optional[float] = None
    
    # Position in range (0 = at low, 1 = at high)
    position_in_range_24h: Optional[float] = None
    position_in_range_7d: Optional[float] = None
    
    # Mean reversion score (-1 to 1, negative = oversold, positive = overbought)
    mean_reversion_score: Optional[float] = None
    
    # Composite volatility score (weighted average)
    composite_volatility_score: Optional[float] = None


class VolatilityAnalyzer:
    """Analyzes price volatility for Polymarket markets."""
    
    # Timeframes in hours
    TIMEFRAMES = {
        "4h": 4,
        "12h": 12,
        "24h": 24,
        "7d": 24 * 7,
        "all": None  # All available data
    }
    
    # Weights for composite score
    VOLATILITY_WEIGHTS = {
        "4h": 0.15,
        "12h": 0.20,
        "24h": 0.30,
        "7d": 0.25,
        "all": 0.10
    }
    
    def __init__(self):
        self.db = db
    
    def calculate_volatility(self, prices: List[float]) -> Optional[float]:
        """
        Calculate volatility as the standard deviation of log returns.
        
        Volatility = std(log(price[t] / price[t-1]))
        """
        if len(prices) < 2:
            return None
        
        prices = np.array(prices)
        # Remove zeros and negatives
        prices = prices[prices > 0]
        
        if len(prices) < 2:
            return None
        
        # Calculate log returns
        log_returns = np.log(prices[1:] / prices[:-1])
        
        # Standard deviation of returns
        volatility = np.std(log_returns)
        
        # Annualize (optional, but common in finance)
        # Assuming ~24 observations per day for hourly data
        # volatility_annualized = volatility * np.sqrt(24 * 365)
        
        return float(volatility)
    
    def calculate_range(self, prices: List[float]) -> Tuple[Optional[float], Optional[float]]:
        """Calculate price range (low, high)."""
        if not prices:
            return None, None
        
        prices = [p for p in prices if p is not None and p > 0]
        if not prices:
            return None, None
        
        return min(prices), max(prices)
    
    def calculate_position_in_range(self, current: float, low: float, high: float) -> Optional[float]:
        """
        Calculate where current price sits in the range.
        Returns 0 if at low, 1 if at high, 0.5 if in middle.
        """
        if low is None or high is None or current is None:
            return None
        
        if high == low:
            return 0.5
        
        position = (current - low) / (high - low)
        return max(0, min(1, position))
    
    def calculate_mean_reversion_score(self, prices: List[float], current: float) -> Optional[float]:
        """
        Calculate mean reversion score based on deviation from moving average.
        
        Returns:
        - Negative values: price below average (potential buy signal)
        - Positive values: price above average (potential sell signal)
        - Range: -1 to 1 (based on standard deviations)
        """
        if len(prices) < 5 or current is None:
            return None
        
        prices = np.array([p for p in prices if p is not None and p > 0])
        if len(prices) < 5:
            return None
        
        mean = np.mean(prices)
        std = np.std(prices)
        
        if std == 0:
            return 0
        
        # Z-score (number of std deviations from mean)
        z_score = (current - mean) / std
        
        # Normalize to -1 to 1 range (using tanh)
        normalized = np.tanh(z_score / 2)
        
        return float(normalized)
    
    def analyze_market(self, market_id: str, token_id: str) -> Optional[VolatilityMetrics]:
        """
        Perform full volatility analysis for a market.
        """
        metrics = VolatilityMetrics(market_id=market_id, token_id=token_id)
        
        # Get all price history
        all_history = self.db.get_price_history(market_id, token_id)
        
        if not all_history:
            logger.debug(f"No price history for {market_id}")
            return None
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(all_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        
        # Use 'price' column, fall back to 'mid' if not available
        price_col = "price" if "price" in df.columns and df["price"].notna().any() else "mid"
        if price_col not in df.columns or df[price_col].isna().all():
            return None
        
        all_prices = df[price_col].dropna().tolist()
        
        if not all_prices:
            return None
        
        # Current price
        metrics.current_price = all_prices[-1] if all_prices else None
        
        now = datetime.utcnow()
        
        # Calculate for each timeframe
        for tf_name, hours in self.TIMEFRAMES.items():
            if hours is not None:
                cutoff = now - timedelta(hours=hours)
                tf_df = df[df["timestamp"] >= cutoff]
            else:
                tf_df = df
            
            prices = tf_df[price_col].dropna().tolist()
            
            if prices:
                vol = self.calculate_volatility(prices)
                low, high = self.calculate_range(prices)
                
                # Set volatility
                setattr(metrics, f"volatility_{tf_name}", vol)
                
                # Set range
                setattr(metrics, f"low_{tf_name}", low)
                setattr(metrics, f"high_{tf_name}", high)
        
        # Calculate position in range
        if metrics.low_24h is not None and metrics.high_24h is not None:
            metrics.position_in_range_24h = self.calculate_position_in_range(
                metrics.current_price, metrics.low_24h, metrics.high_24h
            )
        
        if metrics.low_7d is not None and metrics.high_7d is not None:
            metrics.position_in_range_7d = self.calculate_position_in_range(
                metrics.current_price, metrics.low_7d, metrics.high_7d
            )
        
        # Mean reversion score (using 24h data)
        cutoff_24h = now - timedelta(hours=24)
        prices_24h = df[df["timestamp"] >= cutoff_24h][price_col].dropna().tolist()
        if prices_24h and metrics.current_price:
            metrics.mean_reversion_score = self.calculate_mean_reversion_score(
                prices_24h, metrics.current_price
            )
        
        # 24h price change
        if len(all_prices) >= 2 and prices_24h:
            oldest_24h = prices_24h[0] if prices_24h else all_prices[0]
            if oldest_24h and oldest_24h > 0:
                metrics.price_change_24h = (metrics.current_price - oldest_24h) / oldest_24h
        
        # Composite volatility score
        metrics.composite_volatility_score = self._calculate_composite_score(metrics)
        
        return metrics
    
    def _calculate_composite_score(self, metrics: VolatilityMetrics) -> Optional[float]:
        """
        Calculate weighted composite volatility score.
        Higher score = more volatile = better for scalping.
        """
        total_weight = 0
        weighted_sum = 0
        
        for tf_name, weight in self.VOLATILITY_WEIGHTS.items():
            vol = getattr(metrics, f"volatility_{tf_name}", None)
            if vol is not None:
                weighted_sum += vol * weight
                total_weight += weight
        
        if total_weight == 0:
            return None
        
        return weighted_sum / total_weight
    
    def analyze_markets(self, markets: List[Dict]) -> List[VolatilityMetrics]:
        """
        Analyze volatility for multiple markets.
        """
        from .api_client import client
        
        results = []
        
        for market in markets:
            market_id = market.get("id")
            token_ids = client.parse_clob_token_ids(market)
            
            if not token_ids:
                continue
            
            # Analyze first token (YES outcome)
            token_id = token_ids[0]
            metrics = self.analyze_market(market_id, token_id)
            
            if metrics:
                results.append(metrics)
        
        # Sort by composite volatility score (descending)
        results.sort(
            key=lambda m: m.composite_volatility_score if m.composite_volatility_score else 0,
            reverse=True
        )
        
        logger.info(f"Analyzed volatility for {len(results)} markets")
        return results
    
    def get_top_volatile_markets(self, n: int = 100) -> List[VolatilityMetrics]:
        """
        Get top N most volatile markets.
        Requires price history to be already collected in the database.
        """
        markets = self.db.get_all_markets()
        metrics = self.analyze_markets(markets)
        return metrics[:n]


# Singleton instance
volatility_analyzer = VolatilityAnalyzer()


if __name__ == "__main__":
    print("Testing Volatility Analyzer...")
    
    # This requires price history in the database
    top_volatile = volatility_analyzer.get_top_volatile_markets(10)
    
    for m in top_volatile:
        print(f"\nMarket: {m.market_id}")
        print(f"  Composite Score: {m.composite_volatility_score:.4f}" if m.composite_volatility_score else "  No data")
        print(f"  24h Range: {m.low_24h:.3f} - {m.high_24h:.3f}" if m.low_24h else "  No 24h data")
