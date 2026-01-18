"""
Trade Scorer
=============
Combines volatility, order book, and other metrics into a composite trade score.
Identifies the best scalping opportunities.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

from .volatility_analyzer import VolatilityMetrics, volatility_analyzer
from .orderbook_analyzer import OrderBookMetrics, orderbook_analyzer
from .database import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeScore:
    """Composite trade scoring for a market."""
    market_id: str
    token_id: str
    question: str = ""
    category: str = ""
    
    # Component scores (0-100)
    volatility_score: float = 0
    liquidity_score: float = 0
    opportunity_score: float = 0  # Based on position in range
    momentum_score: float = 0  # Trend strength
    
    # Composite score (0-100)
    total_score: float = 0
    
    # Trading metrics
    current_price: Optional[float] = None
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    
    # Direction
    signal: str = "HOLD"  # BUY, SELL, HOLD
    confidence: str = "low"  # low, medium, high
    
    # Reference data
    volatility_metrics: Optional[VolatilityMetrics] = None
    orderbook_metrics: Optional[OrderBookMetrics] = None
    
    # Additional info
    volume_24h: float = 0
    spread_bps: Optional[float] = None
    range_24h: str = ""
    
    # AI Analysis (if available)
    ai_probability: Optional[float] = None
    ai_bias: Optional[str] = None
    ai_summary: Optional[str] = None


class TradeScorer:
    """
    Scores markets for scalping opportunities.
    
    Scoring weights:
    - Volatility: 35% (more volatile = better for scalping)
    - Liquidity: 25% (need liquidity to enter/exit)
    - Opportunity: 25% (price near range boundaries)
    - Momentum: 15% (mean reversion signal strength)
    """
    
    WEIGHTS = {
        "volatility": 0.35,
        "liquidity": 0.25,
        "opportunity": 0.25,
        "momentum": 0.15
    }
    
    def __init__(self):
        self.db = db
        self.volatility_analyzer = volatility_analyzer
        self.orderbook_analyzer = orderbook_analyzer
    
    def score_market(
        self,
        market: Dict,
        vol_metrics: Optional[VolatilityMetrics] = None,
        ob_metrics: Optional[OrderBookMetrics] = None,
        skip_orderbook: bool = True  # Skip slow orderbook API calls by default
    ) -> Optional[TradeScore]:
        """
        Calculate composite trade score for a market.
        Uses API data directly when database history is not available.
        """
        from .api_client import client
        
        market_id = market.get("id")
        token_ids = client.parse_clob_token_ids(market)
        
        if not token_ids:
            return None
        
        token_id = token_ids[0]
        
        # Get current price from market data
        prices = client.parse_outcome_prices(market)
        current_price = prices[0] if prices else None
        
        if current_price is None:
            return None
        
        # Get volume for liquidity estimation
        volume_24h = float(market.get("volume24hr", 0) or 0)
        
        # Initialize score
        score = TradeScore(
            market_id=market_id,
            token_id=token_id,
            question=market.get("question", "")[:100],
            category=market.get("category", ""),
            volume_24h=volume_24h,
            current_price=current_price
        )
        
        # Try database first, fall back to API data
        if vol_metrics is None:
            vol_metrics = self.volatility_analyzer.analyze_market(market_id, token_id)
        
        # If no database history, create metrics from API data
        if vol_metrics is None:
            vol_metrics = self._create_volatility_from_api(market, current_price)
        
        score.volatility_metrics = vol_metrics
        
        # Get orderbook metrics only if not skipping and not already provided
        if ob_metrics is None and not skip_orderbook:
            ob_metrics = self.orderbook_analyzer.fetch_and_analyze(market_id, token_id)
        
        score.orderbook_metrics = ob_metrics
        
        # Calculate component scores
        score.volatility_score = self._calc_volatility_score(vol_metrics)
        
        # Use volume-based liquidity if no orderbook
        if ob_metrics:
            score.liquidity_score = ob_metrics.liquidity_score
            score.spread_bps = ob_metrics.spread_bps
        else:
            score.liquidity_score = self._calc_liquidity_from_volume(volume_24h)
            score.spread_bps = None
        
        score.opportunity_score = self._calc_opportunity_score(vol_metrics)
        score.momentum_score = self._calc_momentum_score(vol_metrics)
        
        # Calculate total score
        score.total_score = (
            score.volatility_score * self.WEIGHTS["volatility"] +
            score.liquidity_score * self.WEIGHTS["liquidity"] +
            score.opportunity_score * self.WEIGHTS["opportunity"] +
            score.momentum_score * self.WEIGHTS["momentum"]
        )
        
        # Set current price
        if vol_metrics:
            score.current_price = vol_metrics.current_price or current_price
        elif ob_metrics:
            score.current_price = ob_metrics.mid_price
        else:
            score.current_price = current_price
        
        # Set range
        if vol_metrics and vol_metrics.low_24h and vol_metrics.high_24h:
            score.range_24h = f"${vol_metrics.low_24h:.3f} - ${vol_metrics.high_24h:.3f}"
        
        # Determine signal and entry/exit points
        self._calculate_trade_signal(score, vol_metrics, ob_metrics)
        
        # Get AI analysis if available
        ai_analysis = self.db.get_latest_ai_analysis(market_id)
        if ai_analysis:
            score.ai_probability = ai_analysis.get("ai_probability")
            score.ai_bias = ai_analysis.get("bias_direction")
            score.ai_summary = ai_analysis.get("summary")
        
        return score
    
    def _create_volatility_from_api(self, market: Dict, current_price: float) -> VolatilityMetrics:
        """
        Create volatility metrics directly from Polymarket API data.
        Used when no historical database data is available.
        """
        market_id = market.get("id", "")
        
        # Extract relevant data from market
        volume_24h = float(market.get("volume24hr", 0) or 0)
        total_volume = float(market.get("volume", 0) or 0)
        
        # Use spread and price movements to estimate volatility
        # Binary markets oscillate between 0 and 1
        # High volatility = price moving a lot within the day
        
        # Estimate range based on price and typical movement
        # For prediction markets, use the extreme bounds (0.01, 0.99)
        # and the current price to estimate where in the cycle we are
        
        low_estimate = max(0.01, current_price * 0.85)  # 15% below
        high_estimate = min(0.99, current_price * 1.15)  # 15% above
        
        # For prices near extremes, adjust range
        if current_price < 0.2:
            low_estimate = max(0.01, current_price * 0.7)
            high_estimate = min(0.5, current_price * 1.5)
        elif current_price > 0.8:
            low_estimate = max(0.5, current_price * 0.7)
            high_estimate = min(0.99, current_price * 1.3)
        
        # Calculate position in range
        if high_estimate > low_estimate:
            position = (current_price - low_estimate) / (high_estimate - low_estimate)
        else:
            position = 0.5
        
        # Estimate volatility based on volume
        # Higher volume = potentially higher volatility
        vol_normalized = min(1.0, volume_24h / 100000)  # Normalize by 100k
        estimated_volatility = 0.01 + vol_normalized * 0.04  # 1% to 5%
        
        # Mean reversion score based on position
        # Near edges = stronger signal
        if position < 0.3:
            mean_rev = -(0.3 - position) * 3  # -0.9 to 0
        elif position > 0.7:
            mean_rev = (position - 0.7) * 3  # 0 to 0.9
        else:
            mean_rev = 0
        
        return VolatilityMetrics(
            market_id=market_id,
            token_id="",
            current_price=current_price,
            volatility_24h=estimated_volatility,
            volatility_7d=estimated_volatility * 0.8,
            volatility_all=estimated_volatility * 0.6,
            low_24h=low_estimate,
            high_24h=high_estimate,
            low_7d=low_estimate * 0.9,
            high_7d=high_estimate * 1.1,
            position_in_range_24h=position,
            position_in_range_7d=position,
            mean_reversion_score=mean_rev,
            composite_volatility_score=estimated_volatility
        )
    
    def _calc_liquidity_from_volume(self, volume_24h: float) -> float:
        """
        Estimate liquidity score based on 24h volume.
        Maps volume to a 0-100 score.
        """
        if volume_24h <= 0:
            return 20  # Base score for any active market
        
        # Volume tiers:
        # < $1k = 20-30
        # $1k-$10k = 30-50
        # $10k-$100k = 50-70
        # $100k-$1M = 70-90
        # > $1M = 90-100
        
        import math
        # Log scale: log10(volume)
        log_vol = math.log10(max(1, volume_24h))
        
        # Map: log10(1000)=3 -> 30, log10(1000000)=6 -> 90
        score = 20 + (log_vol - 2) * 20  # 2 to 6 -> 20 to 100
        
        return max(20, min(100, score))
    
    def _calc_volatility_score(self, metrics: Optional[VolatilityMetrics]) -> float:
        """
        Convert volatility to a 0-100 score.
        Higher volatility = higher score.
        """
        if not metrics or not metrics.composite_volatility_score:
            return 0
        
        # Volatility typically ranges from 0.001 to 0.1
        # Map to 0-100 scale
        vol = metrics.composite_volatility_score
        
        # Normalize: 0.001 = 10 points, 0.05+ = 100 points
        if vol <= 0.001:
            return 10
        elif vol >= 0.05:
            return 100
        else:
            # Linear interpolation
            return 10 + (vol - 0.001) / (0.05 - 0.001) * 90
    
    def _calc_opportunity_score(self, metrics: Optional[VolatilityMetrics]) -> float:
        """
        Score based on position in price range.
        Near extremes = higher score (better entry/exit points).
        """
        if not metrics:
            return 50  # Neutral
        
        pos_24h = metrics.position_in_range_24h
        pos_7d = metrics.position_in_range_7d
        
        if pos_24h is None and pos_7d is None:
            return 50
        
        # Use 24h if available, else 7d
        pos = pos_24h if pos_24h is not None else pos_7d
        
        # Near extremes (0 or 1) = high score
        # distance from 0.5 determines score
        distance_from_middle = abs(pos - 0.5) * 2  # 0 to 1
        
        return 50 + distance_from_middle * 50
    
    def _calc_momentum_score(self, metrics: Optional[VolatilityMetrics]) -> float:
        """
        Score based on mean reversion potential.
        Strong deviation from mean = higher score.
        """
        if not metrics or metrics.mean_reversion_score is None:
            return 50
        
        # Mean reversion score is -1 to 1
        # Absolute value indicates strength
        strength = abs(metrics.mean_reversion_score)
        
        return 50 + strength * 50
    
    def _calculate_trade_signal(
        self,
        score: TradeScore,
        vol_metrics: Optional[VolatilityMetrics],
        ob_metrics: Optional[OrderBookMetrics]
    ):
        """
        Determine trade signal (BUY/SELL/HOLD) and entry/exit points.
        
        Strategy: Mean reversion scalping
        - BUY when price is in lower portion of range (oversold)
        - SELL when price is in upper portion of range (overbought)
        """
        if not vol_metrics or not score.current_price:
            score.signal = "HOLD"
            score.confidence = "low"
            return
        
        pos = vol_metrics.position_in_range_24h or vol_metrics.position_in_range_7d
        mean_rev = vol_metrics.mean_reversion_score
        
        if pos is None:
            score.signal = "HOLD"
            score.confidence = "low"
            return
        
        low = vol_metrics.low_24h or vol_metrics.low_7d or 0
        high = vol_metrics.high_24h or vol_metrics.high_7d or 1
        current = score.current_price
        
        # Determine signal based on position in range and mean reversion
        if pos < 0.3 and (mean_rev is None or mean_rev < 0):
            # Price in lower 30% of range and below average = BUY
            score.signal = "BUY"
            score.entry_price = current
            score.target_price = current + (high - current) * 0.5  # 50% of upside
            score.stop_loss = max(0.01, low - (current - low) * 0.5)  # 50% below range
            
        elif pos > 0.7 and (mean_rev is None or mean_rev > 0):
            # Price in upper 30% of range and above average = SELL
            score.signal = "SELL"
            score.entry_price = current
            score.target_price = current - (current - low) * 0.5  # 50% of downside
            score.stop_loss = min(0.99, high + (high - current) * 0.5)
            
        else:
            score.signal = "HOLD"
            score.entry_price = current
            score.target_price = current
            score.stop_loss = current
        
        # Calculate risk/reward ratio
        if score.signal != "HOLD" and score.entry_price:
            reward = abs(score.target_price - score.entry_price)
            risk = abs(score.entry_price - score.stop_loss)
            score.risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Determine confidence
        if score.total_score >= 70 and score.risk_reward_ratio and score.risk_reward_ratio >= 2:
            score.confidence = "high"
        elif score.total_score >= 50 and score.risk_reward_ratio and score.risk_reward_ratio >= 1.5:
            score.confidence = "medium"
        else:
            score.confidence = "low"
    
    def score_markets(self, markets: List[Dict]) -> List[TradeScore]:
        """
        Score multiple markets and sort by total score.
        """
        scores = []
        
        for market in markets:
            try:
                score = self.score_market(market)
                if score:
                    scores.append(score)
            except Exception as e:
                logger.debug(f"Error scoring market {market.get('id')}: {e}")
                continue
        
        # Sort by total score descending
        scores.sort(key=lambda s: s.total_score, reverse=True)
        
        logger.info(f"Scored {len(scores)} markets")
        return scores
    
    def get_top_trades(self, n: int = 10, signal_filter: str = None) -> List[TradeScore]:
        """
        Get top N trade opportunities.
        
        Args:
            n: Number of trades to return
            signal_filter: Filter by signal type (BUY, SELL, or None for all)
        """
        from .api_client import client
        
        # Get all active markets
        markets = client.get_all_markets(active=True, min_volume=1000)
        
        # Filter by criteria
        markets = client.filter_markets_by_criteria(
            markets,
            min_age_days=14,
            min_maturity_days=14,
            min_volume=1000
        )
        
        # Score markets
        scores = self.score_markets(markets)
        
        # Filter by signal if requested
        if signal_filter:
            scores = [s for s in scores if s.signal == signal_filter]
        
        # Filter to only actionable signals (not HOLD)
        actionable = [s for s in scores if s.signal != "HOLD"]
        
        return actionable[:n]


# Singleton instance
trade_scorer = TradeScorer()


if __name__ == "__main__":
    print("Testing Trade Scorer...")
    
    top_trades = trade_scorer.get_top_trades(5)
    
    for i, t in enumerate(top_trades, 1):
        print(f"\n#{i} - {t.question[:50]}...")
        print(f"   Signal: {t.signal} | Confidence: {t.confidence}")
        print(f"   Total Score: {t.total_score:.1f}/100")
        print(f"   Entry: ${t.entry_price:.3f} | Target: ${t.target_price:.3f}")
        print(f"   R/R Ratio: {t.risk_reward_ratio:.2f}x" if t.risk_reward_ratio else "")
