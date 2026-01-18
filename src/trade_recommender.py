"""
Trade Recommender
=================
Generates actionable trade recommendations:
- Top 10 trade opportunities
- Entry/exit points
- Risk/reward analysis
- Confidence levels
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from .trade_scorer import TradeScore, trade_scorer
from .ai_analyzer import ai_analyzer
from .database import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeRecommendation:
    """A complete trade recommendation."""
    rank: int
    market_id: str
    question: str
    category: str
    
    # Trade signal
    action: str  # BUY, SELL
    confidence: str  # low, medium, high
    
    # Prices
    current_price: float
    entry_price: float
    target_price: float
    stop_loss: float
    
    # Metrics
    risk_reward_ratio: float
    potential_profit_pct: float
    potential_loss_pct: float
    
    # Scores
    total_score: float
    volatility_score: float
    liquidity_score: float
    
    # Order book
    spread_bps: Optional[float] = None
    bid_depth: Optional[float] = None
    ask_depth: Optional[float] = None
    
    # Range info
    range_24h: str = ""
    position_in_range: Optional[float] = None
    
    # AI Analysis
    ai_probability: Optional[float] = None
    ai_bias: Optional[str] = None
    ai_summary: Optional[str] = None
    
    # Volume
    volume_24h: float = 0
    
    # Timestamp
    generated_at: str = ""


class TradeRecommender:
    """
    Generates and manages trade recommendations.
    """
    
    def __init__(self):
        self.db = db
        self.trade_scorer = trade_scorer
        self.ai_analyzer = ai_analyzer
    
    def generate_recommendations(
        self,
        n: int = 10,
        include_ai: bool = True,
        signal_filter: str = None,
        min_confidence: str = None,
        min_rr_ratio: float = 1.0,
        min_price: float = None,
        max_price: float = None,
        min_age_days: int = 14,
        min_maturity_days: int = 14,
        min_volume: float = 1000
    ) -> List[TradeRecommendation]:
        """
        Generate top N trade recommendations.
        
        Args:
            n: Number of recommendations
            include_ai: Whether to include AI analysis
            signal_filter: Filter by BUY or SELL
            min_confidence: Minimum confidence level (low, medium, high)
            min_rr_ratio: Minimum risk/reward ratio
            min_price: Minimum price (0-1)
            max_price: Maximum price (0-1)
            min_age_days: Minimum market age in days
            min_maturity_days: Minimum days until resolution
            min_volume: Minimum trading volume
        
        Returns:
            List of trade recommendations
        """
        from .api_client import client
        
        logger.info(f"Generating {n} trade recommendations...")
        
        # Get all markets
        markets = client.get_all_markets(active=True, min_volume=min_volume)
        logger.info(f"Fetched {len(markets)} markets")
        
        # Filter by criteria using user-provided parameters
        filtered_markets = client.filter_markets_by_criteria(
            markets,
            min_age_days=min_age_days,
            min_maturity_days=min_maturity_days,
            min_volume=min_volume
        )
        after_criteria_count = len(filtered_markets)
        logger.info(f"Filtered to {after_criteria_count} eligible markets")
        
        # Filter by price range if specified
        if min_price is not None or max_price is not None:
            price_filtered = []
            for market in filtered_markets:
                last_price = market.get("lastTradePrice")
                if last_price is None:
                    continue
                try:
                    price = float(last_price)
                    if min_price is not None and price < min_price:
                        continue
                    if max_price is not None and price > max_price:
                        continue
                    price_filtered.append(market)
                except (ValueError, TypeError):
                    continue
            filtered_markets = price_filtered
            logger.info(f"After price filter ({min_price}-{max_price}): {len(filtered_markets)} markets")
        
        after_price_count = len(filtered_markets)
        
        # Score all markets
        scores = self.trade_scorer.score_markets(filtered_markets)
        
        # Filter actionable signals
        # When price filters are applied, include all signals (not just BUY/SELL)
        # because user wants to see markets in that price range regardless of signal
        if min_price is not None or max_price is not None:
            actionable = scores  # Include all markets when price filtered
        else:
            actionable = [s for s in scores if s.signal in ["BUY", "SELL"]]
        
        # Apply filters
        if signal_filter:
            actionable = [s for s in actionable if s.signal == signal_filter]
        
        if min_confidence:
            conf_order = {"low": 0, "medium": 1, "high": 2}
            min_conf_val = conf_order.get(min_confidence, 0)
            actionable = [s for s in actionable if conf_order.get(s.confidence, 0) >= min_conf_val]
        
        # Only apply RR filter if no price filter (otherwise we want all markets in price range)
        if min_rr_ratio and min_price is None and max_price is None:
            actionable = [s for s in actionable if s.risk_reward_ratio and s.risk_reward_ratio >= min_rr_ratio]
        
        # Take top N
        top_scores = actionable[:n * 2]  # Get extra for AI filtering
        
        # Run AI analysis if enabled
        if include_ai and self.ai_analyzer.is_available():
            logger.info("Running AI analysis on top candidates...")
            
            # Get market data for AI analysis
            market_map = {m["id"]: m for m in filtered_markets}
            
            for score in top_scores[:min(n, 10)]:  # Limit AI calls
                market = market_map.get(score.market_id)
                if market:
                    ai_result = self.ai_analyzer.analyze_market(market)
                    if ai_result:
                        score.ai_probability = ai_result.get("ai_probability")
                        score.ai_bias = ai_result.get("bias_direction")
                        score.ai_summary = ai_result.get("summary")
        
        # Convert to recommendations
        recommendations = []
        for i, score in enumerate(top_scores[:n], 1):
            rec = self._create_recommendation(i, score)
            recommendations.append(rec)
            
            # Store signal in database
            self._store_signal(score)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        # Store stats for display
        self._last_stats = {
            "total_fetched": len(markets),
            "after_criteria": after_criteria_count,
            "after_price_filter": after_price_count,
            "after_scoring": len(actionable)
        }
        
        return recommendations
    
    def get_last_stats(self) -> dict:
        """Get stats from last generate_recommendations call."""
        return getattr(self, '_last_stats', {})
    
    def _create_recommendation(self, rank: int, score: TradeScore) -> TradeRecommendation:
        """
        Convert TradeScore to TradeRecommendation.
        """
        # Calculate profit/loss percentages
        entry = score.entry_price or 0
        target = score.target_price or 0
        stop = score.stop_loss or 0
        
        if entry > 0:
            profit_pct = abs(target - entry) / entry * 100
            loss_pct = abs(entry - stop) / entry * 100
        else:
            profit_pct = 0
            loss_pct = 0
        
        # Get position in range from volatility metrics
        pos_in_range = None
        if score.volatility_metrics:
            pos_in_range = score.volatility_metrics.position_in_range_24h
        
        # Get order book depths
        bid_depth = None
        ask_depth = None
        if score.orderbook_metrics:
            bid_depth = score.orderbook_metrics.bid_depth_total
            ask_depth = score.orderbook_metrics.ask_depth_total
        
        return TradeRecommendation(
            rank=rank,
            market_id=score.market_id,
            question=score.question,
            category=score.category,
            action=score.signal,
            confidence=score.confidence,
            current_price=score.current_price or 0,
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            risk_reward_ratio=score.risk_reward_ratio or 0,
            potential_profit_pct=profit_pct,
            potential_loss_pct=loss_pct,
            total_score=score.total_score,
            volatility_score=score.volatility_score,
            liquidity_score=score.liquidity_score,
            spread_bps=score.spread_bps,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            range_24h=score.range_24h,
            position_in_range=pos_in_range,
            ai_probability=score.ai_probability,
            ai_bias=score.ai_bias,
            ai_summary=score.ai_summary,
            volume_24h=score.volume_24h,
            generated_at=datetime.utcnow().isoformat()
        )
    
    def _store_signal(self, score: TradeScore):
        """
        Store trade signal in database.
        """
        signal = {
            "market_id": score.market_id,
            "token_id": score.token_id,
            "signal_type": score.signal,
            "entry_price": score.entry_price,
            "target_price": score.target_price,
            "stop_loss": score.stop_loss,
            "risk_reward_ratio": score.risk_reward_ratio,
            "confidence_score": score.total_score,
            "volatility_score": score.volatility_score,
            "liquidity_score": score.liquidity_score,
            "ai_bias": score.ai_bias
        }
        self.db.insert_trade_signal(signal)
    
    def format_recommendations_table(self, recommendations: List[TradeRecommendation]) -> str:
        """
        Format recommendations as a text table for console output.
        """
        if not recommendations:
            return "No recommendations available."
        
        lines = []
        lines.append("=" * 120)
        lines.append(" TOP TRADE RECOMMENDATIONS - Polymarket Scanner")
        lines.append(f" Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("=" * 120)
        lines.append("")
        
        for rec in recommendations:
            lines.append(f"#{rec.rank} | {rec.action} | {rec.confidence.upper()} CONFIDENCE")
            lines.append(f"   Market: {rec.question}")
            lines.append(f"   Category: {rec.category}")
            lines.append("")
            lines.append(f"   Entry: ${rec.entry_price:.3f} → Target: ${rec.target_price:.3f} | Stop: ${rec.stop_loss:.3f}")
            lines.append(f"   Risk/Reward: {rec.risk_reward_ratio:.2f}x | Profit: +{rec.potential_profit_pct:.1f}% | Risk: -{rec.potential_loss_pct:.1f}%")
            lines.append(f"   Range 24h: {rec.range_24h} | Position: {rec.position_in_range:.0%}" if rec.position_in_range else f"   Range 24h: {rec.range_24h}")
            lines.append("")
            lines.append(f"   Scores: Total={rec.total_score:.1f} | Vol={rec.volatility_score:.1f} | Liq={rec.liquidity_score:.1f}")
            lines.append(f"   Spread: {rec.spread_bps:.1f} bps | Volume 24h: ${rec.volume_24h:,.0f}" if rec.spread_bps else f"   Volume 24h: ${rec.volume_24h:,.0f}")
            
            if rec.ai_probability is not None:
                lines.append("")
                lines.append(f"   🤖 AI Analysis: {rec.ai_bias} | AI Prob: {rec.ai_probability:.0%}")
                if rec.ai_summary:
                    lines.append(f"   {rec.ai_summary[:100]}...")
            
            lines.append("-" * 120)
        
        return "\n".join(lines)
    
    def to_dict_list(self, recommendations: List[TradeRecommendation]) -> List[Dict]:
        """
        Convert recommendations to list of dictionaries (for JSON/DataFrame).
        """
        from .api_client import client
        
        # Get market map to fetch additional data
        market_ids = [r.market_id for r in recommendations]
        markets = client.get_all_markets(active=True)
        market_map = {m["id"]: m for m in markets if m["id"] in market_ids}
        
        result = []
        for r in recommendations:
            market = market_map.get(r.market_id, {})
            
            # Extract dates and token IDs
            start_date = market.get("startDate") or market.get("createdAt")
            
            # Calculate average daily volume based on market age
            total_volume = float(market.get("volume", 0) or 0)
            from datetime import datetime
            market_age_days = 1  # Default minimum
            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    market_age_days = max(1, (datetime.now(start_dt.tzinfo) - start_dt).days)
                except:
                    pass
            
            # Calculate average daily volume and 7-day volume
            avg_daily_volume = total_volume / market_age_days if market_age_days > 0 else 0
            volume_7d = avg_daily_volume * min(7, market_age_days)  # 7-day or market age, whichever is smaller
            end_date = market.get("endDate") or market.get("end_date_iso")
            
            # Get token IDs (contract addresses)
            clob_token_ids = market.get("clobTokenIds", "")
            token_id = ""
            if isinstance(clob_token_ids, str):
                token_id = clob_token_ids.split(",")[0] if clob_token_ids else ""
            elif isinstance(clob_token_ids, list) and clob_token_ids:
                token_id = str(clob_token_ids[0])
            
            # Get slug and conditionId for Polymarket URL
            slug = market.get("slug", "")
            condition_id = market.get("conditionId", "")
            
            result.append({
                "rank": r.rank,
                "action": r.action,
                "confidence": r.confidence,
                "question": r.question,
                "category": r.category,
                "current_price": r.current_price,
                "entry_price": r.entry_price,
                "target_price": r.target_price,
                "stop_loss": r.stop_loss,
                "risk_reward": r.risk_reward_ratio,
                "profit_pct": r.potential_profit_pct,
                "loss_pct": r.potential_loss_pct,
                "total_score": r.total_score,
                "vol_score": r.volatility_score,
                "liq_score": r.liquidity_score,
                "spread_bps": r.spread_bps,
                "range_24h": r.range_24h,
                "volume_24h": r.volume_24h,
                "volume_7d": volume_7d,
                "volume": total_volume,
                "start_date": start_date,
                "end_date": end_date,
                "token_id": token_id,
                "slug": slug,
                "condition_id": condition_id,
                "ai_prob": r.ai_probability,
                "ai_bias": r.ai_bias,
                "ai_summary": r.ai_summary,
                "market_id": r.market_id
            })
        
        return result


# Singleton instance
trade_recommender = TradeRecommender()


if __name__ == "__main__":
    print("Generating Trade Recommendations...")
    
    recs = trade_recommender.generate_recommendations(
        n=5,
        include_ai=True,
        min_rr_ratio=1.0
    )
    
    print(trade_recommender.format_recommendations_table(recs))
