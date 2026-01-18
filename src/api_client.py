"""
Polymarket API Client
=====================
Fetches market data from Polymarket's public APIs:
- Gamma API: Market metadata, events, prices
- CLOB API: Order books, real-time prices
"""

import requests
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolymarketAPIClient:
    """Client for Polymarket public APIs."""
    
    # API Base URLs
    GAMMA_API_BASE = "https://gamma-api.polymarket.com"
    CLOB_API_BASE = "https://clob.polymarket.com"
    
    # Rate limiting
    REQUEST_DELAY = 0.1  # 100ms between requests
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketVolatilityAnalyzer/1.0"
        })
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a GET request with rate limiting and error handling."""
        self._rate_limit()
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {url} - {e}")
            return None
    
    # =========================================
    # GAMMA API - Market Discovery & Metadata
    # =========================================
    
    def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
        order: str = "volume24hr",
        ascending: bool = False
    ) -> List[Dict]:
        """
        Fetch markets from Gamma API.
        
        Returns market data including:
        - id, question, description
        - clobTokenIds (for order book queries)
        - outcomePrices, volume, liquidity
        - startDate, endDate, createdAt
        """
        url = f"{self.GAMMA_API_BASE}/markets"
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": order,
            "ascending": str(ascending).lower()
        }
        
        result = self._get(url, params)
        return result if result else []
    
    def get_all_markets(
        self,
        active: bool = True,
        closed: bool = False,
        min_volume: float = 0
    ) -> List[Dict]:
        """
        Fetch all markets with pagination.
        Filters by minimum volume to reduce data size.
        """
        all_markets = []
        offset = 0
        limit = 100
        
        while True:
            logger.info(f"Fetching markets: offset={offset}")
            markets = self.get_markets(
                limit=limit,
                offset=offset,
                active=active,
                closed=closed
            )
            
            if not markets:
                break
            
            # Filter by minimum volume
            if min_volume > 0:
                markets = [m for m in markets if float(m.get("volumeNum", 0) or 0) >= min_volume]
            
            all_markets.extend(markets)
            
            if len(markets) < limit:
                break
            
            offset += limit
        
        logger.info(f"Total markets fetched: {len(all_markets)}")
        return all_markets
    
    def get_market_by_id(self, market_id: str) -> Optional[Dict]:
        """Fetch a single market by ID."""
        url = f"{self.GAMMA_API_BASE}/markets"
        params = {"id": market_id}
        result = self._get(url, params)
        return result[0] if result else None
    
    def get_events(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Fetch events (groups of related markets)."""
        url = f"{self.GAMMA_API_BASE}/events"
        params = {"limit": limit, "offset": offset}
        result = self._get(url, params)
        return result if result else []
    
    # =========================================
    # CLOB API - Order Book & Prices
    # =========================================
    
    def get_order_book(self, token_id: str) -> Optional[Dict]:
        """
        Get order book for a token.
        
        Returns:
        - bids: [{"price": "0.45", "size": "100"}, ...]
        - asks: [{"price": "0.55", "size": "50"}, ...]
        - timestamp, tick_size, min_order_size
        """
        url = f"{self.CLOB_API_BASE}/book"
        params = {"token_id": token_id}
        return self._get(url, params)
    
    def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get current price for a token."""
        url = f"{self.CLOB_API_BASE}/price"
        params = {"token_id": token_id, "side": side}
        result = self._get(url, params)
        if result and "price" in result:
            return float(result["price"])
        return None
    
    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token."""
        url = f"{self.CLOB_API_BASE}/midpoint"
        params = {"token_id": token_id}
        result = self._get(url, params)
        if result and "mid" in result:
            return float(result["mid"])
        return None
    
    def get_spread(self, token_id: str) -> Optional[Dict]:
        """Get spread for a token."""
        url = f"{self.CLOB_API_BASE}/spread"
        params = {"token_id": token_id}
        return self._get(url, params)
    
    # =========================================
    # Helper Methods
    # =========================================
    
    def parse_clob_token_ids(self, market: Dict) -> List[str]:
        """
        Extract CLOB token IDs from market data.
        Markets have 2 tokens: YES and NO outcomes.
        """
        clob_token_ids = market.get("clobTokenIds", "")
        if isinstance(clob_token_ids, str):
            # Format: "[\"token1\",\"token2\"]" or "token1,token2"
            clob_token_ids = clob_token_ids.replace("[", "").replace("]", "").replace('"', "")
            return [tid.strip() for tid in clob_token_ids.split(",") if tid.strip()]
        return clob_token_ids if clob_token_ids else []
    
    def parse_outcome_prices(self, market: Dict) -> List[float]:
        """
        Extract outcome prices from market data.
        Returns [YES_price, NO_price]
        """
        prices_str = market.get("outcomePrices", "")
        if isinstance(prices_str, str):
            prices_str = prices_str.replace("[", "").replace("]", "").replace('"', "")
            try:
                return [float(p.strip()) for p in prices_str.split(",") if p.strip()]
            except ValueError:
                return []
        return prices_str if prices_str else []
    
    def filter_markets_by_criteria(
        self,
        markets: List[Dict],
        min_age_days: int = 14,
        min_maturity_days: int = 14,
        min_volume: float = 1000
    ) -> List[Dict]:
        """
        Filter markets by:
        - Age: created at least min_age_days ago
        - Maturity: ends at least min_maturity_days from now
        - Volume: minimum trading volume
        """
        now = datetime.utcnow()
        filtered = []
        
        for market in markets:
            try:
                # Parse dates
                created_at_str = market.get("createdAt") or market.get("startDate")
                end_date_str = market.get("endDate")
                
                if not created_at_str or not end_date_str:
                    continue
                
                # Handle different date formats
                for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"]:
                    try:
                        created_at = datetime.strptime(created_at_str[:26].replace("Z", ""), fmt.replace("Z", ""))
                        break
                    except ValueError:
                        continue
                else:
                    continue
                
                for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"]:
                    try:
                        end_date = datetime.strptime(end_date_str[:26].replace("Z", ""), fmt.replace("Z", ""))
                        break
                    except ValueError:
                        continue
                else:
                    continue
                
                # Check age (market exists for at least X days)
                age = (now - created_at).days
                if age < min_age_days:
                    continue
                
                # Check maturity (market ends in at least X days)
                maturity = (end_date - now).days
                if maturity < min_maturity_days:
                    continue
                
                # Check volume
                volume = float(market.get("volumeNum", 0) or 0)
                if volume < min_volume:
                    continue
                
                # Add computed fields
                market["_age_days"] = age
                market["_maturity_days"] = maturity
                
                filtered.append(market)
                
            except Exception as e:
                logger.debug(f"Error filtering market {market.get('id')}: {e}")
                continue
        
        logger.info(f"Filtered {len(filtered)} markets from {len(markets)}")
        return filtered


# Singleton instance
client = PolymarketAPIClient()


if __name__ == "__main__":
    # Test the client
    print("Testing Polymarket API Client...")
    
    # Fetch some markets
    markets = client.get_markets(limit=5)
    print(f"\nFetched {len(markets)} markets:")
    for m in markets:
        print(f"  - {m.get('question', 'N/A')[:60]}...")
        print(f"    Volume: ${float(m.get('volumeNum', 0) or 0):,.0f}")
        print(f"    Token IDs: {client.parse_clob_token_ids(m)}")
