"""
Real-time Price Streaming
=========================
Provides real-time price updates via optimized polling.
Since Polymarket doesn't expose a public WebSocket, we use:
1. Efficient batch price fetching
2. Short polling intervals (5s)
3. Only fetch active markets being displayed

For true WebSocket when available, this can be extended.
"""

import threading
import time
import logging
from typing import Dict, List, Callable, Optional
from datetime import datetime
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimePriceStream:
    """
    Real-time price streaming using efficient polling.
    Updates prices every few seconds for displayed markets only.
    """
    
    CLOB_API_BASE = "https://clob.polymarket.com"
    GAMMA_API_BASE = "https://gamma-api.polymarket.com"
    
    def __init__(self, update_interval: float = 5.0):
        """
        Initialize the price stream.
        
        Args:
            update_interval: Seconds between price updates (default: 5s)
        """
        self.update_interval = update_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._subscribers: List[Callable[[Dict], None]] = []
        self._watched_tokens: List[str] = []
        self._lock = threading.Lock()
        self._last_prices: Dict[str, Dict] = {}
    
    def subscribe(self, callback: Callable[[Dict], None]) -> None:
        """
        Subscribe to price updates.
        
        Args:
            callback: Function called with price updates dict
                     Format: {token_id: {"price": float, "change": float, "timestamp": str}}
        """
        with self._lock:
            self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[Dict], None]) -> None:
        """Unsubscribe from price updates."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
    
    def watch_tokens(self, token_ids: List[str]) -> None:
        """
        Set which tokens to watch for price updates.
        
        Args:
            token_ids: List of CLOB token IDs to monitor
        """
        with self._lock:
            self._watched_tokens = list(set(token_ids))
        logger.info(f"Now watching {len(self._watched_tokens)} tokens")
    
    def add_token(self, token_id: str) -> None:
        """Add a token to watch list."""
        with self._lock:
            if token_id not in self._watched_tokens:
                self._watched_tokens.append(token_id)
    
    def remove_token(self, token_id: str) -> None:
        """Remove a token from watch list."""
        with self._lock:
            if token_id in self._watched_tokens:
                self._watched_tokens.remove(token_id)
    
    def start(self) -> None:
        """Start the price streaming in background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Price streaming started")
    
    def stop(self) -> None:
        """Stop the price streaming."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Price streaming stopped")
    
    def get_last_prices(self) -> Dict[str, Dict]:
        """Get the last known prices."""
        with self._lock:
            return self._last_prices.copy()
    
    def _run_loop(self) -> None:
        """Main streaming loop."""
        while self._running:
            try:
                self._fetch_and_broadcast()
            except Exception as e:
                logger.error(f"Error in price stream: {e}")
            
            time.sleep(self.update_interval)
    
    def _fetch_and_broadcast(self) -> None:
        """Fetch latest prices and broadcast to subscribers."""
        with self._lock:
            tokens = self._watched_tokens.copy()
            subscribers = self._subscribers.copy()
        
        if not tokens:
            return
        
        # Fetch prices in batches of 20
        updates = {}
        batch_size = 20
        
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size]
            batch_prices = self._fetch_midpoints_batch(batch)
            updates.update(batch_prices)
        
        if not updates:
            return
        
        # Calculate changes from last prices
        timestamp = datetime.utcnow().isoformat()
        for token_id, new_price in updates.items():
            old_data = self._last_prices.get(token_id, {})
            old_price = old_data.get("price", new_price)
            change = new_price - old_price if old_price else 0
            
            updates[token_id] = {
                "price": new_price,
                "change": change,
                "change_pct": (change / old_price * 100) if old_price else 0,
                "timestamp": timestamp
            }
        
        # Update last prices
        with self._lock:
            self._last_prices.update(updates)
        
        # Broadcast to subscribers
        for callback in subscribers:
            try:
                callback(updates)
            except Exception as e:
                logger.error(f"Error in price callback: {e}")
    
    def _fetch_midpoints_batch(self, token_ids: List[str]) -> Dict[str, float]:
        """
        Fetch midpoint prices for a batch of tokens.
        
        Returns:
            Dict of token_id -> midpoint price
        """
        try:
            url = f"{self.CLOB_API_BASE}/midpoints"
            body = [{"token_id": tid} for tid in token_ids]
            
            response = requests.post(url, json=body, timeout=10)
            if response.status_code != 200:
                return {}
            
            results = response.json()
            prices = {}
            
            for i, result in enumerate(results):
                if i < len(token_ids) and "mid" in result:
                    try:
                        prices[token_ids[i]] = float(result["mid"])
                    except (ValueError, TypeError):
                        pass
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching batch prices: {e}")
            return {}
    
    def fetch_single_price(self, token_id: str) -> Optional[float]:
        """
        Fetch a single token's current price (synchronous).
        
        Returns:
            Current midpoint price or None if unavailable
        """
        try:
            url = f"{self.CLOB_API_BASE}/midpoint"
            response = requests.get(url, params={"token_id": token_id}, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if "mid" in data:
                    return float(data["mid"])
            return None
            
        except Exception as e:
            logger.error(f"Error fetching price for {token_id}: {e}")
            return None


# Singleton instance
price_stream = RealtimePriceStream(update_interval=5.0)


if __name__ == "__main__":
    # Test the price stream
    def on_price_update(updates):
        for token_id, data in updates.items():
            print(f"  {token_id[:20]}... : ${data['price']:.4f} ({data['change_pct']:+.2f}%)")
    
    print("Testing Real-time Price Stream...")
    
    # Get some test tokens from API
    response = requests.get(
        "https://gamma-api.polymarket.com/markets",
        params={"limit": 5, "closed": "false", "active": "true", "order": "volume24hr", "ascending": "false"}
    )
    
    if response.status_code == 200:
        markets = response.json()
        
        # Extract token IDs
        token_ids = []
        for m in markets:
            clob_ids = m.get("clobTokenIds", "")
            if isinstance(clob_ids, str):
                clob_ids = clob_ids.replace("[", "").replace("]", "").replace('"', "")
                for tid in clob_ids.split(","):
                    tid = tid.strip()
                    if tid:
                        token_ids.append(tid)
                        break  # Just first token per market
        
        print(f"\nWatching {len(token_ids)} tokens:")
        for tid in token_ids:
            print(f"  - {tid[:30]}...")
        
        # Start streaming
        price_stream.subscribe(on_price_update)
        price_stream.watch_tokens(token_ids)
        price_stream.start()
        
        print("\nStreaming prices (press Ctrl+C to stop)...")
        try:
            while True:
                time.sleep(10)
                print(f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] Update:")
        except KeyboardInterrupt:
            price_stream.stop()
            print("\nStopped.")
    else:
        print(f"Failed to fetch test markets: {response.status_code}")
