"""
Database Module
===============
SQLite database for storing:
- Market metadata
- Price history (for volatility calculations)
- Order book snapshots
- AI analysis results
- Trade signals
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "polymarket.db"


class Database:
    """SQLite database manager for Polymarket data."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self._ensure_db_dir()
        self._init_schema()
    
    def _ensure_db_dir(self):
        """Create database directory if it doesn't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_schema(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Markets table - stores market metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                id TEXT PRIMARY KEY,
                question TEXT,
                description TEXT,
                category TEXT,
                slug TEXT,
                clob_token_ids TEXT,  -- JSON array
                outcomes TEXT,  -- JSON array
                end_date TEXT,
                created_at TEXT,
                volume_num REAL DEFAULT 0,
                liquidity_num REAL DEFAULT 0,
                enable_order_book INTEGER DEFAULT 1,
                neg_risk INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Price history table - for volatility calculations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL,
                bid REAL,
                ask REAL,
                mid REAL,
                volume_24h REAL,
                FOREIGN KEY (market_id) REFERENCES markets(id),
                UNIQUE(market_id, token_id, timestamp)
            )
        """)
        
        # Order book snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                best_bid REAL,
                best_ask REAL,
                spread REAL,
                bid_depth_total REAL,  -- Total size on bid side
                ask_depth_total REAL,  -- Total size on ask side
                bid_levels INTEGER,  -- Number of bid levels
                ask_levels INTEGER,  -- Number of ask levels
                imbalance_ratio REAL,  -- bid_depth / (bid_depth + ask_depth)
                bids_json TEXT,  -- Full orderbook bids as JSON
                asks_json TEXT,  -- Full orderbook asks as JSON
                FOREIGN KEY (market_id) REFERENCES markets(id)
            )
        """)
        
        # AI analysis results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                analyzed_at TEXT NOT NULL,
                ai_probability REAL,
                confidence TEXT,  -- low, medium, high
                bias_direction TEXT,  -- OVERPRICED, UNDERPRICED, FAIR
                key_factors TEXT,  -- JSON array
                summary TEXT,
                model_used TEXT DEFAULT 'gpt-4',
                FOREIGN KEY (market_id) REFERENCES markets(id)
            )
        """)
        
        # Trade signals / recommendations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                signal_type TEXT,  -- BUY, SELL, HOLD
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                risk_reward_ratio REAL,
                confidence_score REAL,
                volatility_score REAL,
                liquidity_score REAL,
                ai_bias TEXT,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (market_id) REFERENCES markets(id)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_market ON price_history(market_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_token ON price_history(token_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orderbook_market ON orderbook_snapshots(market_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_market ON ai_analysis(market_id, analyzed_at)")
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    # =========================================
    # Market Operations
    # =========================================
    
    def upsert_market(self, market: Dict):
        """Insert or update a market."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO markets 
            (id, question, description, category, slug, clob_token_ids, outcomes,
             end_date, created_at, volume_num, liquidity_num, enable_order_book, neg_risk, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market.get("id"),
            market.get("question"),
            market.get("description"),
            market.get("category"),
            market.get("slug"),
            market.get("clobTokenIds") if isinstance(market.get("clobTokenIds"), str) else json.dumps(market.get("clobTokenIds", [])),
            market.get("outcomes") if isinstance(market.get("outcomes"), str) else json.dumps(market.get("outcomes", [])),
            market.get("endDate"),
            market.get("createdAt"),
            float(market.get("volumeNum", 0) or 0),
            float(market.get("liquidityNum", 0) or 0),
            1 if market.get("enableOrderBook") else 0,
            1 if market.get("negRisk") else 0,
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def upsert_markets(self, markets: List[Dict]):
        """Bulk insert/update markets."""
        for market in markets:
            self.upsert_market(market)
        logger.info(f"Upserted {len(markets)} markets")
    
    def get_all_markets(self) -> List[Dict]:
        """Get all markets from database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM markets ORDER BY volume_num DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_market(self, market_id: str) -> Optional[Dict]:
        """Get a single market by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM markets WHERE id = ?", (market_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    # =========================================
    # Price History Operations
    # =========================================
    
    def insert_price(self, market_id: str, token_id: str, price: float, 
                     bid: float = None, ask: float = None, mid: float = None,
                     volume_24h: float = None):
        """Insert a price record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        
        try:
            cursor.execute("""
                INSERT INTO price_history 
                (market_id, token_id, timestamp, price, bid, ask, mid, volume_24h)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (market_id, token_id, timestamp, price, bid, ask, mid, volume_24h))
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # Duplicate entry, ignore
        finally:
            conn.close()
    
    def get_price_history(self, market_id: str, token_id: str = None, 
                          hours: int = None) -> List[Dict]:
        """Get price history for a market."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM price_history WHERE market_id = ?"
        params = [market_id]
        
        if token_id:
            query += " AND token_id = ?"
            params.append(token_id)
        
        if hours:
            from_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            query += " AND timestamp >= ?"
            params.append(from_time)
        
        query += " ORDER BY timestamp ASC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        from datetime import timedelta
        return [dict(row) for row in rows]
    
    # =========================================
    # Order Book Operations
    # =========================================
    
    def insert_orderbook_snapshot(self, market_id: str, token_id: str, 
                                  orderbook_data: Dict):
        """Insert an order book snapshot."""
        conn = self._get_connection()
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        
        bids = orderbook_data.get("bids", [])
        asks = orderbook_data.get("asks", [])
        
        # Calculate metrics
        best_bid = float(bids[0]["price"]) if bids else None
        best_ask = float(asks[0]["price"]) if asks else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        
        bid_depth = sum(float(b["size"]) for b in bids) if bids else 0
        ask_depth = sum(float(a["size"]) for a in asks) if asks else 0
        total_depth = bid_depth + ask_depth
        imbalance = bid_depth / total_depth if total_depth > 0 else 0.5
        
        cursor.execute("""
            INSERT INTO orderbook_snapshots 
            (market_id, token_id, timestamp, best_bid, best_ask, spread,
             bid_depth_total, ask_depth_total, bid_levels, ask_levels,
             imbalance_ratio, bids_json, asks_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_id, token_id, timestamp, best_bid, best_ask, spread,
            bid_depth, ask_depth, len(bids), len(asks), imbalance,
            json.dumps(bids), json.dumps(asks)
        ))
        
        conn.commit()
        conn.close()
    
    def get_latest_orderbook(self, market_id: str, token_id: str) -> Optional[Dict]:
        """Get the latest order book snapshot."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM orderbook_snapshots 
            WHERE market_id = ? AND token_id = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (market_id, token_id))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    # =========================================
    # AI Analysis Operations
    # =========================================
    
    def insert_ai_analysis(self, market_id: str, analysis: Dict):
        """Insert AI analysis result."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO ai_analysis 
            (market_id, analyzed_at, ai_probability, confidence, bias_direction,
             key_factors, summary, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_id,
            datetime.utcnow().isoformat(),
            analysis.get("ai_probability"),
            analysis.get("confidence"),
            analysis.get("bias_direction"),
            json.dumps(analysis.get("key_factors", [])),
            analysis.get("summary"),
            analysis.get("model_used", "gpt-4")
        ))
        
        conn.commit()
        conn.close()
    
    def get_latest_ai_analysis(self, market_id: str) -> Optional[Dict]:
        """Get the latest AI analysis for a market."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM ai_analysis 
            WHERE market_id = ?
            ORDER BY analyzed_at DESC LIMIT 1
        """, (market_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            result = dict(row)
            result["key_factors"] = json.loads(result.get("key_factors", "[]"))
            return result
        return None
    
    # =========================================
    # Trade Signal Operations
    # =========================================
    
    def insert_trade_signal(self, signal: Dict):
        """Insert a trade signal."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trade_signals 
            (market_id, token_id, created_at, signal_type, entry_price,
             target_price, stop_loss, risk_reward_ratio, confidence_score,
             volatility_score, liquidity_score, ai_bias, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.get("market_id"),
            signal.get("token_id"),
            datetime.utcnow().isoformat(),
            signal.get("signal_type"),
            signal.get("entry_price"),
            signal.get("target_price"),
            signal.get("stop_loss"),
            signal.get("risk_reward_ratio"),
            signal.get("confidence_score"),
            signal.get("volatility_score"),
            signal.get("liquidity_score"),
            signal.get("ai_bias"),
            1
        ))
        
        conn.commit()
        conn.close()
    
    def get_active_signals(self, limit: int = 10) -> List[Dict]:
        """Get active trade signals."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ts.*, m.question, m.category
            FROM trade_signals ts
            JOIN markets m ON ts.market_id = m.id
            WHERE ts.is_active = 1
            ORDER BY ts.confidence_score DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]


# Singleton instance
db = Database()


if __name__ == "__main__":
    print("Testing Database...")
    print(f"Database path: {db.db_path}")
    
    # Test insert
    test_market = {
        "id": "test-123",
        "question": "Test Market?",
        "volumeNum": 10000
    }
    db.upsert_market(test_market)
    
    # Test retrieve
    retrieved = db.get_market("test-123")
    print(f"Retrieved: {retrieved}")
