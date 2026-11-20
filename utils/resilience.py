"""
Resilience utilities: retry with backoff, circuit breaker, SQLite state persistence, health monitoring.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import aiosqlite
from loguru import logger
from pydantic import BaseModel, Field

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available. Metrics disabled.")


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    jitter_factor: float = 0.1,
    exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        IOError,
    ),
    before_sleep: Optional[Callable] = None,
) -> Callable:
    """
    Decorator for retry with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Exponential base for backoff
        jitter: Whether to add random jitter
        jitter_factor: Jitter factor (0-1)
        exceptions: Exception types to catch and retry
        before_sleep: Optional callback before sleep

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        if jitter:
                            jitter_amount = delay * jitter_factor * (2 * asyncio.get_event_loop().time() % 1 - 0.5)
                            delay += jitter_amount

                        if before_sleep:
                            before_sleep(attempt, e, delay)

                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All retries exhausted for {func.__name__}: {last_exception}")

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        if jitter:
                            import random
                            jitter_amount = delay * jitter_factor * (2 * random.random() - 1)
                            delay += jitter_amount

                        if before_sleep:
                            before_sleep(attempt, e, delay)

                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All retries exhausted for {func.__name__}: {last_exception}")

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    excluded_exceptions: Tuple[Type[Exception], ...] = ()


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    States:
    - CLOSED: Normal operation, counting failures
    - OPEN: Too many failures, rejecting calls
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout transition."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and \
               time.time() - self._last_failure_time >= self.config.timeout_seconds:
                return CircuitState.HALF_OPEN
        return self._state

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        current_state = self.state

        if current_state == CircuitState.CLOSED:
            return True

        if current_state == CircuitState.OPEN:
            return False

        # HALF_OPEN: allow limited calls
        if current_state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.config.half_open_max_calls

        return False

    async def record_success(self) -> None:
        """Record successful execution."""
        async with self._lock:
            current_state = self.state

            if current_state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker CLOSED after successful recovery")
            elif current_state == CircuitState.CLOSED:
                self._failure_count = 0  # Reset on success

    async def record_failure(self, exception: Optional[Exception] = None) -> None:
        """Record failed execution."""
        async with self._lock:
            current_state = self.state

            # Check if exception should be excluded
            if exception and isinstance(exception, self.config.excluded_exceptions):
                return

            self._last_failure_time = time.time()

            if current_state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning("Circuit breaker OPEN after half-open failure")
            elif current_state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker OPEN after {self._failure_count} failures")

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            'state': self.state.value,
            'failure_count': self._failure_count,
            'success_count': self._success_count,
            'last_failure_time': self._last_failure_time,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'success_threshold': self.config.success_threshold,
                'timeout_seconds': self.config.timeout_seconds,
            }
        }


# =============================================================================
# SQLITE STATE PERSISTENCE
# =============================================================================

class SQLiteState:
    """
    SQLite-based state persistence for trading system.
    Stores positions, orders, daily metrics, signals, news cache, regime history.
    """

    def __init__(self, db_path: str = "data/state/trading_state.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        cursor = self._conn.cursor()

        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                stop_price REAL,
                target_prices TEXT,  -- JSON array
                entry_time TEXT NOT NULL,
                updated_time TEXT NOT NULL,
                pnl REAL DEFAULT 0,
                metadata TEXT  -- JSON
            )
        """)

        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                position_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL,
                status TEXT NOT NULL,
                created_time TEXT NOT NULL,
                updated_time TEXT NOT NULL,
                filled_quantity REAL DEFAULT 0,
                avg_fill_price REAL,
                metadata TEXT,
                FOREIGN KEY (position_id) REFERENCES positions(id)
            )
        """)

        # Daily metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date TEXT PRIMARY KEY,
                equity REAL NOT NULL,
                pnl REAL DEFAULT 0,
                trades INTEGER DEFAULT 0,
                win_rate REAL,
                max_drawdown REAL,
                sharpe REAL,
                metrics TEXT,  -- JSON
                updated_time TEXT NOT NULL
            )
        """)

        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                price REAL,
                strength REAL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                deduplicated INTEGER DEFAULT 0
            )
        """)

        # News cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_cache (
                content_hash TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT,
                link TEXT,
                source TEXT,
                category TEXT,
                symbols TEXT,  -- JSON array
                sentiment_score REAL,
                sentiment_label TEXT,
                published_time TEXT NOT NULL,
                fetched_time TEXT NOT NULL,
                expires_time TEXT NOT NULL
            )
        """)

        # Regime history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                regime TEXT NOT NULL,
                confidence REAL,
                trend_strength REAL,
                volatility REAL,
                adx REAL,
                rsi REAL,
                timestamp TEXT NOT NULL,
                details TEXT
            )
        """)

        # Key-value store for misc state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_time TEXT NOT NULL
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_position ON orders(position_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_symbols ON news_cache(symbols)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_regime_timestamp ON regime_history(timestamp)")

        self._conn.commit()

    # -------------------------------------------------------------------------
    # KEY-VALUE OPERATIONS
    # -------------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Set a key-value pair."""
        import json
        value_str = json.dumps(value) if not isinstance(value, str) else value
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO kv_store (key, value, updated_time)
            VALUES (?, ?, ?)
        """, (key, value_str, datetime.now().isoformat()))
        self._conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key."""
        import json
        cursor = self._conn.cursor()
        cursor.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return row[0]
        return default

    def delete(self, key: str) -> bool:
        """Delete a key."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM kv_store WHERE key = ?", (key,))
        self._conn.commit()
        return cursor.rowcount > 0

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM kv_store WHERE key = ?", (key,))
        return cursor.fetchone() is not None

    # -------------------------------------------------------------------------
    # POSITION OPERATIONS
    # -------------------------------------------------------------------------

    def save_position(self, position: Dict[str, Any]) -> None:
        """Save or update position."""
        import json
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO positions
            (id, symbol, side, quantity, entry_price, current_price, stop_price,
             target_prices, entry_time, updated_time, pnl, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position['id'],
            position['symbol'],
            position['side'],
            position['quantity'],
            position['entry_price'],
            position.get('current_price'),
            position.get('stop_price'),
            json.dumps(position.get('target_prices', [])),
            position['entry_time'],
            datetime.now().isoformat(),
            position.get('pnl', 0),
            json.dumps(position.get('metadata', {})),
        ))
        self._conn.commit()

    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get position by ID."""
        import json
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM positions WHERE id = ?", (position_id,))
        row = cursor.fetchone()
        if row:
            cols = [d[0] for d in cursor.description]
            pos = dict(zip(cols, row))
            pos['target_prices'] = json.loads(pos['target_prices'] or '[]')
            pos['metadata'] = json.loads(pos['metadata'] or '{}')
            return pos
        return None

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all positions, optionally filtered by symbol."""
        import json
        cursor = self._conn.cursor()
        if symbol:
            cursor.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,))
        else:
            cursor.execute("SELECT * FROM positions")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        positions = []
        for row in rows:
            pos = dict(zip(cols, row))
            pos['target_prices'] = json.loads(pos['target_prices'] or '[]')
            pos['metadata'] = json.loads(pos['metadata'] or '{}')
            positions.append(pos)
        return positions

    def delete_position(self, position_id: str) -> bool:
        """Delete position."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM positions WHERE id = ?", (position_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # ORDER OPERATIONS
    # -------------------------------------------------------------------------

    def save_order(self, order: Dict[str, Any]) -> None:
        """Save or update order."""
        import json
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO orders
            (id, position_id, symbol, side, order_type, quantity, price, status,
             created_time, updated_time, filled_quantity, avg_fill_price, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order['id'],
            order.get('position_id'),
            order['symbol'],
            order['side'],
            order['order_type'],
            order['quantity'],
            order.get('price'),
            order['status'],
            order['created_time'],
            datetime.now().isoformat(),
            order.get('filled_quantity', 0),
            order.get('avg_fill_price'),
            json.dumps(order.get('metadata', {})),
        ))
        self._conn.commit()

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order by ID."""
        import json
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE id = ?", (order_id,))
        row = cursor.fetchone()
        if row:
            cols = [d[0] for d in cursor.description]
            order = dict(zip(cols, row))
            order['metadata'] = json.loads(order['metadata'] or '{}')
            return order
        return None

    # -------------------------------------------------------------------------
    # DAILY METRICS
    # -------------------------------------------------------------------------

    def save_daily_metrics(self, date: str, metrics: Dict[str, Any]) -> None:
        """Save daily metrics."""
        import json
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO daily_metrics
            (date, equity, pnl, trades, win_rate, max_drawdown, sharpe, metrics, updated_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date,
            metrics.get('equity', 0),
            metrics.get('pnl', 0),
            metrics.get('trades', 0),
            metrics.get('win_rate'),
            metrics.get('max_drawdown'),
            metrics.get('sharpe'),
            json.dumps(metrics),
            datetime.now().isoformat(),
        ))
        self._conn.commit()

    def get_daily_metrics(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get daily metrics for date range."""
        import json
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM daily_metrics
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        """, (start_date, end_date))
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        metrics = []
        for row in rows:
            m = dict(zip(cols, row))
            m['metrics'] = json.loads(m['metrics'] or '{}')
            metrics.append(m)
        return metrics

    # -------------------------------------------------------------------------
    # SIGNALS
    # -------------------------------------------------------------------------

    def save_signal(self, signal: Dict[str, Any]) -> None:
        """Save trading signal."""
        import json
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO signals
            (id, symbol, signal_type, direction, price, strength, timestamp, metadata, deduplicated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal['id'],
            signal['symbol'],
            signal['signal_type'],
            signal['direction'],
            signal.get('price'),
            signal.get('strength'),
            signal['timestamp'],
            json.dumps(signal.get('metadata', {})),
            signal.get('deduplicated', 0),
        ))
        self._conn.commit()

    def get_recent_signals(self, symbol: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent signals for symbol."""
        import json
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM signals
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (symbol, cutoff))
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        signals = []
        for row in rows:
            s = dict(zip(cols, row))
            s['metadata'] = json.loads(s['metadata'] or '{}')
            signals.append(s)
        return signals

    # -------------------------------------------------------------------------
    # NEWS CACHE
    # -------------------------------------------------------------------------

    def save_news(self, article: Dict[str, Any]) -> None:
        """Save news article to cache."""
        import json
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO news_cache
            (content_hash, title, summary, link, source, category, symbols,
             sentiment_score, sentiment_label, published_time, fetched_time, expires_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article['content_hash'],
            article['title'],
            article.get('summary'),
            article.get('link'),
            article.get('source'),
            article.get('category'),
            json.dumps(article.get('symbols', [])),
            article.get('sentiment_score'),
            article.get('sentiment_label'),
            article['published_time'],
            article['fetched_time'],
            article['expires_time'],
        ))
        self._conn.commit()

    def get_news(self, symbol: str, hours: int = 4) -> List[Dict[str, Any]]:
        """Get cached news for symbol."""
        import json
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM news_cache
            WHERE symbols LIKE ? AND fetched_time > ? AND expires_time > ?
            ORDER BY published_time DESC
        """, (f'%{symbol}%', cutoff, datetime.now().isoformat()))
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        articles = []
        for row in rows:
            a = dict(zip(cols, row))
            a['symbols'] = json.loads(a['symbols'] or '[]')
            articles.append(a)
        return articles

    def cleanup_expired_news(self) -> int:
        """Remove expired news articles."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM news_cache WHERE expires_time < ?", (datetime.now().isoformat(),))
        self._conn.commit()
        return cursor.rowcount

    # -------------------------------------------------------------------------
    # REGIME HISTORY
    # -------------------------------------------------------------------------

    def save_regime(self, regime: Dict[str, Any]) -> None:
        """Save regime state."""
        import json
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO regime_history
            (regime, confidence, trend_strength, volatility, adx, rsi, timestamp, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            regime['regime'],
            regime.get('confidence'),
            regime.get('trend_strength'),
            regime.get('volatility'),
            regime.get('adx'),
            regime.get('rsi'),
            regime['timestamp'],
            json.dumps(regime.get('details', {})),
        ))
        self._conn.commit()

    def get_regime_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get regime history."""
        import json
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM regime_history
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        """, (cutoff,))
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        history = []
        for row in rows:
            r = dict(zip(cols, row))
            r['details'] = json.loads(r['details'] or '{}')
            history.append(r)
        return history

    # -------------------------------------------------------------------------
    # MAINTENANCE
    # -------------------------------------------------------------------------

    def backup(self, backup_path: str) -> None:
        """Create database backup."""
        import shutil
        shutil.copy2(self.db_path, backup_path)

    def vacuum(self) -> None:
        """Vacuum database to reclaim space."""
        self._conn.execute("VACUUM")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# HEALTH MONITOR
# =============================================================================

@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_func: Callable[[], bool]
    threshold: Any = None
    interval_seconds: float = 60.0
    critical: bool = True


@dataclass
class HealthStatus:
    """Health check result."""
    name: str
    healthy: bool
    value: Any
    threshold: Any
    timestamp: datetime
    error: Optional[str] = None


class HealthMonitor:
    """
    Health monitoring with async checks and optional Prometheus metrics.
    """

    def __init__(
        self,
        checks: List[HealthCheck],
        interval_seconds: float = 60.0,
        prometheus_port: Optional[int] = None,
    ):
        self.checks = {c.name: c for c in checks}
        self.interval_seconds = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._latest_results: Dict[str, HealthStatus] = {}
        self._callbacks: List[Callable[[Dict[str, HealthStatus]], None]] = []

        # Prometheus metrics
        self._prometheus_enabled = PROMETHEUS_AVAILABLE and prometheus_port is not None
        if self._prometheus_enabled:
            self._init_prometheus(prometheus_port)

    def _init_prometheus(self, port: int) -> None:
        """Initialize Prometheus metrics."""
        self._health_gauge = Gauge('health_check_status', 'Health check status (1=healthy, 0=unhealthy)', ['check'])
        self._health_value = Gauge('health_check_value', 'Health check numeric value', ['check'])
        self._health_threshold = Gauge('health_check_threshold', 'Health check threshold', ['check'])
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")

    def add_callback(self, callback: Callable[[Dict[str, HealthStatus]], None]) -> None:
        """Add callback for health check results."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")

    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        logger.info("Health monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                results = await self.run_checks()
                self._latest_results = results

                # Call callbacks
                for callback in self._callbacks:
                    try:
                        callback(results)
                    except Exception as e:
                        logger.error(f"Health callback error: {e}")

                # Update Prometheus
                if self._prometheus_enabled:
                    self._update_prometheus(results)

            except Exception as e:
                logger.error(f"Health monitor loop error: {e}")

            await asyncio.sleep(self.interval_seconds)

    async def run_checks(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        results = {}

        for name, check in self.checks.items():
            try:
                start = time.time()
                healthy = await check.check_func() if asyncio.iscoroutinefunction(check.check_func) else check.check_func()
                latency = time.time() - start

                # Evaluate against threshold if provided
                threshold_passed = True
                if check.threshold is not None:
                    # This is simplified - in practice, threshold evaluation depends on check type
                    pass

                status = HealthStatus(
                    name=name,
                    healthy=healthy,
                    value=latency,
                    threshold=check.threshold,
                    timestamp=datetime.now(),
                )

            except Exception as e:
                status = HealthStatus(
                    name=name,
                    healthy=False,
                    value=None,
                    threshold=check.threshold,
                    timestamp=datetime.now(),
                    error=str(e),
                )

            results[name] = status

        return results

    def _update_prometheus(self, results: Dict[str, HealthStatus]) -> None:
        """Update Prometheus metrics."""
        for name, status in results.items():
            self._health_gauge.labels(check=name).set(1 if status.healthy else 0)
            if status.value is not None:
                self._health_value.labels(check=name).set(status.value)
            if status.threshold is not None:
                self._health_threshold.labels(check=name).set(status.threshold)

    def get_status(self) -> Dict[str, HealthStatus]:
        """Get latest health check results."""
        return self._latest_results.copy()

    def is_healthy(self) -> bool:
        """Check if all critical checks are healthy."""
        return all(
            s.healthy for s in self._latest_results.values()
            if self.checks[s.name].critical
        )


# =============================================================================
# PREDEFINED HEALTH CHECKS
# =============================================================================

def create_data_latency_check(data_fetcher, threshold_ms: float = 5000) -> HealthCheck:
    """Create health check for data fetcher latency."""
    async def check():
        health = await data_fetcher.health_check()
        for source, status in health.items():
            if status.latency_ms > threshold_ms:
                return False
        return True

    return HealthCheck(
        name="data_latency",
        check_func=check,
        threshold=threshold_ms,
        interval_seconds=60,
        critical=True,
    )


def create_data_freshness_check(data_fetcher, threshold_seconds: float = 300) -> HealthCheck:
    """Create health check for data freshness."""
    async def check():
        health = await data_fetcher.health_check()
        for source, status in health.items():
            if status.freshness_seconds > threshold_seconds:
                return False
        return True

    return HealthCheck(
        name="data_freshness",
        check_func=check,
        threshold=threshold_seconds,
        interval_seconds=60,
        critical=True,
    )


def create_api_error_check(data_fetcher, threshold_count: int = 10) -> HealthCheck:
    """Create health check for API error count."""
    def check():
        health = await data_fetcher.health_check() if asyncio.iscoroutinefunction(data_fetcher.health_check) else {}
        for source, status in health.items():
            if status.error_count > threshold_count:
                return False
        return True

    return HealthCheck(
        name="api_errors",
        check_func=check,
        threshold=threshold_count,
        interval_seconds=60,
        critical=True,
    )


def create_memory_check(threshold_pct: float = 85) -> HealthCheck:
    """Create health check for memory usage."""
    import psutil
    def check():
        mem = psutil.virtual_memory()
        return mem.percent < threshold_pct

    return HealthCheck(
        name="memory_usage",
        check_func=check,
        threshold=threshold_pct,
        interval_seconds=60,
        critical=True,
    )


def create_disk_check(threshold_pct: float = 90) -> HealthCheck:
    """Create health check for disk usage."""
    import psutil
    def check():
        disk = psutil.disk_usage('/')
        return (disk.used / disk.total * 100) < threshold_pct

    return HealthCheck(
        name="disk_usage",
        check_func=check,
        threshold=threshold_pct,
        interval_seconds=300,
        critical=True,
    )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_circuit_breaker(
    failure_threshold: int = 5,
    timeout_seconds: float = 60.0,
) -> CircuitBreaker:
    """Factory for circuit breaker."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout_seconds=timeout_seconds,
    )
    return CircuitBreaker(config)


def create_sqlite_state(db_path: str = "data/state/trading_state.db") -> SQLiteState:
    """Factory for SQLite state."""
    return SQLiteState(db_path)


def create_health_monitor(
    data_fetcher=None,
    prometheus_port: Optional[int] = None,
) -> HealthMonitor:
    """Factory for health monitor with default checks."""
    checks = []

    if data_fetcher:
        checks.append(create_data_latency_check(data_fetcher))
        checks.append(create_data_freshness_check(data_fetcher))
        checks.append(create_api_error_check(data_fetcher))

    checks.append(create_memory_check())
    checks.append(create_disk_check())

    return HealthMonitor(checks, prometheus_port=prometheus_port)