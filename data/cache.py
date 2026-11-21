"""
OHLCV Cache Layer - SQLite metadata + Parquet data for fast Nifty 200 scanning.
Zero-cost, batch-friendly, rate-limit aware caching for yfinance.
Synchronous version for Streamlit Cloud compatibility.
"""

import os
import sqlite3
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# Cache directory structure
CACHE_ROOT = Path("./data/cache/ohlcv")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH = CACHE_ROOT / "cache_metadata.db"
PARQUET_DIR = CACHE_ROOT / "parquet"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# Default batch settings for yfinance (respect rate limits)
DEFAULT_BATCH_SIZE = 20
DEFAULT_DELAY_SECONDS = 0.5
MAX_RETRIES = 3


@dataclass
class CacheMetadata:
    """Cache metadata for a symbol/timeframe."""
    symbol: str
    timeframe: str
    last_updated: datetime
    row_count: int
    date_min: datetime
    date_max: datetime
    file_path: str


class OHLCVCache:
    """
    Hybrid SQLite + Parquet cache for OHLCV data.

    SQLite stores metadata (fast lookups, date ranges)
    Parquet stores actual OHLCV data (columnar, compressed, fast reads)
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = CACHE_ROOT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
        max_retries: int = MAX_RETRIES,
        default_ttl_hours: int = 24
    ):
        self.cache_dir = Path(cache_dir)
        self.parquet_dir = self.cache_dir / "parquet"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache_metadata.db"
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.default_ttl_hours = default_ttl_hours

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with metadata table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    last_updated TIMESTAMP NOT NULL,
                    row_count INTEGER NOT NULL,
                    date_min TIMESTAMP NOT NULL,
                    date_max TIMESTAMP NOT NULL,
                    file_path TEXT NOT NULL,
                    PRIMARY KEY (symbol, timeframe)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe
                ON cache_metadata(symbol, timeframe)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_updated
                ON cache_metadata(last_updated)
            """)
            conn.commit()

    def _get_parquet_path(self, symbol: str, timeframe: str) -> Path:
        """Get parquet file path for symbol/timeframe."""
        safe_symbol = symbol.replace("&", "").replace("-", "_").replace(".", "_")
        return self.parquet_dir / f"{safe_symbol}_{timeframe}.parquet"

    def _symbol_to_yahoo(self, symbol: str) -> str:
        """Convert NSE symbol to Yahoo Finance format."""
        return f"{symbol}.NS"

    def has_fresh_data(
        self,
        symbol: str,
        timeframe: str,
        max_age_hours: Optional[int] = None
    ) -> bool:
        """Check if cache has fresh data for symbol/timeframe."""
        max_age = max_age_hours or self.default_ttl_hours

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT last_updated FROM cache_metadata WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe)
            ).fetchone()

        if not row:
            return False

        last_updated = datetime.fromisoformat(row['last_updated'])
        return datetime.now() - last_updated < timedelta(hours=max_age)

    def get_date_range(self, symbol: str, timeframe: str) -> Optional[Tuple[datetime, datetime]]:
        """Get cached date range for symbol/timeframe."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT date_min, date_max FROM cache_metadata WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe)
            ).fetchone()

        if not row:
            return None
        return (
            datetime.fromisoformat(row['date_min']),
            datetime.fromisoformat(row['date_max'])
        )

    def get(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get cached OHLCV data for symbol.

        Args:
            symbol: NSE symbol (e.g., "RELIANCE")
            timeframe: "1d", "1h", "15m", etc.
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV columns, datetime index
        """
        parquet_path = self._get_parquet_path(symbol, timeframe)

        if not parquet_path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(parquet_path)
            df.index = pd.to_datetime(df.index)

            # Filter by date range if provided
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]

            return df
        except Exception as e:
            logger.warning(f"Failed to read cache for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def put(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Save OHLCV data to cache.

        Args:
            df: DataFrame with OHLCV columns and datetime index
            symbol: NSE symbol
            timeframe: Timeframe string

        Returns:
            True if successful
        """
        if df.empty:
            return False

        parquet_path = self._get_parquet_path(symbol, timeframe)

        try:
            # Ensure proper format
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} for {symbol}")
                    return False

            df = df[required_cols].sort_index()

            # Save to parquet
            df.to_parquet(parquet_path, compression='snappy')

            # Update metadata
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_metadata
                    (symbol, timeframe, last_updated, row_count, date_min, date_max, file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    timeframe,
                    datetime.now().isoformat(),
                    len(df),
                    df.index.min().isoformat(),
                    df.index.max().isoformat(),
                    str(parquet_path)
                ))
                conn.commit()

            logger.debug(f"Cached {len(df)} rows for {symbol} {timeframe}")
            return True

        except Exception as e:
            logger.error(f"Failed to write cache for {symbol} {timeframe}: {e}")
            return False

    def _download_symbol_sync(self, yf_symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Synchronous yfinance download for a single symbol."""
        return yf.download(
            yf_symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
            group_by='ticker'
        )

    def _download_with_retry(self, yf_symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Download with retry logic (synchronous)."""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                df = self._download_symbol_sync(yf_symbol, period, interval)

                if df.empty:
                    raise ValueError(f"No data returned for {yf_symbol}")

                # Flatten multi-index columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(1)

                # Standardize column names
                df.columns = [c.capitalize() for c in df.columns]
                df.index = pd.to_datetime(df.index)

                # Ensure required columns
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required:
                    if col not in df.columns:
                        raise ValueError(f"Missing column {col}")

                return df[required]

            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check if retryable
                retryable = any(kw in error_msg for kw in [
                    'invalid crumb', '429', 'rate limit', 'too many requests',
                    'connection', 'timeout', 'temporary', 'unavailable'
                ])

                if retryable and attempt < self.max_retries - 1:
                    delay = self.delay_seconds * (2 ** attempt) + np.random.uniform(0, 0.5)
                    logger.warning(f"Retryable error for {yf_symbol} (attempt {attempt+1}): {e}. Waiting {delay:.1f}s...")
                    time.sleep(delay)
                    continue

                raise

        raise last_exception

    def fetch_and_cache(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch and cache data for multiple symbols with batching and delays (synchronous).

        Args:
            symbols: List of NSE symbols
            period: yfinance period (e.g., "2y", "1y", "6mo")
            interval: yfinance interval (e.g., "1d", "1h", "15m")
            progress_callback: Optional callback(current, total, symbol)

        Returns:
            Dict mapping symbol -> DataFrame (empty if failed)
        """
        results = {}
        failed = []

        # Check if interval is 15m/5m/1h and period is too long - yfinance only supports 60 days for intraday
        intraday_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
        if interval in intraday_intervals:
            # Check if period is longer than 60 days
            if period in ['1y', '2y', '5y', '10y', 'max'] or (period.endswith('mo') and int(period[:-2]) > 2):
                logger.warning(f"Interval {interval} only supports last 60 days. Falling back to 1d for period {period}")
                interval = '1d'

        # Process in batches
        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i:i + self.batch_size]

            for j, symbol in enumerate(batch):
                yf_symbol = self._symbol_to_yahoo(symbol)

                # Check cache first
                if self.has_fresh_data(symbol, interval):
                    cached = self.get(symbol, interval)
                    if not cached.empty:
                        logger.debug(f"Cache hit for {symbol} {interval}")
                        results[symbol] = cached
                        if progress_callback:
                            progress_callback(i + j + 1, len(symbols), symbol)
                        continue

                # Fetch fresh data
                try:
                    df = self._download_with_retry(yf_symbol, period, interval)
                    self.put(df, symbol, interval)
                    results[symbol] = df
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol}: {e}")
                    failed.append(symbol)
                    results[symbol] = pd.DataFrame()

                if progress_callback:
                    progress_callback(i + j + 1, len(symbols), symbol)

            # Delay between batches to respect rate limits
            if i + self.batch_size < len(symbols):
                time.sleep(self.delay_seconds)

        if failed:
            logger.info(f"Failed to fetch {len(failed)} symbols: {failed[:10]}...")

        return results

    def get_price_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get price data for symbols (synchronous interface).

        Args:
            symbols: Single symbol or list of symbols
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            period: yfinance period (e.g., "2y", "1y", "6mo")
            interval: yfinance interval (e.g., "1d", "1h", "15m")
            use_cache: Whether to use cached data

        Returns:
            MultiIndex DataFrame: (Symbol, Date) with OHLCV columns
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # Try to get from cache first
        all_data = []
        to_fetch = []

        for symbol in symbols:
            if use_cache and self.has_fresh_data(symbol, interval):
                df = self.get(symbol, interval, start_date, end_date)
                if not df.empty:
                    df['Symbol'] = symbol
                    all_data.append(df)
                    continue
            to_fetch.append(symbol)

        # Fetch missing symbols
        if to_fetch:
            fetch_results = self.fetch_and_cache(to_fetch, period, interval)

            for symbol, df in fetch_results.items():
                if not df.empty:
                    # Apply date filters
                    if start_date:
                        df = df[df.index >= pd.Timestamp(start_date)]
                    if end_date:
                        df = df[df.index <= pd.Timestamp(end_date)]
                    df['Symbol'] = symbol
                    all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        # Combine
        combined = pd.concat(all_data)
        combined = combined.set_index('Symbol', append=True)
        combined = combined.reorder_levels(['Symbol', combined.index.names[0]])
        combined = combined.sort_index()
        return combined

    def get_nifty_index_data(
        self,
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get Nifty 50 index data (^NSEI) - synchronous."""
        yf_symbol = "^NSEI"

        # Check cache
        if self.has_fresh_data("NIFTY50", interval):
            cached = self.get("NIFTY50", interval)
            if not cached.empty:
                return cached

        # Fetch
        df = self._download_with_retry(yf_symbol, period, interval)
        self.put(df, "NIFTY50", interval)
        return df

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT symbol, timeframe, row_count, last_updated, date_min, date_max
                FROM cache_metadata
                ORDER BY last_updated DESC
            """).fetchall()

        total_rows = sum(r['row_count'] for r in rows)
        total_size = sum(
            self._get_parquet_path(r['symbol'], r['timeframe']).stat().st_size
            for r in rows if self._get_parquet_path(r['symbol'], r['timeframe']).exists()
        )

        return {
            'symbols_cached': len(set(r['symbol'] for r in rows)),
            'total_rows': total_rows,
            'total_size_mb': total_size / (1024 * 1024),
            'timeframes': list(set(r['timeframe'] for r in rows)),
            'details': [dict(r) for r in rows]
        }

    def clear_old_cache(self, max_age_days: int = 30):
        """Clear cache entries older than max_age_days."""
        cutoff = datetime.now() - timedelta(days=max_age_days)

        with sqlite3.connect(self.db_path) as conn:
            old_entries = conn.execute(
                "SELECT symbol, timeframe, file_path FROM cache_metadata WHERE last_updated < ?",
                (cutoff.isoformat(),)
            ).fetchall()

            for symbol, timeframe, file_path in old_entries:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass

            conn.execute("DELETE FROM cache_metadata WHERE last_updated < ?", (cutoff.isoformat(),))
            conn.commit()

        logger.info(f"Cleared {len(old_entries)} old cache entries")

    def close(self):
        """Close any resources (no-op for sync version)."""
        pass


# Convenience function for quick access
_default_cache = None


def get_cache() -> OHLCVCache:
    """Get or create default cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = OHLCVCache()
    return _default_cache


async def fetch_nifty200_batch(
    cache: OHLCVCache = None,
    period: str = "2y",
    interval: str = "1d",
    max_symbols: int = 200
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to fetch full Nifty 200 in one call.

    Returns dict of symbol -> DataFrame
    """
    from data.universe import get_nifty200_symbols

    if cache is None:
        cache = get_cache()

    symbols = get_nifty200_symbols()[:max_symbols]
    logger.info(f"Fetching {len(symbols)} Nifty 200 symbols...")

    # Run sync version in executor for async compatibility
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, cache.fetch_and_cache, symbols, period, interval)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test cache
    cache = OHLCVCache()

    # Test single symbol
    print("Testing single symbol fetch...")
    df = cache.get_price_data("RELIANCE", period="3mo", interval="1d")
    print(f"RELIANCE: {len(df)} rows")
    if not df.empty:
        print(df.head())

    # Test batch
    print("\nTesting batch fetch...")
    import asyncio
    async def test_batch():
        results = await cache.fetch_and_cache(["RELIANCE", "TCS", "HDFCBANK"], period="3mo", interval="1d")
        for sym, df in results.items():
            print(f"  {sym}: {len(df)} rows")

    asyncio.run(test_batch())

    # Stats
    print("\nCache stats:")
    stats = cache.get_cache_stats()
    print(f"  Symbols: {stats['symbols_cached']}")
    print(f"  Rows: {stats['total_rows']}")
    print(f"  Size: {stats['total_size_mb']:.2f} MB")