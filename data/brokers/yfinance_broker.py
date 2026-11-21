"""
YFinance Broker Implementation - Free, no API key required.
Uses yfinance for data fetching with robust error handling.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import asyncio
import concurrent.futures

import pandas as pd
import numpy as np
import yfinance as yf

from .base import BrokerBase, BrokerConfig, BrokerType, PriceData, OptionData, OptionChainData, FundamentalsData

logger = logging.getLogger(__name__)


class YFinanceBroker:
    """YFinance Broker - Free, no API key required with robust error handling."""

    def __init__(self, config: BrokerConfig = None):
        self.config = config or BrokerConfig(broker_type=BrokerType.YFINANCE)
        self._initialized = False
        self._ticker_cache = {}
        self._max_retries = 3
        self._base_delay = 1.0
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def initialize(self) -> bool:
        """Initialize - nothing needed for yfinance."""
        self._initialized = True
        return True

    async def close(self):
        """Close connections."""
        self._executor.shutdown(wait=True)

    def _symbol_to_yahoo(self, symbol: str) -> str:
        """Convert NSE symbol to Yahoo Finance format."""
        return f"{symbol}.NS"

    def _get_ticker(self, symbol: str):
        """Get cached ticker object."""
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(f"{symbol}.NS")
        return self._ticker_cache[symbol]

    async def _run_sync_with_retry(self, func, *args, **kwargs):
        """Execute synchronous function with retry logic in thread pool."""
        last_exception = None
        for attempt in range(self._max_retries):
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check for retryable errors
                retryable = any(keyword in error_msg for keyword in [
                    'invalid crumb', '429', 'rate limit', 'too many requests',
                    'connection', 'timeout', 'temporary', 'unavailable'
                ])

                if retryable and attempt < self._max_retries - 1:
                    delay = self._base_delay * (2 ** attempt) + np.random.uniform(0, 0.5)
                    logger.warning(f"Retryable error (attempt {attempt + 1}/{self._max_retries}): {e}. Waiting {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue

                # Non-retryable or max retries reached
                raise

        raise last_exception

    async def get_price_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get price data for symbols with robust error handling."""
        if isinstance(symbols, str):
            symbols = [symbols]

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        all_data = []
        failed_symbols = []

        for symbol in symbols:
            yf_symbol = f"{symbol}.NS"

            try:
                df = await self._run_sync_with_retry(
                    self._download_symbol_sync, yf_symbol, period, interval
                )

                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    failed_symbols.append(symbol)
                    continue

                # Debug: log actual columns received
                logger.debug(f"Raw columns for {symbol}: {list(df.columns)}")

                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    # yfinance with group_by='ticker' creates:
                    # Level 0: Ticker symbol (e.g., 'RELIANCE.NS')
                    # Level 1: OHLCV columns ('Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close')
                    # We need level 1 (OHLCV data)
                    try:
                        df.columns = df.columns.get_level_values(1)
                    except Exception:
                        logger.warning(f"Could not flatten multi-index columns for {symbol}")
                        failed_symbols.append(symbol)
                        continue

                # Standardize column names - handle various formats
                # yfinance may return: 'Open', 'High', 'Low', 'Close', 'Volume'
                # or 'open', 'high', 'low', 'close', 'volume'
                # or 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'
                # Map to standard capitalized format
                col_mapping = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in ('open', 'high', 'low', 'close', 'volume', 'adj close', 'adj_close'):
                        col_mapping[col] = col_lower.capitalize().replace('_', ' ')
                    elif col_lower == 'adj close':
                        col_mapping[col] = 'Adj Close'

                if col_mapping:
                    df = df.rename(columns=col_mapping)

                df.index = pd.to_datetime(df.index)

                # Ensure we have required columns (case-insensitive check)
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required if col not in df.columns]

                # Try alternative column names if standard ones missing
                alt_names = {
                    'Open': ['open', 'OPEN', 'Open Price'],
                    'High': ['high', 'HIGH', 'High Price'],
                    'Low': ['low', 'LOW', 'Low Price'],
                    'Close': ['close', 'CLOSE', 'Close Price', 'Adj Close', 'AdjClose'],
                    'Volume': ['volume', 'VOLUME', 'Vol']
                }

                for req_col in missing_cols[:]:
                    for alt in alt_names.get(req_col, []):
                        if alt in df.columns:
                            df[req_col] = df[alt]
                            missing_cols.remove(req_col)
                            break

                if missing_cols:
                    logger.warning(f"Missing columns {missing_cols} for {symbol} (available: {list(df.columns)})")
                    failed_symbols.append(symbol)
                    continue

                df = df[required]
                df['Symbol'] = symbol
                all_data.append(df)

            except Exception as e:
                error_msg = str(e).lower()
                if 'delisted' in error_msg or 'no data found' in error_msg:
                    logger.info(f"Symbol {symbol} appears to be delisted or unavailable")
                else:
                    logger.warning(f"yfinance fetch failed for {symbol}: {e}")
                failed_symbols.append(symbol)
                continue

        if failed_symbols:
            logger.info(f"Failed to fetch {len(failed_symbols)} symbols: {failed_symbols[:10]}...")

        if not all_data:
            logger.warning("No data fetched for any symbol")
            return pd.DataFrame()

        # Combine all symbols
        combined = pd.concat(all_data)
        combined = combined.set_index('Symbol', append=True)
        combined = combined.reorder_levels(['Symbol', combined.index.names[0]])
        combined = combined.sort_index()
        return combined

    def _download_symbol_sync(self, yf_symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Synchronous download of single symbol."""
        return yf.download(
            yf_symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
            group_by='ticker'
        )

    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Get option chain - not available via yfinance."""
        return pd.DataFrame(columns=[
            'strike', 'expiry', 'option_type', 'iv', 'delta', 'gamma',
            'theta', 'vega', 'rho', 'oi', 'volume', 'bid', 'ask', 'ltp'
        ])

    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data using yfinance."""
        results = []
        for symbol in symbols:
            try:
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)

                # Run in thread pool to avoid blocking
                info = await asyncio.get_event_loop().run_in_executor(
                    self._executor, lambda: ticker.info
                )

                # Extract key metrics
                fundamentals = {
                    'symbol': symbol,
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'roe': info.get('returnOnEquity'),
                    'roic': info.get('returnOnInvestedCapital'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'current_ratio': info.get('currentRatio'),
                    'profit_margin': info.get('profitMargins'),
                    'revenue_growth': info.get('revenueGrowth'),
                    'earnings_growth': info.get('earningsGrowth'),
                    'free_cash_flow': info.get('freeCashflow'),
                    'dividend_yield': info.get('dividendYield'),
                    'beta': info.get('beta'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                }
                results.append(fundamentals)

            except Exception as e:
                logger.warning(f"Fundamentals fetch failed for {symbol}: {e}")

        return pd.DataFrame(results)

    async def get_nifty_index_data(self, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Get Nifty 50 index data for regime detection."""
        try:
            df = await self._run_sync_with_retry(
                self._download_symbol_sync, "^NSEI", period, interval
            )

            if df.empty:
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                # For ^NSEI, level 0 is the ticker, level 1 is OHLCV
                df.columns = df.columns.get_level_values(1)
            df.columns = [c.capitalize() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            logger.error(f"Nifty data fetch failed: {e}")
            return pd.DataFrame()

    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Option chain not available via yfinance."""
        return pd.DataFrame(columns=[
            'strike', 'expiry', 'option_type', 'iv', 'delta', 'gamma',
            'theta', 'vega', 'rho', 'oi', 'volume', 'bid', 'ask', 'ltp'
        ])

    async def health_check(self) -> Dict:
        """Health check."""
        return {"status": "ok", "broker": "yfinance", "available": True}

    async def place_order(self, order_params: dict) -> dict:
        """Place order - not supported for yfinance."""
        return {"error": "Order placement not supported for yfinance broker"}

    async def get_positions(self) -> List[dict]:
        """Get positions - not available for yfinance."""
        return []

    async def get_orders(self) -> List[dict]:
        """Get orders - not available for yfinance."""
        return []