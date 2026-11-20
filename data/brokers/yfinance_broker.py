"""
YFinance Broker Implementation - Free, no API key required.
Uses yfinance for data fetching with robust error handling.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import asyncio
import time

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

    async def initialize(self) -> bool:
        """Initialize - nothing needed for yfinance."""
        self._initialized = True
        return True

    async def close(self):
        """Close connections."""
        pass

    def _symbol_to_yahoo(self, symbol: str) -> str:
        """Convert NSE symbol to Yahoo Finance format."""
        return f"{symbol}.NS"

    def _get_ticker(self, symbol: str):
        """Get cached ticker object."""
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(f"{symbol}.NS")
        return self._ticker_cache[symbol]

    async def _fetch_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic for transient errors."""
        last_exception = None
        for attempt in range(self._max_retries):
            try:
                return await func(*args, **kwargs)
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

    async def _download_symbol(self, yf_symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Download single symbol with error handling."""
        def _download():
            return yf.download(
                yf_symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
                group_by='ticker'
            )

        return await self._fetch_with_retry(_download)

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
                df = await self._download_symbol(yf_symbol, period, interval)

                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    failed_symbols.append(symbol)
                    continue

                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Standardize column names
                df.columns = [c.capitalize() for c in df.columns]
                df.index = pd.to_datetime(df.index)

                # Ensure we have required columns
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns {missing_cols} for {symbol}")
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
        combined.set_index('Symbol', append=True, inplace=True)
        combined = combined.reorder_levels(['Symbol', combined.index.names[0]])
        combined.sort_index(inplace=True)
        return combined

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
                info = await asyncio.get_event_loop().run_in_executor(None, lambda: ticker.info)

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
            df = await self._download_symbol("^NSEI", period, interval)

            if df.empty:
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
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