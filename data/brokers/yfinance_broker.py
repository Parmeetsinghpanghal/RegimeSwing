"""
YFinance Broker Implementation - Free, no API key required.
Uses yfinance for data fetching.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

from .base import BrokerBase, BrokerConfig, BrokerType, PriceData, OptionData, OptionChainData, FundamentalsData

logger = logging.getLogger(__name__)

# Check if yfinance is available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available. Install with: pip install yfinance")

YFINANCE_AVAILABLE = True


class YFinanceBroker:
    """YFinance Broker - Free, no API key required."""

    def __init__(self, config=None):
        self.config = config or {}
        self._initialized = False
        self._ticker_cache = {}

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

    async def initialize(self) -> bool:
        """Initialize - nothing needed for yfinance."""
        self._initialized = True
        return True

    async def close(self):
        """Close connections."""
        pass

    async def get_price_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get price data for symbols."""
        if isinstance(symbols, str):
            symbols = [symbols]

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        all_data = []

        for symbol in symbols:
            try:
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)

                df = yf.download(
                    yf_symbol,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=False
                )

                if df.empty:
                    continue

                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Standardize column names
                df.columns = [c.capitalize() for c in df.columns]
                df.index = pd.to_datetime(df.index)

                # Ensure we have required columns
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required:
                    if col not in df.columns:
                        logger.warning(f"Missing column {col} for {symbol}")
                        return pd.DataFrame()

                df = df[required]
                df['Symbol'] = symbol
                all_data.append(df)

            except Exception as e:
                logger.warning(f"yfinance fetch failed for {symbol}: {e}")

        if not all_data:
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
        if not YFINANCE_AVAILABLE:
            return pd.DataFrame()

        results = []
        for symbol in symbols:
            try:
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info

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
            df = yf.download("^NSEI", period=period, interval=interval, progress=False, auto_adjust=False)
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

    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data using yfinance."""
        if not YFINANCE_AVAILABLE:
            return pd.DataFrame()

        results = []
        for symbol in symbols:
            try:
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info

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
            df = yf.download("^NSEI", period=period, interval=interval, progress=False, auto_adjust=False)
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

    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data using yfinance."""
        if not YFINANCE_AVAILABLE:
            return pd.DataFrame()

        results = []
        for symbol in symbols:
            try:
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info

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
            df = yf.download("^NSEI", period=period, interval=interval, progress=False, auto_adjust=False)
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

    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data using yfinance."""
        results = []
        for symbol in symbols:
            try:
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info

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
            df = yf.download("^NSEI", period=period, interval=interval, progress=False, auto_adjust=False)
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

    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data using yfinance."""
        results = []
        for symbol in symbols:
            try:
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info

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
            df = yf.download("^NSEI", period=period, interval=interval, progress=False, auto_adjust=False)
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

    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data using yfinance."""
        results = []
        for symbol in symbols:
            try:
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info

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
            df = yf.download("^NSEI", period=period, interval=interval, progress=False, auto_adjust=False)
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

    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data using yfinance."""
        results = []
        for symbol in symbols:
            try:
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info

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


# For backward compatibility
YFinanceBroker = YFinanceBroker