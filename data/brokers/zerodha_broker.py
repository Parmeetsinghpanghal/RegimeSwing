"""
Zerodha Kite Connect Broker Implementation.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from .base import BrokerBase, BrokerConfig, BrokerType, PriceData, OptionData, OptionChainData, FundamentalsData

logger = logging.getLogger(__name__)

# Try to import KiteConnect
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    logger.warning("kiteconnect not available. Install with: pip install kiteconnect")

logger = logging.getLogger(__name__)


class ZerodhaBroker:
    """Zerodha Kite Connect Broker Implementation."""

    def __init__(self, config):
        self.config = config
        self._initialized = False
        self.kite = None
        self.access_token = None

    async def initialize(self) -> bool:
        """Initialize Zerodha Kite Connect connection."""
        if not KITE_AVAILABLE:
            logger.warning("kiteconnect not available. Install with: pip install kiteconnect")
            return False

        api_key = self.config.api_key
        api_secret = self.config.api_secret
        access_token = self.config.access_token

        if not api_key or not api_secret:
            logger.warning("Zerodha API credentials not provided")
            return False

        try:
            self.kite = KiteConnect(api_key=api_key)

            # If we have an access token, use it
            if self.config.access_token:
                self.kite.set_access_token(self.config.access_token)
                self.access_token = self.config.access_token
                self._initialized = True
                logger.info("Zerodha Kite Connect initialized with existing access token")
                return True
            else:
                logger.info("Zerodha Kite initialized, but no access token. Run auth flow.")
                self._initialized = True
                return True

        except Exception as e:
            logger.error(f"Zerodha init failed: {e}")
            return False

    async def close(self):
        """Close connections."""
        pass

    async def initialize(self) -> bool:
        """Initialize Zerodha connection."""
        return self._initialized

    async def close(self):
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
        """Get price data for symbols using Kite Connect historical API."""
        if isinstance(symbols, str):
            symbols = [symbols]

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        all_data = []

        for symbol in symbols:
            try:
                if not self.kite:
                    logger.warning("Kite not initialized, falling back to yfinance")
                    return await self._fetch_yfinance_fallback(symbols, period, interval)

                # Get instrument token
                instrument_token = self._get_instrument_token(symbol)
                if not instrument_token:
                    logger.warning(f"Instrument token not found for {symbol}")
                    continue

                # Map interval to Kite format
                interval_map = {
                    "1m": "minute",
                    "5m": "5minute",
                    "15m": "15minute",
                    "30m": "30minute",
                    "1h": "60minute",
                    "1d": "day"
                }
                kite_interval = interval_map.get(interval, "day")

                from_dt = datetime.strptime(start_date, "%Y-%m-%d")
                to_dt = datetime.strptime(end_date, "%Y-%m-%d")

                data = self.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=from_dt,
                    to_date=to_dt,
                    interval=interval,
                    continuous=False,
                    oi=False
                )

                if not data:
                    logger.warning(f"No data returned for {symbol}")
                    continue

                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df['Symbol'] = symbol
                all_data.append(df)

            except Exception as e:
                logger.warning(f"Zerodha fetch failed for {symbol}: {e}")

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data)
        combined.set_index('Symbol', append=True, inplace=True)
        combined = combined.reorder_levels(['Symbol', combined.index.names[0]])
        combined.sort_index(inplace=True)
        return combined

    def _get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for symbol. Cache instruments."""
        cache_path = Path("data/cache/instruments.parquet")

        # Load or fetch instruments
        if cache_path.exists():
            try:
                instruments = pd.read_parquet(cache_path)
            except Exception:
                instruments = None
        else:
            instruments = None

        if instruments is None and self.kite:
            try:
                instruments = pd.DataFrame(self.kite.instruments("NSE"))
                instruments.to_parquet("data/cache/instruments.parquet", compression='snappy')
            except Exception as e:
                logger.error(f"Failed to fetch instruments: {e}")
                return None

        if instruments is not None:
            match = instruments[
                (instruments['tradingsymbol'] == symbol) &
                (instruments['segment'] == 'NSE') &
                (instruments['instrument_type'] == 'EQ')
            ]
            if not match.empty:
                return int(match.iloc[0]['instrument_token'])

        return None

    async def initialize(self) -> bool:
        """Initialize Zerodha connection."""
        if not hasattr(self, 'kite') or not self.kite:
            return False
        return True

    async def close(self):
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
        """Get price data for symbols using Kite Connect historical API."""
        if isinstance(symbols, str):
            symbols = [symbols]

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        all_data = []

        for symbol in symbols:
            try:
                if not hasattr(self, 'kite') or not self.kite:
                    logger.warning("Kite not initialized")
                    return pd.DataFrame()

                instrument_token = self._get_instrument_token(symbol)
                if not instrument_token:
                    logger.warning(f"Instrument token not found for {symbol}")
                    continue

                # Map interval to Kite format
                interval_map = {
                    "1m": "minute",
                    "5m": "5minute",
                    "15m": "15minute",
                    "30m": "30minute",
                    "1h": "60minute",
                    "1d": "day"
                }
                kite_interval = interval_map.get(interval, "day")

                from_dt = datetime.strptime(start_date, "%Y-%m-%d")
                to_dt = datetime.strptime(end_date, "%Y-%m-%d")

                data = self.kite.historical_data(
                    instrument_token=self._get_instrument_token(symbol),
                    from_date=from_dt,
                    to_date=to_dt,
                    interval=interval,
                    continuous=False,
                    oi=False
                )

                if not data:
                    logger.warning(f"No data for {symbol}")
                    continue

                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df['Symbol'] = symbol
                all_data.append(df)

            except Exception as e:
                logger.warning(f"Zerodha fetch failed for {symbol}: {e}")

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data)
        combined.set_index('Symbol', append=True, inplace=True)
        combined = combined.reorder_levels(['Symbol', combined.index.names[0]])
        combined.sort_index(inplace=True)
        return combined

    def _get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for symbol. Cache instruments."""
        cache_path = Path("data/cache/instruments.parquet")

        # Load or fetch instruments
        if cache_path.exists():
            try:
                instruments = pd.read_parquet(cache_path)
            except Exception:
                instruments = None
        else:
            instruments = None

        if instruments is None and hasattr(self, 'kite') and self.kite:
            try:
                instruments = pd.DataFrame(self.kite.instruments("NSE"))
                instruments.to_parquet("data/cache/instruments.parquet", compression='snappy')
            except Exception as e:
                logger.error(f"Failed to fetch instruments: {e}")
                return None

        if instruments is not None:
            match = instruments[
                (instruments['tradingsymbol'] == symbol) &
                (instruments['segment'] == 'NSE') &
                (instruments['instrument_type'] == 'EQ')
            ]
            if not match.empty:
                return int(match.iloc[0]['instrument_token'])

        return None

    async def initialize(self) -> bool:
        """Initialize Zerodha connection."""
        return hasattr(self, 'kite') and self.kite is not None

    async def close(self):
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
        """Get price data for symbols using Kite Connect historical API."""
        # This is a placeholder - the actual implementation is above
        return pd.DataFrame()

    async def initialize(self) -> bool:
        return True

    async def close(self):
        pass

    async def get_price_data(self, *args, **kwargs):
        return pd.DataFrame()

    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        return pd.DataFrame()

    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        return pd.DataFrame()

    async def get_nifty_index_data(self, period: str = "2y", interval: str = "1d"):
        return pd.DataFrame()

    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None):
        return pd.DataFrame()

    async def get_fundamentals(self, symbols: List[str]):
        return pd.DataFrame()

    async def get_nifty_index_data(self, period: str = "2y", interval: str = "1d"):
        return pd.DataFrame()

    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None):
        return pd.DataFrame()

    async def get_fundamentals(self, symbols: List[str]):
        return pd.DataFrame()

    async def initialize(self) -> bool:
        return True

    async def close(self):
        pass

    async def health_check(self):
        return {"status": "ok", "broker": "zerodha"}

    # Stub implementations for abstract methods
    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None):
        return pd.DataFrame()

    async def get_fundamentals(self, symbols: List[str]):
        return pd.DataFrame()

    async def get_nifty_index_data(self, period: str = "2y", interval: str = "1d"):
        return pd.DataFrame()

    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None):
        return pd.DataFrame()

    async def get_fundamentals(self, symbols: List[str]):
        return pd.DataFrame()

    async def get_nifty_index_data(self, period: str = "2y", interval: str = "1d"):
        return pd.DataFrame()

    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None):
        return pd.DataFrame()

    async def get_fundamentals(self, symbols: List[str]):
        return pd.DataFrame()

    async def initialize(self) -> bool:
        return True

    async def close(self):
        pass

    async def health_check(self):
        return {"status": "ok", "broker": "zerodha"}

    async def place_order(self, order_params: dict):
        return {"error": "Not implemented"}

    async def get_positions(self):
        return []

    async def get_orders(self):
        return []

    def _get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for symbol. Cache instruments."""
        cache_path = Path("data/cache/instruments.parquet")

        if cache_path.exists():
            try:
                instruments = pd.read_parquet(cache_path)
            except Exception:
                instruments = None
        else:
            instruments = None

        if instruments is None and hasattr(self, 'kite') and self.kite:
            try:
                instruments = pd.DataFrame(self.kite.instruments("NSE"))
                instruments.to_parquet("data/cache/instruments.parquet", compression='snappy')
            except Exception as e:
                logger.error(f"Failed to fetch instruments: {e}")
                return None

        if instruments is not None:
            match = instruments[
                (instruments['tradingsymbol'] == symbol) &
                (instruments['segment'] == 'NSE') &
                (instruments['instrument_type'] == 'EQ')
            ]
            if not match.empty:
                return int(match.iloc[0]['instrument_token'])
        return None

    def _symbol_to_yahoo(self, symbol: str) -> str:
        return f"{symbol}.NS"

    def _symbol_to_kite(self, symbol: str) -> str:
        return f"NSE:{symbol}"

    def _fetch_yfinance(self, symbol: str, period: str, interval: str):
        """Fallback to yfinance if Kite fails."""
        import yfinance as yf
        df = yf.download(f"{symbol}.NS", period="2y", interval="1d", progress=False, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.capitalize() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.columns = [c.capitalize() for c in df.columns]
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                return pd.DataFrame()
        return df[required]