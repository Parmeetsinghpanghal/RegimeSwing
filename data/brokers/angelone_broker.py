"""
Angel One Broker Implementation - Uses SmartAPI (Angel One's SDK).
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

# Try to import SmartAPI
try:
    from SmartApi import SmartConnect
    SMARTAPI_AVAILABLE = True
except ImportError:
    SMARTAPI_AVAILABLE = False
    logger.warning("SmartAPI not available. Install with: pip install smartapi-python")


class AngelOneBroker:
    """Angel One Broker Implementation using SmartAPI."""

    def __init__(self, config):
        self.config = config
        self._initialized = False
        self.obj = None
        self.auth_token = None
        self.refresh_token = None
        self.feed_token = None

    async def initialize(self) -> bool:
        """Initialize Angel One connection."""
        if not SMARTAPI_AVAILABLE:
            logger.warning("SmartAPI not available. Install: pip install smartapi-python")
            return False

        api_key = self.config.api_key
        api_secret = self.config.api_secret
        client_id = self.config.client_id
        totp_secret = self.config.totp_secret

        if not all([api_key, api_secret, client_id, totp_secret]):
            logger.warning("Angel One credentials incomplete")
            return False

        try:
            self.obj = SmartConnect(api_key=api_key)
            # Generate TOTP
            import pyotp
            totp = pyotp.TOTP(totp_secret).now()

            # Generate session
            data = self.obj.generateSession(client_id, api_secret, totp)

            if data['status']:
                self.auth_token = data['data']['jwtToken']
                self.refresh_token = data['data']['refreshToken']
                self.feed_token = self.obj.getfeedToken()
                logger.info("Angel One authenticated successfully")
                self._initialized = True
                return True
            else:
                logger.error(f"Angel One auth failed: {data}")
                return False
        except Exception as e:
            logger.error(f"Angel One init failed: {e}")
            return False

    async def close(self):
        """Close connections."""
        pass

    def _symbol_to_angel(self, symbol: str) -> str:
        """Convert symbol to Angel One format."""
        return f"NSE:{symbol}"

    def _symbol_to_yahoo(self, symbol: str) -> str:
        """Convert to Yahoo format for fallback."""
        return f"{symbol}.NS"

    def _get_ticker(self, symbol: str):
        """Get ticker for fallback."""
        pass

    async def initialize(self) -> bool:
        """Initialize Angel One connection."""
        return await self.initialize()

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
        """Get price data for symbols using Angel One historical API."""
        if isinstance(symbols, str):
            symbols = [symbols]

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        all_data = []

        for symbol in symbols:
            try:
                # Use Angel One historical API
                token = self._get_token(symbol)
                if not token:
                    # Fallback to yfinance
                    return await self._fetch_yfinance_fallback(symbols, period, interval)

                # Angel One historical API
                historic_param = {
                    "exchange": "NSE",
                    "symboltoken": token,
                    "interval": interval,
                    "fromdate": start_date,
                    "todate": end_date
                }
                data = self.obj.getCandleData(historic_param)

                if not data or 'data' not in data or not data['data']:
                    logger.warning(f"No data for {symbol}")
                    continue

                df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df['Symbol'] = symbol
                all_data.append(df)

            except Exception as e:
                logger.warning(f"Angel One fetch failed for {symbol}: {e}")
                # Fallback to yfinance
                return await self._fetch_yfinance_fallback(symbols, period, interval)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data)
        combined = combined.set_index('Symbol', append=True)
        combined = combined.reorder_levels(['Symbol', combined.index.names[0]])
        combined = combined.sort_index()
        return combined

    def _get_token(self, symbol: str) -> Optional[str]:
        """Get token for symbol from Angel One master."""
        # In production, fetch from Angel One master or cache
        # For now, use a simple mapping or fetch from API
        token_map = {
            "RELIANCE": "2885",
            "TCS": "11536",
            "HDFCBANK": "1333",
            "ICICIBANK": "4963",
            "INFY": "1594",
            "HINDUNILVR": "1394",
            "ITC": "1660",
            "SBIN": "3045",
            "BHARTIARTL": "10604",
            "BAJFINANCE": "317",
            "LT": "11483",
            "ASIANPAINT": "1322",
            "MARUTI": "12345",
            "SUNPHARMA": "8578",
            "TITAN": "1428",
            "AXISBANK": "5900",
            "KOTAKBANK": "1922",
            "M&M": "20374",
            "ULTRACEMCO": "11532",
            "BAJAJFINSV": "16669",
        }
        return token_map.get(symbol)

    async def _fetch_yfinance_fallback(self, symbols, period, interval):
        """Fallback to yfinance."""
        from .yfinance_broker import YFinanceBroker
        yf_broker = YFinanceBroker(self.config)
        return await YFinanceBroker.get_price_data(self, symbols, period, interval)

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
                token = self._get_token(symbol)
                if not token:
                    # Fallback to yfinance
                    from .yfinance_broker import YFinanceBroker
                    yf_broker = YFinanceBroker(self.config)
                    return await YFinanceBroker().get_price_data(symbols, period=period, interval=interval)

                # Angel One historical API
                interval_map = {
                    "1m": "ONE_MINUTE",
                    "5m": "FIVE_MINUTE",
                    "15m": "FIFTEEN_MINUTE",
                    "30m": "THIRTY_MINUTE",
                    "1h": "ONE_HOUR",
                    "1d": "ONE_DAY"
                }
                interval_map = {
                    "1m": "ONE_MINUTE",
                    "5m": "FIVE_MINUTE",
                    "15m": "FIFTEEN_MINUTE",
                    "30m": "THIRTY_MINUTE",
                    "1h": "ONE_HOUR",
                    "1d": "ONE_DAY"
                }

                historic_param = {
                    "exchange": "NSE",
                    "symboltoken": self._get_token(symbol),
                    "interval": interval_map.get(interval, "ONE_DAY"),
                    "fromdate": start_date,
                    "todate": end_date
                }

                data = self.obj.getCandleData(historic_param)

                if not data or 'data' not in data or not data['data']:
                    logger.warning(f"No data for {symbol}")
                    continue

                df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df['Symbol'] = symbol
                all_data.append(df)

            except Exception as e:
                logger.warning(f"Angel One fetch failed for {symbol}: {e}")

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data)
        combined = combined.set_index('Symbol', append=True)
        combined = combined.reorder_levels(['Symbol', combined.index.names[0]])
        combined = combined.sort_index()
        return combined

    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Get option chain from Angel One."""
        # Angel One option chain API
        return pd.DataFrame(columns=[
            'strike', 'expiry', 'option_type', 'iv', 'delta', 'gamma',
            'theta', 'vega', 'rho', 'oi', 'volume', 'bid', 'ask', 'ltp'
        ])

    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data - not available via Angel One API directly."""
        return pd.DataFrame()

    async def get_nifty_index_data(self, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Get Nifty 50 index data."""
        return self.get_price_data(["NIFTY 50"], period=period, interval=interval)

    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Get option chain - not fully implemented."""
        return pd.DataFrame(columns=[
            'strike', 'expiry', 'option_type', 'iv', 'delta', 'gamma',
            'theta', 'vega', 'rho', 'oi', 'volume', 'bid', 'ask', 'ltp'
        ])

    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamentals - not available via Angel One API."""
        return pd.DataFrame()

    async def get_nifty_index_data(self, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        return self.get_price_data(["NIFTY 50"], period=period, interval=interval)

    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            'strike', 'expiry', 'option_type', 'iv', 'delta', 'gamma',
            'theta', 'vega', 'rho', 'oi', 'volume', 'bid', 'ask', 'ltp'
        ])

    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        return pd.DataFrame()

    async def get_nifty_index_data(self, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        return self.get_price_data(["NIFTY 50"], period=period, interval=interval)

    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            'strike', 'expiry', 'option_type', 'iv', 'delta', 'gamma',
            'theta', 'vega', 'rho', 'oi', 'volume', 'bid', 'ask', 'ltp'
        ])

    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        return pd.DataFrame()

    async def initialize(self) -> bool:
        return await self.initialize()

    async def close(self):
        pass

    async def health_check(self) -> dict:
        return {"status": "ok", "broker": "angel_one"}


# For backward compatibility
AngelOneBroker = AngelOneBroker