"""
Data Fetcher Module - Unified interface for price, options, and fundamental data.
Uses Broker Abstraction Layer to support multiple brokers.
"""

import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import asyncio

# Import broker abstraction
from .brokers import BrokerFactory, BrokerConfig, BrokerType

logger = logging.getLogger(__name__)


class PriceData:
    """Container for price data."""
    def __init__(self, symbol: str, open: float, high: float, low: float, close: float, volume: float, timestamp: datetime):
        self.symbol = symbol
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.timestamp = timestamp

    def __repr__(self):
        return f"PriceData({self.symbol}, {self.close})"


class OptionData:
    """Container for option chain data."""
    def __init__(self, symbol: str, strike: float, expiry: datetime, option_type: str,
                 premium: float, delta: float = 0.0, gamma: float = 0.0,
                 theta: float = 0.0, vega: float = 0.0, iv: float = 0.0,
                 oi: int = 0, volume: int = 0, bid: float = 0.0, ask: float = 0.0):
        self.symbol = symbol
        self.strike = strike
        self.expiry = expiry
        self.option_type = option_type  # 'CE' or 'PE'
        self.premium = premium
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        self.iv = iv
        self.oi = oi
        self.volume = volume
        self.bid = bid
        self.ask = ask


class OptionChainData:
    """Container for full option chain data."""
    def __init__(self, symbol: str, expiry: datetime, options: List[dict]):
        self.symbol = symbol
        self.expiry = expiry
        self.options = options

    def get_strikes(self) -> List[float]:
        return sorted(set(opt.strike for opt in self.options))


class FundamentalsData:
    """Container for fundamental data."""
    def __init__(self, symbol: str, market_cap: float = None, pe_ratio: float = None,
                 pb_ratio: float = None, roe: float = None, roic: float = None,
                 debt_to_equity: float = None, current_ratio: float = None,
                 profit_margin: float = None, revenue_growth: float = None,
                 earnings_growth: float = None, free_cash_flow: float = None,
                 dividend_yield: float = None, beta: float = None,
                 sector: str = None, industry: str = None):
        self.symbol = symbol
        self.market_cap = market_cap
        self.pe_ratio = pe_ratio
        self.pb_ratio = pb_ratio
        self.roe = roe
        self.roic = roic
        self.debt_to_equity = debt_to_equity
        self.current_ratio = current_ratio
        self.profit_margin = profit_margin
        self.revenue_growth = revenue_growth
        self.earnings_growth = earnings_growth
        self.free_cash_flow = free_cash_flow
        self.dividend_yield = dividend_yield
        self.beta = beta
        self.sector = sector
        self.industry = industry


class DataFetcher:
    """Unified data fetcher with Broker Abstraction Layer.
    Supports Kite Connect (primary) + yfinance (fallback) + other brokers.
    """

    def __init__(self, config_path: str = "config.yaml", broker_type: str = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_config = self.config.get('data', {})
        self.costs_config = self.config.get('costs', {})
        self.broker_config = self.config.get('broker', {})
        self.cache_dir = Path(self.data_config.get('cache_dir', './data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Determine broker type
        broker_type_str = self.broker_config.get('type', 'yfinance').lower()
        broker_type_map = {
            'yfinance': 'yfinance',
            'zerodha': 'zerodha',
            'angel_one': 'angel_one',
            'groww': 'groww',
            'dhan': 'dhan',
        }
        broker_type_str = broker_type_map.get(broker_type_str, 'yfinance')

        # Create broker config
        broker_config = {
            'api_key': self.broker_config.get('api_key', ''),
            'api_secret': self.broker_config.get('api_secret', ''),
            'access_token': self.broker_config.get('access_token', ''),
            'client_id': self.broker_config.get('client_id', ''),
            'totp_secret': self.broker_config.get('totp_secret', ''),
            'redirect_url': self.broker_config.get('redirect_url', ''),
        }

        # Create broker
        from .brokers import BrokerFactory, BrokerConfig, BrokerType

        broker_type_str = self.broker_config.get('type', 'yfinance').lower()
        broker_type_map = {
            'yfinance': 'yfinance',
            'zerodha': 'zerodha',
            'angel_one': 'angel_one',
            'groww': 'groww',
            'dhan': 'dhan',
        }
        broker_type_str = broker_type_map.get(broker_type_str, 'yfinance')

        broker_config = {
            'api_key': self.broker_config.get('api_key', ''),
            'api_secret': self.broker_config.get('api_secret', ''),
            'access_token': self.broker_config.get('access_token', ''),
            'client_id': self.broker_config.get('client_id', ''),
            'totp_secret': self.broker_config.get('totp_secret', ''),
            'redirect_url': self.broker_config.get('redirect_url', ''),
        }

        # Create broker
        try:
            broker_type_enum = BrokerType(broker_type_str)
        except ValueError:
            logger.warning(f"Unknown broker type {broker_type_str}, defaulting to yfinance")
            broker_type_str = 'yfinance'

        broker_config_obj = BrokerConfig(
            broker_type=BrokerType(broker_type_str),
            api_key=self.broker_config.get('api_key', ''),
            api_secret=self.broker_config.get('api_secret', ''),
            access_token=self.broker_config.get('access_token', ''),
            client_id=self.broker_config.get('client_id', ''),
            totp_secret=self.broker_config.get('totp_secret', ''),
            redirect_url=self.broker_config.get('redirect_url', ''),
            rate_limit=self.broker_config.get('rate_limit', 3),
            cache_ttl_hours=self.data_config.get('cache_ttl_hours', 24),
            fallback_enabled=self.data_config.get('fallback_enabled', True),
        )

        # Create broker
        self.broker = BrokerFactory.create(BrokerConfig(
            broker_type=BrokerType(broker_type_str),
            api_key=self.broker_config.get('api_key', ''),
            api_secret=self.broker_config.get('api_secret', ''),
            access_token=self.broker_config.get('access_token', ''),
            client_id=self.broker_config.get('client_id', ''),
            totp_secret=self.broker_config.get('totp_secret', ''),
            redirect_url=self.broker_config.get('redirect_url', ''),
        ))

        # Initialize broker
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.broker.initialize())

        # Nifty 200 symbols (subset - extend as needed)
        self.nifty200_symbols = self._get_nifty200_symbols()

        # Corporate actions cache
        self.corporate_actions = {}

    def _get_nifty200_symbols(self) -> List[str]:
        """Get Nifty 200 symbols. In production, fetch from NSE."""
        # Core Nifty 50 + Next 50 + midcaps - extend for full 200
        return [
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
            "SBIN", "BHARTIARTL", "BAJFINANCE", "LT", "ASIANPAINT", "MARUTI", "SUNPHARMA",
            "TITAN", "AXISBANK", "KOTAKBANK", "M&M", "ULTRACEMCO", "BAJAJFINSV",
            "NESTLEIND", "POWERGRID", "NTPC", "TECHM", "WIPRO", "ADANIENT", "ADANIPORTS",
            "COALINDIA", "ONGC", "TATASTEEL", "JSWSTEEL", "HINDALCO", "INDUSINDBK",
            "CIPLA", "DRREDDY", "APOLLOHOSP", "DIVISLAB", "BPCL", "IOC", "GRASIM",
            "BRITANNIA", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "TATAMOTORS",
            "TATACONSUM", "UPL", "SHREECEM", "SBILIFE", "HDFCLIFE", "ICICIPRULI",
            "PIDILITIND", "BERGEPAINT", "HAVELLS", "DABUR", "MARICO", "GODREJCP",
            "COLPAL", "UBL", "MCDOWELL-N", "VBL", "RADICO", "SOMANYCERA",
            "KANSAINER", "NOCIL", "NAVINFLUOR", "SRF", "ATUL", "GUJALKALI",
            "DEEPAKNTR", "AARTIIND", "VINATIORGA", "LAURUSLABS", "GRANULES",
            "NATCOPHARM", "ALKEM", "TORNTPHARM", "IPCA", "ABBOTINDIA", "PFIZER",
            "GLAXO", "SANOFI", "PFIZER", "LUPIN", "AUROPHARMA", "BIOCON",
            "CADILAHC", "GLENMARK", "TORNTPHARM", "STRIDES", "CAPLIN", "SUVEN",
        ]

    def _symbol_to_yahoo(self, symbol: str) -> str:
        """Convert NSE symbol to Yahoo Finance format."""
        return f"{symbol}.NS"

    def _symbol_to_kite(self, symbol: str) -> str:
        """Convert to Kite format (NSE:SYMBOL)."""
        return f"NSE:{symbol}"

    def _get_cache_path(self, symbol: str, timeframe: str, data_type: str = "price") -> Path:
        """Generate cache file path."""
        safe_symbol = symbol.replace("&", "").replace("-", "_")
        return self.cache_dir / f"{safe_symbol}_{timeframe}_{data_type}.parquet"

    def _load_from_cache(self, cache_path: Path, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """Load data from cache if fresh enough."""
        if not cache_path.exists():
            return None

        # Check age
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=max_age_hours):
            return None

        try:
            df = pd.read_parquet(cache_path)
            logger.debug(f"Loaded {len(df)} rows from cache: {cache_path.name}")
            return df
        except Exception as e:
            logger.warning(f"Cache read failed for {cache_path}: {e}")
            return None

    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path):
        """Save data to cache."""
        try:
            df.to_parquet(cache_path, compression='snappy')
            logger.debug(f"Saved {len(df)} rows to cache: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Cache write failed for {cache_path}: {e}")

    async def _fetch_kite_historical(self, symbol: str, interval: str, from_date: datetime, to_date: datetime) -> pd.DataFrame:
        """Fetch historical data from Kite Connect."""
        # This would be implemented for Zerodha broker
        # For now, fallback to yfinance
        return await self._fetch_yfinance(symbol, "2y", "1d")

    def _get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for symbol. Cache instruments."""
        cache_path = self.cache_dir / "instruments.parquet"

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
                instruments.to_parquet(cache_path, compression='snappy')
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

    async def _fetch_yfinance(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance with robust error handling."""
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance not available")

        yf_symbol = f"{symbol}.NS"
        max_retries = 3

        for attempt in range(max_retries):
            try:
                df = yf.download(
                    yf_symbol,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    threads=False  # Disable threading to avoid issues
                )

                if df.empty:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return pd.DataFrame()

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

                return df[required]

            except Exception as e:
                logger.warning(f"yfinance fetch attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                logger.error(f"All yfinance attempts failed for {symbol}: {e}")
                return pd.DataFrame()

        return pd.DataFrame()

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
        Get price data for one or more symbols.

        Returns MultiIndex DataFrame: (date, symbol) with OHLCV columns.
        """
        # Use broker to fetch data
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        df = loop.run_until_complete(self.broker.get_price_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            period=period,
            interval=interval,
            use_cache=use_cache
        ))
        return df

    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Get option chain for symbol."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.broker.get_option_chain(symbol, expiry))

    def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.broker.get_fundamentals(symbols))

    def get_nifty_index_data(self, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Get Nifty 50 index data for regime detection."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.broker.get_nifty_index_data(period, interval))

    def health_check(self) -> Dict:
        """Check data pipeline health."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.broker.health_check())


# Convenience functions
def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_fetcher(config_path: str = "config.yaml", broker_type: str = None) -> DataFetcher:
    """Factory function to create DataFetcher."""
    return DataFetcher(config_path, broker_type)


if __name__ == "__main__":
    # Test the fetcher
    logging.basicConfig(level=logging.INFO)

    fetcher = create_fetcher()

    # Health check
    health = fetcher.health_check()
    print("Health Check:", health)

    # Fetch test data
    df = fetcher.get_price_data(["RELIANCE", "TCS"], period="3mo", interval="1d")
    print(f"\nFetched {len(df)} rows for {df.index.get_level_values('Symbol').nunique()} symbols")
    print(df.head(10))

    # Test fundamentals
    fund = fetcher.get_fundamentals(["RELIANCE", "TCS"])
    print(f"\nFundamentals:\n{fund}")