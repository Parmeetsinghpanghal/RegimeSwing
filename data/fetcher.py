"""
Data Fetcher Module - Unified interface for price, options, and fundamental data.
Uses Broker Abstraction Layer with new OHLCVCache and Universe modules.
"""

import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import asyncio

# Import broker abstraction
from .brokers import BrokerFactory, BrokerConfig, BrokerType

# Import new cache and universe modules
from .cache import OHLCVCache, get_cache
from .universe import get_nifty200_symbols, get_nifty50_symbols, get_nifty100_symbols, get_universe

logger = logging.getLogger(__name__)

# Check yfinance availability
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None


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
    """Unified data fetcher with Broker Abstraction Layer + OHLCVCache.
    Supports Kite Connect (primary) + yfinance (fallback) + other brokers.
    """

    def __init__(self, config_path: str = "config.yaml", broker_type: str = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_config = self.config.get('data', {})
        self.costs_config = self.config.get('costs', {})
        self.broker_config = self.config.get('broker', {})

        # Initialize OHLCV cache for yfinance (fast, batched, cached)
        cache_ttl = self.data_config.get('cache_ttl_hours', 24)
        self.cache = OHLCVCache(default_ttl_hours=cache_ttl)

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
        try:
            broker_type_enum = BrokerType(broker_type_str)
        except ValueError:
            logger.warning(f"Unknown broker type {broker_type_str}, defaulting to yfinance")
            broker_type_str = 'yfinance'
            broker_type_enum = BrokerType.YFINANCE

        broker_config_obj = BrokerConfig(
            broker_type=broker_type_enum,
            api_key=self.broker_config.get('api_key', ''),
            api_secret=self.broker_config.get('api_secret', ''),
            access_token=self.broker_config.get('access_token', ''),
            client_id=self.broker_config.get('client_id', ''),
            totp_secret=self.broker_config.get('totp_secret', ''),
            redirect_url=self.broker_config.get('redirect_url', ''),
            rate_limit=self.broker_config.get('rate_limit', 3),
            cache_ttl_hours=cache_ttl,
            fallback_enabled=self.data_config.get('fallback_enabled', True),
        )

        # Create broker
        self.broker = BrokerFactory.create(broker_config_obj)

        # Initialize broker synchronously
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.broker.initialize())
            loop.close()
        except Exception as e:
            logger.warning(f"Broker initialization failed: {e}")

        # Universe symbols - use new dynamic fetcher
        self.nifty200_symbols = get_nifty200_symbols()
        self.nifty50_symbols = get_nifty50_symbols()
        self.nifty100_symbols = get_nifty100_symbols()

        # Corporate actions cache
        self.corporate_actions = {}

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

        Uses OHLCVCache for yfinance (fast, batched, cached).
        Falls back to broker for other sources.

        Returns MultiIndex DataFrame: (Symbol, Date) with OHLCV columns.
        """
        # Use cache for yfinance broker
        if self.broker.config.broker_type == BrokerType.YFINANCE:
            return self.cache.get_price_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                period=period,
                interval=interval,
                use_cache=use_cache
            )

        # For other brokers, use broker directly
        try:
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
            loop.close()
            return df
        except Exception as e:
            logger.error(f"Broker get_price_data failed: {e}")
            return pd.DataFrame()

    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Get option chain for symbol."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            df = loop.run_until_complete(self.broker.get_option_chain(symbol, expiry))
            loop.close()
            return df
        except Exception as e:
            logger.error(f"Broker get_option_chain failed: {e}")
            return pd.DataFrame()

    def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            df = loop.run_until_complete(self.broker.get_fundamentals(symbols))
            loop.close()
            return df
        except Exception as e:
            logger.error(f"Broker get_fundamentals failed: {e}")
            return pd.DataFrame()

    def get_nifty_index_data(self, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Get Nifty 50 index data for regime detection."""
        # Use cache for yfinance (synchronous now)
        if self.broker.config.broker_type == BrokerType.YFINANCE:
            return self.cache.get_nifty_index_data(period, interval)

        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            df = loop.run_until_complete(self.broker.get_nifty_index_data(period, interval))
            loop.close()
            return df
        except Exception as e:
            logger.error(f"Broker get_nifty_index_data failed: {e}")
            return pd.DataFrame()

    def health_check(self) -> Dict:
        """Check data pipeline health."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.broker.health_check())
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Broker health_check failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_universe(self, universe: str = "nifty200") -> List[str]:
        """Get symbols for a universe."""
        return get_universe(universe)

    def refresh_universe(self, force: bool = False):
        """Refresh universe symbols from NSE."""
        if force or not self.nifty200_symbols:
            from .universe import fetch_nifty200_constituents
            self.nifty200_symbols = fetch_nifty200_constituents(force_refresh=force)
            self.nifty50_symbols = self.nifty200_symbols[:50]
            self.nifty100_symbols = self.nifty200_symbols[:100]


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

    # Test universe
    print(f"Nifty 200: {len(fetcher.nifty200_symbols)} symbols")
    print(f"Nifty 50: {len(fetcher.nifty50_symbols)} symbols")
    print(f"Nifty 100: {len(fetcher.nifty100_symbols)} symbols")

    # Fetch test data
    df = fetcher.get_price_data(["RELIANCE", "TCS"], period="3mo", interval="1d")
    print(f"\nFetched {len(df)} rows for {df.index.get_level_values('Symbol').nunique()} symbols")
    print(df.head(10))

    # Test fundamentals
    fund = fetcher.get_fundamentals(["RELIANCE", "TCS"])
    print(f"\nFundamentals:\n{fund}")

    # Cache stats
    print(f"\nCache stats: {fetcher.cache.get_cache_stats()}")