"""
Broker Base Module - Abstract base class and factory for broker implementations.
Supports: yfinance (free), Zerodha (Kite), Angel One, Groww, Dhan
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Supported broker types."""
    YFINANCE = "yfinance"
    ZERODHA = "zerodha"
    ANGEL_ONE = "angel_one"
    GROWW = "groww"
    DHAN = "dhan"


@dataclass
class BrokerConfig:
    """Broker configuration."""
    broker_type: BrokerType
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    client_id: str = ""
    totp_secret: str = ""
    redirect_url: str = ""
    # Rate limiting
    rate_limit: int = 3  # requests per second
    # Cache settings
    cache_ttl_hours: int = 24
    # Fallback
    fallback_enabled: bool = True


@dataclass
class PriceData:
    """Normalized price data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    oi: Optional[float] = None  # Open interest for options


@dataclass
class OptionData:
    """Option chain data."""
    symbol: str
    strike: float
    expiry: datetime
    option_type: str  # 'CE' or 'PE'
    action: str  # 'BUY' or 'SELL'
    premium: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0
    oi: int = 0
    volume: int = 0
    bid: float = 0.0
    ask: float = 0.0


@dataclass
class OptionChainData:
    """Option chain container."""
    symbol: str
    expiry: datetime
    options: List[dict]


@dataclass
class FundamentalsData:
    """Fundamental data container."""
    symbol: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    roic: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    profit_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    free_cash_flow: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None


class BrokerBase(ABC):
    """Abstract base class for all brokers."""

    def __init__(self, config: BrokerConfig):
        self.config = config
        self._initialized = False
        self._rate_limiter = None

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize broker connection."""
        pass

    @abstractmethod
    async def close(self):
        """Close connections."""
        pass

    @abstractmethod
    async def get_price_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get price data for symbols. Returns MultiIndex DataFrame (date, symbol) with OHLCV."""
        pass

    @abstractmethod
    async def get_option_chain(
        self,
        symbol: str,
        expiry: Optional[str] = None
    ) -> pd.DataFrame:
        """Get option chain for symbol."""
        pass

    @abstractmethod
    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data."""
        pass

    @abstractmethod
    async def get_nifty_index_data(self, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Get Nifty 50 index data."""
        pass

    @abstractmethod
    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Get option chain for symbol."""
        pass

    @abstractmethod
    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data."""
        pass

    @abstractmethod
    async def get_nifty_index_data(self, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Get Nifty 50 index data."""
        pass

    @abstractmethod
    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Get option chain for symbol."""
        pass

    @abstractmethod
    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data."""
        pass

    @abstractmethod
    async def get_nifty_index_data(self, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Get Nifty 50 index data."""
        pass

    @abstractmethod
    async def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Get option chain for symbol."""
        pass

    @abstractmethod
    async def get_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Get fundamental data."""
        pass

    @abstractmethod
    async def place_order(self, order_params: dict) -> dict:
        """Place an order."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[dict]:
        """Get current positions."""
        pass

    @abstractmethod
    async def get_orders(self) -> List[dict]:
        """Get order history."""
        pass

    async def health_check(self) -> Dict:
        """Health check."""
        return {"status": "ok", "broker": self.config.broker_type.value}

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to standard format."""
        return symbol.upper().replace("-", "").replace(".NS", "").replace(".BO", "")

    def _to_nse_symbol(self, symbol: str) -> str:
        """Convert to NSE format for yfinance."""
        return f"{self._normalize_symbol(symbol)}.NS"

    def _to_bse_symbol(self, symbol: str) -> str:
        """Convert to BSE format."""
        return f"{self._normalize_symbol(symbol)}.BO"


class BrokerFactory:
    """Factory for creating broker instances."""

    _brokers = {}
    _instance = None

    @classmethod
    def register(cls, broker_type: BrokerType, broker_class):
        """Register a broker implementation."""
        cls._brokers[broker_type] = broker_class

    @classmethod
    def create(cls, config: BrokerConfig) -> BrokerBase:
        """Create broker instance."""
        if config.broker_type not in cls._brokers:
            raise ValueError(f"Broker type {config.broker_type} not supported")

        broker_class = cls._brokers[config.broker_type]
        return broker_class(config)

    @classmethod
    def create_from_env(cls) -> BrokerBase:
        """Create broker instance from environment variables."""
        import os

        broker_type = os.getenv("BROKER_TYPE", "yfinance").lower()

        type_map = {
            "yfinance": BrokerType.YFINANCE,
            "zerodha": BrokerType.ZERODHA,
            "angel_one": BrokerType.ANGEL_ONE,
            "groww": BrokerType.GROWW,
            "dhan": BrokerType.DHAN,
        }

        broker_type = type_map.get(broker_type, BrokerType.YFINANCE)

        config = BrokerConfig(
            broker_type=type_map.get(broker_type, BrokerType.YFINANCE),
            api_key=os.getenv("API_KEY", ""),
            api_secret=os.getenv("API_SECRET", ""),
            access_token=os.getenv("ACCESS_TOKEN", ""),
            client_id=os.getenv("CLIENT_ID", ""),
            totp_secret=os.getenv("TOTP_SECRET", ""),
            redirect_url=os.getenv("REDIRECT_URL", ""),
        )

        return cls.create(BrokerConfig(broker_type=broker_type))

    @classmethod
    def get_available_brokers(cls) -> List[BrokerType]:
        """Get list of registered brokers."""
        return list(cls._brokers.keys())


# Initialize with base broker (yfinance)
from .yfinance_broker import YFinanceBroker
BrokerFactory.register(BrokerType.YFINANCE, YFinanceBroker)

__all__ = [
    # Base classes
    'BrokerBase',
    'BrokerConfig',
    'BrokerType',
    'BrokerFactory',
    'PriceData',
    'OptionData',
    'OptionChainData',
    'FundamentalsData',
    'OptionChainData',
    'BrokerType',
    'BrokerConfig',
    'BrokerFactory',
]