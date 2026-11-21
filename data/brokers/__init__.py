"""
Brokers Package - Multi-broker support for the trading system.
Supports: yfinance (free), Zerodha (Kite), Angel One, Groww, Dhan
"""

from .base import (
    BrokerBase,
    BrokerConfig,
    BrokerType,
    BrokerFactory,
    PriceData,
    OptionData,
    OptionChainData,
    FundamentalsData,
)

from .yfinance_broker import YFinanceBroker
from .zerodha_broker import ZerodhaBroker
from .angelone_broker import AngelOneBroker
from .dhan_broker import DhanBroker
from .groww_broker import GrowwBroker

# Register all brokers with the factory
BrokerFactory.register(BrokerType.YFINANCE, YFinanceBroker)
BrokerFactory.register(BrokerType.ZERODHA, ZerodhaBroker)
BrokerFactory.register(BrokerType.ANGEL_ONE, AngelOneBroker)
BrokerFactory.register(BrokerType.DHAN, DhanBroker)
BrokerFactory.register(BrokerType.GROWW, GrowwBroker)

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

    # Broker implementations
    'YFinanceBroker',
    'ZerodhaBroker',
    'AngelOneBroker',
    'DhanBroker',
    'GrowwBroker',
]