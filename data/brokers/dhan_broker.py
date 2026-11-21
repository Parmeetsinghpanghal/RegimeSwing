"""
Dhan Broker Implementation - Uses DhanHQ API.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests

from .base import BrokerBase, BrokerConfig, BrokerType, PriceData, OptionData, OptionChainData, FundamentalsData

logger = logging.getLogger(__name__)


class DhanBroker:
    """Dhan Broker Implementation using DhanHQ API."""

    def __init__(self, config):
        self.config = config
        self._initialized = False
        self.client_id = config.client_id
        self.access_token = config.access_token
        self.base_url = "https://api.dhan.co"
        self.session = None

    async def initialize(self) -> bool:
        """Initialize Dhan connection."""
        self.access_token = self.config.access_token
        self.client_id = self.config.client_id

        if not self.access_token or not self.config.client_id:
            logger.warning("Dhan credentials not provided")
            return False

        # Test connection
        try:
            headers = {
                "access-token": self.access_token,
                "Content-Type": "application/json"
            }
            response = requests.get(f"{self.base_url}/v2/profile", headers={"access-token": self.access_token})
            if response.status_code == 200:
                self._initialized = True
                logger.info("Dhan connected successfully")
                return True
            else:
                logger.error(f"Dhan auth failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Dhan init failed: {e}")
            return False

    async def close(self):
        pass

    async def initialize(self) -> bool:
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
        """Get price data using Dhan historical API."""
        if isinstance(symbols, str):
            symbols = [symbols]

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        all_data = []

        for symbol in symbols:
            try:
                # Dhan historical API
                url = f"{self.base_url}/v2/charts/historical"
                headers = {
                    "access-token": self.access_token,
                    "Content-Type": "application/json"
                }
                payload = {
                    "securityId": self._get_security_id(symbol),
                    "exchangeSegment": "NSE_EQ",
                    "interval": interval,
                    "fromDate": start_date,
                    "toDate": end_date
                }

                response = requests.post(
                    f"{self.base_url}/v2/charts/intraday",
                    headers={"access-token": self.access_token, "Content-Type": "application/json"},
                    json=payload
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success' and data.get('data'):
                        df = pd.DataFrame(data['data'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        df = df.set_index('timestamp')
                        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                        df['Symbol'] = symbol
                        return df

            except Exception as e:
                logger.warning(f"Dhan fetch failed for {symbol}: {e}")

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
        return {"status": "ok", "broker": "dhan"}


# For backward compatibility
DhanBroker = DhanBroker