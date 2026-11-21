"""
FII/DII Daily Flow & Institutional Data Module.
Fetches FII/DII net cash market flows, bulk/block deals from NSE.
Zero-cost using nsepython or direct NSE CSV endpoints.
"""

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import StringIO

logger = logging.getLogger(__name__)

# NSE URLs for institutional data
NSE_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
NSE_BULK_DEALS_URL = "https://www.nseindia.com/api/historical/bulkDeals"
NSE_BLOCK_DEALS_URL = "https://www.nseindia.com/api/historical/blockDeals"
NSE_BHAVCOPY_URL = "https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&from={from_date}&to={to_date}"

# NSE session headers
NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nseindia.com/',
}

# Cache directory
CACHE_DIR = Path("./data/cache/nse_institutional")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 4  # Refresh every 4 hours


class NSEInstitutionalData:
    """Fetches FII/DII flows, bulk/block deals from NSE."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self._init_session()

    def _init_session(self):
        """Initialize NSE session with cookies."""
        try:
            self.session.get("https://www.nseindia.com", timeout=10)
        except Exception as e:
            logger.warning(f"NSE session init failed: {e}")

    def _get_cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}_{date.today()}.json"

    def _is_cache_fresh(self, path: Path, max_age_hours: int = CACHE_TTL_HOURS) -> bool:
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.now() - mtime < timedelta(hours=max_age_hours)

    def _load_cache(self, path: Path) -> Optional[Dict]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, path: Path, data: Dict):
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Cache save failed for {path}: {e}")

    def _fetch_with_fallback(self, url: str, cache_name: str, static_fallback: any = None) -> any:
        """Fetch from NSE with cache and static fallback."""
        cache_path = self.cache_dir / f"{cache_name}_{date.today()}.json"

        # Try cache first
        if self._is_cache_fresh(cache_path):
            cached = self._load_cache(cache_path)
            if cached:
                return cached

        # Try NSE API
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                self._save_cache(cache_path, data)
                return data
        except Exception as e:
            logger.warning(f"NSE fetch failed for {cache_name}: {e}")

        return static_fallback

    def get_fii_dii_flow(self, days: int = 5) -> Dict:
        """
        Fetch FII + DII net cash market buying/selling for last N days.
        Returns net flows in crores.
        """
        cache_path = self.cache_dir / f"fii_dii_{date.today()}.json"

        if self._is_cache_fresh(cache_path):
            cached = self._load_cache(cache_path)
            if cached:
                return cached

        # Try NSE API
        try:
            resp = self.session.get(NSE_FII_DII_URL, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    recent = df.tail(days)
                    fii_net = recent['fii_net'].sum() if 'fii_net' in recent.columns else 0
                    dii_net = recent['dii_net'].sum() if 'dii_net' in recent.columns else 0

                    result = {
                        'fii_5d_net_cr': float(fii_net),
                        'dii_5d_net_cr': float(dii_net),
                        'combined_net_cr': float(fii_net + dii_net),
                        'signal': 'bullish' if (fii_net + dii_net) > 0 else 'bearish',
                        'details': recent.to_dict('records')
                    }
                    self._save_cache(cache_path, result)
                    return result
        except Exception as e:
            logger.warning(f"FII/DII fetch failed: {e}")

        # Static fallback
        return {
            'fii_5d_net_cr': 0,
            'dii_5d_net_cr': 0,
            'combined_net_cr': 0,
            'signal': 'neutral',
            'note': 'FII/DII data requires nsepython or NSE API'
        }

    def get_bulk_deals(self, days: int = 5) -> pd.DataFrame:
        """Fetch bulk deals from NSE."""
        cache_name = f"bulk_deals_{date.today()}"
        url = f"{NSE_BULK_DEALS_URL}?from={date.today() - timedelta(days=days)}&to={date.today()}"

        data = self._fetch_with_fallback(url, cache_name, [])
        if data:
            return pd.DataFrame(data)
        return pd.DataFrame()

    def get_block_deals(self, days: int = 5) -> pd.DataFrame:
        """Fetch block deals from NSE."""
        cache_name = f"block_deals_{date.today()}"
        url = f"{NSE_BLOCK_DEALS_URL}?from={date.today() - timedelta(days=days)}&to={date.today()}"

        data = self._fetch_with_fallback(url, cache_name, [])
        if data:
            return pd.DataFrame(data)
        return pd.DataFrame()

    def get_institutional_activity(self, symbol: str, days: int = 30) -> Dict:
        """Get institutional activity for a specific symbol."""
        bulk = self.get_bulk_deals(days)
        block = self.get_block_deals(days)

        symbol = symbol.upper().replace(".NS", "")

        bulk_activity = bulk[bulk['symbol'].str.upper() == symbol] if not bulk.empty else pd.DataFrame()
        block_activity = block[block['symbol'].str.upper() == symbol] if not block.empty else pd.DataFrame()

        return {
            'symbol': symbol,
            'bulk_deals': bulk_activity.to_dict('records') if not bulk_activity.empty else [],
            'block_deals': block_activity.to_dict('records') if not block_activity.empty else [],
            'total_bulk_qty': bulk_activity['quantity'].sum() if not bulk_activity.empty else 0,
            'total_block_qty': block_activity['quantity'].sum() if not block_activity.empty else 0
        }

    def get_fii_dii_signal(self) -> Dict:
        """Get FII/DII signal for regime integration."""
        flow = self.get_fii_dii_flow()
        net = flow.get('combined_net_cr', 0)

        if net > 500:
            return {'signal': 'bullish', 'strength': 'strong', 'net_cr': net}
        elif net > 100:
            return {'signal': 'bullish', 'strength': 'moderate', 'net_cr': net}
        elif net > -100:
            return {'signal': 'neutral', 'strength': 'neutral', 'net_cr': net}
        elif net > -500:
            return {'signal': 'bearish', 'strength': 'moderate', 'net_cr': net}
        else:
            return {'signal': 'bearish', 'strength': 'strong', 'net_cr': net}


# Convenience functions
_nse_inst = None

def get_nse_institutional() -> NSEInstitutionalData:
    """Get singleton NSEInstitutionalData instance."""
    global _nse_inst
    if _nse_inst is None:
        _nse_inst = NSEInstitutionalData()
    return _nse_inst


def get_fii_dii_flow(days: int = 5) -> Dict:
    """Get FII/DII flow data."""
    return get_nse_institutional().get_fii_dii_flow(days)


def get_fii_dii_signal() -> Dict:
    """Get FII/DII signal for regime."""
    return get_nse_institutional().get_fii_dii_signal()


def get_bulk_deals(days: int = 5) -> pd.DataFrame:
    """Get bulk deals."""
    return get_nse_institutional().get_bulk_deals(days)


def get_block_deals(days: int = 5) -> pd.DataFrame:
    """Get block deals."""
    return get_nse_institutional().get_block_deals(days)


def get_institutional_activity(symbol: str, days: int = 30) -> Dict:
    """Get institutional activity for symbol."""
    return get_nse_institutional().get_institutional_activity(symbol, days)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    inst = NSEInstitutionalData()

    # Test FII/DII flow
    flow = inst.get_fii_dii_flow(5)
    print(f"FII/DII Flow: {flow}")

    # Test signal
    signal = inst.get_fii_dii_signal()
    print(f"FII/DII Signal: {signal}")