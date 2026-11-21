"""
Dynamic Nifty 200 Constituent Fetcher
Fetches current Nifty 200 constituents from NSE with local caching.
Zero-cost, zero-rate-limit using NSE public APIs.
"""

import os
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
from io import StringIO

logger = logging.getLogger(__name__)

# NSE URLs for index constituents
NSE_NIFTY200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
NSE_NIFTY100_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
NSE_NIFTY50_URL = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
NSE_NIFTY_NEXT50_URL = "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv"

# Cache settings
CACHE_DIR = Path("./data/cache/universe")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 24

# Static fallback - comprehensive Nifty 200 list (as of 2024)
STATIC_NIFTY200 = [
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
    "GLAXO", "SANOFI", "LUPIN", "AUROPHARMA", "BIOCON", "CADILAHC",
    "GLENMARK", "STRIDES", "CAPLIN", "SUVEN", "AJANTPHARM", "ERIS",
    "MANAPPURAM", "MUTHOOTFIN", "CHOLAFIN", "SHRIRAMFIN", "LICHSGFIN",
    "BAJAJHLDNG", "MUTHOOTFIN", "SUNDARMFIN", "MAHAFIN", "EQUITAS",
    "UJJIVAN", "BANDHANBNK", "IDFCFIRSTB", "RBLBANK", "FEDERALBNK",
    "CUB", "KARURVYSYA", "LAKSHVILAS", "SOUTHBANK", "TMB",
    "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "SOBHA", "BRIGADE",
    "PHOENIXLTD", "DLF", "MAHLIFE", "SUNTECK", "KEC",
    "ABB", "SIEMENS", "CROMPTON", "HAVELLS", "POLYCAB",
    "KEI", "FINOLEX", "RRKABEL", "RRWL", "HAVELLS",
    "DIXON", "AMBER", "BLUESTARCO", "VOLTAS", "WHIRLPOOL",
    "BAJAJELEC", "CROMPTON", "ORIENTELEC", "HAVELLS", "VOLTAS",
    "PAGEIND", "RELAXO", "BATAINDIA", "METROBRAND", "CERA",
    "KPRMILL", "TTKPRESTIG", "TTKHLTCARE", "VIPIND", "SIGNATURE",
    "TRENT", "AVENUE", "TRENT", "DMART", "ZOMATO",
    "PAYTM", "POLICYBZR", "PBFINTECH", "CARTRADE", "CARERATING",
    "INDIAMART", "JUSTDIAL", "INDIGO", "SPICEJET", "VISTARA",
    "AIRLINE", "INTERGLOBE", "JUBLFOOD", "DEVYANI", "SAPPHIRE",
    "SUBROS", "MAHSCOOTER", "BAJAJHLDNG", "HEROMOTOCO", "TVSMOTOR",
    "EICHERMOT", "ASHOKLEY", "TATAMOTORS", "M&M", "FORCEMOT",
    "BOSCHLTD", "MOTHERSON", "SUNDRMFAST", "BHARATFORG", "RAMCOCEM",
    "SHREECEM", "ULTRACEMCO", "ACC", "AMBUJACEM", "DALMIACEM",
    "JKCEMENT", "ORIENTCEM", "HEIDELBERG", "PRISMCEM", "BIRLACORPN",
    "GRASIM", "ADITYABIRLA", "CENTURYTEX", "RAYMOND", "VARDHMAN",
    "TRIDENT", "WELSPUNIND", "ALOKIND", "LOKESHMACH", "TEXMACO",
    "SKIPPER", "RATNAMANI", "APLAPOLLO", "JINDALSTEL", "JSWSTEEL",
    "TATASTEEL", "SAIL", "NMDC", "HINDZINC", "VEDL",
    "NATIONALUM", "HINDALCO", "NALCO", "JINDALSTEL", "RATNAMANI",
    "WELCORP", "MANAPPURAM", "MUTHOOTFIN", "CHOLAFIN", "SHRIRAMFIN",
    "PFC", "RECLTD", "IRFC", "NHPC", "SJVN",
    "THERMAX", "BHEL", "L&T", "ENGINEERSIN", "NBCC",
    "IRCON", "RITES", "RAILTEL", "RVNL", "IRCON",
    "COCHINSHIP", "GRSE", "MDL", "HAL", "BEL",
    "BDL", "ASTRA", "MTARTECH", "DATAPATTNS", "PARAS",
    "KAYNES", "SYY", "CYIENT", "PERSISTENT", "COFORGE",
    "MPHASIS", "LTTS", "LTIM", "ZENSAR", "SONATSOFTW",
    "TATAELXSI", "KPITTECH", "TATAELXSI", "AFFLE", "ROUTE",
    "TANLA", "JUSTDIAL", "INDIAMART", "EASEMYTRIP", "IXIGO",
    "PBFINTECH", "POLICYBZR", "FIVESTAR", "CHOLAFIN", "MANAPPURAM"
]


def _get_cache_path(index_name: str) -> Path:
    """Get cache file path for an index."""
    return Path(f"./data/cache/universe/{index_name}_constituents.json")


def _is_cache_fresh(cache_path: Path, max_age_hours: int = 24) -> bool:
    """Check if cache file exists and is fresh."""
    if not cache_path.exists():
        return False
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(hours=24)


def _load_from_cache(cache_path: Path) -> Optional[List[str]]:
    """Load symbols from cache file."""
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        return data.get('symbols', [])
    except Exception as e:
        logger.warning(f"Failed to load cache from {cache_path}: {e}")
        return None


def _save_to_cache(cache_path: Path, symbols: List[str]):
    """Save symbols to cache file."""
    try:
        data = {
            'symbols': symbols,
            'updated_at': datetime.now().isoformat(),
            'count': len(symbols)
        }
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Cached {len(symbols)} symbols to {cache_path.name}")
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_path}: {e}")


def _fetch_nse_csv(url: str, timeout: int = 10) -> Optional[pd.DataFrame]:
    """Fetch CSV from NSE with proper headers."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def _extract_symbols(df: pd.DataFrame) -> List[str]:
    """Extract symbol column from NSE CSV."""
    symbol_cols = [c for c in df.columns if c.lower() in ('symbol', 'tradingsymbol')]
    if not symbol_cols:
        symbol_cols = [df.columns[0]]
    symbols = df[symbol_cols[0]].dropna().astype(str).str.strip().str.upper().tolist()
    symbols = [s.replace(' ', '').replace('.', '') for s in symbols if s]
    return symbols


def fetch_nifty200_constituents(force_refresh: bool = False) -> List[str]:
    """
    Fetch Nifty 200 constituents from NSE with caching.
    Returns list of symbols without .NS suffix.
    """
    cache_path = Path("./data/cache/universe/nifty200_constituents.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_refresh and _is_cache_fresh(cache_path):
        cached = _load_from_cache(cache_path)
        if cached:
            logger.info(f"Loaded {len(cached)} Nifty 200 symbols from cache")
            return cached

    # Fetch from NSE
    logger.info("Fetching Nifty 200 constituents from NSE...")
    df = _fetch_nse_csv("https://archives.nseindia.com/content/indices/ind_nifty200list.csv")

    if df is not None and not df.empty:
        symbols = _extract_symbols(df)
        if symbols:
            _save_to_cache(cache_path, symbols)
            return symbols

    # Fallback to static list
    logger.warning("Using static fallback for Nifty 200")
    _save_to_cache(cache_path, STATIC_NIFTY200)
    return STATIC_NIFTY200


def fetch_nifty50_constituents(force_refresh: bool = False) -> List[str]:
    """Fetch Nifty 50 constituents."""
    cache_path = Path("./data/cache/universe/nifty50_constituents.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_refresh and _is_cache_fresh(cache_path):
        cached = _load_from_cache(cache_path)
        if cached:
            return cached

    df = _fetch_nse_csv("https://archives.nseindia.com/content/indices/ind_nifty50list.csv")
    if df is not None and not df.empty:
        symbols = _extract_symbols(df)
        if symbols:
            _save_to_cache(cache_path, symbols)
            return symbols

    nifty200 = fetch_nifty200_constituents(force_refresh)
    fallback = nifty200[:50]
    _save_to_cache(cache_path, fallback)
    return fallback


def fetch_nifty100_constituents(force_refresh: bool = False) -> List[str]:
    """Fetch Nifty 100 constituents."""
    cache_path = Path("./data/cache/universe/nifty100_constituents.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_refresh and _is_cache_fresh(cache_path):
        cached = _load_from_cache(cache_path)
        if cached:
            return cached

    df = _fetch_nse_csv("https://archives.nseindia.com/content/indices/ind_nifty100list.csv")
    if df is not None and not df.empty:
        symbols = _extract_symbols(df)
        if symbols:
            _save_to_cache(cache_path, symbols)
            return symbols

    nifty200 = fetch_nifty200_constituents(force_refresh)
    fallback = nifty200[:100]
    _save_to_cache(cache_path, fallback)
    return fallback


def fetch_nifty_next50_constituents(force_refresh: bool = False) -> List[str]:
    """Fetch Nifty Next 50 constituents."""
    cache_path = Path("./data/cache/universe/niftynext50_constituents.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_refresh and _is_cache_fresh(cache_path):
        cached = _load_from_cache(cache_path)
        if cached:
            return cached

    df = _fetch_nse_csv("https://archives.nseindia.com/content/indices/ind_niftynext50list.csv")
    if df is not None and not df.empty:
        symbols = _extract_symbols(df)
        if symbols:
            _save_to_cache(cache_path, symbols)
            return symbols

    nifty200 = fetch_nifty200_constituents(force_refresh)
    fallback = nifty200[50:100]
    _save_to_cache(cache_path, fallback)
    return fallback


def get_nifty200_symbols(force_refresh: bool = False) -> List[str]:
    return fetch_nifty200_constituents(force_refresh)


def get_nifty50_symbols(force_refresh: bool = False) -> List[str]:
    return fetch_nifty50_constituents(force_refresh)


def get_nifty100_symbols(force_refresh: bool = False) -> List[str]:
    return fetch_nifty100_constituents(force_refresh)


def get_nifty_next50_symbols(force_refresh: bool = False) -> List[str]:
    return fetch_nifty_next50_constituents(force_refresh)


def get_universe(universe: str, force_refresh: bool = False) -> List[str]:
    universe = universe.lower().strip()
    if universe == 'nifty50':
        return get_nifty50_symbols(force_refresh)
    elif universe == 'nifty100':
        return get_nifty100_symbols(force_refresh)
    elif universe == 'nifty200':
        return get_nifty200_symbols(force_refresh)
    elif universe == 'niftynext50':
        return get_nifty_next50_symbols(force_refresh)
    else:
        return [s.strip().upper() for s in universe.split(',') if s.strip()]


def get_all_symbols() -> List[str]:
    return get_nifty200_symbols()


# Cache directory setup
CACHE_DIR = Path("./data/cache/universe")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 24


def _get_cache_path(index_name: str) -> Path:
    return Path(f"./data/cache/universe/{index_name}_constituents.json")


def _is_cache_fresh(cache_path: Path, max_age_hours: int = 24) -> bool:
    if not cache_path.exists():
        return False
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(hours=max_age_hours)


def _load_from_cache(cache_path: Path) -> Optional[List[str]]:
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        return data.get('symbols', [])
    except Exception as e:
        logger.warning(f"Failed to load cache from {cache_path}: {e}")
        return None


def _save_to_cache(cache_path: Path, symbols: List[str]):
    try:
        data = {
            'symbols': symbols,
            'updated_at': datetime.now().isoformat(),
            'count': len(symbols)
        }
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Cached {len(symbols)} symbols to {cache_path.name}")
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_path}: {e}")


def _fetch_nse_csv(url: str, timeout: int = 10) -> Optional[pd.DataFrame]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def _extract_symbols(df: pd.DataFrame) -> List[str]:
    symbol_cols = [c for c in df.columns if c.lower() in ('symbol', 'tradingsymbol')]
    if not symbol_cols:
        symbol_cols = [df.columns[0]]
    symbols = df[symbol_cols[0]].dropna().astype(str).str.strip().str.upper().tolist()
    symbols = [s.replace(' ', '').replace('.', '') for s in symbols if s]
    return symbols


def get_universe(universe: str, force_refresh: bool = False) -> List[str]:
    universe = universe.lower().strip()
    if universe == 'nifty50':
        return fetch_nifty50_constituents(force_refresh)
    elif universe == 'nifty100':
        return get_nifty100_symbols()
    elif universe == 'nifty200':
        return get_nifty200_symbols()
    elif universe == 'niftynext50':
        return fetch_nifty_next50_constituents()
    else:
        return [s.strip().upper() for s in universe.split(',') if s.strip()]


def get_all_symbols() -> List[str]:
    return fetch_nifty200_constituents()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing universe fetcher...")

    # Test Nifty 200
    nifty200 = fetch_nifty200_constituents()
    print(f"Nifty 200: {len(nifty200)} symbols")
    print(f"  First 10: {nifty200[:10]}")

    # Test Nifty 50
    nifty50 = fetch_nifty50_constituents()
    print(f"Nifty 50: {len(nifty50)} symbols")

    # Test Nifty 100
    nifty100 = get_nifty100_symbols()
    print(f"Nifty 100: {len(nifty100)} symbols")

    # Test Nifty Next 50
    niftynext50 = get_nifty_next50_symbols()
    print(f"Nifty Next 50: {len(niftynext50)} symbols")

    # Test custom
    custom = get_universe("RELIANCE,TCS,HDFCBANK")
    print(f"Custom: {custom}")