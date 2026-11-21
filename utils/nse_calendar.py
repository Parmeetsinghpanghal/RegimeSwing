"""
NSE Calendar Utility - Handles NSE holidays, earnings calendar, ASM/GSM lists.
Zero-cost, resilient with graceful fallbacks for NSE rate limits.
"""

import os
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Dict, Optional, Set
from functools import lru_cache

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path("./data/cache/nse_calendar")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# NSE URLs
NSE_HOLIDAYS_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
NSE_ASM_GSM_URL = "https://www.nseindia.com/api/report-daily-asm"
NSE_EARNINGS_URL = "https://www.nseindia.com/api/corporate-announcements?index=equities&from_date={}&to_date={}"

# Static fallback holidays (2024-2025 major ones)
STATIC_HOLIDAYS = {
    2024: [
        "2024-01-26",  # Republic Day
        "2024-03-08",  # Mahashivratri
        "2024-03-25",  # Holi
        "2024-03-29",  # Good Friday
        "2024-04-11",  # Id-Ul-Fitr
        "2024-04-17",  # Ram Navami
        "2024-05-01",  # Maharashtra Day
        "2024-06-17",  # Bakri Id
        "2024-07-17",  # Muharram
        "2024-08-15",  # Independence Day
        "2024-09-02",  # Ganesh Chaturthi
        "2024-10-02",  # Gandhi Jayanti
        "2024-11-01",  # Diwali Laxmi Pujan
        "2024-11-15",  # Guru Nanak Jayanti
        "2024-12-25",  # Christmas
    ],
    2025: [
        "2025-01-26",  # Republic Day
        "2025-02-26",  # Mahashivratri
        "2025-03-14",  # Holi
        "2025-03-31",  # Id-Ul-Fitr
        "2025-04-10",  # Mahavir Jayanti
        "2025-04-14",  # Ambedkar Jayanti
        "2025-04-18",  # Good Friday
        "2025-05-01",  # Maharashtra Day
        "2025-06-07",  # Bakri Id
        "2025-07-06",  # Muharram
        "2025-08-15",  # Independence Day
        "2025-08-27",  # Ganesh Chaturthi
        "2025-10-02",  # Gandhi Jayanti
        "2025-10-21",  # Diwali
        "2025-11-05",  # Guru Nanak Jayanti
        "2025-12-25",  # Christmas
    ]
}

# NSE session headers
NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nseindia.com/',
}

# Cache TTL (hours)
CACHE_TTL_HOURS = 24


class NSECalendar:
    """Manages NSE trading calendar, holidays, earnings, ASM/GSM lists."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        # Initialize session with NSE
        self._init_session()

    def _init_session(self):
        """Initialize NSE session with cookies."""
        try:
            self.session.get("https://www.nseindia.com", timeout=10)
        except Exception as e:
            logger.warning(f"NSE session init failed: {e}")

    def _get_cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.json"

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

    def _fetch_with_fallback(self, url: str, cache_name: str,
                             static_fallback: any = None) -> any:
        """Fetch from NSE with cache and static fallback."""
        cache_path = self._get_cache_path(cache_name)

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

        # Return static fallback
        if static_fallback is not None:
            logger.info(f"Using static fallback for {cache_name}")
            return static_fallback

        return None

    def get_holidays(self, year: int = None) -> List[str]:
        """Get NSE trading holidays for a year."""
        if year is None:
            year = datetime.now().year

        # Try NSE API
        data = self._fetch_with_fallback(
            NSE_HOLIDAYS_URL,
            f"holidays_{year}",
            static_fallback={"FO": [], "CM": []}
        )

        holidays = []
        if data and isinstance(data, dict):
            # NSE returns dict with FO, CM, etc.
            for segment in ["CM", "FO", "CD", "SLB", "WDM"]:
                if segment in data:
                    for h in data[segment]:
                        if h.get("tradingDate"):
                            holidays.append(h["tradingDate"])

        # Add static fallback for the year
        if year in STATIC_HOLIDAYS:
            holidays.extend(STATIC_HOLIDAYS[year])

        # Deduplicate and sort
        holidays = sorted(set(holidays))
        return holidays

    def is_trading_day(self, check_date: date = None) -> bool:
        """Check if a date is a trading day (not weekend, not holiday)."""
        if check_date is None:
            check_date = date.today()

        # Weekend check
        if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Holiday check
        year = check_date.year
        holidays = self.get_holidays(year)
        date_str = check_date.strftime("%Y-%m-%d")

        return date_str not in holidays

    def get_next_trading_day(self, from_date: date = None) -> date:
        """Get next trading day after given date."""
        if from_date is None:
            from_date = date.today()

        next_day = from_date + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day

    def get_asm_gsm_list(self) -> Dict[str, Set[str]]:
        """Get ASM (Additional Surveillance Measure) and GSM (Graded Surveillance Measure) lists."""
        cache_name = f"asm_gsm_{date.today()}"

        # Static fallback - major known ASM/GSM stocks (update periodically)
        static_asm = {"SUZLON", "RPOWER", "JPPOWER", "HFCL", "GTL", "IDEA"}
        static_gsm = {"BANKINDIA", "CENTRALBK", "UCOBANK", "IOB", "PNB"}

        data = self._fetch_with_fallback(
            NSE_ASM_GSM_URL,
            cache_name,
            static_fallback={"asm": [], "gsm": []}
        )

        asm_stocks = set(static_asm)
        gsm_stocks = set(static_gsm)

        if data and isinstance(data, list):
            for item in data:
                symbol = item.get("symbol", "").upper()
                stage = item.get("stage", "").upper()
                if symbol:
                    if "ASM" in stage:
                        asm_stocks.add(symbol)
                    elif "GSM" in stage:
                        gsm_stocks.add(symbol)

        return {"asm": asm_stocks, "gsm": gsm_stocks}

    def is_asm_gsm(self, symbol: str) -> Dict[str, bool]:
        """Check if a stock is in ASM or GSM."""
        lists = self.get_asm_gsm_list()
        sym = symbol.upper().replace(".NS", "")
        return {
            "asm": sym in lists["asm"],
            "gsm": sym in lists["gsm"],
            "restricted": sym in lists["asm"] or sym in lists["gsm"]
        }

    def get_earnings_calendar(self, days_ahead: int = 30) -> List[Dict]:
        """Get upcoming earnings announcements."""
        from_date = date.today()
        to_date = from_date + timedelta(days=days_ahead)

        cache_name = f"earnings_{from_date}_{to_date}"

        url = NSE_EARNINGS_URL.format(
            from_date.strftime("%d-%m-%Y"),
            to_date.strftime("%d-%m-%Y")
        )

        static_fallback = []

        data = self._fetch_with_fallback(url, cache_name, static_fallback)

        earnings = []
        if data and isinstance(data, list):
            for item in data:
                try:
                    symbol = item.get("symbol", "").upper()
                    subject = item.get("subject", "").lower()
                    broadcast_date = item.get("broadcast_date", "")

                    # Filter for earnings-related announcements
                    if any(kw in subject for kw in ["results", "earnings", "quarterly", "financial"]):
                        earnings.append({
                            "symbol": symbol,
                            "date": broadcast_date,
                            "subject": item.get("subject", "")
                        })
                except Exception:
                    continue

        return earnings

    def has_earnings_soon(self, symbol: str, days: int = 5) -> bool:
        """Check if a stock has earnings within specified days."""
        earnings = self.get_earnings_calendar(days)
        sym = symbol.upper().replace(".NS", "")
        for e in earnings:
            if e["symbol"] == sym:
                return True
        return False

    def get_nse_session_status(self) -> Dict:
        """Get current NSE session status."""
        now = datetime.now()
        today = date.today()

        is_trading = self.is_trading_day(today)

        # Market hours: 9:15 AM - 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        pre_market_start = now.replace(hour=8, minute=45, second=0, microsecond=0)

        is_market_hours = market_open <= now <= market_close
        is_pre_market = pre_market_start <= now < market_open

        return {
            "is_trading_day": is_trading,
            "is_market_hours": is_market_hours and is_trading,
            "is_pre_market": is_pre_market and is_trading,
            "market_open": market_open.strftime("%H:%M"),
            "market_close": market_close.strftime("%H:%M"),
            "current_time": now.strftime("%H:%M:%S"),
            "timezone": "IST"
        }

    def get_fii_dii_flow(self, days: int = 5) -> Dict:
        """
        Fetch FII + DII net cash market buying/selling.
        Note: This uses a fallback since NSE doesn't provide free API.
        For production, consider using nsepython library.
        """
        # Static fallback - in production, use nsepython or NSE API
        # This is a placeholder - real implementation would fetch from NSE
        return {
            'fii_5d_net_cr': 0,
            'dii_5d_net_cr': 0,
            'combined_net_cr': 0,
            'signal': 'neutral',
            'note': 'FII/DII data requires nsepython or NSE API subscription'
        }


# Convenience functions
_nse_calendar = None

def get_nse_calendar() -> NSECalendar:
    """Get singleton NSECalendar instance."""
    global _nse_calendar
    if _nse_calendar is None:
        _nse_calendar = NSECalendar()
    return _nse_calendar


def is_trading_day(check_date: date = None) -> bool:
    """Check if date is a trading day."""
    return get_nse_calendar().is_trading_day(check_date)


def is_asm_gsm(symbol: str) -> Dict[str, bool]:
    """Check if symbol is in ASM/GSM."""
    return get_nse_calendar().is_asm_gsm(symbol)


def has_earnings_soon(symbol: str, days: int = 5) -> bool:
    """Check if symbol has earnings within days."""
    return get_nse_calendar().has_earnings_soon(symbol, days)


def get_nse_session_status() -> Dict:
    """Get NSE session status."""
    return get_nse_calendar().get_nse_session_status()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    cal = NSECalendar()

    # Test holidays
    print("Holidays 2024:", cal.get_holidays(2024)[:5], "...")

    # Test trading day
    print("Today trading:", cal.is_trading_day())

    # Test ASM/GSM
    print("RELIANCE ASM/GSM:", cal.is_asm_gsm("RELIANCE"))
    print("SUZLON ASM/GSM:", cal.is_asm_gsm("SUZLON"))

    # Test session status
    print("Session status:", cal.get_nse_session_status())

    # Test next trading day
    print("Next trading day:", cal.get_next_trading_day())