"""
Macro Chain-Reaction Monitor.
Monitors global macro tickers (Brent, Gold, SOX, US 10Y, USD/INR, India VIX)
and derives sector tilts for Indian markets.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)

# Macro tickers (all free on yfinance)
MACRO_TICKERS = {
    'brent_crude': 'BZ=F',
    'gold': 'GC=F',
    'semiconductor': '^SOX',
    'us_10y': '^TNX',
    'usd_inr': 'INR=X',
    'india_vix': '^INDIAVIX',
}

MACRO_THRESHOLDS = {
    'brent_crude_pct': 0.04,      # +4% daily spike
    'gold_pct': 0.03,             # +3% daily spike
    'semiconductor_pct': 0.05,    # +5% weekly
    'us_10y_drop_bps': 15,        # -15bps drop
    'vix_threshold': 22,          # India VIX > 22
}

# NSE Sector mapping for macro tilts
NSE_SECTOR_STOCKS = {
    'PAINTS': ['ASIANPAINT', 'BERGEPAINT', 'KANSAINER', 'NEROLAC'],
    'AVIATION': ['INDIGO', 'SPICEJET'],
    'OIL_MARKETING': ['BPCL', 'IOC', 'HPCL'],
    'TYRES': ['MRF', 'APOLLOTYRE', 'CEAT', 'BALKRISIND'],
    'DEFENCE': ['BEL', 'HAL', 'ASTRA', 'MTARTECH', 'DATAPATTNS'],
    'GOLD': ['TITAN', 'KALYANKJIL', 'PCJEWELLER'],
    'GOLD_LOANS': ['MUTHOOTFIN', 'MANAPPURAM'],
    'UPSTREAM_OIL': ['ONGC', 'OIL'],
    'JEWELLERY': ['TITAN', 'KALYANKJIL', 'PCJEWELLER', 'TBZ'],
    'EMS': ['DIXON', 'KAYNES', 'CGPOWER', 'TATAELXSI', 'SYY', 'CYIENT'],
    'SEMICONDUCTOR': ['TATAELXSI', 'KAYNES', 'SPELSEMI', 'ASM'],
    'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'PERSISTENT', 'COFORGE'],
    'RATE_SENSITIVE_BANKS': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN', 'BAJFINANCE'],
}

SECTOR_MAP = {
    'brent_crude_spike': {
        'avoid': ['PAINTS', 'AVIATION', 'OIL_MARKETING', 'TYRES'],
        'favor': ['DEFENCE', 'GOLD', 'UPSTREAM_OIL'],
    },
    'gold_spike': {
        'favor': ['JEWELLERY', 'GOLD_LOANS'],
    },
    'semiconductor_rally': {
        'favor': ['EMS', 'SEMICONDUCTOR'],
    },
    'us_10y_drop': {
        'favor': ['IT', 'RATE_SENSITIVE_BANKS'],
    },
    'rupee_strength': {
        'favor': ['IT', 'RATE_SENSITIVE_BANKS'],
    },
    'vix_high': {
        'size_mult': 0.25,  # 75% size cut
    },
}


def fetch_macro_data(lookback_days: int = 10) -> Dict:
    """
    Fetch all macro tickers and compute signals.

    Returns dict with ticker data and computed changes.
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    signals = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                           end=end.strftime('%Y-%m-%d'), progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.capitalize() for c in df.columns]
                df.index = pd.to_datetime(df.index)

                close = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2] if len(df) > 1 else close
                pct_change = (close - prev) / prev if prev > 0 else 0

                # Weekly change for semiconductor
                if name == 'semiconductor':
                    week_ago = df['Close'].iloc[-5] if len(df) >= 5 else prev
                    weekly_change = (close - week_ago) / week_ago if week_ago > 0 else 0
                    signals[name] = {'close': close, 'daily_pct': pct_change, 'weekly_pct': weekly_change}
                else:
                    signals[name] = {'close': close, 'daily_pct': pct_change}

        except Exception as e:
            logger.warning(f"Macro fetch failed for {name} ({ticker}): {e}")
            signals[name] = {'error': str(e)}

    return signals


def evaluate_macro_signals(signals: Dict) -> Dict:
    """
    Evaluate macro signals and return sector tilts + size multiplier.

    Returns:
        dict with 'avoid' (set), 'favor' (set), 'size_mult' (float)
    """
    tilts = {'avoid': set(), 'favor': set(), 'size_mult': 1.0}

    # Brent Crude spike
    if signals.get('brent_crude', {}).get('daily_pct', 0) > MACRO_THRESHOLDS['brent_crude_pct']:
        for s in SECTOR_MAP['brent_crude_spike']['avoid']:
            tilts['avoid'].update(NSE_SECTOR_STOCKS.get(s, []))
        for s in SECTOR_MAP['brent_crude_spike']['favor']:
            tilts['favor'].update(NSE_SECTOR_STOCKS.get(s, []))

    # Gold spike
    if signals.get('gold', {}).get('daily_pct', 0) > MACRO_THRESHOLDS['gold_pct']:
        for s in SECTOR_MAP['gold_spike']['favor']:
            tilts['favor'].update(NSE_SECTOR_STOCKS.get(s, []))

    # Semiconductor rally
    if signals.get('semiconductor', {}).get('weekly_pct', 0) > MACRO_THRESHOLDS['semiconductor_pct']:
        for s in SECTOR_MAP['semiconductor_rally']['favor']:
            tilts['favor'].update(NSE_SECTOR_STOCKS.get(s, []))

    # US 10Y drop
    if signals.get('us_10y', {}).get('daily_pct', 0) < -MACRO_THRESHOLDS['us_10y_drop_bps'] / 10000:
        for s in SECTOR_MAP['us_10y_drop']['favor']:
            tilts['favor'].update(NSE_SECTOR_STOCKS.get(s, []))

    # Rupee strength (USD/INR down = INR up)
    if signals.get('usd_inr', {}).get('daily_pct', 0) < -0.003:
        for s in SECTOR_MAP['rupee_strength']['favor']:
            tilts['favor'].update(NSE_SECTOR_STOCKS.get(s, []))

    # VIX high
    if signals.get('india_vix', {}).get('close', 0) > MACRO_THRESHOLDS['vix_threshold']:
        tilts['size_mult'] = min(tilts['size_mult'], SECTOR_MAP['vix_high']['size_mult'])

    return tilts


def get_macro_summary(signals: Dict) -> str:
    """Generate human-readable macro summary."""
    lines = ["📊 MACRO MONITOR:"]
    for name, data in signals.items():
        if 'error' in data:
            lines.append(f"  {name}: Error - {data['error']}")
        elif 'daily_pct' in data:
            pct = data['daily_pct'] * 100
            direction = "📈" if pct > 0 else "📉" if pct < 0 else "➡️"
            lines.append(f"  {name.replace('_', ' ').title()}: {direction} {pct:+.2f}%")
            if 'weekly_pct' in data:
                wpct = data['weekly_pct'] * 100
                lines.append(f"    Weekly: {wpct:+.2f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    signals = fetch_macro_data()
    print(get_macro_summary(signals))

    tilts = evaluate_macro_signals(signals)
    print("\n🎯 SECTOR TILTS:")
    print(f"  Favor: {', '.join(sorted(tilts['favor'])) if tilts['favor'] else 'None'}")
    print(f"  Avoid: {', '.join(sorted(tilts['avoid'])) if tilts['avoid'] else 'None'}")
    print(f"  Size Multiplier: {tilts['size_mult']:.2f}")