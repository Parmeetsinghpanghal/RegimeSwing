"""
Multi-Timeframe Confluence Analysis
Aligns Weekly, Daily, 4H, 1H, 15m timeframes for high-probability setups.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


class Timeframe(Enum):
    WEEKLY = "W"
    DAILY = "D"
    H4 = "4h"
    H1 = "1h"
    M15 = "15m"
    M5 = "5m"
    M1 = "1m"


@dataclass
class TFState:
    """State of a single timeframe."""
    timeframe: str
    trend: TrendDirection
    trend_strength: float          # 0-1 (ADX normalized)
    price_vs_ema: Dict[str, float] # Distance from key EMAs
    rsi: float
    macd_signal: str               # 'bullish', 'bearish', 'neutral'
    structure: str                 # 'higher_highs', 'lower_lows', 'ranging'
    key_levels: Dict[str, float]   # Support/Resistance
    volume_trend: str              # 'increasing', 'decreasing', 'flat'


@dataclass
class ConfluenceResult:
    """Multi-timeframe confluence analysis."""
    symbol: str
    current_price: float
    timeframes: Dict[str, TFState]
    alignment_score: float         # 0-1, how aligned all TFs are
    dominant_trend: TrendDirection
    confluence_zones: List[Dict]   # Price zones where multiple TFs have levels
    signal: str                    # 'strong_long', 'long', 'neutral', 'short', 'strong_short'
    signal_strength: float         # 0-1
    entry_quality: str             # 'A+', 'A', 'B', 'C', 'D'
    risk_reward_estimate: float
    notes: List[str]


# Timeframe hierarchy for swing trading
SWING_TIMEFRAMES = [
    Timeframe.WEEKLY,
    Timeframe.DAILY,
    Timeframe.H4,
    Timeframe.H1,
]

INTRADAY_TIMEFRAMES = [
    Timeframe.DAILY,
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
]

ALL_TIMEFRAMES = [
    Timeframe.WEEKLY,
    Timeframe.DAILY,
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
]


def resample_to_timeframe(
    df: pd.DataFrame,
    timeframe: str
) -> pd.DataFrame:
    """
    Resample OHLCV data to target timeframe.

    Args:
        df: DataFrame with DatetimeIndex and OHLCV columns
        timeframe: Pandas offset string ('W', 'D', '4h', '1h', '15min', '5min')

    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Resample
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }

    resampled = df.resample(timeframe).apply(ohlc_dict).dropna()
    return resampled


def analyze_single_timeframe(
    df: pd.DataFrame,
    timeframe: str
) -> TFState:
    """
    Analyze a single timeframe for trend, momentum, structure.

    Args:
        df: Resampled OHLCV DataFrame
        timeframe: Timeframe string

    Returns:
        TFState object
    """
    if len(df) < 50:
        return TFState(
            timeframe=timeframe,
            trend=TrendDirection.SIDEWAYS,
            trend_strength=0,
            price_vs_ema={},
            rsi=50,
            macd_signal='neutral',
            structure='insufficient_data',
            key_levels={},
            volume_trend='flat'
        )

    # Add indicators
    from indicators.technical import add_all_indicators, ensure_multiindex

    # Ensure single symbol format
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel('Symbol', axis=0) if 'Symbol' in df.index.names else df

    df_ind = add_all_indicators(df.copy(), {})

    latest = df_ind.iloc[-1]
    prev = df_ind.iloc[-2] if len(df_ind) > 1 else latest

    # Trend determination
    close = latest['Close']
    ema20 = latest.get('EMA_20', close)
    ema50 = latest.get('EMA_50', close)
    ema200 = latest.get('EMA_200', close)
    adx = latest.get('ADX_14', 0)

    # Trend direction
    if close > ema50 > ema200:
        trend = TrendDirection.UP
    elif close < ema50 < ema200:
        trend = TrendDirection.DOWN
    else:
        trend = TrendDirection.SIDEWAYS

    # Trend strength (ADX normalized)
    trend_strength = min(adx / 50.0, 1.0)

    # Price vs EMAs (as percentages)
    price_vs_ema = {
        'ema20': (close - ema20) / close * 100,
        'ema50': (close - ema50) / close * 100,
        'ema200': (close - ema200) / close * 100,
    }

    # RSI
    rsi = latest.get('RSI_14', 50)

    # MACD Signal
    macd = latest.get('MACD_12_26_9', 0)
    macd_signal_line = latest.get('MACDs_12_26_9', 0)
    macd_hist = latest.get('MACDh_12_26_9', 0)

    if macd > macd_signal_line and macd_hist > 0:
        macd_sig = 'bullish'
    elif macd < macd_signal_line and macd_hist < 0:
        macd_sig = 'bearish'
    else:
        macd_sig = 'neutral'

    # Market Structure
    highs = df_ind['High'].values
    lows = df_ind['Low'].values

    # Check last 20 bars for structure
    recent_highs = highs[-20:]
    recent_lows = lows[-20:]

    hh = recent_highs[-1] > recent_highs[-5:-1].max()
    ll = recent_lows[-1] < recent_lows[-5:-1].min()
    hl = recent_lows[-1] > recent_lows[-5:-1].min()
    lh = recent_highs[-1] < recent_highs[-5:-1].max()

    if hh and hl:
        structure = 'higher_highs_higher_lows'  # Uptrend
    elif lh and ll:
        structure = 'lower_highs_lower_lows'    # Downtrend
    else:
        structure = 'ranging'

    # Key Levels (simplified - recent swing high/low)
    lookback = min(50, len(df_ind))
    recent_high = df_ind['High'].tail(lookback).max()
    recent_low = df_ind['Low'].tail(lookback).min()
    poc = latest.get('VP_POC', close)  # Volume Profile POC if available

    key_levels = {
        'recent_high': recent_high,
        'recent_low': recent_low,
        'poc': poc if not np.isnan(poc) else close,
        'vwap': latest.get('VWAP', close) if not np.isnan(latest.get('VWAP', np.nan)) else close,
    }

    # Volume Trend
    vol_sma_20 = latest.get('Volume_SMA_20', latest['Volume'])
    vol_ratio = latest.get('Volume_Ratio', 1.0)

    if vol_ratio > 1.2:
        vol_trend = 'increasing'
    elif vol_ratio < 0.8:
        vol_trend = 'decreasing'
    else:
        vol_trend = 'flat'

    return TFState(
        timeframe=timeframe,
        trend=trend,
        trend_strength=trend_strength,
        price_vs_ema=price_vs_ema,
        rsi=rsi,
        macd_signal=macd_sig,
        structure=structure,
        key_levels=key_levels,
        volume_trend=vol_trend
    )


def find_confluence_zones(
    tf_states: Dict[str, TFState],
    current_price: float,
    tolerance_pct: float = 0.02
) -> List[Dict]:
    """
    Find price zones where multiple timeframes have key levels.

    Args:
        tf_states: Dict of timeframe -> TFState
        current_price: Current price
        tolerance_pct: Tolerance for level matching (%)

    Returns:
        List of confluence zones with timeframe count and levels
    """
    # Collect all levels from all timeframes
    all_levels = []

    for tf_name, state in tf_states.items():
        for level_name, level_price in state.key_levels.items():
            if np.isnan(level_price) or level_price <= 0:
                continue
            all_levels.append({
                'price': level_price,
                'timeframe': tf_name,
                'level_type': level_name,
                'trend': state.trend.value
            })

    if not all_levels:
        return []

    # Cluster nearby levels
    levels_df = pd.DataFrame(all_levels)
    prices = levels_df['price'].values

    # Hierarchical clustering
    from scipy.cluster.hierarchy import linkage, fcluster
    price_range = prices.max() - prices.min()
    distance_threshold = price_range * tolerance_pct

    if len(prices) > 1:
        Z = linkage(prices.reshape(-1, 1), method='ward')
        clusters = fcluster(Z, distance_threshold, criterion='distance')
    else:
        clusters = np.array([1])

    # Build confluence zones
    zones = []
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        cluster_levels = levels_df[mask]

        zone_price = cluster_levels['price'].mean()
        zone_tfs = cluster_levels['timeframe'].tolist()
        zone_types = cluster_levels['level_type'].tolist()

        # Only count as confluence if multiple timeframes
        unique_tfs = set(cluster_levels['timeframe'])
        if len(unique_tfs) >= 2:
            zones.append({
                'price': zone_price,
                'price_low': cluster_levels['price'].min(),
                'price_high': cluster_levels['price'].max(),
                'timeframes': zone_tfs,
                'unique_timeframes': list(unique_tfs),
                'tf_count': len(unique_tfs),
                'level_types': zone_types,
                'distance_from_current_pct': (zone_price - current_price) / current_price * 100
            })

    # Sort by confluence strength (tf_count) and proximity
    zones.sort(key=lambda x: (-x['tf_count'], abs(x['distance_from_current_pct'])))

    return zones


def calculate_alignment_score(tf_states: Dict[str, TFState]) -> Tuple[float, TrendDirection]:
    """
    Calculate how aligned all timeframes are.

    Returns:
        (alignment_score, dominant_trend)
    """
    if not tf_states:
        return 0.0, TrendDirection.SIDEWAYS

    # Weight timeframes (higher TFs get more weight)
    tf_weights = {
        'W': 1.0,
        'D': 0.8,
        '4h': 0.6,
        '1h': 0.4,
        '15m': 0.3,
        '5m': 0.2,
    }

    trend_scores = {'up': 0, 'down': 0, 'sideways': 0}
    total_weight = 0

    for tf_name, state in tf_states.items():
        weight = tf_weights.get(tf_name, 0.2)
        trend_key = state.trend.value
        trend_scores[trend_key] += weight * state.trend_strength
        total_weight += weight

    # Normalize
    for k in trend_scores:
        trend_scores[k] /= total_weight if total_weight > 0 else 1

    # Dominant trend
    dominant = max(trend_scores, key=trend_scores.get)
    dominant_trend = TrendDirection(dominant)

    # Alignment = dominant score - second best
    sorted_scores = sorted(trend_scores.values(), reverse=True)
    alignment = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

    return alignment, dominant_trend


def analyze_multitf_confluence(
    symbol: str,
    data_by_tf: Dict[str, pd.DataFrame],
    current_price: float = None
) -> ConfluenceResult:
    """
    Main function: Analyze multi-timeframe confluence for a symbol.

    Args:
        symbol: Symbol name
        data_by_tf: Dict of timeframe -> OHLCV DataFrame (already resampled)
        current_price: Current price (uses last close if None)

    Returns:
        ConfluenceResult with full analysis
    """
    if not data_by_tf:
        return ConfluenceResult(
            symbol=symbol,
            current_price=current_price or 0,
            timeframes={},
            alignment_score=0,
            dominant_trend=TrendDirection.SIDEWAYS,
            confluence_zones=[],
            signal='no_data',
            signal_strength=0,
            entry_quality='F',
            risk_reward_estimate=0,
            notes=['No data provided']
        )

    # Analyze each timeframe
    tf_states = {}
    for tf_name, df in data_by_tf.items():
        if df is not None and len(df) > 20:
            tf_states[tf_name] = analyze_single_timeframe(df, tf_name)

    # Current price
    if current_price is None:
        # Use daily close
        daily_df = data_by_tf.get('D') or data_by_tf.get('daily')
        if daily_df is not None and len(daily_df) > 0:
            current_price = daily_df['Close'].iloc[-1]
        else:
            current_price = 0

    # Alignment
    alignment, dominant_trend = calculate_alignment_score(tf_states)

    # Confluence zones
    zones = find_confluence_zones(tf_states, current_price)

    # Signal determination
    signal, strength, quality, rr = _determine_signal(
        tf_states, zones, current_price, alignment, dominant_trend
    )

    # Notes
    notes = _generate_notes(tf_states, zones, alignment, dominant_trend)

    return ConfluenceResult(
        symbol=symbol,
        current_price=current_price,
        timeframes=tf_states,
        alignment_score=alignment,
        dominant_trend=dominant_trend,
        confluence_zones=zones,
        signal=signal,
        signal_strength=strength,
        entry_quality=quality,
        risk_reward_estimate=rr,
        notes=notes
    )


def _determine_signal(
    tf_states: Dict[str, TFState],
    zones: List[Dict],
    current_price: float,
    alignment: float,
    dominant_trend: TrendDirection
) -> Tuple[str, float, str, float]:
    """Determine trading signal from confluence analysis."""

    # Base signal from dominant trend
    if dominant_trend == TrendDirection.UP:
        base_signal = 'long'
    elif dominant_trend == TrendDirection.DOWN:
        base_signal = 'short'
    else:
        base_signal = 'neutral'

    # Strength factors
    strength_factors = []

    # 1. Alignment strength
    strength_factors.append(alignment)

    # 2. Confluence zones near current price
    near_zones = [z for z in zones if abs(z['distance_from_current_pct']) < 2.0]
    zone_bonus = min(len(near_zones) * 0.1, 0.3)
    strength_factors.append(zone_bonus)

    # 3. RSI alignment (not overbought/oversold against trend)
    rsi_alignment = 0
    for tf_name, state in tf_states.items():
        if dominant_trend == TrendDirection.UP and state.rsi < 70:
            rsi_alignment += 0.1
        elif dominant_trend == TrendDirection.DOWN and state.rsi > 30:
            rsi_alignment += 0.1
    strength_factors.append(min(rsi_alignment, 0.2))

    # 4. Volume confirmation
    vol_confirm = sum(1 for s in tf_states.values() if s.volume_trend == 'increasing') / len(tf_states) if tf_states else 0
    strength_factors.append(vol_confirm * 0.15)

    # 5. MACD alignment
    macd_align = sum(1 for s in tf_states.values()
                     if (dominant_trend == TrendDirection.UP and s.macd_signal == 'bullish') or
                     (dominant_trend == TrendDirection.DOWN and s.macd_signal == 'bearish')) / len(tf_states) if tf_states else 0
    strength_factors.append(macd_align * 0.15)

    total_strength = sum(strength_factors)

    # Determine signal and quality
    if dominant_trend == TrendDirection.UP:
        if total_strength >= 0.75:
            signal = 'strong_long'
            quality = 'A+'
        elif total_strength >= 0.6:
            signal = 'long'
            quality = 'A'
        elif total_strength >= 0.45:
            signal = 'long'
            quality = 'B'
        elif total_strength >= 0.3:
            signal = 'long'
            quality = 'C'
        else:
            signal = 'weak_long'
            quality = 'D'
    elif dominant_trend == TrendDirection.DOWN:
        if total_strength >= 0.75:
            signal = 'strong_short'
            quality = 'A+'
        elif total_strength >= 0.6:
            signal = 'short'
            quality = 'A'
        elif total_strength >= 0.45:
            signal = 'short'
            quality = 'B'
        elif total_strength >= 0.3:
            signal = 'short'
            quality = 'C'
        else:
            signal = 'weak_short'
            quality = 'D'
    else:
        signal = 'neutral'
        quality = 'D'
        total_strength = 0.2

    # Estimate R:R from confluence zones
    rr = 1.5  # Default
    resistance_zones = [z for z in zones if z['price'] > current_price]
    support_zones = [z for z in zones if z['price'] < current_price]

    if dominant_trend == TrendDirection.UP and resistance_zones:
        nearest_resistance = min(resistance_zones, key=lambda x: x['price'])
        risk = (current_price - nearest_resistance['price']) / current_price
        if risk < 0:
            risk = abs(risk)
        reward = (nearest_resistance['price'] - current_price) / current_price
        if risk > 0:
            rr = reward / risk
    elif dominant_trend == TrendDirection.DOWN and support_zones:
        nearest_support = max(support_zones, key=lambda x: x['price'])
        risk = (nearest_support['price'] - current_price) / current_price
        if risk < 0:
            risk = abs(risk)
        reward = (current_price - nearest_support['price']) / current_price
        if risk > 0:
            rr = reward / risk

    return base_signal if signal == 'neutral' else signal, min(total_strength, 1.0), quality, max(rr, 0.5)


def _generate_notes(
    tf_states: Dict[str, TFState],
    zones: List[Dict],
    alignment: float,
    dominant_trend: TrendDirection
) -> List[str]:
    """Generate human-readable notes."""
    notes = []

    # Trend alignment
    trend_tfs = [tf for tf, s in tf_states.items() if s.trend == dominant_trend]
    notes.append(f"Dominant trend: {dominant_trend.value.upper()} across {len(trend_tfs)}/{len(tf_states)} TFs")

    # Alignment
    if alignment > 0.5:
        notes.append(f"Strong alignment ({alignment:.0%}) - high probability setup")
    elif alignment > 0.2:
        notes.append(f"Moderate alignment ({alignment:.0%}) - decent setup")
    else:
        notes.append(f"Weak alignment ({alignment:.0%}) - conflicting timeframes")

    # Confluence zones
    if zones:
        top_zone = zones[0]
        notes.append(f"Key confluence at {top_zone['price']:.2f} ({top_zone['tf_count']} TFs: {', '.join(top_zone['unique_timeframes'])})")

    # RSI warnings
    for tf, state in tf_states.items():
        if state.rsi > 75:
            notes.append(f"⚠️ {tf} RSI overbought ({state.rsi:.0f})")
        elif state.rsi < 25:
            notes.append(f"⚠️ {tf} RSI oversold ({state.rsi:.0f})")

    # Volume
    increasing_tfs = [tf for tf, s in tf_states.items() if s.volume_trend == 'increasing']
    if increasing_tfs:
        notes.append(f"📈 Volume increasing on {', '.join(increasing_tfs)}")

    return notes


def fetch_multitf_data(
    fetcher,
    symbol: str,
    timeframes: List[str] = None,
    lookback_days: Dict[str, int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple timeframes.

    Args:
        fetcher: DataFetcher instance
        symbol: Symbol to fetch
        timeframes: List of timeframe strings
        lookback_days: Dict of timeframe -> lookback days

    Returns:
        Dict of timeframe -> DataFrame
    """
    if timeframes is None:
        timeframes = ['W', 'D', '4h', '1h', '15min']

    if lookback_days is None:
        lookback_days = {
            'W': 104 * 7,    # 2 years weekly
            'D': 730,        # 2 years daily
            '4h': 180,       # 6 months 4h
            '1h': 60,        # 2 months 1h
            '15min': 30,     # 1 month 15m
        }

    result = {}

    # Fetch daily as base (most reliable)
    daily_df = fetcher.get_price_data(symbol, period="2y", interval="1d")

    if daily_df.empty:
        logger.warning(f"No daily data for {symbol}")
        return {}

    # Extract symbol data
    if isinstance(daily_df.index, pd.MultiIndex):
        daily_df = daily_df.xs(symbol, level='Symbol')

    daily_df.index = pd.to_datetime(daily_df.index)
    result['D'] = daily_df

    # Resample to other timeframes
    for tf in timeframes:
        if tf == 'D':
            continue

        lookback = lookback_days.get(tf, 30)
        # Use appropriate source data
        if tf in ['4h', '1h', '15min', '5min']:
            # Need intraday data - fetch separately
            try:
                intraday_df = fetcher.get_price_data([symbol], period=f"{lookback}d", interval=tf)
                if not intraday_df.empty:
                    if isinstance(intraday_df.index, pd.MultiIndex):
                        intraday_df = intraday_df.xs(symbol, level='Symbol')
                    intraday_df.index = pd.to_datetime(intraday_df.index)
                    result[tf] = intraday_df
            except Exception as e:
                logger.warning(f"Could not fetch {tf} for {symbol}: {e}")
                # Fallback: resample daily (less accurate)
                if tf in ['4h', '1h']:
                    result[tf] = resample_to_timeframe(daily_df, tf)
        else:
            # Resample from daily
            result[tf] = resample_to_timeframe(daily_df, tf)

    return result


def scan_universe_confluence(
    fetcher,
    symbols: List[str],
    timeframes: List[str] = None
) -> List[ConfluenceResult]:
    """
    Scan multiple symbols for confluence setups.

    Returns:
        List of ConfluenceResult sorted by signal strength
    """
    results = []

    for symbol in symbols:
        try:
            data_by_tf = fetch_multitf_data(fetcher, symbol, timeframes)
            if not data_by_tf:
                continue

            confluence = analyze_multitf_confluence(symbol, data_by_tf)
            results.append(confluence)
        except Exception as e:
            logger.warning(f"Confluence analysis failed for {symbol}: {e}")

    # Sort by signal strength
    results.sort(key=lambda x: x.signal_strength, reverse=True)

    return results


if __name__ == "__main__":
    import yfinance as yf
    from data.fetcher import create_fetcher

    # Test with RELIANCE
    fetcher = create_fetcher()

    # Fetch multi-timeframe data
    symbol = "RELIANCE"
    data_by_tf = fetch_multitf_data(fetcher, symbol)

    print(f"Fetched timeframes: {list(data_by_tf.keys())}")
    for tf, df in data_by_tf.items():
        print(f"  {tf}: {len(df)} bars")

    # Analyze confluence
    confluence = analyze_multitf_confluence(symbol, data_by_tf)

    print(f"\n=== CONFLUENCE ANALYSIS: {symbol} ===")
    print(f"Current Price: {confluence.current_price:.2f}")
    print(f"Dominant Trend: {confluence.dominant_trend.value.upper()}")
    print(f"Alignment Score: {confluence.alignment_score:.2%}")
    print(f"Signal: {confluence.signal} (Strength: {confluence.signal_strength:.2%})")
    print(f"Entry Quality: {confluence.entry_quality}")
    print(f"Est. R:R: {confluence.risk_reward_estimate:.2f}")

    print(f"\nTimeframe States:")
    for tf, state in confluence.timeframes.items():
        print(f"  {tf:4} | {state.trend.value:9} | ADX: {state.trend_strength:.2f} | "
              f"RSI: {state.rsi:5.1f} | MACD: {state.macd_signal:8} | "
              f"Structure: {state.structure}")

    print(f"\nConfluence Zones:")
    for zone in confluence.confluence_zones[:5]:
        print(f"  {zone['price']:.2f} | {zone['tf_count']} TFs ({', '.join(zone['unique_timeframes'])}) | "
              f"Dist: {zone['distance_from_current_pct']:+.2f}% | Types: {zone['level_types']}")

    print(f"\nNotes:")
    for note in confluence.notes:
        print(f"  - {note}")