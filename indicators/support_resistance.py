"""
Support & Resistance Detection Module
Identifies key support/resistance levels using fractal pivots, volume weighting, and clustering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.signal import argrelextrema
import logging

logger = logging.getLogger(__name__)


@dataclass
class SRZone:
    """Support/Resistance Zone."""
    price: float                  # Zone center price
    price_low: float              # Zone lower bound
    price_high: float             # Zone upper bound
    zone_type: str                # 'support', 'resistance', 'both'
    strength: float               # 0-1 strength score
    touches: int                  # Number of tests
    total_volume: float           # Volume at tests
    avg_volume_per_touch: float
    last_test_date: pd.Timestamp
    first_test_date: pd.Timestamp
    is_broken: bool = False       # Whether zone was broken
    break_date: Optional[pd.Timestamp] = None


@dataclass
class SRAnalysis:
    """Complete S/R Analysis."""
    zones: List[SRZone]
    current_price: float
    nearest_support: Optional[SRZone]
    nearest_resistance: Optional[SRZone]
    current_zone: Optional[SRZone]      # Zone price is currently in
    signals: Dict                       # Trading signals


def find_fractal_pivots(
    df: pd.DataFrame,
    left_bars: int = 5,
    right_bars: int = 5,
    pivot_type: str = 'both'  # 'high', 'low', 'both'
) -> pd.DataFrame:
    """
    Find fractal pivot points (Williams Fractals).

    Args:
        df: DataFrame with OHLCV
        left_bars: Bars to left for confirmation
        right_bars: Bars to right for confirmation
        pivot_type: 'high' (resistance), 'low' (support), 'both'

    Returns:
        DataFrame with pivot points
    """
    highs = df['High'].values
    lows = df['Low'].values
    dates = df.index

    pivots = []

    if pivot_type in ['high', 'both']:
        # Resistance pivots (highs)
        high_pivots = argrelextrema(highs, np.greater_equal, order=left_bars)[0]
        for idx in high_pivots:
            if idx >= left_bars and idx < len(highs) - right_bars:
                window = highs[idx-left_bars:idx+right_bars+1]
                if highs[idx] == window.max():
                    pivots.append({
                        'date': dates[idx],
                        'price': highs[idx],
                        'type': 'resistance',
                        'index': idx
                    })

    if pivot_type in ['low', 'both']:
        # Support pivots (lows)
        low_pivots = argrelextrema(lows, np.less_equal, order=left_bars)[0]
        for idx in low_pivots:
            if idx >= left_bars and idx < len(lows) - right_bars:
                window = lows[idx-left_bars:idx+right_bars+1]
                if lows[idx] == window.min():
                    pivots.append({
                        'date': dates[idx],
                        'price': lows[idx],
                        'type': 'support',
                        'index': idx
                    })

    return pd.DataFrame(pivots).sort_values('date') if pivots else pd.DataFrame(
        columns=['date', 'price', 'type', 'index']
    )


def find_swing_points(
    df: pd.DataFrame,
    lookback: int = 10,
    min_swing_pct: float = 0.01
) -> pd.DataFrame:
    """
    Find swing highs and lows using ZigZag logic.

    Args:
        df: DataFrame with OHLCV
        lookback: Minimum bars between swings
        min_swing_pct: Minimum swing size as percentage

    Returns:
        DataFrame with swing points
    """
    highs = df['High'].values
    lows = df['Low'].values
    dates = df.index

    swings = []
    last_pivot_type = None  # 'high' or 'low'
    last_pivot_price = None
    last_pivot_idx = 0

    for i in range(lookback, len(highs) - lookback):
        # Check for swing high
        window_high = highs[i-lookback:i+lookback+1]
        if highs[i] == window_high.max():
            if last_pivot_type != 'high':
                if last_pivot_price is not None:
                    swing_size = abs(highs[i] - last_pivot_price) / last_pivot_price
                    if swing_size >= min_swing_pct:
                        swings.append({
                            'date': dates[i],
                            'price': highs[i],
                            'type': 'resistance',
                            'index': i
                        })
                        last_pivot_type = 'high'
                        last_pivot_price = highs[i]
                        last_pivot_idx = i

        # Check for swing low
        window_low = lows[i-lookback:i+lookback+1]
        if lows[i] == window_low.min():
            if last_pivot_type != 'low':
                if last_pivot_price is not None:
                    swing_size = abs(lows[i] - last_pivot_price) / last_pivot_price
                    if swing_size >= min_swing_pct:
                        swings.append({
                            'date': dates[i],
                            'price': lows[i],
                            'type': 'support',
                            'index': i
                        })
                        last_pivot_type = 'low'
                        last_pivot_price = lows[i]
                        last_pivot_idx = i

    return pd.DataFrame(swings).sort_values('date') if swings else pd.DataFrame(
        columns=['date', 'price', 'type', 'index']
    )


def cluster_price_levels(
    prices: np.ndarray,
    volumes: np.ndarray,
    max_distance_pct: float = 0.015,
    min_cluster_size: int = 2
) -> List[Tuple[float, float, float, int, float]]:
    """
    Cluster price levels using hierarchical clustering.

    Returns:
        List of (center_price, low, high, total_volume, count)
    """
    if len(prices) < min_cluster_size:
        return []

    # Hierarchical clustering
    X = prices.reshape(-1, 1)

    # Distance threshold
    price_range = prices.max() - prices.min()
    distance_threshold = price_range * max_distance_pct

    if len(prices) > 1:
        Z = linkage(X, method='ward', metric='euclidean')
        clusters = fcluster(Z, distance_threshold, criterion='distance')
    else:
        clusters = np.array([1])

    results = []
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        cluster_prices = prices[mask]
        cluster_volumes = volumes[mask]

        if len(cluster_prices) >= min_cluster_size:
            # Weighted center by volume
            center = np.average(cluster_prices, weights=cluster_volumes)
            total_vol = cluster_volumes.sum()
            results.append((
                center,
                cluster_prices.min(),
                cluster_prices.max(),
                total_vol,
                len(cluster_prices)
            ))

    return sorted(results, key=lambda x: x[0])


def calculate_zone_strength(
    zone_price: float,
    zone_low: float,
    zone_high: float,
    touches: int,
    total_volume: float,
    avg_daily_volume: float,
    days_since_last_test: int,
    price_range: float
) -> float:
    """
    Calculate zone strength score (0-1).

    Factors:
    - Number of touches (more = stronger)
    - Volume at touches (higher = stronger)
    - Recency (recent tests = stronger)
    - Zone tightness (tighter = stronger)
    """
    # Touch factor (diminishing returns)
    touch_score = min(touches / 5.0, 1.0) * 0.3

    # Volume factor
    vol_ratio = total_volume / (avg_daily_volume * touches) if touches > 0 else 0
    vol_score = min(vol_ratio / 2.0, 1.0) * 0.3

    # Recency factor (exponential decay)
    recency_score = np.exp(-days_since_last_test / 30.0) * 0.2

    # Tightness factor
    zone_width_pct = (zone_high - zone_low) / zone_price
    tightness_score = max(0, 1 - zone_width_pct / 0.03) * 0.2  # 3% width = 0 score

    return min(touch_score + vol_score + recency_score + tightness_score, 1.0)


def detect_sr_zones(
    df: pd.DataFrame,
    lookback: int = 100,
    left_bars: int = 5,
    right_bars: int = 5,
    cluster_distance_pct: float = 0.015,
    min_touches: int = 2,
    min_strength: float = 0.3
) -> List[SRZone]:
    """
    Detect Support/Resistance Zones from price action.

    Process:
    1. Find fractal pivots (swing highs/lows)
    2. Cluster nearby price levels
    3. Calculate strength for each cluster
    4. Filter by minimum strength
    5. Classify as support/resistance/both

    Args:
        df: DataFrame with OHLCV
        lookback: Bars to analyze
        left_bars: Left bars for fractal
        right_bars: Right bars for fractal
        cluster_distance_pct: Max distance for clustering (% of price)
        min_touches: Minimum touches for valid zone
        min_strength: Minimum strength score

    Returns:
        List of SRZone objects
    """
    data = df.tail(lookback).copy()
    if len(data) < left_bars + right_bars + 10:
        return []

    avg_volume = data['Volume'].mean()
    price_range = data['High'].max() - data['Low'].min()

    # Find pivots
    pivots = find_fractal_pivots(data, left_bars, right_bars, 'both')

    if pivots.empty:
        return []

    # Separate support and resistance pivots
    support_pivots = pivots[pivots['type'] == 'support']
    resistance_pivots = pivots[pivots['type'] == 'resistance']

    all_zones = []

    for pivot_type, pivot_df in [('support', support_pivots), ('resistance', resistance_pivots)]:
        if pivot_df.empty:
            continue

        prices = pivot_df['price'].values
        dates = pivot_df['date'].values
        indices = pivot_df['index'].values

        # Get volume at each pivot
        volumes = []
        for idx in pivot_df['index']:
            if idx < len(data):
                volumes.append(data['Volume'].iloc[idx])
            else:
                volumes.append(avg_volume)
        volumes = np.array(volumes)

        # Cluster nearby levels
        clusters = cluster_price_levels(
            prices, volumes,
            max_distance_pct=cluster_distance_pct,
            min_cluster_size=min_touches
        )

        for center, low, high, total_vol, count in clusters:
            # Get touch dates
            cluster_mask = (prices >= low) & (prices <= high)
            touch_dates = dates[cluster_mask]
            touch_indices = indices[cluster_mask]

            last_test = touch_dates.max() if len(touch_dates) > 0 else data.index[-1]
            first_test = touch_dates.min() if len(touch_dates) > 0 else data.index[0]
            days_since_last = (data.index[-1] - last_test).days if hasattr(data.index[-1], '__sub__') else 0

            # Calculate strength
            strength = calculate_zone_strength(
                center, low, high, count, total_vol, avg_volume,
                days_since_last, price_range
            )

            if strength < min_strength:
                continue

            # Determine if zone is broken
            current_price = data['Close'].iloc[-1]
            is_broken = False
            break_date = None

            if pivot_type == 'support' and current_price < low * 0.99:
                is_broken = True
            elif pivot_type == 'resistance' and current_price > high * 1.01:
                is_broken = True

            zone = SRZone(
                price=center,
                price_low=low,
                price_high=high,
                zone_type=pivot_type,
                strength=strength,
                touches=count,
                total_volume=total_vol,
                avg_volume_per_touch=total_vol / count if count > 0 else 0,
                last_test_date=last_test,
                first_test_date=first_test,
                is_broken=is_broken,
                break_date=break_date
            )
            all_zones.append(zone)

    # Merge overlapping support/resistance into 'both' type
    merged_zones = []
    for zone in all_zones:
        opposite_type = 'resistance' if zone.zone_type == 'support' else 'support'
        overlapping = [z for z in all_zones
                       if z.zone_type == opposite_type
                       and z.price_low <= zone.price_high
                       and z.price_high >= zone.price_low]

        if overlapping:
            for other in overlapping:
                merged_low = min(zone.price_low, other.price_low)
                merged_high = max(zone.price_high, other.price_high)
                merged_center = (merged_low + merged_high) / 2
                merged_strength = max(zone.strength, other.strength)
                merged_touches = zone.touches + other.touches
                merged_volume = zone.total_volume + other.total_volume

                merged = SRZone(
                    price=merged_center,
                    price_low=merged_low,
                    price_high=merged_high,
                    zone_type='both',
                    strength=merged_strength,
                    touches=merged_touches,
                    total_volume=merged_volume,
                    avg_volume_per_touch=merged_volume / merged_touches,
                    last_test_date=max(zone.last_test_date, other.last_test_date),
                    first_test_date=min(zone.first_test_date, other.first_test_date),
                    is_broken=zone.is_broken or other.is_broken
                )
                merged_zones.append(merged)
        else:
            merged_zones.append(zone)

    # Remove duplicates (keep stronger)
    final_zones = []
    for zone in merged_zones:
        is_dup = False
        for existing in final_zones:
            if (abs(zone.price - existing.price) / existing.price < 0.005 and
                zone.zone_type == existing.zone_type):
                if zone.strength > existing.strength:
                    final_zones.remove(existing)
                else:
                    is_dup = True
                break
        if not is_dup:
            final_zones.append(zone)

    return sorted(final_zones, key=lambda x: x.price)


def get_sr_signals(
    current_price: float,
    zones: List[SRZone],
    atr: float = None
) -> Dict:
    """
    Generate trading signals from S/R zones.

    Returns:
        Dict with signals and key levels
    """
    signals = {
        'nearest_support': None,
        'nearest_resistance': None,
        'at_support': False,
        'at_resistance': False,
        'near_support': False,
        'near_resistance': False,
        'support_distance_pct': None,
        'resistance_distance_pct': None,
        'signal': 'neutral',
        'strength': 0,
        'entry_zone': None,
        'stop_zone': None,
        'target_zones': []
    }

    if not zones:
        return signals

    # Separate supports and resistances
    supports = [z for z in zones if z.zone_type in ['support', 'both'] and not z.is_broken]
    resistances = [z for z in zones if z.zone_type in ['resistance', 'both'] and not z.is_broken]

    # Find nearest support (below current price)
    valid_supports = [z for z in supports if z.price_high < current_price * 1.001]
    if valid_supports:
        nearest_support = max(valid_supports, key=lambda z: z.price)
        signals['nearest_support'] = nearest_support
        dist = (current_price - nearest_support.price) / current_price * 100
        signals['support_distance_pct'] = dist

        tolerance = max(atr / current_price * 100 * 1.5, 0.5) if atr else 0.5
        if dist < tolerance:
            signals['at_support'] = True
            signals['signal'] = 'at_support'
            signals['strength'] = nearest_support.strength
            signals['entry_zone'] = nearest_support.price
            signals['stop_zone'] = nearest_support.price_low * 0.995
        elif dist < tolerance * 3:
            signals['near_support'] = True

    # Find nearest resistance (above current price)
    valid_resistances = [z for z in resistances if z.price_low > current_price * 0.999]
    if valid_resistances:
        nearest_resistance = min(valid_resistances, key=lambda z: z.price)
        signals['nearest_resistance'] = nearest_resistance
        dist = (nearest_resistance.price - current_price) / current_price * 100
        signals['resistance_distance_pct'] = dist

        tolerance = max(atr / current_price * 100 * 1.5, 0.5) if atr else 0.5
        if dist < tolerance:
            signals['at_resistance'] = True
            signals['signal'] = 'at_resistance'
            signals['strength'] = max(signals['strength'], nearest_resistance.strength)
        elif dist < tolerance * 3:
            signals['near_resistance'] = True

    # Target zones (next resistance levels for longs)
    if valid_resistances:
        targets = sorted(valid_resistances, key=lambda z: z.price)[:3]
        signals['target_zones'] = [
            {'price': z.price, 'strength': z.strength, 'type': z.zone_type}
            for z in targets
        ]

    return signals


def add_sr_zones_to_dataframe(
    df: pd.DataFrame,
    lookback: int = 100,
    left_bars: int = 5,
    right_bars: int = 5
) -> pd.DataFrame:
    """
    Add S/R zone columns to DataFrame (rolling calculation).
    Adds: SR_Support_1, SR_Support_2, SR_Resistance_1, SR_Resistance_2, SR_Signal, etc.
    """
    if isinstance(df.index, pd.MultiIndex):
        symbols = df.index.get_level_values('Symbol').unique()
        results = []

        for symbol in symbols:
            sym_df = df.xs(symbol, level='Symbol').copy()
            sym_df = _add_sr_rolling(sym_df, lookback, left_bars, right_bars)
            sym_df['Symbol'] = symbol
            results.append(sym_df)

        if results:
            combined = pd.concat(results)
            combined.set_index('Symbol', append=True, inplace=True)
            combined = combined.reorder_levels(['Symbol', combined.index.names[0]])
            combined.sort_index(inplace=True)
            return combined
        return df
    else:
        return _add_sr_rolling(df, lookback, left_bars, right_bars)


def _add_sr_rolling(
    df: pd.DataFrame,
    lookback: int,
    left_bars: int,
    right_bars: int
) -> pd.DataFrame:
    """Rolling S/R calculation."""
    compute_interval = 20

    sr_support_1 = [np.nan] * len(df)
    sr_support_2 = [np.nan] * len(df)
    sr_resistance_1 = [np.nan] * len(df)
    sr_resistance_2 = [np.nan] * len(df)
    sr_signal = ['none'] * len(df)
    sr_at_support = [False] * len(df)
    sr_at_resistance = [False] * len(df)

    last_zones = None

    for i in range(len(df)):
        if i < lookback:
            continue

        # Recompute periodically
        if i % 20 == 0 or last_zones is None:
            window_df = df.iloc[i-lookback:i+1]
            try:
                last_zones = detect_sr_zones(
                    window_df, lookback=lookback,
                    left_bars=left_bars, right_bars=right_bars
                )
            except Exception:
                last_zones = []

        if last_zones:
            current_price = df['Close'].iloc[i]
            atr = df.get('ATR_14', pd.Series([np.nan]*len(df))).iloc[i]

            signals = get_sr_signals(current_price, last_zones, atr)

            if signals['nearest_support']:
                sr_support_1[i] = signals['nearest_support'].price
                if signals['nearest_support'].zone_type == 'both':
                    sr_support_2[i] = signals['nearest_support'].price_high
                elif len([z for z in last_zones if z.zone_type in ['support','both'] and z.price < current_price]) > 1:
                    other_supports = [z for z in last_zones if z.zone_type in ['support','both'] and z.price < current_price and z != signals['nearest_support']]
                    if other_supports:
                        sr_support_2[i] = max(other_supports, key=lambda x: x.price).price

            if signals['nearest_resistance']:
                sr_resistance_1[i] = signals['nearest_resistance'].price
                if signals['nearest_resistance'].zone_type == 'both':
                    sr_resistance_2[i] = signals['nearest_resistance'].price_low
                elif len([z for z in last_zones if z.zone_type in ['resistance','both'] and z.price > current_price]) > 1:
                    other_resistances = [z for z in last_zones if z.zone_type in ['resistance','both'] and z.price > current_price and z != signals['nearest_resistance']]
                    if other_resistances:
                        sr_resistance_2[i] = min(other_resistances, key=lambda x: x.price).price

            sr_signal[i] = signals['signal']
            sr_at_support[i] = signals['at_support']
            sr_at_resistance[i] = signals['at_resistance']

    df['SR_Support_1'] = sr_support_1
    df['SR_Support_2'] = sr_support_2
    df['SR_Resistance_1'] = sr_resistance_1
    df['SR_Resistance_2'] = sr_resistance_2
    df['SR_Signal'] = sr_signal
    df['SR_At_Support'] = sr_at_support
    df['SR_At_Resistance'] = sr_at_resistance

    return df


if __name__ == "__main__":
    import yfinance as yf

    df = yf.download("RELIANCE.NS", period="6mo", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.capitalize() for c in df.columns]

    # Detect zones
    zones = detect_sr_zones(df, lookback=100)

    print(f"Found {len(zones)} S/R zones:")
    for z in zones:
        print(f"  {z.zone_type.upper():10} | {z.price:8.2f} ({z.price_low:.2f}-{z.price_high:.2f}) | "
              f"Strength: {z.strength:.2f} | Touches: {z.touches} | "
              f"Broken: {z.is_broken}")

    # Test signals
    current = df['Close'].iloc[-1]
    atr = 30
    signals = get_sr_signals(current, zones, atr)
    print(f"\nCurrent: {current:.2f}")
    print(f"Signal: {signals['signal']}")
    print(f"Nearest Support: {signals['nearest_support'].price if signals['nearest_support'] else 'None'}")
    print(f"Nearest Resistance: {signals['nearest_resistance'].price if signals['nearest_resistance'] else 'None'}")

    # Add to dataframe
    df_sr = add_sr_zones_to_dataframe(df)
    sr_cols = [c for c in df_sr.columns if c.startswith('SR_')]
    print(f"\nAdded columns: {sr_cols}")