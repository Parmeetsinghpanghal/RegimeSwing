"""
Volume Profile & TPO (Time Price Opportunity) Analysis
Professional-grade volume profile for identifying institutional zones.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfile:
    """Volume Profile results."""
    poc: float                    # Point of Control (highest volume price)
    vah: float                    # Value Area High (70% volume)
    val: float                    # Value Area Low (70% volume)
    hvn: List[float]              # High Volume Nodes
    lvn: List[float]              # Low Volume Nodes
    single_prints: List[float]    # Single print prices (excess)
    profile: pd.DataFrame         # Full profile: price, volume, tpo_count
    value_area_volume: float      # 70% of total volume
    total_volume: float
    price_bins: np.ndarray


@dataclass
class TPOProfile:
    """TPO (Time Price Opportunity) Profile results."""
    poc: float
    vah: float
    val: float
    initial_balance_high: float   # First 60-90 min range high
    initial_balance_low: float    # First 60-90 min range low
    tpo_profile: pd.DataFrame     # price, tpo_count, time_periods
    total_tpos: int


def calculate_volume_profile(
    df: pd.DataFrame,
    lookback: int = 50,
    num_bins: int = 50,
    value_area_pct: float = 0.70
) -> VolumeProfile:
    """
    Calculate Volume Profile for given lookback period.

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of bars to include
        num_bins: Number of price bins
        value_area_pct: Value area percentage (default 70%)

    Returns:
        VolumeProfile object
    """
    # Use recent data
    data = df.tail(lookback).copy()

    if len(data) < 10:
        raise ValueError("Insufficient data for volume profile")

    # Price range
    price_min = data['Low'].min()
    price_max = data['High'].max()

    # Create price bins
    bins = np.linspace(price_min, price_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Distribute volume across bins for each bar
    profile_volumes = np.zeros(num_bins)

    for _, row in data.iterrows():
        typical_price = (row['High'] + row['Low'] + row['Close']) / 3
        # Find which bin the typical price falls into
        bin_idx = np.searchsorted(bins, typical_price) - 1
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)
        profile_volumes[bin_idx] += row['Volume']

    # Also distribute volume across the bar's range (more accurate)
    for _, row in data.iterrows():
        bar_bins = np.searchsorted(bins, [row['Low'], row['High']])
        start_bin = max(0, bar_bins[0] - 1)
        end_bin = min(num_bins, bar_bins[1])
        if end_bin > start_bin:
            vol_per_bin = row['Volume'] / (end_bin - start_bin)
            profile_volumes[start_bin:end_bin] += vol_per_bin

    # Create profile DataFrame
    profile_df = pd.DataFrame({
        'price': bin_centers,
        'volume': profile_volumes
    })
    profile_df = profile_df[profile_df['volume'] > 0].sort_values('price')

    if profile_df.empty:
        raise ValueError("Empty volume profile")

    total_volume = profile_df['volume'].sum()
    target_va_volume = total_volume * value_area_pct

    # Find POC (Point of Control)
    poc_idx = profile_df['volume'].idxmax()
    poc_price = profile_df.loc[poc_idx, 'price']

    # Find Value Area (70% volume around POC)
    poc_position = profile_df.index.get_loc(poc_idx)

    va_volume = profile_df.loc[poc_idx, 'volume']
    lower_idx = poc_position
    upper_idx = poc_position

    while va_volume < target_va_volume:
        lower_vol = profile_df.iloc[lower_idx - 1]['volume'] if lower_idx > 0 else 0
        upper_vol = profile_df.iloc[upper_idx + 1]['volume'] if upper_idx < len(profile_df) - 1 else 0

        if lower_vol >= upper_vol and lower_idx > 0:
            lower_idx -= 1
            va_volume += lower_vol
        elif upper_idx < len(profile_df) - 1:
            upper_idx += 1
            va_volume += upper_vol
        else:
            break

    vah_price = profile_df.iloc[upper_idx]['price']
    val_price = profile_df.iloc[lower_idx]['price']

    # High Volume Nodes (local maxima in volume)
    hvn_prices = []
    volumes = profile_df['volume'].values
    prices = profile_df['price'].values

    for i in range(1, len(volumes) - 1):
        if volumes[i] > volumes[i-1] and volumes[i] > volumes[i+1]:
            if volumes[i] > volumes.mean() * 1.5:
                hvn_prices.append(prices[i])

    # Low Volume Nodes (local minima)
    lvn_prices = []
    for i in range(1, len(volumes) - 1):
        if volumes[i] < volumes[i-1] and volumes[i] < volumes[i+1]:
            if volumes[i] < volumes.mean() * 0.5:
                lvn_prices.append(prices[i])

    # Single prints (excess) - prices with very low volume at extremes
    single_prints = []
    top_threshold = profile_df['price'].quantile(0.9)
    bottom_threshold = profile_df['price'].quantile(0.1)

    top_low_vol = profile_df[
        (profile_df['price'] >= top_threshold) &
        (profile_df['volume'] < volumes.mean() * 0.3)
    ]['price'].tolist()

    bottom_low_vol = profile_df[
        (profile_df['price'] <= bottom_threshold) &
        (profile_df['volume'] < volumes.mean() * 0.3)
    ]['price'].tolist()

    single_prints = top_low_vol + bottom_low_vol

    return VolumeProfile(
        poc=poc_price,
        vah=vah_price,
        val=val_price,
        hvn=hvn_prices,
        lvn=lvn_prices,
        single_prints=single_prints,
        profile=profile_df,
        value_area_volume=va_volume,
        total_volume=total_volume,
        price_bins=bins
    )


def calculate_tpo_profile(
    df: pd.DataFrame,
    lookback: int = 10,
    tick_size: float = 0.05,
    ib_periods: int = 6  # First 60-90 min range
) -> TPOProfile:
    """
    Calculate TPO (Time Price Opportunity) Profile.
    Used for Market Profile analysis.

    Args:
        df: DataFrame with OHLCV (intraday timeframe)
        lookback: Number of sessions/days
        tick_size: Minimum price increment
        ib_periods: Number of periods for Initial Balance

    Returns:
        TPOProfile object
    """
    # For daily charts, we can't do true TPO without intraday data
    # This is a simplified version using daily bars
    data = df.tail(lookback).copy()

    if len(data) < 2:
        raise ValueError("Insufficient data for TPO")

    price_min = data['Low'].min()
    price_max = data['High'].max()
    bins = np.arange(price_min, price_max + tick_size, tick_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    tpo_counts = np.zeros(len(bin_centers))

    # Each bar = 1 TPO at each price level it touched
    for _, row in data.iterrows():
        bar_bins = np.searchsorted(bins, [row['Low'], row['High']])
        start_bin = max(0, bar_bins[0] - 1)
        end_bin = min(len(bin_centers), bar_bins[1])
        tpo_counts[start_bin:end_bin] += 1

    tpo_df = pd.DataFrame({
        'price': bin_centers,
        'tpo_count': tpo_counts
    })
    tpo_df = tpo_df[tpo_df['tpo_count'] > 0].sort_values('price')

    if tpo_df.empty:
        raise ValueError("Empty TPO profile")

    # POC
    poc_idx = tpo_df['tpo_count'].idxmax()
    poc_price = tpo_df.loc[poc_idx, 'price']

    # Value Area (70% of TPOs)
    total_tpos = tpo_df['tpo_count'].sum()
    target_va = total_tpos * 0.70

    poc_pos = tpo_df.index.get_loc(poc_idx)
    va_tpos = tpo_df.loc[poc_idx, 'tpo_count']
    lower_idx = poc_pos
    upper_idx = poc_pos

    while va_tpos < target_va:
        lower_tpos = tpo_df.iloc[lower_idx - 1]['tpo_count'] if lower_idx > 0 else 0
        upper_tpos = tpo_df.iloc[upper_idx + 1]['tpo_count'] if upper_idx < len(tpo_df) - 1 else 0

        if lower_tpos >= upper_tpos and lower_idx > 0:
            lower_idx -= 1
            va_tpos += lower_tpos
        elif upper_idx < len(tpo_df) - 1:
            upper_idx += 1
            va_tpos += upper_tpos
        else:
            break

    vah_price = tpo_df.iloc[upper_idx]['price']
    val_price = tpo_df.iloc[lower_idx]['price']

    # Initial Balance (first ib_periods)
    ib_high = data.head(ib_periods)['High'].max()
    ib_low = data.head(ib_periods)['Low'].min()

    return TPOProfile(
        poc=poc_price,
        vah=vah_price,
        val=val_price,
        initial_balance_high=ib_high,
        initial_balance_low=ib_low,
        tpo_profile=tpo_df,
        total_tpos=int(total_tpos)
    )


def calculate_composite_volume_profile(
    df: pd.DataFrame,
    session_lookback: int = 20,
    num_bins: int = 100
) -> VolumeProfile:
    """
    Calculate composite volume profile across multiple sessions.
    More robust for identifying long-term institutional zones.
    """
    return calculate_volume_profile(df, lookback=session_lookback, num_bins=num_bins)


def get_volume_profile_signals(
    current_price: float,
    vp: VolumeProfile,
    tp: Optional[TPOProfile] = None
) -> Dict:
    """
    Generate trading signals from Volume Profile.

    Returns:
        Dict with signal type, strength, and levels
    """
    signals = {
        'at_poc': False,
        'at_vah': False,
        'at_val': False,
        'near_hvn': False,
        'near_lvn': False,
        'in_value_area': False,
        'above_value_area': False,
        'below_value_area': False,
        'poc_distance_pct': 0,
        'vah_distance_pct': 0,
        'val_distance_pct': 0,
        'signal': 'neutral',
        'strength': 0
    }

    # Distances as percentages
    signals['poc_distance_pct'] = (current_price - vp.poc) / vp.poc * 100
    signals['vah_distance_pct'] = (current_price - vp.vah) / vp.vah * 100
    signals['val_distance_pct'] = (current_price - vp.val) / vp.val * 100

    # Check positions
    tolerance = 0.005  # 0.5% tolerance

    if abs(signals['poc_distance_pct']) < tolerance * 100:
        signals['at_poc'] = True
        signals['signal'] = 'poc_test'
        signals['strength'] = 0.7

    if abs(signals['vah_distance_pct']) < tolerance * 100:
        signals['at_vah'] = True
        signals['signal'] = 'vah_rejection'
        signals['strength'] = 0.8

    if abs(signals['val_distance_pct']) < tolerance * 100:
        signals['at_val'] = True
        signals['signal'] = 'val_support'
        signals['strength'] = 0.8

    # Check HVN/LVN proximity
    for hvn in vp.hvn:
        dist = abs(current_price - hvn) / hvn
        if dist < tolerance:
            signals['near_hvn'] = True
            signals['signal'] = 'hvn_magnet'
            signals['strength'] = max(signals['strength'], 0.75)
            break

    for lvn in vp.lvn:
        dist = abs(current_price - lvn) / lvn
        if dist < tolerance:
            signals['near_lvn'] = True
            signals['signal'] = 'lvn_gap'
            signals['strength'] = max(signals['strength'], 0.7)
            break

    # Value Area position
    if vp.val <= current_price <= vp.vah:
        signals['in_value_area'] = True
    elif current_price > vp.vah:
        signals['above_value_area'] = True
        signals['signal'] = 'breakout_above_va'
        signals['strength'] = 0.6
    elif current_price < vp.val:
        signals['below_value_area'] = True
        signals['signal'] = 'breakdown_below_va'
        signals['strength'] = 0.6

    return signals


def add_volume_profile_to_dataframe(
    df: pd.DataFrame,
    lookback: int = 50,
    num_bins: int = 50
) -> pd.DataFrame:
    """
    Add Volume Profile columns to DataFrame (rolling calculation).
    Adds: VP_POC, VP_VAH, VP_VAL, VP_HVN_1, VP_HVN_2, VP_LVN_1, etc.
    """
    if isinstance(df.index, pd.MultiIndex):
        symbols = df.index.get_level_values('Symbol').unique()
        results = []

        for symbol in symbols:
            sym_df = df.xs(symbol, level='Symbol').copy()
            sym_df = _add_vp_rolling(sym_df, lookback, num_bins)
            sym_df['Symbol'] = symbol
            results.append(sym_df)

        if results:
            combined = pd.concat(results)
            combined = combined.set_index('Symbol', append=True)
            combined = combined.reorder_levels(['Symbol', combined.index.names[0]])
            combined = combined.sort_index()
            return combined
        return df
    else:
        return _add_vp_rolling(df, lookback, num_bins)


def _add_vp_rolling(
    df: pd.DataFrame,
    lookback: int,
    num_bins: int
) -> pd.DataFrame:
    """Rolling volume profile calculation."""
    vp_poc = []
    vp_vah = []
    vp_val = []
    vp_hvn1 = []
    vp_hvn2 = []
    vp_lvn1 = []
    vp_lvn2 = []
    vp_signal = []

    for i in range(len(df)):
        if i < lookback:
            vp_poc.append(np.nan)
            vp_vah.append(np.nan)
            vp_val.append(np.nan)
            vp_hvn1.append(np.nan)
            vp_hvn2.append(np.nan)
            vp_lvn1.append(np.nan)
            vp_lvn2.append(np.nan)
            vp_signal.append('insufficient_data')
        else:
            try:
                window_df = df.iloc[i-lookback:i+1]
                vp = calculate_volume_profile(window_df, lookback=lookback, num_bins=num_bins)

                vp_poc.append(vp.poc)
                vp_vah.append(vp.vah)
                vp_val.append(vp.val)
                vp_hvn1.append(vp.hvn[0] if len(vp.hvn) > 0 else np.nan)
                vp_hvn2.append(vp.hvn[1] if len(vp.hvn) > 1 else np.nan)
                vp_lvn1.append(vp.lvn[0] if len(vp.lvn) > 0 else np.nan)
                vp_lvn2.append(vp.lvn[1] if len(vp.lvn) > 1 else np.nan)

                # Signal for current price
                current_price = df['Close'].iloc[i]
                sig = get_volume_profile_signals(current_price, vp)
                vp_signal.append(sig['signal'])

            except Exception:
                vp_poc.append(np.nan)
                vp_vah.append(np.nan)
                vp_val.append(np.nan)
                vp_hvn1.append(np.nan)
                vp_hvn2.append(np.nan)
                vp_lvn1.append(np.nan)
                vp_lvn2.append(np.nan)
                vp_signal.append('error')

    df['VP_POC'] = vp_poc
    df['VP_VAH'] = vp_vah
    df['VP_VAL'] = vp_val
    df['VP_HVN_1'] = vp_hvn1
    df['VP_HVN_2'] = vp_hvn2
    df['VP_LVN_1'] = vp_lvn1
    df['VP_LVN_2'] = vp_lvn2
    df['VP_Signal'] = vp_signal

    return df


if __name__ == "__main__":
    import yfinance as yf

    # Test with RELIANCE
    df = yf.download("RELIANCE.NS", period="6mo", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.capitalize() for c in df.columns]

    # Calculate volume profile
    vp = calculate_volume_profile(df, lookback=50)

    print(f"POC: {vp.poc:.2f}")
    print(f"VAH: {vp.vah:.2f}")
    print(f"VAL: {vp.val:.2f}")
    print(f"HVN: {[f'{h:.2f}' for h in vp.hvn]}")
    print(f"LVN: {[f'{l:.2f}' for l in vp.lvn]}")
    print(f"Single Prints: {[f'{s:.2f}' for s in vp.single_prints[:5]]}")

    # Test signals
    current = df['Close'].iloc[-1]
    sig = get_volume_profile_signals(current, vp)
    print(f"\nCurrent: {current:.2f}")
    print(f"Signal: {sig}")

    # Add to dataframe
    df_with_vp = add_volume_profile_to_dataframe(df)
    print(f"\nColumns added: {[c for c in df_with_vp.columns if c.startswith('VP_')]}")