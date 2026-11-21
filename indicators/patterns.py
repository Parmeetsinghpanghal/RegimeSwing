"""
Candlestick Pattern Detection
Top 15 statistically validated patterns for swing trading.
Each pattern returns: name, confidence (0-1), direction, target projection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PatternDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class CandlestickPattern:
    """Detected candlestick pattern."""
    name: str
    direction: PatternDirection
    confidence: float          # 0-1
    index: int                 # Bar index where pattern completed
    price: float               # Close price at completion
    target: Optional[float] = None      # Projected target
    stop: Optional[float] = None        # Suggested stop
    metadata: Dict = None              # Additional info


def get_candle_properties(row: pd.Series) -> Dict:
    """Extract candle properties for pattern detection."""
    open_p = row['Open']
    high = row['High']
    low = row['Low']
    close = row['Close']

    body = abs(close - open_p)
    upper_wick = high - max(close, open_p)
    lower_wick = min(close, open_p) - low
    total_range = high - low

    is_bullish = close > open_p
    is_bearish = close < open_p
    is_doji = body / total_range < 0.1 if total_range > 0 else False

    body_pct = body / total_range if total_range > 0 else 0
    upper_wick_pct = upper_wick / total_range if total_range > 0 else 0
    lower_wick_pct = lower_wick / total_range if total_range > 0 else 0

    return {
        'open': open_p, 'high': high, 'low': low, 'close': close,
        'body': body, 'upper_wick': upper_wick, 'lower_wick': lower_wick,
        'total_range': total_range, 'is_bullish': is_bullish,
        'is_bearish': is_bearish, 'is_doji': is_doji,
        'body_pct': body_pct, 'upper_wick_pct': upper_wick_pct,
        'lower_wick_pct': lower_wick_pct
    }


# ============================================================
# SINGLE CANDLE PATTERNS
# ============================================================

def detect_hammer(row: pd.Series, prev_row: pd.Series = None) -> Optional[CandlestickPattern]:
    """
    Hammer / Hanging Man
    - Small body, long lower wick (2-3x body), little/no upper wick
    - Bullish in downtrend (Hammer), Bearish in uptrend (Hanging Man)
    """
    props = get_candle_properties(row)
    if props['is_doji'] or props['total_range'] == 0:
        return None

    # Long lower wick, small upper wick, small body
    if (props['lower_wick'] >= 2 * props['body'] and
        props['upper_wick'] <= 0.5 * props['body'] and
        props['body_pct'] <= 0.3):

        # Determine direction based on trend (need context)
        # Default: Hammer = bullish reversal
        confidence = min(0.6 + props['lower_wick'] / props['total_range'] * 0.3, 0.85)

        return CandlestickPattern(
            name='Hammer',
            direction=PatternDirection.BULLISH,
            confidence=confidence,
            index=row.name if hasattr(row, 'name') else 0,
            price=row['Close'],
            target=row['High'] + props['body'],
            stop=row['Low'],
            metadata={'lower_wick_ratio': props['lower_wick'] / props['body']}
        )
    return None


def detect_shooting_star(row: pd.Series, prev_row: pd.Series = None) -> Optional[CandlestickPattern]:
    """
    Shooting Star / Inverted Hammer
    - Small body, long upper wick (2-3x body), little/no lower wick
    - Bearish in uptrend (Shooting Star), Bullish in downtrend (Inverted Hammer)
    """
    props = get_candle_properties(row)
    if props['is_doji'] or props['total_range'] == 0:
        return None

    if (props['upper_wick'] >= 2 * props['body'] and
        props['lower_wick'] <= 0.5 * props['body'] and
        props['body_pct'] <= 0.3):

        confidence = min(0.6 + props['upper_wick'] / props['total_range'] * 0.3, 0.85)

        return CandlestickPattern(
            name='Shooting_Star',
            direction=PatternDirection.BEARISH,
            confidence=confidence,
            index=row.name if hasattr(row, 'name') else 0,
            price=row['Close'],
            target=row['Low'] - props['body'],
            stop=row['High'],
            metadata={'upper_wick_ratio': props['upper_wick'] / props['body']}
        )
    return None


def detect_doji(row: pd.Series, prev_row: pd.Series = None) -> Optional[CandlestickPattern]:
    """
    Doji - Open ≈ Close, indicates indecision
    - Variations: Long-legged, Dragonfly, Gravestone
    """
    props = get_candle_properties(row)
    if not props['is_doji']:
        return None

    # Classify doji type
    if props['lower_wick'] > 3 * props['body'] and props['upper_wick'] < props['body']:
        # Dragonfly Doji - bullish reversal
        return CandlestickPattern(
            name='Dragonfly_Doji',
            direction=PatternDirection.BULLISH,
            confidence=0.55,
            index=row.name if hasattr(row, 'name') else 0,
            price=row['Close'],
            target=row['High'] + props['total_range'] * 0.5,
            stop=row['Low']
        )
    elif props['upper_wick'] > 3 * props['body'] and props['lower_wick'] < props['body']:
        # Gravestone Doji - bearish reversal
        return CandlestickPattern(
            name='Gravestone_Doji',
            direction=PatternDirection.BEARISH,
            confidence=0.55,
            index=row.name if hasattr(row, 'name') else 0,
            price=row['Close'],
            target=row['Low'] - props['total_range'] * 0.5,
            stop=row['High']
        )
    elif props['upper_wick'] > 2 * props['body'] and props['lower_wick'] > 2 * props['body']:
        # Long-legged Doji - high volatility, trend continuation likely
        return CandlestickPattern(
            name='Long_Legged_Doji',
            direction=PatternDirection.NEUTRAL,
            confidence=0.45,
            index=row.name if hasattr(row, 'name') else 0,
            price=row['Close'],
            metadata={'type': 'continuation'}
        )
    else:
        # Standard Doji - indecision
        return CandlestickPattern(
            name='Doji',
            direction=PatternDirection.NEUTRAL,
            confidence=0.4,
            index=row.name if hasattr(row, 'name') else 0,
            price=row['Close'],
            metadata={'type': 'indecision'}
        )
    return None


def detect_marubozu(row: pd.Series, prev_row: pd.Series = None) -> Optional[CandlestickPattern]:
    """
    Marubozu - No wicks, strong trend continuation
    - Full body, open=low & close=high (bullish) or open=high & close=low (bearish)
    """
    props = get_candle_properties(row)
    if props['total_range'] == 0:
        return None

    # Very small wicks
    if props['upper_wick_pct'] < 0.05 and props['lower_wick_pct'] < 0.05:
        if props['is_bullish']:
            return CandlestickPattern(
                name='Bullish_Marubozu',
                direction=PatternDirection.BULLISH,
                confidence=0.75,
                index=row.name if hasattr(row, 'name') else 0,
                price=row['Close'],
                target=row['Close'] + props['total_range'],
                stop=row['Open'],
                metadata={'strength': 'very_strong'}
            )
        else:
            return CandlestickPattern(
                name='Bearish_Marubozu',
                direction=PatternDirection.BEARISH,
                confidence=0.75,
                index=row.name if hasattr(row, 'name') else 0,
                price=row['Close'],
                target=row['Close'] - props['total_range'],
                stop=row['Open'],
                metadata={'strength': 'very_strong'}
            )
    return None


# ============================================================
# TWO CANDLE PATTERNS
# ============================================================

def detect_engulfing(curr: pd.Series, prev: pd.Series) -> Optional[CandlestickPattern]:
    """
    Bullish/Bearish Engulfing
    - Current body completely engulfs previous body
    - Opposite colors
    """
    curr_props = get_candle_properties(curr)
    prev_props = get_candle_properties(prev)

    if curr_props['is_doji'] or prev_props['is_doji']:
        return None

    # Bullish Engulfing: prev bearish, curr bullish, curr engulfs prev
    if (prev_props['is_bearish'] and curr_props['is_bullish'] and
        curr['Open'] < prev['Close'] and curr['Close'] > prev['Open']):

        engulf_ratio = curr_props['body'] / prev_props['body']
        confidence = min(0.65 + engulf_ratio * 0.1, 0.85)

        return CandlestickPattern(
            name='Bullish_Engulfing',
            direction=PatternDirection.BULLISH,
            confidence=confidence,
            index=curr.name if hasattr(curr, 'name') else 0,
            price=curr['Close'],
            target=curr['High'] + curr_props['body'],
            stop=prev['Low'],
            metadata={'engulf_ratio': engulf_ratio}
        )

    # Bearish Engulfing: prev bullish, curr bearish, curr engulfs prev
    if (prev_props['is_bullish'] and curr_props['is_bearish'] and
        curr['Open'] > prev['Close'] and curr['Close'] < prev['Open']):

        engulf_ratio = curr_props['body'] / prev_props['body']
        confidence = min(0.65 + engulf_ratio * 0.1, 0.85)

        return CandlestickPattern(
            name='Bearish_Engulfing',
            direction=PatternDirection.BEARISH,
            confidence=confidence,
            index=curr.name if hasattr(curr, 'name') else 0,
            price=curr['Close'],
            target=curr['Low'] - curr_props['body'],
            stop=prev['High'],
            metadata={'engulf_ratio': engulf_ratio}
        )
    return None


def detect_harami(curr: pd.Series, prev: pd.Series) -> Optional[CandlestickPattern]:
    """
    Harami - Current body inside previous body (opposite of engulfing)
    - Potential reversal, weaker than engulfing
    """
    curr_props = get_candle_properties(curr)
    prev_props = get_candle_properties(prev)

    if curr_props['is_doji'] or prev_props['is_doji']:
        return None

    # Bullish Harami: prev bearish, curr bullish, curr inside prev
    if (prev_props['is_bearish'] and curr_props['is_bullish'] and
        curr['Open'] > prev['Close'] and curr['Close'] < prev['Open']):

        return CandlestickPattern(
            name='Bullish_Harami',
            direction=PatternDirection.BULLISH,
            confidence=0.55,
            index=curr.name if hasattr(curr, 'name') else 0,
            price=curr['Close'],
            target=curr['High'] + curr_props['body'],
            stop=prev['Low'],
            metadata={'type': 'reversal'}
        )

    # Bearish Harami: prev bullish, curr bearish, curr inside prev
    if (prev_props['is_bullish'] and curr_props['is_bearish'] and
        curr['Open'] < prev['Close'] and curr['Close'] > prev['Open']):

        return CandlestickPattern(
            name='Bearish_Harami',
            direction=PatternDirection.BEARISH,
            confidence=0.55,
            index=curr.name if hasattr(curr, 'name') else 0,
            price=curr['Close'],
            target=curr['Low'] - curr_props['body'],
            stop=prev['High'],
            metadata={'type': 'reversal'}
        )
    return None


def detect_piercing_line(curr: pd.Series, prev: pd.Series) -> Optional[CandlestickPattern]:
    """
    Piercing Line (Bullish) / Dark Cloud Cover (Bearish)
    - Two candle reversal patterns
    """
    curr_props = get_candle_properties(curr)
    prev_props = get_candle_properties(prev)

    # Piercing Line: prev bearish, curr bullish, curr opens below prev low,
    # closes above 50% of prev body
    if (prev_props['is_bearish'] and curr_props['is_bullish'] and
        curr['Open'] < prev['Low'] and
        curr['Close'] > (prev['Open'] + prev['Close']) / 2 and
        curr['Close'] < prev['Open']):  # Not fully engulfing

        penetration = (curr['Close'] - prev['Close']) / prev_props['body']
        confidence = min(0.6 + penetration * 0.2, 0.8)

        return CandlestickPattern(
            name='Piercing_Line',
            direction=PatternDirection.BULLISH,
            confidence=confidence,
            index=curr.name if hasattr(curr, 'name') else 0,
            price=curr['Close'],
            target=curr['High'] + curr_props['body'],
            stop=prev['Low'],
            metadata={'penetration': penetration}
        )

    # Dark Cloud Cover: prev bullish, curr bearish, curr opens above prev high,
    # closes below 50% of prev body
    if (prev_props['is_bullish'] and curr_props['is_bearish'] and
        curr['Open'] > prev['High'] and
        curr['Close'] < (prev['Open'] + prev['Close']) / 2 and
        curr['Close'] > prev['Open']):

        penetration = (prev['Open'] - curr['Close']) / prev_props['body']
        confidence = min(0.6 + penetration * 0.2, 0.8)

        return CandlestickPattern(
            name='Dark_Cloud_Cover',
            direction=PatternDirection.BEARISH,
            confidence=confidence,
            index=curr.name if hasattr(curr, 'name') else 0,
            price=curr['Close'],
            target=curr['Low'] - curr_props['body'],
            stop=prev['High'],
            metadata={'penetration': penetration}
        )
    return None


# ============================================================
# THREE CANDLE PATTERNS
# ============================================================

def detect_morning_star(c1: pd.Series, c2: pd.Series, c3: pd.Series) -> Optional[CandlestickPattern]:
    """
    Morning Star (Bullish) / Evening Star (Bearish)
    - 3 candle reversal pattern
    - c1: Strong trend candle
    - c2: Small body (star) - gap down/up
    - c3: Strong opposite candle, closes well into c1 body
    """
    p1, p2, p3 = get_candle_properties(c1), get_candle_properties(c2), get_candle_properties(c3)

    # Morning Star: Bearish -> Small -> Bullish
    if (p1['is_bearish'] and p1['body_pct'] > 0.5 and
        p2['body_pct'] < 0.3 and
        p3['is_bullish'] and p3['body_pct'] > 0.5 and
        c2['High'] < c1['Low'] and  # Gap down
        c3['Close'] > (c1['Open'] + c1['Close']) / 2):  # Close above mid of c1

        return CandlestickPattern(
            name='Morning_Star',
            direction=PatternDirection.BULLISH,
            confidence=0.75,
            index=c3.name if hasattr(c3, 'name') else 0,
            price=c3['Close'],
            target=c3['High'] + p3['body'] * 2,
            stop=c2['Low'],
            metadata={'gap_down': True, 'c1_body': p1['body']}
        )

    # Evening Star: Bullish -> Small -> Bearish
    if (p1['is_bullish'] and p1['body_pct'] > 0.5 and
        p2['body_pct'] < 0.3 and
        p3['is_bearish'] and p3['body_pct'] > 0.5 and
        c2['Low'] > c1['High'] and  # Gap up
        c3['Close'] < (c1['Open'] + c1['Close']) / 2):  # Close below mid of c1

        return CandlestickPattern(
            name='Evening_Star',
            direction=PatternDirection.BEARISH,
            confidence=0.75,
            index=c3.name if hasattr(c3, 'name') else 0,
            price=c3['Close'],
            target=c3['Low'] - p3['body'] * 2,
            stop=c2['High'],
            metadata={'gap_up': True, 'c1_body': p1['body']}
        )
    return None


def detect_three_white_soldiers(c1: pd.Series, c2: pd.Series, c3: pd.Series) -> Optional[CandlestickPattern]:
    """
    Three White Soldiers (Bullish) / Three Black Crows (Bearish)
    - Three consecutive strong candles in same direction
    - Each opens within previous body, closes near high/low
    """
    p1, p2, p3 = get_candle_properties(c1), get_candle_properties(c2), get_candle_properties(c3)

    # Three White Soldiers: All bullish, each higher
    if (p1['is_bullish'] and p2['is_bullish'] and p3['is_bullish'] and
        p1['body_pct'] > 0.5 and p2['body_pct'] > 0.5 and p3['body_pct'] > 0.5 and
        c2['Open'] > c1['Open'] and c2['Open'] < c1['Close'] and
        c3['Open'] > c2['Open'] and c3['Open'] < c2['Close'] and
        c1['Close'] < c2['Close'] < c3['Close']):

        return CandlestickPattern(
            name='Three_White_Soldiers',
            direction=PatternDirection.BULLISH,
            confidence=0.8,
            index=c3.name if hasattr(c3, 'name') else 0,
            price=c3['Close'],
            target=c3['Close'] + (c3['Close'] - c1['Open']),
            stop=c1['Open'],
            metadata={'type': 'strong_continuation'}
        )

    # Three Black Crows: All bearish, each lower
    if (p1['is_bearish'] and p2['is_bearish'] and p3['is_bearish'] and
        p1['body_pct'] > 0.5 and p2['body_pct'] > 0.5 and p3['body_pct'] > 0.5 and
        c2['Open'] < c1['Open'] and c2['Open'] > c1['Close'] and
        c3['Open'] < c2['Open'] and c3['Open'] > c2['Close'] and
        c1['Close'] > c2['Close'] > c3['Close']):

        return CandlestickPattern(
            name='Three_Black_Crows',
            direction=PatternDirection.BEARISH,
            confidence=0.8,
            index=c3.name if hasattr(c3, 'name') else 0,
            price=c3['Close'],
            target=c3['Close'] - (c1['Open'] - c3['Close']),
            stop=c1['Open'],
            metadata={'type': 'strong_continuation'}
        )
    return None


# ============================================================
# PATTERN REGISTRY
# ============================================================

SINGLE_CANDLE_PATTERNS = [
    ('Hammer', detect_hammer),
    ('Shooting_Star', detect_shooting_star),
    ('Doji', detect_doji),
    ('Marubozu', detect_marubozu),
]

TWO_CANDLE_PATTERNS = [
    ('Engulfing', detect_engulfing),
    ('Harami', detect_harami),
    ('Piercing_Line', detect_piercing_line),
]

THREE_CANDLE_PATTERNS = [
    ('Morning_Star', detect_morning_star),
    ('Evening_Star', detect_morning_star),  # Same function handles both
    ('Three_White_Soldiers', detect_three_white_soldiers),
    ('Three_Black_Crows', detect_three_white_soldiers),  # Same function handles both
]

ALL_PATTERN_DETECTORS = {
    'single': SINGLE_CANDLE_PATTERNS,
    'two': TWO_CANDLE_PATTERNS,
    'three': THREE_CANDLE_PATTERNS,
}


def detect_all_patterns(df: pd.DataFrame) -> List[CandlestickPattern]:
    """
    Detect all candlestick patterns in DataFrame.

    Returns:
        List of CandlestickPattern objects sorted by index
    """
    patterns = []

    for i in range(len(df)):
        curr = df.iloc[i]

        # Single candle patterns
        for name, func in SINGLE_CANDLE_PATTERNS:
            try:
                pattern = func(curr)
                if pattern:
                    pattern.index = i
                    patterns.append(pattern)
            except Exception:
                pass

        # Two candle patterns
        if i >= 1:
            prev = df.iloc[i-1]
            for name, func in TWO_CANDLE_PATTERNS:
                try:
                    pattern = func(curr, prev)
                    if pattern:
                        pattern.index = i
                        patterns.append(pattern)
                except Exception:
                    pass

        # Three candle patterns
        if i >= 2:
            c1, c2 = df.iloc[i-2], df.iloc[i-1]
            for name, func in THREE_CANDLE_PATTERNS:
                try:
                    pattern = func(c1, c2, curr)
                    if pattern:
                        pattern.index = i
                        patterns.append(pattern)
                except Exception:
                    pass

    return patterns


def add_patterns_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pattern detection columns to DataFrame.
    Adds: Pattern_Name, Pattern_Direction, Pattern_Confidence, Pattern_Target, Pattern_Stop
    """
    if isinstance(df.index, pd.MultiIndex):
        symbols = df.index.get_level_values('Symbol').unique()
        results = []

        for symbol in symbols:
            sym_df = df.xs(symbol, level='Symbol').copy()
            sym_df = _add_patterns_rolling(sym_df)
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
        return _add_patterns_rolling(df)


def _add_patterns_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Add pattern columns (computed once for all data)."""
    patterns = detect_all_patterns(df)

    # Initialize columns
    pattern_name = [''] * len(df)
    pattern_direction = [''] * len(df)
    pattern_confidence = [0.0] * len(df)
    pattern_target = [np.nan] * len(df)
    pattern_stop = [np.nan] * len(df)
    pattern_count = [0] * len(df)

    # Count patterns per bar
    pattern_at_bar = {}
    for p in patterns:
        if p.index not in pattern_at_bar:
            pattern_at_bar[p.index] = []
        pattern_at_bar[p.index].append(p)

    for i in range(len(df)):
        if i in pattern_at_bar:
            bar_patterns = pattern_at_bar[i]
            pattern_count[i] = len(bar_patterns)

            # Keep highest confidence pattern
            best = max(bar_patterns, key=lambda x: x.confidence)
            pattern_name[i] = best.name
            pattern_direction[i] = best.direction.value
            pattern_confidence[i] = best.confidence
            pattern_target[i] = best.target if best.target else np.nan
            pattern_stop[i] = best.stop if best.stop else np.nan

    df['Pattern_Name'] = pattern_name
    df['Pattern_Direction'] = pattern_direction
    df['Pattern_Confidence'] = pattern_confidence
    df['Pattern_Target'] = pattern_target
    df['Pattern_Stop'] = pattern_stop
    df['Pattern_Count'] = pattern_count

    return df


def get_latest_patterns(df: pd.DataFrame, n_bars: int = 5) -> List[CandlestickPattern]:
    """Get patterns from last N bars."""
    patterns = detect_all_patterns(df)
    recent = [p for p in patterns if p.index >= len(df) - n_bars]
    return sorted(recent, key=lambda x: x.confidence, reverse=True)


# ============================================================
# PATTERN STATISTICS (for validation)
# ============================================================

def calculate_pattern_statistics(
    df: pd.DataFrame,
    forward_bars: int = 10
) -> pd.DataFrame:
    """
    Calculate win rate, avg return for each pattern type.
    Used for validating pattern effectiveness.
    """
    patterns = detect_all_patterns(df)
    stats = []

    for p in patterns:
        idx = p.index
        if idx + forward_bars >= len(df):
            continue

        entry_price = df['Close'].iloc[idx]
        future_high = df['High'].iloc[idx+1:idx+forward_bars+1].max()
        future_low = df['Low'].iloc[idx+1:idx+forward_bars+1].min()

        if p.direction == PatternDirection.BULLISH:
            hit_target = future_high >= (p.target if p.target else entry_price * 1.02)
            hit_stop = future_low <= (p.stop if p.stop else entry_price * 0.98)
            max_favorable = (future_high - entry_price) / entry_price
            max_adverse = (entry_price - future_low) / entry_price
        elif p.direction == PatternDirection.BEARISH:
            hit_target = future_low <= (p.target if p.target else entry_price * 0.98)
            hit_stop = future_high >= (p.stop if p.stop else entry_price * 1.02)
            max_favorable = (entry_price - future_low) / entry_price
            max_adverse = (future_high - entry_price) / entry_price
        else:
            continue

        stats.append({
            'pattern': p.name,
            'direction': p.direction.value,
            'confidence': p.confidence,
            'hit_target': hit_target,
            'hit_stop': hit_stop,
            'max_favorable': max_favorable,
            'max_adverse': max_adverse,
            'mfe_mae_ratio': max_favorable / max_adverse if max_adverse > 0 else 0
        })

    if not stats:
        return pd.DataFrame()

    stats_df = pd.DataFrame(stats)

    # Aggregate by pattern
    agg = stats_df.groupby(['pattern', 'direction']).agg(
        count=('pattern', 'count'),
        avg_confidence=('confidence', 'mean'),
        win_rate=('hit_target', 'mean'),
        stop_rate=('hit_stop', 'mean'),
        avg_mfe=('max_favorable', 'mean'),
        avg_mae=('max_adverse', 'mean'),
        avg_mfe_mae=('mfe_mae_ratio', 'mean')
    ).round(4)

    return agg.reset_index()


if __name__ == "__main__":
    import yfinance as yf

    df = yf.download("RELIANCE.NS", period="6mo", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.capitalize() for c in df.columns]

    # Detect patterns
    patterns = detect_all_patterns(df)
    print(f"Found {len(patterns)} patterns")

    # Show recent
    recent = get_latest_patterns(df, 10)
    for p in recent[:10]:
        print(f"  {p.name:20} | {p.direction.value:8} | Conf: {p.confidence:.2f} | "
              f"Idx: {p.index:3} | Target: {p.target:.2f if p.target else 'N/A':>7} | "
              f"Stop: {p.stop:.2f if p.stop else 'N/A':>7}")

    # Statistics
    print("\nCalculating pattern statistics...")
    stats = calculate_pattern_statistics(df, forward_bars=10)
    if not stats.empty:
        print(stats.to_string(index=False))

    # Add to dataframe
    df_pat = add_patterns_to_dataframe(df)
    pat_cols = [c for c in df_pat.columns if c.startswith('Pattern_')]
    print(f"\nAdded columns: {pat_cols}")