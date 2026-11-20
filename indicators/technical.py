"""
Technical Indicators Module - All indicators computed via pandas (no pandas_ta dependency).
Fixed for pandas >= 2.0 compatibility and Streamlit Cloud deployment.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)


def ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has (Symbol, Date) MultiIndex."""
    if isinstance(df.index, pd.MultiIndex):
        if df.index.nlevels == 2:
            return df
    if 'Symbol' in df.columns:
        df = df.set_index('Symbol', append=True).reorder_levels(['Symbol', df.index.names[0]])
    elif df.index.name == 'Symbol' or (isinstance(df.index, pd.MultiIndex) and 'Symbol' in df.index.names):
        pass
    else:
        df = df.copy()
        df['Symbol'] = 'UNKNOWN'
        df = df.set_index('Symbol', append=True).reorder_levels(['Symbol', df.index.names[0]])
    df.sort_index(inplace=True)
    return df


def get_symbol_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Extract single symbol data from MultiIndex DataFrame."""
    if isinstance(df.index, pd.MultiIndex) and 'Symbol' in df.index.names:
        return df.xs(symbol, level='Symbol')
    return df


def apply_per_symbol(df: pd.DataFrame, func, *args, **kwargs) -> pd.DataFrame:
    """Apply function per symbol and combine results."""
    if not isinstance(df.index, pd.MultiIndex):
        return func(df, *args, **kwargs)

    symbols = df.index.get_level_values('Symbol').unique()
    results = []

    for symbol in symbols:
        sym_df = df.xs(symbol, level='Symbol')
        try:
            result = func(sym_df, *args, **kwargs)
            result['Symbol'] = symbol
            results.append(result)
        except Exception as e:
            logger.warning(f"Indicator failed for {symbol}: {e}")

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results)
    combined.set_index('Symbol', append=True, inplace=True)
    combined = combined.reorder_levels(['Symbol', combined.index.names[0]])
    combined.sort_index(inplace=True)
    return combined


# ============================================================
# TREND INDICATORS
# ============================================================

def add_ema(df: pd.DataFrame, periods: List[int] = None, column: str = 'Close') -> pd.DataFrame:
    """Add Exponential Moving Averages."""
    if periods is None:
        periods = [9, 20, 50, 100, 200]

    def _ema(sdf):
        for p in periods:
            sdf[f'EMA_{p}'] = sdf[column].ewm(span=p, adjust=False).mean()
        return sdf

    return apply_per_symbol(df, _ema)


def add_sma(df: pd.DataFrame, periods: List[int] = None, column: str = 'Close') -> pd.DataFrame:
    """Add Simple Moving Averages."""
    if periods is None:
        periods = [20, 50, 200]

    def _sma(sdf):
        for p in periods:
            sdf[f'SMA_{p}'] = sdf[column].rolling(window=p, min_periods=1).mean()
        return sdf

    return apply_per_symbol(df, _sma)


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average Directional Index."""
    def _adx(sdf):
        high = sdf['High']
        low = sdf['Low']
        close = sdf['Close']

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smooth with Wilder's smoothing (EWM with alpha=1/period)
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=sdf.index).ewm(alpha=alpha, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=sdf.index).ewm(alpha=alpha, adjust=False).mean() / atr

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        sdf['ADX_14'] = dx.ewm(alpha=alpha, adjust=False).mean()
        return sdf

    return apply_per_symbol(df, _adx)


def add_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Add Supertrend indicator using pure-pandas."""
    def _supertrend(sdf):
        if len(sdf) < 10:
            return sdf

        high = sdf['High']
        low = sdf['Low']
        close = sdf['Close']

        # Calculate ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()

        # Supertrend calculation
        hl2 = (high + low) / 2
        basic_ub = hl2 + mult * atr
        basic_lb = hl2 - mult * atr

        final_ub = basic_ub.copy()
        final_lb = basic_lb.copy()

        for i in range(1, len(sdf)):
            if basic_ub.iloc[i] < final_ub.iloc[i-1] or close.iloc[i-1] > final_ub.iloc[i-1]:
                final_ub.iloc[i] = basic_ub.iloc[i]
            else:
                final_ub.iloc[i] = final_ub.iloc[i-1]

            if basic_lb.iloc[i] > final_lb.iloc[i-1] or close.iloc[i-1] < final_lb.iloc[i-1]:
                final_lb.iloc[i] = basic_lb.iloc[i]
            else:
                final_lb.iloc[i] = final_lb.iloc[i-1]

        supertrend = np.where(close <= final_lb, final_ub, final_lb)
        sdf['supertrend'] = supertrend
        sdf['supertrend_signal'] = np.where(close <= supertrend, 'buy', 'sell')

        # Also add Chandelier exits
        sdf['Chandelier_Long'] = close - (mult * atr)
        sdf['Chandelier_Short'] = close + (mult * atr)

        return sdf

    return apply_per_symbol(df, _supertrend)


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD using pure-pandas."""
    def _macd(sdf):
        fast_ema = sdf['Close'].ewm(span=fast, adjust=False).mean()
        slow_ema = sdf['Close'].ewm(span=slow, adjust=False).mean()
        sdf['MACD_12_26_9'] = fast_ema - slow_ema
        sdf['MACDs_12_26_9'] = sdf['MACD_12_26_9'].ewm(span=signal, adjust=False).mean()
        sdf['MACDh_12_26_9'] = sdf['MACD_12_26_9'] - sdf['MACDs_12_26_9']
        return sdf

    return apply_per_symbol(df, _macd)


# ============================================================
# MOMENTUM INDICATORS
# ============================================================

def add_rsi(df: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.DataFrame:
    """Add Relative Strength Index using pure-pandas."""
    def _rsi(sdf):
        delta = sdf[column].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's smoothing
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        sdf['RSI_14'] = 100 - (100 / (1 + rs))
        sdf['RSI_14'] = sdf['RSI_14'].fillna(50)
        return sdf

    return apply_per_symbol(df, _rsi)


def add_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    """Add Stochastic Oscillator using pure-pandas."""
    def _stoch(sdf):
        lowest_low = sdf['Low'].rolling(window=k, min_periods=1).min()
        highest_high = sdf['High'].rolling(window=k, min_periods=1).max()
        sdf['STOCH_K'] = 100 * (sdf['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        sdf['STOCH_D'] = sdf['STOCH_K'].rolling(window=d, min_periods=1).mean()
        return sdf

    return apply_per_symbol(df, _stoch)


def add_roc(df: pd.DataFrame, period: int = 10, column: str = 'Close') -> pd.DataFrame:
    """Add Rate of Change using pure-pandas."""
    def _roc(sdf):
        sdf[f'ROC_{period}'] = (sdf[column] / sdf[column].shift(period) - 1) * 100
        return sdf

    return apply_per_symbol(df, _roc)


# ============================================================
# VOLATILITY INDICATORS
# ============================================================

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range using pure-pandas."""
    def _atr(sdf):
        high_low = sdf['High'] - sdf['Low']
        high_close = (sdf['High'] - sdf['Close'].shift(1)).abs()
        low_close = (sdf['Low'] - sdf['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        sdf['ATR_14'] = tr.rolling(window=period, min_periods=1).mean()
        return sdf

    return apply_per_symbol(df, _atr)


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands using pure-pandas."""
    def _bb(sdf):
        sma = sdf['Close'].rolling(window=period, min_periods=1).mean()
        std_dev = sdf['Close'].rolling(window=period, min_periods=1).std()
        sdf['BB_upper'] = sma + (std * std_dev)
        sdf['BB_lower'] = sma - (std * std_dev)
        sdf['BB_middle'] = sma
        sdf['BB_width'] = (sdf['BB_upper'] - sdf['BB_lower']) / sma
        return sdf

    return apply_per_symbol(df, _bb)


def add_keltner_channel(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """Add Keltner Channels using pure-pandas."""
    def _kc(sdf):
        sdf['KC_middle'] = sdf['Close'].rolling(window=period, min_periods=1).mean()
        high_low = sdf['High'] - sdf['Low']
        high_close = (sdf['High'] - sdf['Close'].shift(1)).abs()
        low_close = (sdf['Low'] - sdf['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        sdf['KC_upper'] = sdf['KC_middle'] + (mult * atr)
        sdf['KC_lower'] = sdf['KC_middle'] - (mult * atr)
        return sdf

    return apply_per_symbol(df, _kc)


def add_donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Donchian Channel using pure-pandas."""
    def _dc(sdf):
        sdf['DC_upper'] = sdf['High'].rolling(window=period, min_periods=1).max()
        sdf['DC_lower'] = sdf['Low'].rolling(window=period, min_periods=1).min()
        sdf['DC_middle'] = (sdf['DC_upper'] + sdf['DC_lower']) / 2
        return sdf

    return apply_per_symbol(df, _dc)


# ============================================================
# VOLUME INDICATORS
# ============================================================

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume using pure-pandas."""
    def _obv(sdf):
        sdf['OBV'] = (np.sign(sdf['Close'].diff()) * sdf['Volume']).fillna(0).cumsum()
        return sdf

    return apply_per_symbol(df, _obv)


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add Volume Weighted Average Price using pure-pandas."""
    def _vwap(sdf):
        typical_price = (sdf['High'] + sdf['Low'] + sdf['Close']) / 3
        sdf['VWAP'] = (typical_price * sdf['Volume']).cumsum() / sdf['Volume'].cumsum()
        return sdf

    return apply_per_symbol(df, _vwap)


def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Volume Simple Moving Average using pure-pandas."""
    def _vsma(sdf):
        sdf['Volume_SMA_20'] = sdf['Volume'].rolling(window=period, min_periods=1).mean()
        sdf['Volume_Ratio'] = sdf['Volume'] / sdf['Volume_SMA_20'].replace(0, np.nan)
        return sdf

    return apply_per_symbol(df, _vsma)


# ============================================================
# ADVANCED / COMPOSITE INDICATORS
# ============================================================

def add_vwap_bands(df: pd.DataFrame, period: int = 20, std_mult: float = 1.0) -> pd.DataFrame:
    """Add VWAP with Standard Deviation Bands using pure-pandas."""
    def _vwap_bands(sdf):
        typical = (sdf['High'] + sdf['Low'] + sdf['Close']) / 3
        sdf['VWAP'] = (typical * sdf['Volume']).rolling(window=period, min_periods=1).sum() / sdf['Volume'].rolling(window=period, min_periods=1).sum()
        vwap_std = sdf['Close'].rolling(window=period, min_periods=1).std()
        sdf['VWAP_Upper_1SD'] = sdf['VWAP'] + (std_mult * vwap_std)
        sdf['VWAP_Lower_1SD'] = sdf['VWAP'] - (std_mult * vwap_std)
        return sdf

    return apply_per_symbol(df, _vwap_bands)


def add_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """Add Fibonacci retracement levels using pure-pandas."""
    def _fib(sdf):
        high = sdf['High'].rolling(window=lookback, min_periods=1).max()
        low = sdf['Low'].rolling(window=lookback, min_periods=1).min()
        diff = high - low

        sdf['Fib_0.236'] = high - 0.236 * diff
        sdf['Fib_0.382'] = high - 0.382 * diff
        sdf['Fib_0.500'] = high - 0.500 * diff
        sdf['Fib_0.618'] = high - 0.618 * diff
        sdf['Fib_0.786'] = high - 0.786 * diff
        return sdf

    return apply_per_symbol(df, _fib)


def add_kase_permission_stochastic(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Kase Permission Stochastic using pure-pandas."""
    def _kps(sdf):
        sdf['Kase_Permission'] = sdf['Close'].rolling(window=period, min_periods=1).mean() / sdf['High'].rolling(window=period, min_periods=1).max() * 100
        sdf['Kase_Stress'] = 100 - sdf['Kase_Permission']
        return sdf

    return apply_per_symbol(df, _kps)


# ============================================================
# OPTIONS-SPECIFIC INDICATORS
# ============================================================

def calculate_iv_rank(iv_series: pd.Series, lookback: int = 252) -> pd.Series:
    """Calculate IV Rank (current IV percentile over lookback)."""
    return iv_series.rolling(lookback, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() > x.min() else 50
    )


def calculate_iv_percentile(iv_series: pd.Series, lookback: int = 252) -> pd.Series:
    """Calculate IV Percentile (percentile rank of current IV)."""
    return iv_series.rolling(lookback, min_periods=1).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x) * 100
    )


# ============================================================
# MASTER FUNCTION: Add All Indicators
# ============================================================

def add_all_indicators(
    df: pd.DataFrame,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Add all technical indicators to DataFrame.

    Args:
        df: DataFrame with OHLCV data, MultiIndex (Symbol, Date) or single symbol
        config: Configuration dict with indicator parameters

    Returns:
        DataFrame with all indicators added
    """
    if config is None:
        config = {}

    # Ensure MultiIndex
    df = ensure_multiindex(df)

    # Get indicator config - FIX: was using undefined ind_config
    ind_config = config.get('indicators', {})

    logger.info(f"Adding indicators for {df.index.get_level_values('Symbol').nunique()} symbols...")

    # Trend
    df = add_ema(df, ind_config.get('ema_periods', [9, 20, 50, 100, 200]))
    df = add_sma(df, ind_config.get('sma_periods', [20, 50, 200]))
    df = add_adx(df, ind_config.get('adx_period', 14))
    df = add_supertrend(
        df,
        ind_config.get('supertrend_period', 10),
        ind_config.get('supertrend_mult', 3.0)
    )
    df = add_macd(
        df,
        ind_config.get('macd_fast', 12),
        ind_config.get('macd_slow', 26),
        ind_config.get('macd_signal', 9)
    )

    # Momentum
    df = add_rsi(df, ind_config.get('rsi_period', 14))
    df = add_stochastic(df, ind_config.get('stoch_k', 14), ind_config.get('stoch_d', 3))
    df = add_roc(df, ind_config.get('roc_period', 10))

    # Volatility
    df = add_atr(df, ind_config.get('atr_period', 14))
    df = add_bollinger_bands(df, ind_config.get('bb_period', 20), ind_config.get('bb_std', 2.0))
    df = add_keltner_channel(
        df,
        ind_config.get('keltner_period', 20),
        ind_config.get('keltner_mult', 2.0)
    )
    df = add_donchian_channel(df, ind_config.get('donchian_period', 20))

    # Volume
    df = add_obv(df)
    df = add_vwap(df)
    df = add_volume_sma(df, ind_config.get('volume_sma_period', 20))

    # Advanced indicators
    df = add_vwap_bands(df)
    df = add_fibonacci_levels(df)
    df = add_kase_permission_stochastic(df)

    logger.info("All indicators added successfully")
    return df


# ============================================================
# INCREMENTAL CALCULATION (for real-time updates)
# ============================================================

class IncrementalIndicators:
    """Calculate indicators incrementally for new bars (avoid full recompute)."""

    def __init__(self, config: dict):
        self.config = config
        self.ind_config = config.get('indicators', {})
        self.state = {}  # Symbol -> indicator state

    def update(self, symbol: str, new_bar: pd.Series) -> Dict:
        """Update indicators with new bar. Returns dict of new indicator values."""
        return {}

    def get_latest(self, symbol: str) -> Dict:
        """Get latest indicator values for symbol."""
        return self.state.get(symbol, {})


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Test with sample data
    df = pd.DataFrame({
        'Close': np.random.random(100),
        'High': np.random.random(100),
        'Low': np.random.random(100),
        'Open': np.random.random(100),
        'Volume': np.random.random(100)
    })
    df['Symbol'] = 'TEST'
    df.set_index('Symbol', append=True, inplace=True)
    df = df.reorder_levels(['Symbol', 'Date'])

    config = {
        'indicators': {
            'ema_periods': [20, 50, 200],
            'rsi_period': 14,
            'atr_period': 14,
            'bb_period': 20,
        }
    }

    df_with_ind = add_all_indicators(df, config)
    print(f"Columns: {df_with_ind.columns.tolist()}")
    print(df_with_ind.tail())