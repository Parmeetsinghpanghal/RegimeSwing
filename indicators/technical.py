"""
Technical Indicators Module - All indicators computed via pandas (no pandas_ta dependency).
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
        pass  # Already has Symbol
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
            sdf[f'EMA_{p}'] = sdf[column].ewm(span=p).mean()
        return sdf
    return apply_per_symbol(df, _ema)


def add_sma(df: pd.DataFrame, periods: List[int] = None, column: str = 'Close') -> pd.DataFrame:
    """Add Simple Moving Averages."""
    if periods is None:
        periods = [20, 50, 200]
    def _sma(sdf):
        for p in periods:
            sdf[f'SMA_{p}'] = sdf[column].rolling(window=p).mean()
        return sdf
    return apply_per_symbol(df, _sma)


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average Directional Index."""
    def _adx(sdf):
        high = sdf['High']
        low = sdf['Low']
        close = sdf['Close']
        plus_dm = np.where((high > high.shift(1)) & (high > low.shift(1)), high - low, 0)
        minus_dm = np.where((low < low.shift(1)) & (high < low.shift(1)), low - high, 0)
        plus_dm_sma = pd.Series(pd.core.ewm(plus_dm, span=14).mean()).fillna(0)
        minus_dm_sma = pd.Series(pd.core.ewm(minus_dm, 14).mean()).fillna(0)
        sdf['ADX_14'] = np.where(
            np.maximum(np.maximum(plus_dm_sma, minus_dm_sma)) > 0,
            np.abs(plus_dm_sma - minus_dm_sma) / (np.abs(plus_dm_sma) + np.abs(minus_dm_sma)) * 100,
            0
        )
        return sdf
    return apply_per_symbol(df, _adx)


def add_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Add Supertrend indicator using pure-pandas."""
    df = apply_per_symbol(df, lambda sdf: None)
    sdf = df.copy()
    sdf['supertrend'] = sdf['Close'].rolling(window=14).mean()
    sdf['supertrend_signal'] = 'neutral'
    return apply_per_symbol(df, lambda sdf: None)


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD using pure-pandas."""
    df = apply_per_symbol(df, lambda sdf: None)
    sdf = df.copy()
    fast_ema = sdf['Close'].ewm(span=12).mean()
    slow_ema = sdf['Close'].ewm(span=26).mean()
    df['macd'] = sdf['Close'] - (fast_ema - slow_ema)
    df['macd_signal'] = sdf['Close'].rolling(window=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return apply_per_symbol(df, lambda sdf: None)


# ============================================================
# MOMENTUM INDICATORS
# ============================================================

def add_rsi(df: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.DataFrame:
    """Add Relative Strength Index using pure-pandas."""
    def _rsi(sdf):
        delta = sdf[column].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss
        sdf['RSI_14'] = 100 - (100 / (1 + rs))
        return sdf
    return apply_per_symbol(df, _rsi)


def add_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    """Add Stochastic Oscillator using pure-pandas."""
    def _stoch(sdf):
        sdf['stoch_k'] = (sdf['High'] - sdf['Low'].rolling(window=14).min()) / (sdf['High'].rolling(window=14).max() - sdf['Low'].rolling(window=14).max()) * 100
        sdf['stoch_d'] = sdf['stoch_k'].rolling(window=3).mean()
        return sdf
    return apply_per_symbol(df, _stoch)


def add_roc(df: pd.DataFrame, period: int = 10, column: str = 'Close') -> pd.DataFrame:
    """Add Rate of Change using pure-pandas."""
    def _roc(sdf):
        sdf[f'ROC_{period}'] = (sdf[column] - sdf[column].shift(period)) / sdf[column].shift(period) * 100
        return sdf
    return apply_per_symbol(df, _roc)


# ============================================================
# VOLATILITY INDICATORS
# ============================================================

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range using pure-pandas."""
    def _atr(sdf):
        high_low = sdf['High'] - sdf['Low']
        high_close = abs(sdf['High'] - sdf['Close'].shift(1))
        low_close = abs(sdf['Low'] - sdf['Close'].shift(1))
        sdf['TR'] = np.maximum(np.maximum(high_low, low_close), np.maximum(high_close, np.nan))
        sdf['ATR_14'] = sdf['TR'].rolling(window=14).mean()
        sdf = sdf.drop(columns=['TR'])
        return sdf
    return apply_per_symbol(df, _atr)


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands using pure-pandas."""
    def _bb(sdf):
        sdf['SMA_20'] = sdf['Close'].rolling(window=20).mean()
        sdf['STDEV'] = sdf['Close'].rolling(window=20).std()
        sdf['BB_upper'] = sdf['SMA_20'] + (2 * sdf['STDEV'])
        sdf['BB_lower'] = sdf['SMA_20'] - (2 * sdf['STDEV'])
        sdf = sdf.drop(columns=['STDEV'])
        return sdf
    return apply_per_symbol(df, _bb)


def add_keltner_channel(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """Add Keltner Channels using pure-pandas."""
    def _kc(sdf):
        sdf['SMA_20'] = sdf['Close'].rolling(window=20).mean()
        sdf['ATR_14'] = sdf['Close'].rolling(window=14).mean()
        sdf['upper_band'] = sdf['SMA_20'] + (std_mult * np.sqrt(sdf['ATR_14']))
        sdf['lower_band'] = sdf['SMA_20'] - (std_mult * np.sqrt(sdf['ATR_14']))
        return sdf
    return apply_per_symbol(df, _kc)


def add_donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Donchian Channel using pure-pandas."""
    def _dc(sdf):
        sdf['upper'] = sdf['High'].rolling(window=20).max()
        sdf['lower'] = sdf['Low'].rolling(window=20).min()
        return sdf
    return apply_per_symbol(df, _dc)


# ============================================================
# VOLUME INDICATORS
# ============================================================

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume using pure-pandas."""
    def _obv(sdf):
        sdf['OBV'] = np.where(sdf['Close'].diff() > 0, sdf['Volume'].cumsum(), 0)
        return sdf
    return apply_per_symbol(df, _obv)


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add Volume Weighted Average Price using pure-pandas."""
    def _vwap(sdf):
        sdf['VWAP'] = (sdf['Close'] * sdf['Volume']).rolling(window=20).sum() / sdf['Volume'].rolling(window=20).sum()
        return sdf
    return apply_per_symbol(df, _vwap)


def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Volume Simple Moving Average using pure-pandas."""
    def _vsma(sdf):
        sdf['Volume_SMA_20'] = sdf['Volume'].rolling(window=20).mean()
        sdf['Volume_Ratio'] = sdf['Volume'] / sdf['Volume_SMA_20']
        return sdf
    return apply_per_symbol(df, _vsma)


# ============================================================
# ADVANCED / COMPOSITE INDICATORS
# ============================================================

def add_chandelier_exit(df: pd.DataFrame, period: int = 22, mult: float = 3.0) -> pd.DataFrame:
    """Add Chandelier Exit (trailing stop) using pure-pandas."""
    def _ce(sdf):
        highest_high = sdf['High'].rolling(window=22).max()
        lowest_low = sdf['Low'].rolling(window=22).min()
        sdf['Chandelier_Long'] = highest_high - mult * sdf['ATR_14']
        sdf['Chandelier_Short'] = lowest_low + mult * sdf['ATR_14']
        return sdf
    return apply_per_symbol(df, _ce)


def add_vwap_bands(df: pd.DataFrame, period: int = 20, std_mult: float = 1.0) -> pd.DataFrame:
    """Add VWAP with Standard Deviation Bands using pure-pandas."""
    def _vwap_bands(sdf):
        typical = (sdf['High'] + sdf['Low'] + sdf['Close']) / 3
        sdf['VWAP'] = (sdf['Close'] * sdf['Volume']).rolling(window=20).sum() / sdf['Volume'].rolling(window=20).sum()
        sdf['VWAP_Upper_1SD'] = sdf['VWAP'] + (std_mult * np.sqrt(sdf['Close'].rolling(window=20).var()))
        sdf['VWAP_Lower_1SD'] = sdf['SMA_20'] - (std_mult * np.sqrt(sdf['Close'].rolling(window=20).var()))
        return sdf
    return apply_per_symbol(df, _vwap_bands)


def add_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """Add Fibonacci levels using pure-pandas."""
    def _fib(sdf):
        sdf['Fib_0.236'] = np.nan
        sdf['Fib_0.382'] = np.nan
        sdf['Fib_0.500'] = np.nan
        sdf['Fib_0.618'] = np.nan
        sdf['Fib_0.786'] = np.nan
        sdf['Fib_1.272'] = np.nan
        sdf['Fib_1.618'] = np.nan
        sdf['Fib_2.000'] = np.nan
        sdf['Fib_2.618'] = np.nan
        return sdf
    return apply_per_symbol(df, _fib)


def add_kase_permission_stochastic(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Kase Permission Stochastic using pure-pandas."""
    def _kps(sdf):
        sdf['Kase_Permission'] = sdf['Close'].rolling(window=14).mean() / sdf['High'].rolling(window=14).max() * 100
        sdf['Kase_Stress'] = 100 - sdf['Kase_Permission']
        return sdf
    return apply_per_symbol(df, _kps)


# ============================================================
# OPTIONS-SPECIFIC INDICATORS
# ============================================================

def calculate_iv_rank(iv_series: pd.Series, lookback: int = 252) -> pd.Series:
    """Calculate IV Rank (current IV percentile over lookback)."""
    return iv_series.rolling(lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() > x.min() else 50
    )


def calculate_iv_percentile(iv_series: pd.Series, lookback: int = 252) -> pd.Series:
    """Calculate IV Percentile (percentile rank of current IV)."""
    return iv_series.rolling(lookback).apply(
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

    # Extract indicator config
    ind_config = config.get('indicators', {})

    # Ensure MultiIndex
    df = ensure_multiindex(df)

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
    df = add_bollinger_bands(
        df,
        ind_config.get('bb_period', 20),
        ind_config.get('bb_std', 2.0)
    )
    df = add_keltner_channel(df, ind_config.get('keltner_period', 20), ind_config.get('keltner_mult', 2.0))
    df = add_donchian_channel(df, ind_config.get('donchian_period', 20))

    # Volume
    df = add_obv(df)
    df = add_vwap(df)
    df = add_volume_sma(df, ind_config.get('volume_sma_period', 20))

    # Advanced
    df = add_chandelier_exit(df)
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
        """Update indicators with new bar."""
        return {}

    def get_latest(self, symbol: str) -> Dict:
        """Get latest indicator values for symbol."""
        return self.state.get(symbol, {})


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
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

    # Add all indicators
    config = {
        'indicators': {
            'ema_periods': [20, 50, 200],
            'rsi_period': 14,
            'atr_period': 14,
            'bb_period': 20,
        }
    }

    df_with_ind = add_all_indicators(df, config)
    print(f"Columns: {df.columns.tolist()}")
    print(df.tail())