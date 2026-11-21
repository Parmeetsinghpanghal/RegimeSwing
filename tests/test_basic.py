"""
Basic smoke tests for RegimeSwing toolkit.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))


def test_imports():
    """Test that all main modules can be imported."""
    from data.fetcher import create_fetcher, DataFetcher
    from indicators.technical import add_all_indicators, ensure_multiindex
    from indicators.regime import RegimeDetector
    from strategies.swing import run_all_strategies, get_all_strategies
    from strategies.scoring import create_scorer, SwingScore
    from backtest import BacktestConfig, WalkForwardBacktester
    from papertrade import PaperTradingEngine

    assert True  # If imports work, test passes


def test_universe_fetcher():
    """Test universe fetcher returns symbols."""
    from data.universe import get_nifty50_symbols, get_nifty200_symbols

    nifty50 = get_nifty50_symbols()
    nifty200 = get_nifty200_symbols()

    assert len(nifty50) >= 50
    assert len(nifty200) >= 200
    assert "RELIANCE" in nifty50
    assert "TCS" in nifty50


def test_cache_initialization():
    """Test OHLCVCache initializes correctly."""
    from data.cache import OHLCVCache

    cache = OHLCVCache()
    assert cache.db_path.exists()
    assert cache.parquet_dir.exists()
    cache.close()


def test_indicators_add_all():
    """Test add_all_indicators works on sample data."""
    from indicators.technical import add_all_indicators, ensure_multiindex

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(100, 200, 100),
        'Low': np.random.uniform(100, 200, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.uniform(1000000, 5000000, 100)
    }, index=dates)

    # Ensure High >= Low, Close between High/Low
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)

    df['Symbol'] = 'TEST'
    df = ensure_multiindex(df)

    # Add indicators
    config = {'indicators': {'ema_periods': [20, 50], 'rsi_period': 14}}
    df_with_ind = add_all_indicators(df, config)

    # Check key indicators exist
    assert 'EMA_20' in df_with_ind.columns
    assert 'EMA_50' in df_with_ind.columns
    assert 'RSI_14' in df_with_ind.columns
    assert 'ATR_14' in df_with_ind.columns


def test_regime_detector():
    """Test RegimeDetector runs without error."""
    from indicators.technical import add_all_indicators, ensure_multiindex
    from indicators.regime import RegimeDetector

    # Create sample Nifty data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    nifty = pd.DataFrame({
        'Open': np.random.uniform(19000, 20000, 100),
        'High': np.random.uniform(19000, 20000, 100),
        'Low': np.random.uniform(19000, 20000, 100),
        'Close': np.random.uniform(19000, 20000, 100),
        'Volume': np.random.uniform(1000000, 5000000, 100)
    }, index=dates)

    nifty['High'] = nifty[['Open', 'High', 'Low', 'Close']].max(axis=1)
    nifty['Low'] = nifty[['Open', 'High', 'Low', 'Close']].min(axis=1)

    nifty = add_all_indicators(nifty, {})

    detector = RegimeDetector({})
    regime = detector.detect_nifty_regime(nifty)

    assert regime.regime in ['bull', 'bear', 'sideways', 'volatile', 'unknown']
    assert 0 <= regime.confidence <= 1
    assert regime.nifty_trend in ['up', 'down', 'flat']


def test_swing_strategies():
    """Test swing strategies generate signals (or run without error)."""
    from indicators.technical import add_all_indicators, ensure_multiindex
    from indicators.regime import RegimeDetector
    from strategies.swing import run_all_strategies, get_all_strategies

    # Create sample multi-symbol data
    symbols = ["RELIANCE", "TCS"]
    dfs = []
    for sym in symbols:
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 5000000, 100)
        }, index=dates)
        df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
        df['Symbol'] = sym
        dfs.append(df)

    combined = pd.concat(dfs)
    combined = ensure_multiindex(combined)
    combined = add_all_indicators(combined, {})

    # Regime
    nifty = combined.xs("RELIANCE", level='Symbol').copy()  # Use one as proxy
    nifty = add_all_indicators(nifty, {})
    detector = RegimeDetector({})
    regime = detector.detect_nifty_regime(nifty)

    regime_info = {
        'regime': regime.regime,
        'size_mult': detector.get_size_multiplier(regime),
        'nifty_trend': regime.nifty_trend,
        'should_trade_swing': detector.should_trade_swing,
        'should_sell_options': detector.should_sell_options
    }

    config = {
        'swing': {
            'trend_pullback': {'enabled': True, 'adx_min': 20, 'rsi_min': 40, 'rsi_max': 60},
            'breakout_volume': {'enabled': True},
            'mean_reversion_bb': {'enabled': True}
        },
        'capital': {'initial': 500000, 'max_risk_per_trade': 0.015, 'max_position_pct': 0.05},
        'targets': {}
    }

    signals = run_all_strategies(combined, config, regime_info)
    # Just verify it runs - signals may be empty due to random data
    assert isinstance(signals, list)
    assert True  # Pass if no exception


def test_scorer():
    """Test multi-factor scorer."""
    from indicators.technical import add_all_indicators, ensure_multiindex
    from indicators.regime import RegimeDetector
    from strategies.scoring import create_scorer

    # Sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(100, 200, 100),
        'Low': np.random.uniform(100, 200, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.uniform(1000000, 5000000, 100)
    }, index=dates)
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
    df['Symbol'] = 'TEST'
    df = ensure_multiindex(df)
    df = add_all_indicators(df, {})

    nifty = df.xs("TEST", level='Symbol').copy()
    nifty = add_all_indicators(nifty, {})
    detector = RegimeDetector({})
    regime = detector.detect_nifty_regime(nifty)

    regime_info = {
        'regime': regime.regime,
        'confidence': regime.confidence,
        'size_mult': detector.get_size_multiplier(regime),
        'nifty_trend': regime.nifty_trend
    }

    scorer = create_scorer()
    scores = scorer.score_universe(df, regime_info, 'trend_pullback', min_score=0)

    assert len(scores) == 1
    assert hasattr(scores[0], 'total_score')
    assert 0 <= scores[0].total_score <= 100


def test_config_load():
    """Test config.yaml loads correctly."""
    import yaml

    config_path = ROOT_DIR / "config.yaml"
    assert config_path.exists()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check required sections
    assert 'capital' in config
    assert 'swing' in config
    assert 'regime' in config
    assert 'data' in config


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])