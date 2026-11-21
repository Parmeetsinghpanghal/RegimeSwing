"""
Indicators Package - All technical indicators for swing trading.
"""

from .technical import (
    add_all_indicators,
    ensure_multiindex,
    apply_per_symbol,
    get_symbol_data,
    add_ema, add_sma, add_adx, add_supertrend, add_macd,
    add_rsi, add_stochastic, add_roc,
    add_atr, add_bollinger_bands, add_keltner_channel, add_donchian_channel,
    add_obv, add_vwap, add_volume_sma,
    add_chandelier_exit, add_vwap_bands, add_fibonacci_levels,
    add_kase_permission_stochastic,
    calculate_iv_rank, calculate_iv_percentile,
    IncrementalIndicators
)

from .targets import (
    TargetCalculator,
    StopMethod,
    TargetMethod,
    StopTarget,
    calculate_position_size,
    calculate_option_position_size,
    calculate_position_size_atr_parity,
    analyze_risk_reward
)

from .regime import (
    RegimeDetector,
    RegimeState,
    analyze_historical_regimes,
    regime_performance_attribution
)

from .volume_profile import (
    VolumeProfile,
    TPOProfile,
    calculate_volume_profile,
    calculate_tpo_profile,
    calculate_composite_volume_profile,
    get_volume_profile_signals,
    add_volume_profile_to_dataframe
)

from .support_resistance import (
    SRZone,
    SRAnalysis,
    find_fractal_pivots,
    find_swing_points,
    cluster_price_levels,
    calculate_zone_strength,
    detect_sr_zones,
    get_sr_signals,
    add_sr_zones_to_dataframe
)

from .patterns import (
    CandlestickPattern,
    PatternDirection,
    detect_all_patterns,
    add_patterns_to_dataframe,
    get_latest_patterns,
    calculate_pattern_statistics
)

from .multitf import (
    TFState,
    ConfluenceResult,
    Timeframe,
    TrendDirection,
    analyze_single_timeframe,
    analyze_multitf_confluence,
    fetch_multitf_data,
    scan_universe_confluence,
    resample_to_timeframe,
    find_confluence_zones,
    calculate_alignment_score
)

__all__ = [
    # Technical
    'add_all_indicators', 'ensure_multiindex', 'apply_per_symbol', 'get_symbol_data',
    'add_ema', 'add_sma', 'add_adx', 'add_supertrend', 'add_macd',
    'add_rsi', 'add_stochastic', 'add_roc',
    'add_atr', 'add_bollinger_bands', 'add_keltner_channel', 'add_donchian_channel',
    'add_obv', 'add_vwap', 'add_volume_sma',
    'add_chandelier_exit', 'add_vwap_bands', 'add_fibonacci_levels',
    'add_kase_permission_stochastic',
    'calculate_iv_rank', 'calculate_iv_percentile',
    'IncrementalIndicators',

    # Targets
    'TargetCalculator', 'StopMethod', 'TargetMethod', 'StopTarget',
    'calculate_position_size', 'calculate_option_position_size', 'analyze_risk_reward',

    # Regime
    'RegimeDetector', 'RegimeState',
    'analyze_historical_regimes', 'regime_performance_attribution',

    # Volume Profile
    'VolumeProfile', 'TPOProfile',
    'calculate_volume_profile', 'calculate_tpo_profile',
    'calculate_composite_volume_profile', 'get_volume_profile_signals',
    'add_volume_profile_to_dataframe',

    # Support/Resistance
    'SRZone', 'SRAnalysis',
    'find_fractal_pivots', 'find_swing_points', 'cluster_price_levels',
    'calculate_zone_strength', 'detect_sr_zones', 'get_sr_signals',
    'add_sr_zones_to_dataframe',

    # Patterns
    'CandlestickPattern', 'PatternDirection',
    'detect_all_patterns', 'add_patterns_to_dataframe',
    'get_latest_patterns', 'calculate_pattern_statistics',

    # Multi-timeframe
    'TFState', 'ConfluenceResult', 'Timeframe', 'TrendDirection',
    'analyze_single_timeframe', 'analyze_multitf_confluence',
    'fetch_multitf_data', 'scan_universe_confluence',
    'resample_to_timeframe', 'find_confluence_zones', 'calculate_alignment_score',
]