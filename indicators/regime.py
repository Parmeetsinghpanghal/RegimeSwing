"""
Market Regime Detection - Identifies bull/bear/sideways/volatile regimes.
Used to adjust strategy allocation and position sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


RegimeType = Literal['bull', 'bear', 'sideways', 'volatile', 'unknown']


@dataclass
class RegimeState:
    """Current market regime state."""
    regime: RegimeType
    confidence: float          # 0-1
    nifty_trend: str           # 'up', 'down', 'flat'
    vix_level: float
    adx_value: float
    ema_alignment: str         # 'bullish', 'bearish', 'mixed'
    breadth: float             # % stocks above EMA50
    timestamp: pd.Timestamp


class RegimeDetector:
    """
    Detect market regime using Nifty 50 and breadth indicators.
    """

    def __init__(self, config: dict):
        self.config = config
        self.regime_config = config.get('regime', {})
        self.indicators_config = config.get('indicators', {})

        # Thresholds
        self.vix_high = self.regime_config.get('volatile_conditions', {}).get('vix_threshold', 22)
        self.atr_pct_high = self.regime_config.get('volatile_conditions', {}).get('atr_pct_threshold', 0.02)
        self.adx_trend = self.regime_config.get('bull_conditions', {}).get('adx_min', 20)
        self.ema_tolerance = self.regime_config.get('sideways_conditions', {}).get('ema_tolerance', 0.03)

        # Size multipliers by regime
        self.size_mult = self.regime_config.get('size_multiplier', {
            'bull': 1.0,
            'bear': 0.0,
            'sideways': 0.7,
            'volatile': 0.5
        })

    def detect_nifty_regime(self, nifty_df: pd.DataFrame) -> RegimeState:
        """
        Detect regime from Nifty 50 data.

        Args:
            nifty_df: DataFrame with OHLCV and indicators for Nifty 50

        Returns:
            RegimeState object
        """
        if nifty_df.empty or len(nifty_df) < 50:
            return RegimeState(
                regime='unknown',
                confidence=0.0,
                nifty_trend='unknown',
                vix_level=0,
                adx_value=0,
                ema_alignment='unknown',
                breadth=0,
                timestamp=pd.Timestamp.now()
            )

        latest = nifty_df.iloc[-1]

        # Helper to safely extract numeric values (handle tuples from yfinance)
        def _as_float(val, default=0.0):
            """Convert value to float, handling tuples/lists from yfinance."""
            if val is None:
                return default
            if isinstance(val, (tuple, list)):
                return float(val[0]) if val else default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        close = _as_float(latest.get('Close'))
        if close <= 0:
            close = _as_float(latest.get('Adj Close', 10000.0))

        # Get indicators with safe extraction
        ema20 = _as_float(latest.get('EMA_20'), close)
        ema50 = _as_float(latest.get('EMA_50'), close)
        ema200 = _as_float(latest.get('EMA_200'), close)
        adx = _as_float(latest.get('ADX_14'), 0)
        atr = _as_float(latest.get('ATR_14'), 0)
        atr_pct = atr / close if close > 0 else 0

        # VIX (if available, else estimate from ATR)
        vix = _as_float(latest.get('VIX', atr_pct * 100 * np.sqrt(252)))

        # EMA alignment
        ema_alignment = 'mixed'
        if close > ema50 > ema200:
            ema_alignment = 'bullish'
        elif close < ema50 < ema200:
            ema_alignment = 'bearish'

        # Trend direction
        nifty_trend = 'flat'
        if close > ema50 and ema50 > ema200:
            nifty_trend = 'up'
        elif close < ema50 and ema50 < ema200:
            nifty_trend = 'down'

        # Regime scoring
        scores = {
            'bull': 0.0,
            'bear': 0.0,
            'sideways': 0.0,
            'volatile': 0.0
        }

        # Bull conditions
        if ema_alignment == 'bullish':
            scores['bull'] += 0.4
        if adx > 20:  # self.adx_trend
            scores['bull'] += 0.3
        if nifty_trend == 'up':
            scores['bull'] += 0.3

        # Bear conditions
        if ema_alignment == 'bearish':
            scores['bear'] += 0.4
        if adx > 20:
            scores['bear'] += 0.3
        if nifty_trend == 'down':
            scores['bear'] += 0.3

        # Sideways conditions
        if ema_alignment == 'mixed':
            scores['sideways'] += 0.3
        if adx < 20:
            scores['sideways'] += 0.3
        if abs(ema50 - ema200) / ema200 < 0.03:
            scores['sideways'] += 0.4

        # Volatile conditions
        if atr_pct > 0.02:
            scores['volatile'] += 0.5
        if adx > 30:
            scores['volatile'] += 0.5

        # Determine regime
        regime = max(scores, key=scores.get)
        confidence = scores[regime]

        # Override: Volatile trumps others if extreme
        if scores['volatile'] > 0.7:
            regime = 'volatile'
            confidence = scores['volatile']

        # Breadth (simplified - would need all stocks data)
        breadth = 0.5  # Placeholder

        return RegimeState(
            regime=regime,
            confidence=confidence,
            nifty_trend=nifty_trend,
            vix_level=vix,
            adx_value=adx,
            ema_alignment=ema_alignment,
            breadth=breadth,
            timestamp=pd.Timestamp.now()
        )

    def detect_sector_regime(self, sector_df: pd.DataFrame, nifty_df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect regime for each sector relative to Nifty.

        Args:
            sector_df: MultiIndex DataFrame with sector data
            nifty_df: Nifty 50 DataFrame

        Returns:
            Dict mapping sector -> regime ('outperforming', 'underperforming', 'neutral')
        """
        if sector_df.empty or nifty_df.empty:
            return {}

        nifty_return = nifty_df['Close'].pct_change(20).iloc[-1]  # 20-day return

        regimes = {}
        for sector in sector_df.index.get_level_values('Symbol').unique():
            try:
                sec_data = sector_df.xs(sector, level='Symbol')
                sec_return = sec_data['Close'].pct_change(20).iloc[-1]

                diff = sec_return - nifty_return
                if diff > 0.02:
                    regimes[sector] = 'outperforming'
                elif diff < -0.02:
                    regimes[sector] = 'underperforming'
                else:
                    regimes[sector] = 'neutral'
            except Exception:
                regimes[sector] = 'neutral'

        return regimes

    def should_trade_swing(self, regime: RegimeState, strategy: str) -> Tuple[bool, str]:
        """
        Determine if swing strategy should trade in current regime.

        Args:
            regime: Current regime state
            strategy: Strategy name ('trend_pullback', 'breakout_volume', 'mean_reversion_bb')

        Returns:
            (should_trade, reason)
        """
        r = regime.regime

        if strategy == 'trend_pullback':
            if r == 'bull':
                return True, "Bull regime favors trend following"
            elif r == 'volatile':
                return True, "Volatile but trend exists - reduce size"
            elif r == 'sideways':
                return False, "Sideways market - trend pullbacks fail"
            elif r == 'bear':
                return False, "Bear market - no long trend pullbacks"
            return False, "Unknown regime"

        elif strategy == 'breakout_volume':
            if r == 'bull':
                return True, "Breakouts work in bull markets"
            elif r == 'volatile':
                return True, "High volatility breakouts - wider stops"
            elif r == 'sideways':
                return False, "Sideways - breakouts fail (false breaks)"
            elif r == 'bear':
                return False, "Bear market - breakouts are traps"
            return False, "Unknown regime"

        elif strategy == 'mean_reversion_bb':
            if r == 'sideways':
                return True, "Mean reversion works in ranging markets"
            elif r == 'bull' and 20 < 20:  # adx < 20
                return True, "Bull but low ADX - ranging within trend"
            elif r == 'volatile':
                return False, "Volatile - bands expand, false signals"
            elif r == 'bear':
                return False, "Bear market - don't catch falling knives"
            return False, "Unknown regime"

        return True, "Default allow"

    def should_sell_options(self, regime: RegimeState) -> Tuple[bool, str, Dict]:
        """
        Determine options selling parameters for regime.

        Returns:
            (should_sell, reason, params_dict)
        """
        r = regime.regime
        params = {
            'iv_rank_min': 50,
            'delta_target': 0.25,
            'dte_range': (30, 45),
            'size_mult': 1.0
        }

        if r == 'bull':
            return True, "Bull market - sell CSP on quality stocks", {
                **params, 'iv_rank_min': 50, 'delta_target': 0.25, 'dte_range': (30, 45), 'size_mult': 1.0
            }
        elif r == 'bear':
            return True, "Bear market - sell CSP wider, CC on longs", {
                **params, 'iv_rank_min': 40, 'delta_target': 0.15, 'dte_range': (21, 35), 'size_mult': 0.5
            }
        elif r == 'sideways':
            return True, "Sideways - iron condors, credit spreads", {
                **params, 'iv_rank_min': 60, 'delta_target': 0.20, 'dte_range': (21, 30), 'size_mult': 0.7
            }
        elif r == 'volatile':
            return True, "High IV - sell premium aggressively", {
                **params, 'iv_rank_min': 70, 'delta_target': 0.30, 'dte_range': (14, 30), 'size_mult': 0.5
            }

        return False, "Unknown regime", params

    def get_size_multiplier(self, regime: RegimeState) -> float:
        """Get position size multiplier for regime."""
        return self.size_mult.get(regime.regime, 0.5)

    def get_regime_summary(self, regime: RegimeState) -> str:
        """Get human-readable regime summary."""
        icons = {
            'bull': '🟢',
            'bear': '🔴',
            'sideways': '🟡',
            'volatile': '⚡',
            'unknown': '⚪'
        }
        icon = icons.get(regime.regime, '⚪')

        return (
            f"{icon} Regime: {regime.regime.upper()} (confidence: {regime.confidence:.0%})\n"
            f"   Nifty Trend: {regime.nifty_trend} | EMA: {regime.ema_alignment}\n"
            f"   ADX: {regime.adx_value:.1f} | VIX: {regime.vix_level:.1f} | Breadth: {regime.breadth:.0%}"
        )


# ============================================================
# HISTORICAL REGIME ANALYSIS
# ============================================================

def analyze_historical_regimes(
    nifty_df: pd.DataFrame,
    detector: 'RegimeDetector'
) -> pd.DataFrame:
    """
    Analyze regime over historical period.

    Returns:
        DataFrame with regime for each date
    """
    regimes = []
    window = 50  # Need 50 bars for indicators

    for i in range(window, len(nifty_df)):
        subset = nifty_df.iloc[i-window:i+1]
        regime = detector.detect_nifty_regime(subset)
        regimes.append({
            'date': subset.index[-1],
            'regime': regime.regime,
            'confidence': regime.confidence,
            'nifty_trend': regime.nifty_trend,
            'adx': regime.adx_value,
            'vix': regime.vix_level,
            'ema_alignment': regime.ema_alignment
        })

    return pd.DataFrame(regimes).set_index('date')


def regime_performance_attribution(
    trades_df: pd.DataFrame,
    regimes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Attribute trade performance to market regimes.

    Args:
        trades_df: DataFrame with trades (entry_date, exit_date, pnl)
        regimes_df: DataFrame with regime per date

    Returns:
        Performance by regime
    """
    # Merge trades with entry regime
    trades_df = trades_df.copy()
    trades_df['entry_regime'] = trades_df['entry_date'].map(
        regimes_df['regime']
    )

    perf = trades_df.groupby('entry_regime').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum'),
        avg_r=('r_multiple', 'mean') if 'r_multiple' in trades_df.columns else ('pnl', 'mean')
    ).round(4)

    return perf


if __name__ == "__main__":
    import yfinance as yf
    from indicators.technical import add_all_indicators, ensure_multiindex

    logging.basicConfig(level=logging.INFO)

    # Fetch Nifty data
    nifty = yf.download("^NSEI", period="2y", interval="1d", progress=False)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty.columns = [c.capitalize() for c in nifty.columns]

    # Add indicators
    nifty = add_all_indicators(nifty, {})

    # Detect regime
    detector = RegimeDetector({})
    regime = detector.detect_nifty_regime(nifty)

    print(detector.get_regime_summary(regime))

    # Test strategy decisions
    for strat in ['trend_pullback', 'breakout_volume', 'mean_reversion_bb']:
        should, reason = detector.should_trade_swing(regime, strat)
        print(f"  {strat}: {'✅' if should else '❌'} {reason}")

    # Options
    should, reason, params = detector.should_sell_options(regime)
    print(f"\n  Options Selling: {'✅' if should else '❌'} {reason}")
    print(f"  Params: {params}")