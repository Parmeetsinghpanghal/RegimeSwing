"""
Swing Trading Strategies - Rule-based implementations with config-driven parameters.
Each strategy generates signals with entry, stop, targets, and position size.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal with all execution details."""
    symbol: str
    strategy: str
    timestamp: pd.Timestamp
    signal_type: str          # 'entry', 'exit', 'adjust'
    side: str                 # 'long', 'short'
    entry_price: float
    stop_price: float
    targets: List[float]
    position_size: int
    risk_pct: float
    r_multiple_targets: List[float]
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'strategy': self.strategy,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, pd.Timestamp) else str(self.timestamp),
            'signal_type': self.signal_type,
            'side': self.side,
            'entry_price': self.entry_price,
            'stop_price': self.stop_price,
            'targets': self.targets,
            'position_size': self.position_size,
            'risk_pct': self.risk_pct,
            'r_multiple_targets': self.r_multiple_targets,
            'metadata': self.metadata
        }


class BaseSwingStrategy(ABC):
    """Base class for swing strategies."""

    def __init__(self, config: dict, name: str):
        self.config = config
        self.name = name
        self.strategy_config = config.get('swing', {}).get(name, {})
        self.capital_config = config.get('capital', {})
        self.targets_config = config.get('targets', {})
        self.enabled = self.strategy_config.get('enabled', True)

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, regime_info: dict = None) -> List[Signal]:
        """Generate entry signals for all symbols in DataFrame."""
        pass

    def _get_latest_bar(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Get latest bar for symbol."""
        if isinstance(df.index, pd.MultiIndex):
            sym_df = df.xs(symbol, level='Symbol')
        else:
            sym_df = df
        return sym_df.iloc[-1]

    def _get_symbol_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Get all data for symbol."""
        if isinstance(df.index, pd.MultiIndex):
            return df.xs(symbol, level='Symbol')
        return df

    def _check_basic_filters(self, latest: pd.Series, sym_df: pd.DataFrame, config: dict) -> Tuple[bool, str]:
        """Check basic entry filters (volume, trend, etc.)."""
        # Volume filter
        vol_ratio = latest.get('Volume_Ratio', 1.0)
        min_vol = config.get('volume_mult', 1.0)
        if vol_ratio < min_vol:
            return False, f"Volume ratio {vol_ratio:.2f} < {min_vol}"

        return True, "OK"

    def _create_signal(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        entry: float,
        stop: float,
        targets: List[float],
        position_size: int,
        risk_pct: float,
        r_targets: List[float],
        metadata: Dict = None
    ) -> Signal:
        """Create Signal object."""
        return Signal(
            symbol=symbol,
            strategy=self.name,
            timestamp=timestamp,
            signal_type='entry',
            side='long',
            entry_price=entry,
            stop_price=stop,
            targets=targets,
            position_size=position_size,
            risk_pct=risk_pct,
            r_multiple_targets=r_targets,
            metadata=metadata or {}
        )


class TrendPullbackStrategy(BaseSwingStrategy):
    """
    Trend Pullback Strategy:
    - Trend: Price > EMA50 > EMA200, ADX > 25
    - Pullback: Price near EMA20 (within 0.5*ATR), RSI 40-55
    - Volume: Current > 1.2x 20-day avg
    - Entry: Close back above EMA20 or bullish candle at support
    """

    def generate_signals(self, df: pd.DataFrame, regime_info: dict = None) -> List[Signal]:
        if not self.enabled:
            return []

        signals = []
        params = self.strategy_config.get('trend_pullback', {})
        capital = self.capital_config.get('initial', 500000)
        max_risk = self.capital_config.get('max_risk_per_trade', 0.015)
        max_pos_pct = self.capital_config.get('max_position_pct', 0.05)

        symbols = df.index.get_level_values('Symbol').unique() if isinstance(df.index, pd.MultiIndex) else ['UNKNOWN']

        for symbol in symbols:
            try:
                sym_df = self._get_symbol_data(df, symbol)
                if len(sym_df) < 50:
                    continue

                latest = sym_df.iloc[-1]
                prev = sym_df.iloc[-2] if len(sym_df) > 1 else latest

                # --- Trend Filter ---
                close = latest['Close']
                ema20 = latest.get('EMA_20', close)
                ema50 = latest.get('EMA_50', close)
                ema200 = latest.get('EMA_200', close)
                adx = latest.get('ADX_14', 0)

                # Trend: Price > EMA50 > EMA200, ADX > 25
                if not (close > ema50 > ema200):
                    continue
                if adx < params.get('adx_min', 25):
                    continue

                # Regime filter
                if regime_info:
                    should, _ = regime_info.get('should_trade_swing', lambda r, s: (True, ""))(regime_info, 'trend_pullback')
                    if not should:
                        continue

                # --- Pullback Detection ---
                atr = latest.get('ATR_14', close * 0.02)
                pullback_zone = params.get('pullback_zone_atr_mult', 0.5) * atr

                # Price near EMA20
                dist_to_ema20 = abs(close - ema20)
                if dist_to_ema20 > pullback_zone:
                    continue

                # RSI in pullback zone (40-55)
                rsi = latest.get('RSI_14', 50)
                if not (params.get('rsi_min', 40) <= rsi <= params.get('rsi_max', 55)):
                    continue

                # Volume confirmation
                vol_ratio = latest.get('Volume_Ratio', 1.0)
                if vol_ratio < params.get('volume_mult', 1.2):
                    continue

                # --- Entry Trigger ---
                # Bullish candle closing above EMA20 or hammer/bullish engulfing
                is_bullish = close > latest['Open']
                closed_above_ema20 = close > ema20

                if not (is_bullish and closed_above_ema20):
                    # Check for bullish patterns
                    body = abs(close - latest['Open'])
                    lower_wick = min(close, latest['Open']) - latest['Low']
                    upper_wick = latest['High'] - max(close, latest['Open'])

                    # Hammer: small body, long lower wick
                    is_hammer = (lower_wick > 2 * body) and (upper_wick < body)

                    # Bullish engulfing
                    prev_bearish = prev['Close'] < prev['Open']
                    bullish_engulf = prev_bearish and is_bullish and \
                                     close > prev['Open'] and latest['Open'] < prev['Close']

                    if not (is_hammer or bullish_engulf):
                        continue

                # --- Calculate Stop & Targets ---
                from indicators.targets import TargetCalculator, StopMethod, TargetMethod
                calc = TargetCalculator(self.config)

                # Stop: Chandelier or structure
                stop = calc.calculate_stop(
                    entry_price=close,
                    atr=atr,
                    df=sym_df,
                    method=StopMethod.CHANDELIER
                )

                # Targets
                targets = calc.calculate_targets(
                    entry_price=close,
                    stop_price=stop,
                    atr=atr,
                    df=sym_df
                )

                if not targets or stop >= close:
                    continue

                # Position sizing
                from indicators.targets import calculate_position_size
                position_size = calculate_position_size(
                    capital=capital,
                    risk_per_trade=max_risk * regime_info.get('size_mult', 1.0) if regime_info else max_risk,
                    entry_price=close,
                    stop_price=stop,
                    max_position_pct=max_pos_pct
                )

                if position_size == 0:
                    continue

                risk_pct = abs(close - stop) / close
                risk_amount = risk_pct * position_size * close
                r_targets = [(t - close) / risk_pct / close for t in targets]  # R multiples

                # Create signal
                signal = self._create_signal(
                    symbol=symbol,
                    timestamp=sym_df.index[-1],
                    entry=close,
                    stop=stop,
                    targets=targets,
                    position_size=position_size,
                    risk_pct=risk_pct,
                    r_targets=r_targets,
                    metadata={
                        'ema20': ema20,
                        'ema50': ema50,
                        'ema200': ema200,
                        'adx': adx,
                        'rsi': rsi,
                        'vol_ratio': vol_ratio,
                        'pullback_dist': dist_to_ema20
                    }
                )
                signals.append(signal)

            except Exception as e:
                logger.warning(f"TrendPullback error for {symbol}: {e}")

        return signals


class BreakoutVolumeStrategy(BaseSwingStrategy):
    """
    Breakout Volume Strategy:
    - Consolidation: 10+ days, range < 8%
    - Breakout: Close > 20-day high, Volume > 1.5x avg
    - Trend filter: Nifty > EMA50
    """

    def generate_signals(self, df: pd.DataFrame, regime_info: dict = None) -> List[Signal]:
        if not self.enabled:
            return []

        signals = []
        params = self.strategy_config.get('breakout_volume', {})
        capital = self.capital_config.get('initial', 500000)
        max_risk = self.capital_config.get('max_risk_per_trade', 0.015)
        max_pos_pct = self.capital_config.get('max_position_pct', 0.05)

        symbols = df.index.get_level_values('Symbol').unique() if isinstance(df.index, pd.MultiIndex) else ['UNKNOWN']

        for symbol in symbols:
            try:
                sym_df = self._get_symbol_data(df, symbol)
                if len(sym_df) < 50:
                    continue

                latest = sym_df.iloc[-1]
                close = latest['Close']

                # --- Consolidation Detection ---
                lookback = params.get('lookback_days', 20)
                recent = sym_df.tail(lookback)

                high_20 = recent['High'].max()
                low_20 = recent['Low'].min()
                consolidation_range = (high_20 - low_20) / low_20

                # Must be in consolidation
                if consolidation_range > params.get('max_consolidation_range', 0.08):
                    continue

                # --- Breakout Detection ---
                # Close above 20-day high
                if close <= high_20:
                    continue

                # Volume > 1.5x average
                vol_ratio = latest.get('Volume_Ratio', 1.0)
                if vol_ratio < params.get('volume_mult', 1.5):
                    continue

                # Market regime filter
                if params.get('market_regime_filter', True):
                    if regime_info and regime_info.get('nifty_trend') != 'up':
                        continue

                # --- Calculate Stop & Targets ---
                from indicators.targets import TargetCalculator, StopMethod, calculate_position_size
                calc = TargetCalculator(self.config)
                atr = latest.get('ATR_14', close * 0.02)

                # Stop: Below breakout candle low or 1.5*ATR
                breakout_low = latest['Low']
                stop_atr = close - params.get('stop_atr_mult', 1.5) * atr
                stop = max(breakout_low * 0.995, stop_atr)  # Slightly below breakout low

                # Targets
                targets = calc.calculate_targets(
                    entry_price=close,
                    stop_price=stop,
                    atr=atr,
                    df=sym_df
                )

                if not targets or stop >= close:
                    continue

                # Position sizing
                position_size = calculate_position_size(
                    capital=capital,
                    risk_per_trade=max_risk * regime_info.get('size_mult', 1.0) if regime_info else max_risk,
                    entry_price=close,
                    stop_price=stop,
                    max_position_pct=max_pos_pct
                )

                if position_size == 0:
                    continue

                risk_pct = abs(close - stop) / close
                r_targets = [(t - close) / (close - stop) for t in targets]

                # Create signal
                signal = self._create_signal(
                    symbol=symbol,
                    timestamp=sym_df.index[-1],
                    entry=close,
                    stop=stop,
                    targets=targets,
                    position_size=position_size,
                    risk_pct=risk_pct,
                    r_targets=r_targets,
                    metadata={
                        'consolidation_days': lookback,
                        'consolidation_range': consolidation_range,
                        'high_20': high_20,
                        'vol_ratio': vol_ratio,
                        'breakout_type': '20d_high'
                    }
                )
                signals.append(signal)

            except Exception as e:
                logger.warning(f"BreakoutVolume error for {symbol}: {e}")

        return signals


class MeanReversionBBStrategy(BaseSwingStrategy):
    """
    Mean Reversion (Bollinger Bands):
    - Price at lower BB (2,20), RSI < 30, ADX < 20 (ranging)
    - Entry: Close back inside BB + bullish candle
    - Quick exit if no move in 3 days
    """

    def generate_signals(self, df: pd.DataFrame, regime_info: dict = None) -> List[Signal]:
        if not self.enabled:
            return []

        signals = []
        params = self.strategy_config.get('mean_reversion_bb', {})
        capital = self.capital_config.get('initial', 500000)
        max_risk = self.capital_config.get('max_risk_per_trade', 0.015)
        max_pos_pct = self.capital_config.get('max_position_pct', 0.05)

        symbols = df.index.get_level_values('Symbol').unique() if isinstance(df.index, pd.MultiIndex) else ['UNKNOWN']

        for symbol in symbols:
            try:
                sym_df = self._get_symbol_data(df, symbol)
                if len(sym_df) < 50:
                    continue

                latest = sym_df.iloc[-1]
                prev = sym_df.iloc[-2] if len(sym_df) > 1 else latest
                close = latest['Close']

                # --- Regime Check ---
                # Only in bull/sideways, never in bear
                if regime_info:
                    r = regime_info.get('regime')
                    if r == 'bear':
                        continue
                    if r == 'volatile':
                        continue

                # --- Setup Conditions ---
                bb_lower = latest.get('BBL_20_2.0', close * 0.95)
                bb_mid = latest.get('BBM_20_2.0', close)
                bb_upper = latest.get('BBU_20_2.0', close * 1.05)
                rsi = latest.get('RSI_14', 50)
                adx = latest.get('ADX_14', 25)

                # Price at or below lower BB
                if close > bb_lower * 1.01:  # Allow 1% buffer
                    continue

                # RSI oversold
                if rsi > params.get('rsi_max', 30):
                    continue

                # ADX low (ranging market)
                if adx > params.get('adx_max', 20):
                    continue

                # Market trend filter
                if params.get('market_regime_filter', True):
                    ema200 = latest.get('EMA_200', close)
                    if close < ema200:
                        continue

                # --- Entry Trigger ---
                # Close back inside BB + bullish candle
                inside_bb = close > bb_lower and close < bb_upper
                is_bullish = close > latest['Open']

                # Or bullish reversal pattern
                body = abs(close - latest['Open'])
                lower_wick = min(close, latest['Open']) - latest['Low']
                is_hammer = (lower_wick > 2 * body)

                if not (inside_bb and (is_bullish or is_hammer)):
                    continue

                # --- Calculate Stop & Targets ---
                from indicators.targets import TargetCalculator, calculate_position_size
                calc = TargetCalculator(self.config)
                atr = latest.get('ATR_14', close * 0.02)

                # Stop: Below entry candle low
                stop = min(latest['Low'], prev['Low']) - 0.5 * atr

                # Targets: BB mid, then BB upper
                targets = [bb_mid, bb_upper]

                if not targets or stop >= close:
                    continue

                # Position sizing
                position_size = calculate_position_size(
                    capital=capital,
                    risk_per_trade=max_risk * regime_info.get('size_mult', 1.0) if regime_info else max_risk,
                    entry_price=close,
                    stop_price=stop,
                    max_position_pct=max_pos_pct
                )

                if position_size == 0:
                    continue

                risk_pct = abs(close - stop) / close
                r_targets = [(t - close) / (close - stop) for t in targets]

                # Create signal
                signal = self._create_signal(
                    symbol=symbol,
                    timestamp=sym_df.index[-1],
                    entry=close,
                    stop=stop,
                    targets=targets,
                    position_size=position_size,
                    risk_pct=risk_pct,
                    r_targets=r_targets,
                    metadata={
                        'bb_lower': bb_lower,
                        'bb_mid': bb_mid,
                        'bb_upper': bb_upper,
                        'rsi': rsi,
                        'adx': adx,
                        'setup_type': 'bb_mean_reversion'
                    }
                )
                signals.append(signal)

            except Exception as e:
                logger.warning(f"MeanReversionBB error for {symbol}: {e}")

        return signals


# ============================================================
# STRATEGY REGISTRY
# ============================================================

STRATEGY_REGISTRY = {
    'trend_pullback': TrendPullbackStrategy,
    'breakout_volume': BreakoutVolumeStrategy,
    'mean_reversion_bb': MeanReversionBBStrategy,
}


def create_strategy(name: str, config: dict) -> BaseSwingStrategy:
    """Factory function to create strategy by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](config, name)


def get_all_strategies(config: dict) -> Dict[str, BaseSwingStrategy]:
    """Create all enabled strategies."""
    strategies = {}
    for name, cls in STRATEGY_REGISTRY.items():
        strat = cls(config, name)
        if strat.enabled:
            strategies[name] = strat
    return strategies


def run_all_strategies(df: pd.DataFrame, config: dict, regime_info: dict = None) -> List:
    """Run all enabled strategies and combine signals."""
    all_signals = []
    strategies = get_all_strategies(config)

    for name, strat in strategies.items():
        try:
            signals = strat.generate_signals(df, regime_info)
            all_signals.extend(signals)
            logger.info(f"{name}: Generated {len(signals)} signals")
        except Exception as e:
            logger.error(f"Strategy {name} failed: {e}")

    return all_signals


if __name__ == "__main__":
    import yfinance as yf
    from indicators.technical import add_all_indicators, ensure_multiindex
    from indicators.regime import RegimeDetector

    logging.basicConfig(level=logging.INFO)

    # Test data
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
    dfs = []
    for sym in symbols:
        df = yf.download(f"{sym}.NS", period="6mo", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.capitalize() for c in df.columns]
        df['Symbol'] = sym
        dfs.append(df)

    combined = pd.concat(dfs)
    combined = ensure_multiindex(combined)
    combined = add_all_indicators(combined)

    # Regime detection
    nifty = yf.download("^NSEI", period="6mo", interval="1d", progress=False)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty.columns = [c.capitalize() for c in nifty.columns]
    nifty = add_all_indicators(nifty)

    config = {
        'swing': {
            'trend_pullback': {'enabled': True, 'adx_min': 25, 'rsi_min': 40, 'rsi_max': 55, 'volume_mult': 1.2},
            'breakout_volume': {'enabled': True, 'volume_mult': 1.5},
            'mean_reversion_bb': {'enabled': True, 'rsi_max': 30, 'adx_max': 20},
        },
        'capital': {'initial': 500000, 'max_risk_per_trade': 0.015, 'max_position_pct': 0.05},
        'targets': {}
    }

    detector = RegimeDetector(config)
    regime = detector.detect_nifty_regime(nifty)
    regime_info = {
        'regime': regime.regime,
        'size_mult': detector.get_size_multiplier(regime),
        'nifty_trend': regime.nifty_trend,
        'should_trade_swing': detector.should_trade_swing
    }

    # Run all strategies
    signals = run_all_strategies(combined, config, regime_info)

    print(f"\nTotal Signals: {len(signals)}")
    for s in signals:
        print(f"  {s.symbol} | {s.strategy} | Entry: {s.entry_price:.2f} | Stop: {s.stop_price:.2f} | "
              f"Targets: {[f'{t:.2f}' for t in s.targets]} | Size: {s.position_size} | Risk: {s.risk_pct:.2%}")