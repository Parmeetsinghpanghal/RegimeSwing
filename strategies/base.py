"""
Base Strategy Classes for Swing-Options Toolkit
Defines the common interface for all trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# SIGNAL DATA CLASSES
# ============================================================

@dataclass
class EquitySignal:
    """Signal for equity trade."""
    symbol: str
    date: pd.Timestamp
    signal: int  # 1 = long, -1 = short, 0 = flat
    entry_price: float
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    position_size: Optional[float] = None  # Fraction of capital
    metadata: Dict = field(default_factory=dict)


@dataclass
class OptionSignal:
    """Signal for options trade."""
    symbol: str
    date: pd.Timestamp
    strategy: str  # covered_call, cash_secured_put, iron_condor, etc.
    legs: List[Dict]  # Each leg: strike, expiry, option_type, action, quantity
    net_premium: float
    max_profit: float
    max_loss: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    """Open position tracking."""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    quantity: float
    side: str  # long, short
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    strategy: str = ''
    highest_price: float = 0  # For trailing stop
    lowest_price: float = 0


# ============================================================
# BASE STRATEGY CLASS
# ============================================================

class BaseStrategy(ABC):
    """Abstract base class for all strategies."""

    def __init__(self, params: Dict = None):
        self.params = params or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def required_data(self) -> List[str]:
        """Return list of required data columns."""
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate trading signals.

        Args:
            data: MultiIndex DataFrame (date, symbol) with OHLCV and indicators
            **kwargs: Additional parameters (option_chains, config, etc.)

        Returns:
            DataFrame with signals indexed by (date, symbol) containing:
            - signal: 1 (long), -1 (short), 0 (flat/close)
            - entry_price, stop_loss, target (for equity)
            - position_size (optional)
            - For options: strategy, legs, net_premium, etc.
        """
        pass

    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        """Validate strategy parameters. Returns (is_valid, error_message)."""
        return True, ""

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get parameter with default."""
        return self.params.get(key, default)


# ============================================================
# SWING STRATEGY BASE
# ============================================================

class SwingStrategy(BaseStrategy):
    """Base class for swing trading strategies."""

    def required_data(self) -> List[str]:
        return ['Open', 'High', 'Low', 'Close', 'Volume']

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate swing trading signals.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add required indicators to data."""
        from indicators.technical import add_all_indicators
        config = kwargs.get('config', {})
        return add_all_indicators(data, config)

    def _calculate_position_size(self, capital: float, entry: float,
                                  stop: float, risk_pct: float) -> int:
        """Calculate position size based on fixed risk."""
        if stop is None or entry == stop:
            return 0
        risk_amount = capital * risk_pct / 100
        risk_per_share = abs(entry - stop)
        return int(risk_amount / risk_per_share)


# ============================================================
# OPTIONS STRATEGY BASE
# ============================================================

class OptionsStrategy(BaseStrategy):
    """Base class for options strategies."""

    def required_data(self) -> List[str]:
        return ['Open', 'High', 'Low', 'Close', 'Volume']

    def generate_signals(self, data: pd.DataFrame,
                         option_chains: Dict = None, **kwargs) -> pd.DataFrame:
        """
        Generate options trading signals.

        Args:
            data: Equity price data with indicators
            option_chains: Dict of symbol -> option chain DataFrame
            **kwargs: Additional parameters
        """
        raise NotImplementedError

    def _select_strikes(self, chain: pd.DataFrame, spot: float,
                        target_delta: float, option_type: str) -> Optional[float]:
        """Select strike closest to target delta."""
        from indicators.options import find_strike_by_delta
        return find_strike_by_delta(chain, target_delta, option_type)

    def _calculate_leg_greeks(self, legs: List[Dict], spot: float) -> Dict:
        """Calculate net Greeks for option legs."""
        from indicators.options import position_greeks
        return position_greeks(legs, spot)

    def _validate_iv_rank(self, chain: pd.DataFrame, iv_rank_min: float) -> bool:
        """Check if IV Rank meets minimum threshold."""
        from indicators.options import iv_rank
        if chain.empty:
            return False
        iv_series = chain.groupby('expiry')['IV'].mean()
        if len(iv_series) < 2:
            return True
        current_rank = iv_rank(iv_series).iloc[-1]
        return current_rank >= iv_rank_min


# ============================================================
# CONCRETE SWING STRATEGIES
# ============================================================

class TrendPullbackStrategy(SwingStrategy):
    """
    Trend Following with Pullback Entry.
    - Identify trend using EMAs
    - Enter on pullback to support/EMA
    - Exit on trend reversal or target hit
    """

    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        required = ['ema_fast', 'ema_medium', 'ema_slow', 'pullback_depth_pct',
                    'atr_multiplier_sl', 'atr_multiplier_tp', 'rsi_pullback_max']
        for key in required:
            if key not in params:
                return False, f"Missing required parameter: {key}"
        return True, ""

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        config = kwargs.get('config', {})
        data = self._prepare_data(data)

        signals_list = []

        for (date, symbol), row in data.iterrows():
            # Trend identification
            trend_up = (
                row['EMA_20'] > row['EMA_50'] > row['EMA_200'] and
                row['ADX'] > self.get_param('adx_threshold', 25)
            )
            trend_down = (
                row['EMA_20'] < row['EMA_50'] < row['EMA_200'] and
                row['ADX'] > self.get_param('adx_threshold', 25)
            )

            signal = 0
            entry_price = row['Close']
            stop_loss = None
            target = None

            if trend_up:
                # Pullback entry: price near EMA_20 or recent low
                pullback_level = row['Low'] <= row['EMA_20'] * (1 + self.get_param('pullback_depth_pct', 8) / 100)
                rsi_ok = row['RSI'] < self.get_param('rsi_pullback_max', 50)
                volume_ok = row['Volume'] > row['Volume'].rolling(20).mean() * 1.5

                if pullback_level and rsi_ok:
                    signal = 1
                    atr = row['ATR']
                    stop_loss = entry_price - self.get_param('atr_multiplier_sl', 2.0) * atr
                    target = entry_price + self.get_param('atr_multiplier_tp', 3.0) * atr

            elif trend_down:
                # Short pullback
                pullback_level = row['High'] >= row['EMA_20'] * (1 - self.get_param('pullback_depth_pct', 8) / 100)
                rsi_ok = row['RSI'] > (100 - self.get_param('rsi_pullback_max', 50))
                volume_ok = row['Volume'] > row['Volume'].rolling(20).mean() * 1.5

                if pullback_level and rsi_ok:
                    signal = -1
                    atr = row['ATR']
                    stop_loss = entry_price + self.get_param('atr_multiplier_sl', 2.0) * atr
                    target = entry_price - self.get_param('atr_multiplier_tp', 3.0) * atr

            signals_list.append({
                'date': date,
                'symbol': symbol,
                'signal': signal,
                'entry_price': entry_price if signal != 0 else np.nan,
                'stop_loss': stop_loss if signal != 0 else np.nan,
                'target': target if signal != 0 else np.nan,
                'position_size': self.get_param('position_size', 0.1),
                'trend_up': trend_up,
                'trend_down': trend_down,
                'rsi': row['RSI'],
                'adx': row['ADX']
            })

        signals_df = pd.DataFrame(signals_list)
        if not signals_df.empty:
            signals_df = signals_df.set_index(['date', 'symbol'])
        return signals_df


class BreakoutStrategy(SwingStrategy):
    """
    Breakout Strategy.
    - Enter on break of N-day high/low with volume confirmation
    - Stop at breakout candle low/high or ATR-based
    - Trail with ATR
    """

    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        required = ['breakout_period', 'volume_surge_pct', 'atr_multiplier_sl', 'atr_multiplier_tp']
        for key in required:
            if key not in params:
                return False, f"Missing required parameter: {key}"
        return True, ""

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        data = self._prepare_data(data)

        period = self.get_param('breakout_period', 20)
        volume_surge = self.get_param('volume_surge_pct', 50) / 100

        # Calculate rolling high/low
        data['Rolling_High'] = data.groupby(level='symbol')['High'].transform(
            lambda x: x.rolling(period).max().shift(1)
        )
        data['Rolling_Low'] = data.groupby(level='symbol')['Low'].transform(
            lambda x: x.rolling(period).min().shift(1)
        )
        data['Avg_Volume'] = data.groupby(level='symbol')['Volume'].transform(
            lambda x: x.rolling(20).mean()
        )

        signals_list = []

        for (date, symbol), row in data.iterrows():
            signal = 0
            entry_price = row['Close']
            stop_loss = None
            target = None

            volume_ok = row['Volume'] > row['Avg_Volume'] * (1 + volume_surge)

            # Long breakout
            if row['Close'] > row['Rolling_High'] and volume_ok:
                signal = 1
                atr = row['ATR']
                stop_loss = row['Low']  # Stop at breakout candle low
                target = entry_price + self.get_param('atr_multiplier_tp', 3.0) * atr

            # Short breakdown
            elif row['Close'] < row['Rolling_Low'] and volume_ok:
                signal = -1
                atr = row['ATR']
                stop_loss = row['High']  # Stop at breakout candle high
                target = entry_price - self.get_param('atr_multiplier_tp', 3.0) * atr

            signals_list.append({
                'date': date,
                'symbol': symbol,
                'signal': signal,
                'entry_price': entry_price if signal != 0 else np.nan,
                'stop_loss': stop_loss if signal != 0 else np.nan,
                'target': target if signal != 0 else np.nan,
                'position_size': self.get_param('position_size', 0.1),
                'breakout_high': row['Rolling_High'],
                'breakout_low': row['Rolling_Low']
            })

        signals_df = pd.DataFrame(signals_list)
        if not signals_df.empty:
            signals_df = signals_df.set_index(['date', 'symbol'])
        return signals_df


class MeanReversionStrategy(SwingStrategy):
    """
    Mean Reversion Strategy.
    - Enter when price deviates significantly from mean (Bollinger Bands/Keltner)
    - RSI oversold/overbought confirmation
    - Exit at mean or opposite band
    """

    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        required = ['bb_period', 'bb_std', 'rsi_oversold', 'rsi_overbought', 'atr_multiplier_sl']
        for key in required:
            if key not in params:
                return False, f"Missing required parameter: {key}"
        return True, ""

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        data = self._prepare_data(data)

        signals_list = []

        for (date, symbol), row in data.iterrows():
            signal = 0
            entry_price = row['Close']
            stop_loss = None
            target = None

            # Long: price at lower BB, RSI oversold
            bb_oversold = row['Close'] <= row['BB_Lower']
            rsi_oversold = row['RSI'] < self.get_param('rsi_oversold', 30)
            # Squeeze: BB inside KC (low volatility)
            squeeze = (row['BB_Upper'] < row['KC_Upper']) & (row['BB_Lower'] > row['KC_Lower'])

            if bb_oversold and rsi_oversold:
                signal = 1
                atr = row['ATR']
                stop_loss = entry_price - self.get_param('atr_multiplier_sl', 2.0) * atr
                target = row['BB_Middle']  # Target middle band

            # Short: price at upper BB, RSI overbought
            bb_overbought = row['Close'] >= row['BB_Upper']
            rsi_overbought = row['RSI'] > self.get_param('rsi_overbought', 70)

            if bb_overbought and rsi_overbought:
                signal = -1
                atr = row['ATR']
                stop_loss = entry_price + self.get_param('atr_multiplier_sl', 2.0) * atr
                target = row['BB_Middle']

            signals_list.append({
                'date': date,
                'symbol': symbol,
                'signal': signal,
                'entry_price': entry_price if signal != 0 else np.nan,
                'stop_loss': stop_loss if signal != 0 else np.nan,
                'target': target if signal != 0 else np.nan,
                'position_size': self.get_param('position_size', 0.1),
                'bb_pct_b': row['BB_PctB'],
                'squeeze': squeeze
            })

        signals_df = pd.DataFrame(signals_list)
        if not signals_df.empty:
            signals_df = signals_df.set_index(['date', 'symbol'])
        return signals_df


# ============================================================
# CONCRETE OPTIONS STRATEGIES
# ============================================================

class CoveredCallStrategy(OptionsStrategy):
    """
    Covered Call Strategy.
    - Own 100 shares (or 1 lot), sell OTM call
    - Target delta: 0.20-0.35
    - DTE: 30-45 days
    - IV Rank > 50
    """

    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        required = ['target_delta', 'min_dte', 'max_dte', 'iv_rank_min', 'profit_target_pct']
        for key in required:
            if key not in params:
                return False, f"Missing required parameter: {key}"
        return True, ""

    def generate_signals(self, data: pd.DataFrame,
                         option_chains: Dict = None, **kwargs) -> pd.DataFrame:
        if option_chains is None:
            return pd.DataFrame()

        signals_list = []

        for (date, symbol), row in data.iterrows():
            if symbol not in option_chains:
                continue

            chain = option_chains[symbol]
            spot = row['Close']

            # Only if we have long position (simplified: assume we do)
            # In reality, check portfolio for existing long position

            # Filter chain for suitable expiries
            expiries = chain['expiry'].unique()
            for expiry in expiries:
                exp_chain = chain[chain['expiry'] == expiry]
                dte = (pd.to_datetime(expiry) - date).days

                if not (self.get_param('min_dte', 14) <= dte <= self.get_param('max_dte', 45)):
                    continue

                # Check IV Rank
                if not self._validate_iv_rank(exp_chain, self.get_param('iv_rank_min', 50)):
                    continue

                # Find call strike near target delta
                target_delta = self.get_param('target_delta', 0.30)
                strike = self._select_strikes(exp_chain, spot, target_delta, 'call')

                if strike is None:
                    continue

                call = exp_chain[(exp_chain['strike'] == strike) &
                                 (exp_chain['option_type'] == 'call')]
                if call.empty:
                    continue

                call_row = call.iloc[0]
                premium = call_row['LTP']

                # Create signal
                signals_list.append({
                    'date': date,
                    'symbol': symbol,
                    'signal': 1,  # Sell call
                    'strategy': 'covered_call',
                    'legs': [{
                        'strike': strike,
                        'expiry': expiry,
                        'option_type': 'call',
                        'action': 'sell',
                        'quantity': 1,
                        'entry_price': premium,
                        'delta': call_row.get('Delta', 0)
                    }],
                    'net_premium': premium,
                    'max_profit': premium + (strike - spot) if strike > spot else premium,
                    'max_loss': spot - strike + premium if strike < spot else spot,
                    'position_size': 1,
                    'dte': dte,
                    'iv_rank': self.get_param('iv_rank_min', 50)
                })

        signals_df = pd.DataFrame(signals_list)
        if not signals_df.empty:
            signals_df = signals_df.set_index(['date', 'symbol'])
        return signals_df


class CashSecuredPutStrategy(OptionsStrategy):
    """
    Cash Secured Put Strategy.
    - Sell OTM put, cash secured
    - Target delta: 0.20-0.35
    - Willing to buy at strike price
    - IV Rank > 50
    """

    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        required = ['target_delta', 'min_dte', 'max_dte', 'iv_rank_min', 'profit_target_pct']
        for key in required:
            if key not in params:
                return False, f"Missing required parameter: {key}"
        return True, ""

    def generate_signals(self, data: pd.DataFrame,
                         option_chains: Dict = None, **kwargs) -> pd.DataFrame:
        if option_chains is None:
            return pd.DataFrame()

        signals_list = []

        for (date, symbol), row in data.iterrows():
            if symbol not in option_chains:
                continue

            chain = option_chains[symbol]
            spot = row['Close']

            # Trend should be neutral to bullish
            if row.get('Trend_Down', False):
                continue

            expiries = chain['expiry'].unique()
            for expiry in expiries:
                exp_chain = chain[chain['expiry'] == expiry]
                dte = (pd.to_datetime(expiry) - date).days

                if not (self.get_param('min_dte', 14) <= dte <= self.get_param('max_dte', 45)):
                    continue

                if not self._validate_iv_rank(exp_chain, self.get_param('iv_rank_min', 50)):
                    continue

                target_delta = self.get_param('target_delta', 0.30)
                strike = self._select_strikes(exp_chain, spot, target_delta, 'put')

                if strike is None:
                    continue

                put = exp_chain[(exp_chain['strike'] == strike) &
                                (exp_chain['option_type'] == 'put')]
                if put.empty:
                    continue

                put_row = put.iloc[0]
                premium = put_row['LTP']

                # Only if we're willing to own at strike
                if strike > spot * 1.05:  # Don't sell puts too far ITM
                    continue

                signals_list.append({
                    'date': date,
                    'symbol': symbol,
                    'signal': 1,  # Sell put
                    'strategy': 'cash_secured_put',
                    'legs': [{
                        'strike': strike,
                        'expiry': expiry,
                        'option_type': 'put',
                        'action': 'sell',
                        'quantity': 1,
                        'entry_price': premium,
                        'delta': put_row.get('Delta', 0)
                    }],
                    'net_premium': premium,
                    'max_profit': premium,
                    'max_loss': strike - premium,  # If assigned at 0
                    'position_size': 1,
                    'dte': dte,
                    'strike_pct_below_spot': (spot - strike) / spot * 100
                })

        signals_df = pd.DataFrame(signals_list)
        if not signals_df.empty:
            signals_df = signals_df.set_index(['date', 'symbol'])
        return signals_df


class IronCondorStrategy(OptionsStrategy):
    """
    Iron Condor Strategy.
    - Sell OTM call spread + sell OTM put spread
    - Target delta: 0.15-0.25 for short strikes
    - Width: 50-100 points (NIFTY)
    - IV Rank > 60 (high IV for premium selling)
    - DTE: 30-45
    """

    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        required = ['call_delta', 'put_delta', 'spread_width', 'min_dte', 'max_dte', 'iv_rank_min']
        for key in required:
            if key not in params:
                return False, f"Missing required parameter: {key}"
        return True, ""

    def generate_signals(self, data: pd.DataFrame,
                         option_chains: Dict = None, **kwargs) -> pd.DataFrame:
        if option_chains is None:
            return pd.DataFrame()

        signals_list = []

        for (date, symbol), row in data.iterrows():
            if symbol not in option_chains:
                continue

            chain = option_chains[symbol]
            spot = row['Close']

            # Prefer low ADX (range-bound market)
            if row.get('ADX', 0) > 25:
                continue

            expiries = chain['expiry'].unique()
            for expiry in expiries:
                exp_chain = chain[chain['expiry'] == expiry]
                dte = (pd.to_datetime(expiry) - date).days

                if not (self.get_param('min_dte', 21) <= dte <= self.get_param('max_dte', 45)):
                    continue

                if not self._validate_iv_rank(exp_chain, self.get_param('iv_rank_min', 60)):
                    continue

                # Find short call strike
                call_delta = self.get_param('call_delta', 0.20)
                call_strike = self._select_strikes(exp_chain, spot, call_delta, 'call')

                # Find short put strike
                put_delta = self.get_param('put_delta', 0.20)
                put_strike = self._select_strikes(exp_chain, spot, put_delta, 'put')

                if call_strike is None or put_strike is None:
                    continue

                width = self.get_param('spread_width', 100)
                long_call_strike = call_strike + width
                long_put_strike = put_strike - width

                # Get all four legs
                legs = []
                net_premium = 0

                for strike, opt_type, action in [
                    (call_strike, 'call', 'sell'),
                    (long_call_strike, 'call', 'buy'),
                    (put_strike, 'put', 'sell'),
                    (long_put_strike, 'put', 'buy')
                ]:
                    leg = exp_chain[(exp_chain['strike'] == strike) &
                                    (exp_chain['option_type'] == opt_type)]
                    if leg.empty:
                        break
                    leg_row = leg.iloc[0]
                    qty = 1 if action == 'sell' else 1
                    price = leg_row['LTP']
                    legs.append({
                        'strike': strike,
                        'expiry': expiry,
                        'option_type': opt_type,
                        'action': action,
                        'quantity': qty,
                        'entry_price': price,
                        'delta': leg_row.get('Delta', 0)
                    })
                    net_premium += price if action == 'sell' else -price

                if len(legs) != 4:
                    continue

                # Validate: short strikes should be OTM
                if call_strike <= spot or put_strike >= spot:
                    continue

                max_profit = net_premium
                max_loss = width - net_premium

                signals_list.append({
                    'date': date,
                    'symbol': symbol,
                    'signal': 1,
                    'strategy': 'iron_condor',
                    'legs': legs,
                    'net_premium': net_premium,
                    'max_profit': max_profit,
                    'max_loss': max_loss,
                    'position_size': 1,
                    'dte': dte,
                    'call_strike': call_strike,
                    'put_strike': put_strike,
                    'width': width
                })

        signals_df = pd.DataFrame(signals_list)
        if not signals_df.empty:
            signals_df = signals_df.set_index(['date', 'symbol'])
        return signals_df


class CreditSpreadStrategy(OptionsStrategy):
    """
    Credit Spread (Bull Put / Bear Call).
    - Directional credit spread
    - Sell closer strike, buy further strike
    - IV Rank > 50
    """

    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        required = ['direction', 'short_delta', 'spread_width', 'min_dte', 'max_dte', 'iv_rank_min']
        for key in required:
            if key not in params:
                return False, f"Missing required parameter: {key}"
        return True, ""

    def generate_signals(self, data: pd.DataFrame,
                         option_chains: Dict = None, **kwargs) -> pd.DataFrame:
        if option_chains is None:
            return pd.DataFrame()

        direction = self.get_param('direction', 'bull_put')  # bull_put, bear_call
        signals_list = []

        for (date, symbol), row in data.iterrows():
            if symbol not in option_chains:
                continue

            chain = option_chains[symbol]
            spot = row['Close']

            # Direction filter
            if direction == 'bull_put' and row.get('Trend_Down', False):
                continue
            if direction == 'bear_call' and row.get('Trend_Up', False):
                continue

            expiries = chain['expiry'].unique()
            for expiry in expiries:
                exp_chain = chain[chain['expiry'] == expiry]
                dte = (pd.to_datetime(expiry) - date).days

                if not (self.get_param('min_dte', 14) <= dte <= self.get_param('max_dte', 45)):
                    continue

                if not self._validate_iv_rank(exp_chain, self.get_param('iv_rank_min', 50)):
                    continue

                short_delta = self.get_param('short_delta', 0.30)
                width = self.get_param('spread_width', 100)

                if direction == 'bull_put':
                    short_strike = self._select_strikes(exp_chain, spot, short_delta, 'put')
                    if short_strike is None:
                        continue
                    long_strike = short_strike - width
                    opt_type = 'put'
                else:
                    short_strike = self._select_strikes(exp_chain, spot, short_delta, 'call')
                    if short_strike is None:
                        continue
                    long_strike = short_strike + width
                    opt_type = 'call'

                # Get legs
                legs = []
                net_premium = 0

                for strike, action in [(short_strike, 'sell'), (long_strike, 'buy')]:
                    leg = exp_chain[(exp_chain['strike'] == strike) &
                                    (exp_chain['option_type'] == opt_type)]
                    if leg.empty:
                        break
                    leg_row = leg.iloc[0]
                    price = leg_row['LTP']
                    legs.append({
                        'strike': strike,
                        'expiry': expiry,
                        'option_type': opt_type,
                        'action': action,
                        'quantity': 1,
                        'entry_price': price,
                        'delta': leg_row.get('Delta', 0)
                    })
                    net_premium += price if action == 'sell' else -price

                if len(legs) != 2:
                    continue

                max_profit = net_premium
                max_loss = width - net_premium

                signals_list.append({
                    'date': date,
                    'symbol': symbol,
                    'signal': 1,
                    'strategy': direction,
                    'legs': legs,
                    'net_premium': net_premium,
                    'max_profit': max_profit,
                    'max_loss': max_loss,
                    'position_size': 1,
                    'dte': dte
                })

        signals_df = pd.DataFrame(signals_list)
        if not signals_df.empty:
            signals_df = signals_df.set_index(['date', 'symbol'])
        return signals_df


# ============================================================
# STRATEGY FACTORY
# ============================================================

class StrategyFactory:
    """Factory for creating strategy instances."""

    _strategies = {
        # Swing strategies
        'trend_pullback': TrendPullbackStrategy,
        'breakout': BreakoutStrategy,
        'mean_reversion': MeanReversionStrategy,
        # Options strategies
        'covered_call': CoveredCallStrategy,
        'cash_secured_put': CashSecuredPutStrategy,
        'iron_condor': IronCondorStrategy,
        'credit_spread': CreditSpreadStrategy,
    }

    @classmethod
    def create(cls, name: str, params: Dict = None) -> BaseStrategy:
        """Create strategy instance by name."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name](params)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List available strategy names."""
        return list(cls._strategies.keys())

    @classmethod
    def register(cls, name: str, strategy_class: type):
        """Register a custom strategy."""
        cls._strategies[name] = strategy_class


# ============================================================
# SIGNAL PROCESSING UTILITIES
# ============================================================

def combine_signals(signals_list: List[pd.DataFrame],
                    method: str = 'consensus') -> pd.DataFrame:
    """
    Combine signals from multiple strategies.

    method: 'consensus' (all agree), 'any' (any signals), 'weighted'
    """
    if not signals_list:
        return pd.DataFrame()

    combined = pd.concat(signals_list)
    combined = combined.groupby(['date', 'symbol']).agg({
        'signal': 'sum',  # Sum of signals
        'entry_price': 'first',
        'stop_loss': 'first',
        'target': 'first',
    }).reset_index()

    if method == 'consensus':
        # Only keep if all strategies agree (same direction)
        combined['signal'] = np.where(combined['signal'] > 0, 1,
                               np.where(combined['signal'] < 0, -1, 0))
    elif method == 'any':
        combined['signal'] = np.sign(combined['signal'])

    return combined.set_index(['date', 'symbol'])


def filter_signals(signals: pd.DataFrame,
                   min_adv: float = None,
                   max_volatility: float = None,
                   min_price: float = None,
                   max_price: float = None) -> pd.DataFrame:
    """Filter signals based on liquidity and price criteria."""
    # This would need additional data columns
    # Placeholder for filtering logic
    return signals


def generate_exit_signals(positions: List[Position],
                          data: pd.DataFrame,
                          config: Dict) -> List[Tuple[str, str]]:
    """
    Generate exit signals for open positions.

    Returns list of (symbol, exit_reason).
    """
    exits = []
    for pos in positions:
        try:
            row = data.loc[(data.index.get_level_values('date')[-1], pos.symbol)]
        except KeyError:
            continue

        current_price = row['Close']
        high = row['High']
        low = row['Low']

        # Update trailing stop
        if pos.side == 'long':
            pos.highest_price = max(pos.highest_price, high)
            trailing_sl = pos.highest_price - config['swing']['risk']['trailing_atr_multiplier'] * row['ATR']
            if trailing_sl > pos.stop_loss:
                pos.stop_loss = trailing_sl

            # Check exits
            if low <= pos.stop_loss:
                exits.append((pos.symbol, 'stop_loss'))
            elif pos.target and high >= pos.target:
                exits.append((pos.symbol, 'target'))
        else:
            pos.lowest_price = min(pos.lowest_price, low)
            trailing_sl = pos.lowest_price + config['swing']['risk']['trailing_atr_multiplier'] * row['ATR']
            if trailing_sl < pos.stop_loss:
                pos.stop_loss = trailing_sl

            if high >= pos.stop_loss:
                exits.append((pos.symbol, 'stop_loss'))
            elif pos.target and low <= pos.target:
                exits.append((pos.symbol, 'target'))

        # Time exit
        hold_days = (data.index.get_level_values('date')[-1] - pos.entry_date).days
        if hold_days >= config['swing']['risk']['max_hold_days']:
            exits.append((pos.symbol, 'time_exit'))

    return exits


# ============================================================
# MAIN EXPORT
# ============================================================

__all__ = [
    'BaseStrategy', 'SwingStrategy', 'OptionsStrategy',
    'EquitySignal', 'OptionSignal', 'Position',
    'TrendPullbackStrategy', 'BreakoutStrategy', 'MeanReversionStrategy',
    'CoveredCallStrategy', 'CashSecuredPutStrategy',
    'IronCondorStrategy', 'CreditSpreadStrategy',
    'StrategyFactory',
    'combine_signals', 'filter_signals', 'generate_exit_signals'
]


if __name__ == "__main__":
    # Quick test
    print("Available strategies:", StrategyFactory.list_strategies())

    # Test creating strategies
    trend_strat = StrategyFactory.create('trend_pullback', {
        'ema_fast': 20, 'ema_medium': 50, 'ema_slow': 200,
        'pullback_depth_pct': 8, 'atr_multiplier_sl': 2.0,
        'atr_multiplier_tp': 3.0, 'rsi_pullback_max': 50
    })
    print(f"Created: {trend_strat.name}")

    cc_strat = StrategyFactory.create('covered_call', {
        'target_delta': 0.30, 'min_dte': 14, 'max_dte': 45,
        'iv_rank_min': 50, 'profit_target_pct': 50
    })
    print(f"Created: {cc_strat.name}")