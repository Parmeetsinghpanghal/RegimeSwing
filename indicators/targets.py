"""
Target & Exit Calculator - Computes stops, targets, trailing stops, and time-based exits.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StopMethod(Enum):
    ATR = "atr"
    CHANDELIER = "chandelier"
    STRUCTURE = "structure"
    PERCENTAGE = "percentage"
    VWAP = "vwap"


class TargetMethod(Enum):
    R_MULTIPLE = "r_multiple"
    FIBONACCI = "fibonacci"
    STRUCTURE = "structure"
    MAX_PAIN = "max_pain"
    VWAP_BANDS = "vwap_bands"
    CHANDELIER_TRAIL = "chandelier_trail"


@dataclass
class StopTarget:
    """Container for stop and target prices."""
    entry_price: float
    stop_price: float
    targets: List[float]
    trailing_stop: Optional[float] = None
    risk_per_share: float = 0.0
    risk_pct: float = 0.0
    r_multiple_targets: List[float] = None

    def __post_init__(self):
        if self.risk_per_share == 0:
            self.risk_per_share = abs(self.entry_price - self.stop_price)
        if self.entry_price > 0:
            self.risk_pct = self.risk_per_share / self.entry_price


class TargetCalculator:
    """
    Calculate stops, targets, and trailing exits based on multiple methods.
    """

    def __init__(self, config: dict):
        self.config = config
        self.targets_config = config.get('targets', {})
        self.swing_config = config.get('swing', {})

        # Default parameters
        self.atr_period = self.targets_config.get('atr_period', 14)
        self.stop_atr_mult = self.targets_config.get('stop_atr_mult', 2.0)
        self.target_atr_mults = self.targets_config.get('target_atr_mult', [1.5, 2.5, 3.5])
        self.chandelier_period = self.targets_config.get('chandelier_period', 22)
        self.chandelier_mult = self.targets_config.get('chandelier_mult', 3.0)
        self.fib_extensions = self.targets_config.get('fib_extensions', [1.272, 1.618, 2.0, 2.618])
        self.use_structure = self.targets_config.get('use_structure_stops', True)
        self.structure_lookback = self.targets_config.get('structure_lookback', 20)
        self.force_close_hour = self.targets_config.get('force_close_hour', 15.15)
        self.options_expiry_exit_days = self.targets_config.get('options_expiry_exit_days', 3)
        self.use_max_pain = self.targets_config.get('use_max_pain_target', True)

    def calculate_stop(
        self,
        entry_price: float,
        atr: float,
        df: pd.DataFrame,
        method: StopMethod = StopMethod.ATR,
        side: str = "long"
    ) -> float:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price
            atr: Current ATR value
            df: DataFrame with indicators (must have High, Low, Chandelier levels)
            method: Stop calculation method
            side: "long" or "short"

        Returns:
            Stop price
        """
        if method == StopMethod.ATR:
            if side == "long":
                return entry_price - self.stop_atr_mult * atr
            else:
                return entry_price + self.stop_atr_mult * atr

        elif method == StopMethod.CHANDELIER:
            if 'Chandelier_Long' in df.columns and 'Chandelier_Short' in df.columns:
                latest = df.iloc[-1]
                if side == "long":
                    return latest['Chandelier_Long']
                else:
                    return latest['Chandelier_Short']
            # Fallback to ATR
            return self.calculate_stop(entry_price, atr, df, StopMethod.ATR, side)

        elif method == StopMethod.STRUCTURE:
            if self.use_structure and len(df) >= self.structure_lookback:
                recent = df.tail(self.structure_lookback)
                if side == "long":
                    # Use swing low minus buffer
                    swing_low = recent['Low'].min()
                    buffer = 0.5 * atr
                    return swing_low - buffer
                else:
                    swing_high = recent['High'].max()
                    buffer = 0.5 * atr
                    return swing_high + buffer
            return self.calculate_stop(entry_price, atr, df, StopMethod.ATR, side)

        elif method == StopMethod.VWAP:
            if 'VWAP' in df.columns:
                vwap = df['VWAP'].iloc[-1]
                if side == "long":
                    return min(vwap, entry_price - self.stop_atr_mult * atr)
                else:
                    return max(vwap, entry_price + self.stop_atr_mult * atr)
            return self.calculate_stop(entry_price, atr, df, StopMethod.ATR, side)

        elif method == StopMethod.PERCENTAGE:
            pct = self.targets_config.get('stop_percentage', 0.05)
            if side == "long":
                return entry_price * (1 - pct)
            else:
                return entry_price * (1 + pct)

        return self.calculate_stop(entry_price, atr, df, StopMethod.ATR, side)

    def calculate_targets(
        self,
        entry_price: float,
        stop_price: float,
        atr: float,
        df: pd.DataFrame,
        max_pain: Optional[float] = None,
        methods: List[TargetMethod] = None
    ) -> List[float]:
        """
        Calculate multiple target prices.

        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            atr: Current ATR
            df: DataFrame with indicators
            max_pain: Max pain strike (for options)
            methods: List of target methods to use

        Returns:
            List of target prices (sorted)
        """
        if methods is None:
            methods = [
                TargetMethod.R_MULTIPLE,
                TargetMethod.FIBONACCI,
                TargetMethod.CHANDELIER_TRAIL
            ]

        risk = abs(entry_price - stop_price)
        targets = []

        for method in methods:
            if method == TargetMethod.R_MULTIPLE:
                for mult in self.target_atr_mults:
                    target = entry_price + mult * risk
                    targets.append(target)

            elif method == TargetMethod.FIBONACCI:
                # Use Fibonacci extensions from recent swing
                if 'Fib_1.272' in df.columns:
                    recent = df.tail(1)
                    for fib_level in self.fib_extensions:
                        col = f'Fib_{fib_level:.3f}'.rstrip('0').rstrip('.')
                        if col in recent.columns:
                            targets.append(recent[col].values[0])

            elif method == TargetMethod.STRUCTURE:
                if self.use_structure and len(df) >= self.structure_lookback:
                    recent = df.tail(self.structure_lookback)
                    swing_high = recent['High'].max()
                    if entry_price < swing_high:
                        targets.append(swing_high)

            elif method == TargetMethod.MAX_PAIN:
                if self.use_max_pain and max_pain:
                    targets.append(max_pain)

            elif method == TargetMethod.VWAP_BANDS:
                if 'VWAP_Upper_2SD' in df.columns:
                    vwap_upper = df['VWAP_Upper_2SD'].iloc[-1]
                    if entry_price < vwap_upper:
                        targets.append(vwap_upper)

            elif method == TargetMethod.CHANDELIER_TRAIL:
                # This is a trailing target - handled separately
                pass

        # Deduplicate and sort
        targets = sorted(set([round(t, 2) for t in targets if t > entry_price]))
        return targets

    def calculate_trailing_stop(
        self,
        current_price: float,
        highest_since_entry: float,
        atr: float,
        method: str = "chandelier"
    ) -> float:
        """
        Calculate trailing stop for open position.

        Args:
            current_price: Current price
            highest_since_entry: Highest price since entry (for longs)
            atr: Current ATR
            method: "chandelier", "atr", "percentage"

        Returns:
            Trailing stop price
        """
        if method == "chandelier":
            return highest_since_entry - self.chandelier_mult * atr

        elif method == "atr":
            return highest_since_entry - self.stop_atr_mult * atr

        elif method == "percentage":
            trail_pct = self.targets_config.get('trail_percentage', 0.05)
            return highest_since_entry * (1 - trail_pct)

        return highest_since_entry - self.chandelier_mult * atr

    def time_based_exit(
        self,
        entry_time: pd.Timestamp,
        current_time: pd.Timestamp,
        force_close_hour: float = None
    ) -> bool:
        """
        Check if time-based exit should trigger.

        Args:
            entry_time: Entry timestamp
            current_time: Current timestamp
            force_close_hour: Hour to force close (e.g., 15.15 for 3:15 PM)

        Returns:
            True if should exit
        """
        if force_close_hour is None:
            force_close_hour = self.force_close_hour

        # Force close at specified hour
        if current_time.hour + current_time.minute / 60 >= force_close_hour:
            return True

        # Max hold days
        max_days = self.targets_config.get('max_hold_days_default', 20)
        if (current_time - entry_time).days >= max_days:
            return True

        return False

    def options_expiry_exit(self, dte: int, days_before: int = None) -> bool:
        """Check if should exit due to options expiry approaching."""
        if days_before is None:
            days_before = self.options_expiry_exit_days
        return dte <= days_before

    def quick_exit_check(
        self,
        entry_price: float,
        current_price: float,
        days_held: int,
        min_move_pct: float = 0.02,
        max_days_no_move: int = 3
    ) -> bool:
        """
        Quick exit for mean reversion trades that aren't working.

        Args:
            entry_price: Entry price
            current_price: Current price
            days_held: Days since entry
            min_move_pct: Minimum expected move %
            max_days_no_move: Max days without min move

        Returns:
            True if should exit
        """
        if days_held >= max_days_no_move:
            move = abs(current_price - entry_price) / entry_price
            if move < min_move_pct:
                return True
        return False

    def build_stop_target_object(
        self,
        entry_price: float,
        atr: float,
        df: pd.DataFrame,
        side: str = "long",
        max_pain: Optional[float] = None,
        stop_method: StopMethod = StopMethod.CHANDELIER
    ) -> StopTarget:
        """Build complete StopTarget object with all levels."""
        stop = self.calculate_stop(entry_price, atr, df, stop_method, side)
        targets = self.calculate_targets(entry_price, stop, atr, df, max_pain)
        trail = self.calculate_trailing_stop(entry_price, entry_price, atr)

        return StopTarget(
            entry_price=entry_price,
            stop_price=stop,
            targets=targets,
            trailing_stop=trail,
            risk_per_share=abs(entry_price - stop),
            risk_pct=abs(entry_price - stop) / entry_price if entry_price > 0 else 0
        )


# ============================================================
# POSITION SIZING
# ============================================================

def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    entry_price: float,
    stop_price: float,
    max_position_pct: float = 0.05,
    lot_size: int = 1
) -> int:
    """
    Calculate position size based on risk.

    Args:
        capital: Total capital
        risk_per_trade: Risk per trade as fraction (e.g., 0.015 = 1.5%)
        entry_price: Entry price
        stop_price: Stop loss price
        max_position_pct: Maximum position as % of capital
        lot_size: Minimum lot size (1 for equity, 75 for Nifty options, etc.)

    Returns:
        Number of shares/contracts (rounded to lot size)
    """
    risk_amount = capital * risk_per_trade
    risk_per_share = abs(entry_price - stop_price)

    if risk_per_share <= 0:
        return 0

    # Size based on risk
    shares_by_risk = risk_amount / risk_per_share

    # Size based on max position %
    max_shares = (capital * max_position_pct) / entry_price

    # Take minimum
    shares = min(shares_by_risk, max_shares)

    # Round to lot size
    shares = int(shares / lot_size) * lot_size

    return max(shares, 0)


def calculate_option_position_size(
    capital: float,
    risk_per_trade: float,
    premium: float,
    lot_size: int,
    max_positions: int = 3
) -> int:
    """
    Calculate option selling position size (CSP/CC).

    For CSP: Need cash = strike * lot_size * contracts
    For CC: Need shares = lot_size * contracts
    """
    # Risk per contract = max loss per contract
    # For CSP: max loss = (strike - premium) * lot_size ≈ strike * lot_size (conservative)
    # For simplicity, use premium * lot_size as margin proxy

    max_risk_amount = capital * risk_per_trade
    margin_per_contract = premium * lot_size  # Approximate

    contracts_by_risk = max_risk_amount / margin_per_contract
    contracts_by_capital = (capital / max_positions) / margin_per_contract

    contracts = min(contracts_by_risk, contracts_by_capital)
    contracts = int(contracts)

    return max(contracts, 0)


# ============================================================
# RISK/REWARD ANALYSIS
# ============================================================

def analyze_risk_reward(
    entry: float,
    stop: float,
    targets: List[float],
    position_size: int
) -> Dict:
    """Analyze risk/reward for a trade setup."""
    risk_per_share = abs(entry - stop)
    total_risk = risk_per_share * position_size

    rewards = []
    for i, target in enumerate(targets):
        reward_per_share = target - entry
        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        total_reward = reward_per_share * position_size
        rewards.append({
            'target_num': i + 1,
            'price': target,
            'reward_per_share': reward_per_share,
            'rr_ratio': round(rr_ratio, 2),
            'total_reward': total_reward
        })

    return {
        'entry': entry,
        'stop': stop,
        'risk_per_share': risk_per_share,
        'total_risk': total_risk,
        'position_size': position_size,
        'targets': rewards,
        'best_rr': max([r['rr_ratio'] for r in rewards]) if rewards else 0
    }


if __name__ == "__main__":
    # Test
    import yfinance as yf
    from indicators.technical import add_all_indicators, ensure_multiindex

    logging.basicConfig(level=logging.INFO)

    df = yf.download("RELIANCE.NS", period="6mo", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.capitalize() for c in df.columns]
    df['Symbol'] = 'RELIANCE'
    df.set_index('Symbol', append=True, inplace=True)
    df = df.reorder_levels(['Symbol', 'Date'])

    df = add_all_indicators(df, {
        'indicators': {
            'atr_period': 14,
            'chandelier_period': 22,
            'chandelier_mult': 3.0,
        }
    })

    # Test calculator
    config = {
        'targets': {
            'atr_period': 14,
            'stop_atr_mult': 2.0,
            'target_atr_mult': [1.5, 2.5, 3.5],
            'chandelier_period': 22,
            'chandelier_mult': 3.0,
            'fib_extensions': [1.272, 1.618, 2.0, 2.618],
            'use_structure_stops': True,
            'structure_lookback': 20,
            'force_close_hour': 15.15,
            'options_expiry_exit_days': 3,
            'use_max_pain_target': True,
        }
    }

    calc = TargetCalculator(config)

    latest = df.xs('RELIANCE', level='Symbol').iloc[-1]
    entry = latest['Close']
    atr = latest.get('ATR_14', 30)

    st = calc.build_stop_target_object(entry, atr, df.xs('RELIANCE', level='Symbol'))

    print(f"Entry: {entry:.2f}")
    print(f"Stop: {st.stop_price:.2f}")
    print(f"Risk: {st.risk_pct:.2%}")
    print(f"Targets: {[f'{t:.2f}' for t in st.targets]}")
    print(f"Trailing Stop: {st.trailing_stop:.2f}")

    # Position sizing
    shares = calculate_position_size(500000, 0.015, entry, st.stop_price)
    print(f"\nPosition Size: {shares} shares")

    rr = analyze_risk_reward(entry, st.stop_price, st.targets, shares)
    for t in rr['targets']:
        print(f"  T{t['target_num']}: {t['price']:.2f} | R:R = {t['rr_ratio']} | Reward: ₹{t['total_reward']:,.0f}")