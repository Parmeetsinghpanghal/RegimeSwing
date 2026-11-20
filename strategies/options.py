"""
Options Selling Strategies - Cash-Secured Puts, Covered Calls, Wheel, Credit Spreads.
Rule-based with IV rank, delta targeting, and risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class OptionType(Enum):
    CALL = "CE"
    PUT = "PE"


class OptionAction(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OptionLeg:
    """Single option leg."""
    symbol: str           # Underlying symbol
    strike: float
    expiry: pd.Timestamp
    option_type: OptionType
    action: OptionAction  # BUY or SELL
    quantity: int         # Number of lots
    premium: float        # Premium per share
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0

    @property
    def total_premium(self) -> float:
        return self.premium * self.quantity * self.lot_size

    @property
    def lot_size(self) -> int:
        # Standard lot sizes (approximate)
        lot_sizes = {
            'NIFTY': 50, 'BANKNIFTY': 25, 'FINNIFTY': 40,
            'MIDCPNIFTY': 75
        }
        return lot_sizes.get(self.symbol.replace('.NS', '').split('_')[0], 1)


@dataclass
class OptionsSignal:
    """Options strategy signal."""
    strategy: str
    underlying: str
    timestamp: pd.Timestamp
    legs: List[OptionLeg]
    max_profit: float
    max_loss: float
    breakeven: List[float]
    probability_of_profit: float
    required_capital: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'strategy': self.strategy,
            'underlying': self.underlying,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, pd.Timestamp) else str(self.timestamp),
            'legs': [
                {
                    'strike': leg.strike,
                    'expiry': leg.expiry.isoformat() if isinstance(leg.expiry, pd.Timestamp) else str(leg.expiry),
                    'type': leg.option_type.value,
                    'action': leg.action.value,
                    'quantity': leg.quantity,
                    'premium': leg.premium,
                    'delta': leg.delta,
                    'iv': leg.iv
                }
                for leg in self.legs
            ],
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'breakeven': self.breakeven,
            'pop': self.probability_of_profit,
            'capital': self.required_capital,
            'metadata': self.metadata
        }


class BaseOptionsStrategy(ABC):
    """Base class for options strategies."""

    def __init__(self, config: dict, name: str):
        self.config = config
        self.name = name
        self.strategy_config = config.get('options', {}).get(name, {})
        self.capital_config = config.get('capital', {})
        self.enabled = self.strategy_config.get('enabled', True)

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
        option_chains: Dict[str, pd.DataFrame],
        regime_info: dict = None
    ) -> List[OptionsSignal]:
        """Generate options signals."""
        pass

    def _filter_strikes(
        self,
        chain: pd.DataFrame,
        spot: float,
        delta_range: Tuple[float, float],
        dte_range: Tuple[int, int],
        option_type: OptionType
    ) -> pd.DataFrame:
        """Filter option chain by delta, DTE, and type."""
        if chain.empty:
            return chain

        filtered = chain.copy()

        # Filter by option type
        if 'option_type' in filtered.columns:
            filtered = filtered[filtered['option_type'] == option_type.value]

        # Filter by DTE
        if 'dte' in filtered.columns:
            filtered = filtered[
                (filtered['dte'] >= dte_range[0]) &
                (filtered['dte'] <= dte_range[1])
            ]

        # Filter by delta
        if 'delta' in filtered.columns:
            if option_type == OptionType.PUT:
                # Put delta is negative, we want -0.30 to -0.20 (abs 0.20-0.30)
                filtered = filtered[
                    (filtered['delta'] >= -delta_range[1]) &
                    (filtered['delta'] <= -delta_range[0])
                ]
            else:
                filtered = filtered[
                    (filtered['delta'] >= delta_range[0]) &
                    (filtered['delta'] <= delta_range[1])
                ]

        return filtered

    def _select_best_strike(
        self,
        filtered: pd.DataFrame,
        spot: float,
        option_type: OptionType,
        prefer_higher_premium: bool = True
    ) -> Optional[pd.Series]:
        """Select best strike from filtered chain."""
        if filtered.empty:
            return None

        # Sort by premium (for selling) or delta proximity
        if prefer_higher_premium and 'premium' in filtered.columns:
            return filtered.loc[filtered['premium'].idxmax()]
        elif 'delta' in filtered.columns:
            # Closest to target delta
            target_delta = 0.25 if option_type == OptionType.PUT else 0.25
            filtered = filtered.copy()
            filtered['delta_diff'] = abs(filtered['delta'].abs() - target_delta)
            return filtered.loc[filtered['delta_diff'].idxmin()]

        return filtered.iloc[0]


class CashSecuredPutsStrategy(BaseOptionsStrategy):
    """
    Cash-Secured Puts (CSP):
    - Sell OTM puts on quality stocks in uptrend
    - IV Rank > 60, Delta 0.25-0.30, DTE 30-45
    - Manage: Close at 50% profit, roll at 21 DTE, stop at 2x premium
    """

    def generate_signals(
        self,
        df: pd.DataFrame,
        option_chains: Dict[str, pd.DataFrame],
        regime_info: dict = None
    ) -> List[OptionsSignal]:
        if not self.enabled:
            return []

        signals = []
        params = self.strategy_config.get('cash_secured_puts', {})
        capital = self.capital_config.get('initial', 500000)

        symbols = df.index.get_level_values('Symbol').unique() if isinstance(df.index, pd.MultiIndex) else ['UNKNOWN']

        for symbol in symbols:
            try:
                if symbol not in option_chains:
                    continue

                chain = option_chains[symbol]
                if chain.empty:
                    continue

                sym_df = df.xs(symbol, level='Symbol') if isinstance(df.index, pd.MultiIndex) else df
                latest = sym_df.iloc[-1]
                spot = latest['Close']

                # --- Filters ---
                # 1. Trend: Stock > EMA50 > EMA200
                ema50 = latest.get('EMA_50', spot)
                ema200 = latest.get('EMA_200', spot)
                if not (spot > ema50 > ema200):
                    continue

                # 2. Delivery > 50%
                delivery_pct = latest.get('Delivery_Pct', 100)
                if delivery_pct < params.get('delivery_pct_min', 50):
                    continue

                # 3. IV Rank > 60
                iv_rank = latest.get('IV_Rank', 0)
                if iv_rank < params.get('iv_rank_min', 60):
                    continue

                # 4. No earnings in DTE window
                # (Would need earnings calendar - placeholder)
                earnings_buffer = params.get('earnings_buffer_days', 5)
                # Skip earnings check for now

                # Regime check
                if regime_info:
                    should, _, opt_params = regime_info.get('should_sell_options', lambda r: (True, "", {}))(regime_info)
                    if not should:
                        continue
                    # Use regime-adjusted params
                    iv_rank_min = opt_params.get('iv_rank_min', params.get('iv_rank_min', 60))
                    delta_target = opt_params.get('delta_target', params.get('delta_target_min', 0.25))
                    size_mult = opt_params.get('size_mult', 1.0)
                else:
                    iv_rank_min = params.get('iv_rank_min', 60)
                    delta_target = params.get('delta_target_min', 0.25)
                    size_mult = 1.0

                # --- Select Strike ---
                delta_min = params.get('delta_target_min', 0.25)
                delta_max = params.get('delta_target_max', 0.30)
                dte_min = params.get('dte_min', 30)
                dte_max = params.get('dte_max', 45)

                filtered = self._filter_strikes(
                    chain, spot, (delta_min, delta_max), (dte_min, dte_max), OptionType.PUT
                )

                if filtered.empty:
                    continue

                best = self._select_best_strike(filtered, spot, OptionType.PUT)
                if best is None:
                    continue

                strike = best['strike']
                premium = best.get('premium', 0)
                delta = best.get('delta', 0)
                iv = best.get('iv', 0)
                dte = best.get('dte', 30)
                expiry = best.get('expiry', pd.Timestamp.now() + pd.Timedelta(days=dte))

                # Premium check: > 1.5% of strike per month
                premium_pct_month = (premium / strike) * (30 / dte) * 12
                if premium_pct_month < params.get('premium_per_month_min', 0.015) * 100:
                    continue

                # --- Position Sizing ---
                # Cash needed = strike * lot_size * contracts
                lot_size = self._get_lot_size(symbol)
                max_contracts = int(capital * 0.2 / (strike * lot_size))  # Max 20% capital per CSP
                max_contracts = max_contracts * size_mult
                contracts = min(max_contracts, 3)  # Cap at 3 contracts

                if contracts == 0:
                    continue

                required_capital = strike * lot_size * contracts

                # --- Risk Metrics ---
                max_profit = premium * lot_size * contracts
                max_loss = (strike - premium) * lot_size * contracts  # If assigned at 0
                breakeven = strike - premium
                # Approximate POP using delta
                pop = 1 - abs(delta)  # Rough approximation

                leg = OptionLeg(
                    symbol=symbol,
                    strike=strike,
                    expiry=expiry,
                    option_type=OptionType.PUT,
                    action=OptionAction.SELL,
                    quantity=contracts,
                    premium=premium,
                    delta=delta,
                    iv=iv
                )

                signal = OptionsSignal(
                    strategy='cash_secured_puts',
                    underlying=symbol,
                    timestamp=sym_df.index[-1],
                    legs=[leg],
                    max_profit=max_profit,
                    max_loss=max_loss,
                    breakeven=[breakeven],
                    probability_of_profit=pop,
                    required_capital=required_capital,
                    metadata={
                        'iv_rank': iv_rank,
                        'delta': delta,
                        'dte': dte,
                        'premium_pct_month': premium_pct_month,
                        'contracts': contracts,
                        'management': {
                            'close_at_profit': 0.5,
                            'roll_at_dte': 21,
                            'stop_loss_mult': 2.0
                        }
                    }
                )
                signals.append(signal)

            except Exception as e:
                logger.warning(f"CSP error for {symbol}: {e}")

        return signals

    def _get_lot_size(self, symbol: str) -> int:
        """Get lot size for symbol."""
        index_lots = {'NIFTY': 50, 'BANKNIFTY': 25, 'FINNIFTY': 40}
        for idx, lot in index_lots.items():
            if idx in symbol:
                return lot
        return 1  # Stock options - varies, default 1


class CoveredCallsStrategy(BaseOptionsStrategy):
    """
    Covered Calls (CC):
    - Sell OTM calls on existing long positions > 10% gain
    - Delta 0.30-0.35, DTE 30-45
    - Roll if ITM near expiry
    """

    def generate_signals(
        self,
        df: pd.DataFrame,
        option_chains: Dict[str, pd.DataFrame],
        existing_positions: Dict[str, Dict] = None,
        regime_info: dict = None
    ) -> List[OptionsSignal]:
        if not self.enabled:
            return []

        signals = []
        params = self.strategy_config.get('covered_calls', {})

        if not existing_positions:
            return signals

        for symbol, position in existing_positions.items():
            try:
                if symbol not in option_chains:
                    continue

                chain = option_chains[symbol]
                if chain.empty:
                    continue

                sym_df = df.xs(symbol, level='Symbol') if isinstance(df.index, pd.MultiIndex) else df
                latest = sym_df.iloc[-1]
                spot = latest['Close']

                # Check unrealized gain > 10%
                entry_price = position.get('entry_price', spot)
                qty = position.get('quantity', 0)
                unrealized_pct = (spot - entry_price) / entry_price

                if unrealized_pct < params.get('min_unrealized_gain', 0.10):
                    continue

                if qty == 0:
                    continue

                # Select strike
                delta_min = params.get('delta_target_min', 0.30)
                delta_max = params.get('delta_target_max', 0.35)
                dte_min = params.get('dte_min', 30)
                dte_max = params.get('dte_max', 45)

                filtered = self._filter_strikes(
                    chain, spot, (delta_min, delta_max), (dte_min, dte_max), OptionType.CALL
                )

                if filtered.empty:
                    continue

                best = self._select_best_strike(filtered, spot, OptionType.CALL)
                if best is None:
                    continue

                strike = best['strike']
                premium = best.get('premium', 0)
                delta = best.get('delta', 0)
                iv = best.get('iv', 0)
                dte = best.get('dte', 30)
                expiry = best.get('expiry', pd.Timestamp.now() + pd.Timedelta(days=dte))

                # Strike must be above cost basis
                if params.get('strike_above_cost', True) and strike < entry_price:
                    continue

                # Contracts = shares / lot_size
                lot_size = self._get_lot_size(symbol)
                contracts = qty // lot_size
                contracts = min(contracts, 5)  # Cap

                if contracts == 0:
                    continue

                max_profit = premium * lot_size * contracts
                # Max loss is stock decline - premium received
                breakeven = strike + premium

                leg = OptionLeg(
                    symbol=symbol,
                    strike=strike,
                    expiry=expiry,
                    option_type=OptionType.CALL,
                    action=OptionAction.SELL,
                    quantity=contracts,
                    premium=premium,
                    delta=delta,
                    iv=iv
                )

                signal = OptionsSignal(
                    strategy='covered_calls',
                    underlying=symbol,
                    timestamp=sym_df.index[-1],
                    legs=[leg],
                    max_profit=max_profit,
                    max_loss=entry_price * lot_size * contracts,  # Stock to 0
                    breakeven=[breakeven],
                    probability_of_profit=1 - delta,
                    required_capital=0,  # No additional capital (shares already held)
                    metadata={
                        'entry_price': entry_price,
                        'unrealized_pct': unrealized_pct,
                        'delta': delta,
                        'dte': dte,
                        'contracts': contracts,
                        'management': {
                            'roll_if_itm_dte': 5,
                            'roll_strikes_up': 1,
                            'roll_dte_out': 21
                        }
                    }
                )
                signals.append(signal)

            except Exception as e:
                logger.warning(f"CC error for {symbol}: {e}")

        return signals


class WheelStrategy(BaseOptionsStrategy):
    """
    Wheel Strategy:
    State machine: CSP -> Assignment -> CC -> Called -> CSP
    Tracks cost basis, premium collected, annualized return.
    """

    def __init__(self, config: dict):
        super().__init__(config, 'wheel')
        self.wheels = {}  # symbol -> wheel state

    def generate_signals(
        self,
        df: pd.DataFrame,
        option_chains: Dict[str, pd.DataFrame],
        regime_info: dict = None
    ) -> List[OptionsSignal]:
        if not self.enabled:
            return []

        signals = []
        params = self.strategy_config.get('wheel', {})

        # This strategy manages existing wheels
        # New wheel initiation uses CSP logic
        # For now, return signals for new wheel initiation on quality stocks

        symbols = df.index.get_level_values('Symbol').unique() if isinstance(df.index, pd.MultiIndex) else ['UNKNOWN']
        max_wheels = params.get('max_concurrent_wheels', 3)
        min_capital = params.get('min_capital_per_wheel', 200000)

        active_wheels = len([w for w in self.wheels.values() if w.get('active', False)])
        available_slots = max_wheels - active_wheels

        if available_slots <= 0:
            return signals

        capital = self.capital_config.get('initial', 500000)

        for symbol in symbols[:available_slots]:
            try:
                if symbol in self.wheels and self.wheels[symbol].get('active', False):
                    continue

                if symbol not in option_chains:
                    continue

                chain = option_chains[symbol]
                if chain.empty:
                    continue

                sym_df = df.xs(symbol, level='Symbol') if isinstance(df.index, pd.MultiIndex) else df
                latest = sym_df.iloc[-1]
                spot = latest['Close']

                # Quality filters for wheel
                ema50 = latest.get('EMA_50', spot)
                ema200 = latest.get('EMA_200', spot)
                if not (spot > ema50 > ema200):
                    continue

                delivery_pct = latest.get('Delivery_Pct', 100)
                if delivery_pct < 50:
                    continue

                iv_rank = latest.get('IV_Rank', 0)
                if iv_rank < 50:
                    continue

                # Initiate CSP leg
                csp_params = self.config.get('options', {}).get('cash_secured_puts', {})
                delta_min = csp_params.get('delta_target_min', 0.25)
                delta_max = csp_params.get('delta_target_max', 0.30)
                dte_min = csp_params.get('dte_min', 30)
                dte_max = csp_params.get('dte_max', 45)

                filtered = self._filter_strikes(
                    chain, spot, (delta_min, delta_max), (dte_min, dte_max), OptionType.PUT
                )

                if filtered.empty:
                    continue

                best = self._select_best_strike(filtered, spot, OptionType.PUT)
                if best is None:
                    continue

                strike = best['strike']
                premium = best.get('premium', 0)
                delta = best.get('delta', 0)
                iv = best.get('iv', 0)
                dte = best.get('dte', 30)
                expiry = best.get('expiry', pd.Timestamp.now() + pd.Timedelta(days=dte))

                lot_size = self._get_lot_size(symbol)
                contracts = 1  # Start with 1 contract per wheel
                required_capital = strike * lot_size * contracts

                if required_capital > capital * 0.3:  # Max 30% per wheel
                    continue

                # Record wheel state
                self.wheels[symbol] = {
                    'active': True,
                    'state': 'CSP',  # CSP -> ASSIGNED -> CC -> CALLED
                    'strike': strike,
                    'expiry': expiry,
                    'contracts': contracts,
                    'premium_collected': premium * lot_size * contracts,
                    'cost_basis': strike,  # If assigned
                    'entry_date': latest.name if hasattr(latest, 'name') else pd.Timestamp.now()
                }

                max_profit = premium * lot_size * contracts
                breakeven = strike - premium

                leg = OptionLeg(
                    symbol=symbol,
                    strike=strike,
                    expiry=expiry,
                    option_type=OptionType.PUT,
                    action=OptionAction.SELL,
                    quantity=contracts,
                    premium=premium,
                    delta=delta,
                    iv=iv
                )

                signal = OptionsSignal(
                    strategy='wheel',
                    underlying=symbol,
                    timestamp=sym_df.index[-1],
                    legs=[leg],
                    max_profit=max_profit,
                    max_loss=(strike - premium) * lot_size * contracts,
                    breakeven=[breakeven],
                    probability_of_profit=1 - abs(delta),
                    required_capital=required_capital,
                    metadata={
                        'wheel_state': 'CSP',
                        'delta': delta,
                        'dte': dte,
                        'iv_rank': iv_rank,
                        'management': {
                            'close_at_profit': 0.5,
                            'roll_at_dte': 21,
                            'assignment_action': 'sell_cc'
                        }
                    }
                )
                signals.append(signal)

            except Exception as e:
                logger.warning(f"Wheel error for {symbol}: {e}")

        return signals

    def update_wheel_state(self, symbol: str, event: str, price: float = None):
        """Update wheel state on events (assignment, call away, etc.)."""
        if symbol not in self.wheels:
            return

        wheel = self.wheels[symbol]

        if event == 'assigned':
            wheel['state'] = 'ASSIGNED'
            wheel['cost_basis'] = wheel['strike'] - (wheel['premium_collected'] / (wheel['contracts'] * self._get_lot_size(symbol)))
        elif event == 'called_away':
            wheel['state'] = 'COMPLETED'
            wheel['active'] = False
            # Calculate total return
        elif event == 'cc_sold':
            wheel['state'] = 'CC'
            wheel['premium_collected'] += price * wheel['contracts'] * self._get_lot_size(symbol)


class CreditSpreadsStrategy(BaseOptionsStrategy):
    """
    Credit Spreads:
    - Bull Put Spread (bullish): Sell higher strike put, buy lower strike put
    - Bear Call Spread (bearish): Sell lower strike call, buy higher strike call
    - Iron Condor (neutral): Bull put + Bear call
    """

    def generate_signals(
        self,
        df: pd.DataFrame,
        option_chains: Dict[str, pd.DataFrame],
        regime_info: dict = None
    ) -> List[OptionsSignal]:
        if not self.enabled:
            return []

        signals = []
        params = self.strategy_config.get('credit_spreads', {})

        symbols = df.index.get_level_values('Symbol').unique() if isinstance(df.index, pd.MultiIndex) else ['UNKNOWN']

        for symbol in symbols:
            try:
                if symbol not in option_chains:
                    continue

                chain = option_chains[symbol]
                if chain.empty:
                    continue

                sym_df = df.xs(symbol, level='Symbol') if isinstance(df.index, pd.MultiIndex) else df
                latest = sym_df.iloc[-1]
                spot = latest['Close']

                # Regime-based strategy selection
                if regime_info:
                    r = regime_info.get('regime')
                    if r == 'bull':
                        spread_type = 'bull_put'
                    elif r == 'bear':
                        spread_type = 'bear_call'
                    elif r == 'sideways':
                        spread_type = 'iron_condor'
                    else:
                        spread_type = 'bull_put'  # Default
                else:
                    spread_type = params.get('default_type', 'bull_put')

                # Only trade if IV Rank > 70 for iron condor
                iv_rank = latest.get('IV_Rank', 0)
                if spread_type == 'iron_condor' and iv_rank < params.get('ic_iv_rank_min', 70):
                    continue

                # Generate spread legs
                legs = self._build_spread(chain, spot, spread_type, params)
                if not legs:
                    continue

                # Calculate metrics
                net_premium = sum(leg.premium * leg.quantity for leg in legs if leg.action == OptionAction.SELL) - \
                              sum(leg.premium * leg.quantity for leg in legs if leg.action == OptionAction.BUY)
                lot_size = self._get_lot_size(symbol)

                # Width between strikes
                strikes = [leg.strike for leg in legs]
                width = max(strikes) - min(strikes)
                max_loss = (width - net_premium) * lot_size
                max_profit = net_premium * lot_size
                breakeven = self._calc_breakevens(legs, net_premium, spread_type)

                signal = OptionsSignal(
                    strategy=f'credit_spread_{spread_type}',
                    underlying=symbol,
                    timestamp=sym_df.index[-1],
                    legs=legs,
                    max_profit=max_profit,
                    max_loss=max_loss,
                    breakeven=breakeven,
                    probability_of_profit=0.65,  # Approximate for credit spreads
                    required_capital=max_loss,  # Margin ≈ max loss
                    metadata={
                        'spread_type': spread_type,
                        'width': width,
                        'net_premium': net_premium,
                        'iv_rank': iv_rank
                    }
                )
                signals.append(signal)

            except Exception as e:
                logger.warning(f"CreditSpread error for {symbol}: {e}")

        return signals

    def _build_spread(
        self,
        chain: pd.DataFrame,
        spot: float,
        spread_type: str,
        params: dict
    ) -> List[OptionLeg]:
        """Build spread legs based on type."""
        # Filter for near-term expiries (21-30 DTE)
        dte_min = params.get('dte_min', 21)
        dte_max = params.get('dte_max', 30)

        if spread_type == 'bull_put':
            # Sell put at ~0.20 delta, buy put at ~0.10 delta (5-10 strikes lower)
            short = self._filter_strikes(chain, spot, (0.18, 0.22), (dte_min, dte_max), OptionType.PUT)
            long = self._filter_strikes(chain, spot, (0.08, 0.12), (dte_min, dte_max), OptionType.PUT)

            if short.empty or long.empty:
                return []

            short_leg = self._select_best_strike(short, spot, OptionType.PUT)
            long_leg = self._select_best_strike(long, spot, OptionType.PUT)

            if short_leg is None or long_leg is None:
                return []

            return [
                OptionLeg(symbol='', strike=short_leg['strike'], expiry=short_leg['expiry'],
                          option_type=OptionType.PUT, action=OptionAction.SELL, quantity=1,
                          premium=short_leg['premium'], delta=short_leg['delta']),
                OptionLeg(symbol='', strike=long_leg['strike'], expiry=long_leg['expiry'],
                          option_type=OptionType.PUT, action=OptionAction.BUY, quantity=1,
                          premium=long_leg['premium'], delta=long_leg['delta'])
            ]

        elif spread_type == 'bear_call':
            # Sell call at ~0.20 delta, buy call at ~0.10 delta
            short = self._filter_strikes(chain, spot, (0.18, 0.22), (dte_min, dte_max), OptionType.CALL)
            long = self._filter_strikes(chain, spot, (0.08, 0.12), (dte_min, dte_max), OptionType.CALL)

            if short.empty or long.empty:
                return []

            short_leg = self._select_best_strike(short, spot, OptionType.CALL)
            long_leg = self._select_best_strike(long, spot, OptionType.CALL)

            if short_leg is None or long_leg is None:
                return []

            return [
                OptionLeg(symbol='', strike=short_leg['strike'], expiry=short_leg['expiry'],
                          option_type=OptionType.CALL, action=OptionAction.SELL, quantity=1,
                          premium=short_leg['premium'], delta=short_leg['delta']),
                OptionLeg(symbol='', strike=long_leg['strike'], expiry=long_leg['expiry'],
                          option_type=OptionType.CALL, action=OptionAction.BUY, quantity=1,
                          premium=long_leg['premium'], delta=long_leg['delta'])
            ]

        elif spread_type == 'iron_condor':
            # Combine bull put + bear call
            put_legs = self._build_spread(chain, spot, 'bull_put', params)
            call_legs = self._build_spread(chain, spot, 'bear_call', params)
            return put_legs + call_legs

        return []

    def _calc_breakevens(self, legs: List[OptionLeg], net_premium: float, spread_type: str) -> List[float]:
        """Calculate breakeven points."""
        if spread_type == 'bull_put':
            short_strike = max(leg.strike for leg in legs if leg.action == OptionAction.SELL)
            return [short_strike - net_premium]
        elif spread_type == 'bear_call':
            short_strike = min(leg.strike for leg in legs if leg.action == OptionAction.SELL)
            return [short_strike + net_premium]
        elif spread_type == 'iron_condor':
            put_short = max(leg.strike for leg in legs if leg.option_type == OptionType.PUT and leg.action == OptionAction.SELL)
            call_short = min(leg.strike for leg in legs if leg.option_type == OptionType.CALL and leg.action == OptionAction.SELL)
            return [put_short - net_premium, call_short + net_premium]
        return []


# ============================================================
# OPTIONS REGISTRY
# ============================================================

OPTIONS_STRATEGY_REGISTRY = {
    'cash_secured_puts': CashSecuredPutsStrategy,
    'covered_calls': CoveredCallsStrategy,
    'wheel': WheelStrategy,
    'credit_spreads': CreditSpreadsStrategy,
}


def create_options_strategy(name: str, config: dict) -> BaseOptionsStrategy:
    """Factory for options strategies."""
    if name not in OPTIONS_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown options strategy: {name}")
    return OPTIONS_STRATEGY_REGISTRY[name](config)


def get_all_options_strategies(config: dict) -> Dict[str, BaseOptionsStrategy]:
    """Create all enabled options strategies."""
    strategies = {}
    for name, cls in OPTIONS_STRATEGY_REGISTRY.items():
        strat = cls(config)
        if strat.enabled:
            strategies[name] = strat
    return strategies


def run_all_options_strategies(
    df: pd.DataFrame,
    option_chains: Dict[str, pd.DataFrame],
    config: dict,
    regime_info: dict = None,
    existing_positions: Dict = None
) -> List[OptionsSignal]:
    """Run all enabled options strategies."""
    all_signals = []
    strategies = get_all_options_strategies(config)

    for name, strat in strategies.items():
        try:
            if name == 'covered_calls':
                signals = strat.generate_signals(df, option_chains, existing_positions, regime_info)
            else:
                signals = strat.generate_signals(df, option_chains, regime_info)
            all_signals.extend(signals)
            logger.info(f"{name}: Generated {len(signals)} signals")
        except Exception as e:
            logger.error(f"Options strategy {name} failed: {e}")

    return all_signals


if __name__ == "__main__":
    # Test with mock data
    import yfinance as yf
    from indicators.technical import add_all_indicators, ensure_multiindex
    from indicators.regime import RegimeDetector

    logging.basicConfig(level=logging.INFO)

    # This would need real option chains to test fully
    print("Options strategies module loaded successfully")
    print("Available strategies:", list(OPTIONS_STRATEGY_REGISTRY.keys()))