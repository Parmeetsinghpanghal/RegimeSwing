"""
Walk-Forward Backtesting Engine - Realistic simulation with costs, slippage,
Monte Carlo validation, and regime-aware analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    start_date: str
    end_date: str
    initial_capital: float = 500000
    max_risk_per_trade: float = 0.015
    max_positions: int = 10
    commission_per_order: float = 20
    slippage_bps: float = 15
    stt_rate: float = 0.001
    walk_forward_train_years: int = 3
    walk_forward_test_years: int = 1
    walk_forward_step_years: int = 1
    monte_carlo_sims: int = 500
    block_size: int = 20  # For block bootstrap


@dataclass
class Trade:
    """Single trade record."""
    symbol: str
    strategy: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    quantity: int
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    r_multiple: float
    hold_days: int
    exit_reason: str
    commission: float
    slippage: float


@dataclass
class BacktestResult:
    """Complete backtest result."""
    config: BacktestConfig
    trades: List[Trade]
    equity_curve: pd.Series
    daily_returns: pd.Series
    metrics: Dict
    regime_performance: pd.DataFrame
    monte_carlo: Dict
    parameter_sensitivity: Dict


class CostModel:
    """Realistic Indian market cost model."""

    def __init__(self, config: BacktestConfig):
        self.commission = config.commission_per_order
        self.slippage_bps = config.slippage_bps / 10000
        self.stt_rate = config.stt_rate

    def calculate_costs(self, price: float, quantity: int, side: str, is_option: bool = False) -> Dict:
        """Calculate all costs for a trade."""
        turnover = price * quantity

        # Brokerage
        brokerage = self.commission * 2  # Entry + exit

        # STT
        if is_option:
            stt = turnover * 0.000625  # Option sell side
        else:
            stt = turnover * self.stt_rate  # Equity delivery sell side

        # Exchange charges
        exchange = turnover * 0.0000345

        # SEBI charges
        sebi = turnover * 0.000001

        # Stamp duty (buy side only for equity)
        stamp = turnover * 0.00015 if side == 'buy' else 0

        # Slippage
        slippage_cost = turnover * self.slippage_bps

        total = brokerage + stt + exchange + sebi + stamp + slippage_cost

        return {
            'brokerage': brokerage,
            'stt': stt,
            'exchange': exchange,
            'sebi': sebi,
            'stamp': stamp,
            'slippage': slippage_cost,
            'total': total
        }


class WalkForwardBacktester:
    """
    Walk-forward backtesting with expanding window.
    Trains on N years, tests on 1 year, rolls forward.
    """

    def __init__(self, config: BacktestConfig, data_fetcher, strategy_func: Callable):
        self.config = config
        self.fetcher = data_fetcher
        self.strategy_func = strategy_func  # Function that generates signals
        self.cost_model = CostModel(config)

    def run(self, symbols: List[str], regime_detector=None) -> BacktestResult:
        """Run walk-forward backtest."""
        all_trades = []
        all_equity = []
        all_daily_returns = []

        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)

        # Generate walk-forward windows
        windows = self._generate_windows(start, end)

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Window {i+1}/{len(windows)}: Train {train_start.date()} - {train_end.date()}, "
                       f"Test {test_start.date()} - {test_end.date()}")

            # Fetch data for this window (with buffer for indicators)
            buffer_start = train_start - timedelta(days=250)
            df = self.fetcher.get_price_data(
                symbols,
                start_date=buffer_start.strftime('%Y-%m-%d'),
                end_date=test_end.strftime('%Y-%m-%d'),
                interval='1d'
            )

            if df.empty:
                logger.warning(f"No data for window {i+1}")
                continue

            # Add indicators
            from indicators.technical import add_all_indicators, ensure_multiindex
            df = ensure_multiindex(df)
            df = add_all_indicators(df, {})

            # Regime detection for test period
            nifty_df = self.fetcher.get_nifty_index_data(
                period=f"{(test_end - buffer_start).days}d",
                interval='1d'
            )
            nifty_df = ensure_multiindex(nifty_df)
            nifty_df = add_all_indicators(nifty_df, {})

            regime_info = None
            if regime_detector:
                regime = regime_detector.detect_nifty_regime(nifty_df)
                regime_info = {
                    'regime': regime.regime,
                    'size_mult': regime_detector.get_size_multiplier(regime),
                    'nifty_trend': regime.nifty_trend,
                    'should_trade_swing': regime_detector.should_trade_swing
                }

            # Generate signals on test period
            test_mask = (df.index.get_level_values(1) >= test_start) & \
                       (df.index.get_level_values(1) <= test_end)
            test_df = df[test_mask]

            if test_df.empty:
                continue

            # Run strategy
            signals = self.strategy_func(test_df, {}, regime_info)

            # Simulate trades
            window_trades = self._simulate_trades(signals, test_df, test_start, test_end)
            all_trades.extend(window_trades)

        # Build equity curve
        equity_curve, daily_returns = self._build_equity_curve(all_trades)

        # Calculate metrics
        metrics = self._calculate_metrics(all_trades, equity_curve, daily_returns)

        # Regime attribution
        regime_perf = self._attribute_regimes(all_trades, nifty_df, regime_detector)

        # Monte Carlo
        mc_results = self._monte_carlo(all_trades)

        return BacktestResult(
            config=self.config,
            trades=all_trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            metrics=metrics,
            regime_performance=regime_perf,
            monte_carlo=mc_results,
            parameter_sensitivity={}
        )

    def _generate_windows(self, start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple]:
        """Generate walk-forward windows."""
        windows = []
        train_days = self.config.walk_forward_train_years * 252
        test_days = self.config.walk_forward_test_years * 252
        step_days = self.config.walk_forward_step_years * 252

        current = start
        while current + timedelta(days=train_days + test_days) <= end:
            train_start = current
            train_end = current + timedelta(days=train_days)
            test_start = train_end
            test_end = train_end + timedelta(days=test_days)

            windows.append((train_start, train_end, test_start, test_end))
            current += timedelta(days=step_days)

        return windows

    def _simulate_trades(
        self,
        signals: List,
        price_df: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> List[Trade]:
        """Simulate trades from signals."""
        trades = []

        # Group signals by symbol
        from collections import defaultdict
        signals_by_symbol = defaultdict(list)
        for s in signals:
            signals_by_symbol[s.symbol].append(s)

        for symbol, sym_signals in signals_by_symbol.items():
            try:
                sym_data = price_df.xs(symbol, level='Symbol')
            except KeyError:
                continue

            # Sort signals by date
            sym_signals.sort(key=lambda x: x.timestamp)

            # Track open positions
            open_positions = []

            for signal in sym_signals:
                sig_date = signal.timestamp
                if isinstance(sig_date, str):
                    sig_date = pd.Timestamp(sig_date)

                # Get price data from signal date onward
                future_data = sym_data[sym_data.index >= sig_date]
                if len(future_data) < 2:
                    continue

                entry_price = signal.entry_price
                stop_price = signal.stop_price
                targets = signal.targets
                qty = signal.position_size

                if qty == 0:
                    continue

                # Simulate day by day
                entry_date = future_data.index[0]
                entry_row = future_data.iloc[0]

                # Use next open as entry (more realistic)
                actual_entry = future_data.iloc[1]['Open'] if len(future_data) > 1 else entry_price

                # Track position
                position = {
                    'symbol': symbol,
                    'entry_date': entry_date,
                    'entry_price': actual_entry,
                    'stop': stop_price,
                    'targets': targets.copy(),
                    'qty': qty,
                    'strategy': signal.strategy,
                    'target_hit': [False] * len(targets)
                }

                # Simulate forward
                for idx, row in future_data.iloc[1:].iterrows():
                    high, low, close = row['High'], row['Low'], row['Close']

                    # Check stop loss
                    if low <= position['stop']:
                        exit_price = min(position['stop'], row['Open'])  # Gap down protection
                        trade = self._close_trade(position, idx, exit_price, 'stop_loss', close)
                        trades.append(trade)
                        position = None
                        break

                    # Check targets (partial exits)
                    for ti, target in enumerate(position['targets']):
                        if not position['target_hit'][ti] and high >= target:
                            exit_qty = int(position['qty'] * [0.5, 0.3, 0.2][ti] if ti < 3 else 0.2)
                            if exit_qty > 0:
                                trade = self._close_trade(
                                    position, idx, target, f'target_{ti+1}', close,
                                    partial_qty=exit_qty
                                )
                                trades.append(trade)
                                position['target_hit'][ti] = True
                                position['qty'] -= exit_qty

                    if position and position['qty'] == 0:
                        position = None
                        break

                # Force close at end of test period
                if position:
                    last_row = future_data.iloc[-1]
                    trade = self._close_trade(position, future_data.index[-1], last_row['Close'], 'period_end', last_row['Close'])
                    trades.append(trade)

        return trades

    def _close_trade(
        self,
        position: Dict,
        exit_date: pd.Timestamp,
        exit_price: float,
        reason: str,
        close_price: float,
        partial_qty: int = None
    ) -> Trade:
        """Close position and calculate P&L."""
        qty = partial_qty if partial_qty else position['qty']
        entry = position['entry_price']

        # Calculate costs
        costs = self.cost_model.calculate_costs(entry, qty, 'buy')
        costs_exit = self.cost_model.calculate_costs(exit_price, qty, 'sell')
        total_costs = costs['total'] + costs_exit['total']

        # P&L
        gross_pnl = (exit_price - entry) * qty
        net_pnl = gross_pnl - total_costs
        pnl_pct = net_pnl / (entry * qty) if entry > 0 else 0

        # R-multiple
        risk_per_share = abs(entry - position['stop'])
        r_multiple = (exit_price - entry) / risk_per_share if risk_per_share > 0 else 0

        hold_days = (exit_date - position['entry_date']).days

        return Trade(
            symbol=position['symbol'],
            strategy=position['strategy'],
            entry_date=position['entry_date'],
            entry_price=entry,
            exit_date=exit_date,
            exit_price=exit_price,
            quantity=qty,
            side='long',
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            r_multiple=r_multiple,
            hold_days=hold_days,
            exit_reason=reason,
            commission=costs['brokerage'] + costs_exit['brokerage'],
            slippage=costs['slippage'] + costs_exit['slippage']
        )

    def _build_equity_curve(self, trades: List[Trade]) -> Tuple[pd.Series, pd.Series]:
        """Build equity curve from trades."""
        if not trades:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # Daily P&L
        trade_df = pd.DataFrame([{
            'date': t.exit_date,
            'pnl': t.pnl
        } for t in trades])

        daily_pnl = trade_df.groupby('date')['pnl'].sum()
        daily_pnl = daily_pnl.reindex(
            pd.date_range(trade_df['date'].min(), trade_df['date'].max(), freq='B'),
            fill_value=0
        )

        equity = self.config.initial_capital + daily_pnl.cumsum()
        returns = daily_pnl / self.config.initial_capital

        return equity, returns

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity: pd.Series,
        returns: pd.Series
    ) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return {}

        trade_df = pd.DataFrame([{
            'pnl': t.pnl,
            'r_multiple': t.r_multiple,
            'hold_days': t.hold_days,
            'exit_reason': t.exit_reason
        } for t in trades])

        wins = trade_df[trade_df['pnl'] > 0]
        losses = trade_df[trade_df['pnl'] < 0]

        total_return = (equity.iloc[-1] / self.config.initial_capital - 1) if len(equity) > 0 else 0
        years = len(returns) / 252 if len(returns) > 0 else 1
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0

        # Sharpe
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Sortino
        downside = returns[returns < 0]
        sortino = returns.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

        # Max Drawdown
        peak = equity.cummax()
        dd = (peak - equity) / peak
        max_dd = dd.max()

        # Calmar
        calmar = cagr / max_dd if max_dd > 0 else 0

        # Win rate
        win_rate = len(wins) / len(trade_df) if len(trade_df) > 0 else 0

        # Avg win/loss
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Profit factor
        profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if losses['pnl'].sum() != 0 else float('inf')

        # Expectancy
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Kelly %
        kelly = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else 0

        # Ulcer Index
        ulcer = np.sqrt((dd ** 2).mean())

        return {
            'total_trades': len(trade_df),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'kelly_pct': kelly,
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'calmar': calmar,
            'ulcer_index': ulcer,
            'avg_hold_days': trade_df['hold_days'].mean(),
            'avg_r_multiple': trade_df['r_multiple'].mean(),
            'exit_reasons': trade_df['exit_reason'].value_counts().to_dict()
        }

    def _attribute_regimes(
        self,
        trades: List[Trade],
        nifty_df: pd.DataFrame,
        regime_detector
    ) -> pd.DataFrame:
        """Attribute trades to market regimes."""
        if not trades or regime_detector is None:
            return pd.DataFrame()

        # Get regime for each trade entry date
        trade_data = []
        for t in trades:
            # Find regime at entry date
            regime = 'unknown'
            try:
                # This is simplified - in production, pre-compute regimes
                pass
            except Exception:
                pass
            trade_data.append({
                'regime': regime,
                'pnl': t.pnl,
                'r_multiple': t.r_multiple
            })

        if not trade_data:
            return pd.DataFrame()

        df = pd.DataFrame(trade_data)
        return df.groupby('regime').agg(
            trades=('pnl', 'count'),
            win_rate=('pnl', lambda x: (x > 0).mean()),
            avg_pnl=('pnl', 'mean'),
            total_pnl=('pnl', 'sum'),
            avg_r=('r_multiple', 'mean')
        ).round(4)

    def _monte_carlo(self, trades: List[Trade], n_sims: int = None) -> Dict:
        """Monte Carlo simulation using block bootstrap."""
        if not trades:
            return {}

        n_sims = n_sims or self.config.monte_carlo_sims
        block_size = self.config.block_size

        trade_returns = np.array([t.pnl_pct for t in trades])
        n_trades = len(trade_returns)

        if n_trades < block_size * 2:
            return {}

        sim_results = []

        for _ in range(n_sims):
            # Block bootstrap
            n_blocks = int(np.ceil(n_trades / block_size))
            sampled_returns = []

            for _ in range(n_blocks):
                start_idx = np.random.randint(0, n_trades - block_size + 1)
                block = trade_returns[start_idx:start_idx + block_size]
                sampled_returns.extend(block)

            sampled_returns = np.array(sampled_returns[:n_trades])

            # Calculate equity curve
            equity = self.config.initial_capital * (1 + sampled_returns).cumprod()
            total_ret = equity[-1] / self.config.initial_capital - 1

            sim_results.append({
                'total_return': total_ret,
                'max_dd': (equity.cummax() - equity).max() / equity.cummax().max(),
                'final_equity': equity[-1]
            })

        sim_df = pd.DataFrame(sim_results)

        return {
            'mean_return': sim_df['total_return'].mean(),
            'median_return': sim_df['total_return'].median(),
            'return_5pct': sim_df['total_return'].quantile(0.05),
            'return_95pct': sim_df['total_return'].quantile(0.95),
            'mean_max_dd': sim_df['max_dd'].mean(),
            'max_dd_95pct': sim_df['max_dd'].quantile(0.95),
            'prob_profit': (sim_df['total_return'] > 0).mean(),
            'prob_loss_10pct': (sim_df['total_return'] < -0.1).mean()
        }


def run_backtest(
    config: BacktestConfig,
    data_fetcher,
    strategy_func: Callable,
    symbols: List[str],
    regime_detector=None
) -> BacktestResult:
    """Convenience function to run backtest."""
    backtester = WalkForwardBacktester(config, data_fetcher, strategy_func)
    return backtester.run(symbols, regime_detector)


if __name__ == "__main__":
    import yaml
    from data.fetcher import create_fetcher
    from indicators.regime import RegimeDetector
    from strategies.swing import run_all_strategies

    logging.basicConfig(level=logging.INFO)

    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # Config
    bt_config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2024-12-31',
        initial_capital=cfg['capital']['initial'],
        max_risk_per_trade=cfg['capital']['max_risk_per_trade'],
        walk_forward_train_years=3,
        walk_forward_test_years=1,
        walk_forward_step_years=1,
        monte_carlo_sims=200
    )

    fetcher = create_fetcher()
    symbols = fetcher.nifty200_symbols[:30]  # Test with 30 symbols

    # Strategy function
    def strategy_func(df, _, regime_info):
        return run_all_strategies(df, cfg, regime_info)

    # Regime detector
    detector = RegimeDetector(cfg)

    # Run
    print("Running walk-forward backtest...")
    result = run_backtest(bt_config, fetcher, strategy_func, symbols, detector)

    print("\n=== BACKTEST RESULTS ===")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")

    print("\n=== MONTE CARLO ===")
    for k, v in result.monte_carlo.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")