"""
Backtesting module for the trading system.
"""
from .engine import (
    BacktestEngine,
    create_backtest_engine,
    run_simple_backtest,
    BacktestResult,
    Trade,
    MetricsCalculator,
    SlippageModel,
    OptionsPricer,
    WalkForwardAnalyzer,
    MonteCarloSimulator,
    example_strategy,
)

__all__ = [
    'BacktestEngine',
    'create_backtest_engine',
    'run_simple_backtest',
    'BacktestResult',
    'Trade',
    'MetricsCalculator',
    'SlippageModel',
    'OptionsPricer',
    'WalkForwardAnalyzer',
    'MonteCarloSimulator',
    'example_strategy',
]