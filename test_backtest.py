import sys
sys.path.insert(0, '.')
try:
    from backtest.engine import BacktestConfig, WalkForwardBacktester, run_backtest, BacktestConfig, Trade, BacktestResult
    from backtest import BacktestConfig, WalkForwardBacktester, run_backtest
    print('Backtest imports successful')
except Exception as e:
    import traceback
    print(f'Import failed: {e}')
    traceback.print_exc()