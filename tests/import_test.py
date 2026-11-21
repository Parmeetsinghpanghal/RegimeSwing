import sys
sys.path.insert(0, '.')
try:
    from data.fetcher import create_fetcher
    from indicators.technical import add_all_indicators
    from indicators.regime import RegimeDetector
    from strategies.swing import run_all_strategies
    from strategies.scoring import create_scorer
    from backtest import WalkForwardBacktester, BacktestConfig
    from papertrade import PaperTradingEngine
    print('All imports successful')
except Exception as e:
    import traceback
    print(f'Import failed: {e}')
    traceback.print_exc()