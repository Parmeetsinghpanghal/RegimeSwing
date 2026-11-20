#!/usr/bin/env python3
"""
Swing Trading + Options Selling Toolkit - Main Entry Point

Usage:
    python main.py backtest --strategy trend_pullback --start 2020-01-01
    python main.py papertrade --config config.yaml
    python main.py dashboard
    python main.py scan --symbols RELIANCE,TCS
    python main.py auth --auto
"""

import argparse
import sys
import os
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.fetcher import create_fetcher, DataFetcher
from indicators.technical import add_all_indicators, ensure_multiindex
from indicators.regime import RegimeDetector
from indicators.targets import TargetCalculator
from indicators.multitf import analyze_multitf_confluence, fetch_multitf_data, scan_universe_confluence
from indicators.patterns import detect_all_patterns, calculate_pattern_statistics
from indicators.volume_profile import calculate_volume_profile, get_volume_profile_signals
from indicators.support_resistance import detect_sr_zones, get_sr_signals
from strategies.swing import run_all_strategies, get_all_strategies
from strategies.options import run_all_options_strategies, get_all_options_strategies
from papertrade.engine import PaperTradingEngine


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cmd_backtest(args):
    """Run backtest for specified strategy."""
    config = load_config(args.config)
    setup_logging(args.log_level)

    print(f"🔄 Running backtest for {args.strategy}...")
    print(f"📅 Period: {args.start} to {args.end}")

    # Initialize components
    fetcher = create_fetcher(args.config)

    # Fetch data
    if args.universe == "nifty200":
        symbols = fetcher.nifty200_symbols
    elif args.universe == "nifty50":
        symbols = fetcher.nifty200_symbols[:50]
    else:
        symbols = [s.strip() for s in args.universe.split(',')]

    print(f"📊 Universe: {len(symbols)} symbols")

    # Fetch price data
    df = fetcher.get_price_data(
        symbols,
        start_date=args.start,
        end_date=args.end,
        interval=args.timeframe
    )

    if df.empty:
        print("❌ No data fetched")
        return

    print(f"✅ Fetched {len(df)} bars for {df.index.get_level_values('Symbol').nunique()} symbols")

    # Add indicators
    df = ensure_multiindex(df)
    df = add_all_indicators(df, config)

    # Regime detection
    nifty_df = fetcher.get_nifty_index_data(
        period=f"{(datetime.strptime(args.end, '%Y-%m-%d') - datetime.strptime(args.start, '%Y-%m-%d')).days}d",
        interval=args.timeframe
    )
    nifty_df = ensure_multiindex(nifty_df)
    nifty_df = add_all_indicators(nifty_df, config)

    detector = RegimeDetector(config)
    regime = detector.detect_nifty_regime(nifty_df)
    regime_info = {
        'regime': regime.regime,
        'size_mult': detector.get_size_multiplier(regime),
        'nifty_trend': regime.nifty_trend,
        'should_trade_swing': detector.should_trade_swing,
        'should_sell_options': detector.should_sell_options
    }

    print(f"📈 Regime: {regime.regime} (confidence: {regime.confidence:.0%})")

    # Run strategies
    if args.strategy == "all":
        swing_signals = run_all_strategies(df, config, regime_info)
        print(f"\n📊 Generated {len(swing_signals)} swing signals")

        # Group by strategy
        from collections import defaultdict
        by_strat = defaultdict(list)
        for s in swing_signals:
            by_strat[s.strategy].append(s)

        for strat, sigs in by_strat.items():
            print(f"  {strat}: {len(sigs)} signals")
            for s in sigs[:5]:  # Show first 5
                print(f"    {s.symbol} @ {s.entry_price:.2f} | SL: {s.stop_price:.2f} | "
                      f"Targets: {[f'{t:.2f}' for t in s.targets]} | Size: {s.position_size}")

    else:
        # Single strategy
        strategies = get_all_strategies(config)
        if args.strategy not in strategies:
            print(f"❌ Unknown strategy: {args.strategy}")
            return

        strat = strategies[args.strategy]
        signals = strat.generate_signals(df, regime_info)
        print(f"\n📊 {args.strategy}: {len(signals)} signals")
        for s in signals[:10]:
            print(f"  {s.symbol} @ {s.entry_price:.2f} | SL: {s.stop_price:.2f} | "
                  f"Targets: {[f'{t:.2f}' for t in s.targets]} | Size: {s.position_size}")

    # Save signals
    if args.output:
        import json
        output_data = [s.to_dict() for s in swing_signals] if args.strategy == "all" else [s.to_dict() for s in signals]
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n💾 Signals saved to {args.output}")


def cmd_papertrade(args):
    """Run paper trading engine."""
    config = load_config(args.config)
    setup_logging(args.log_level)

    print("📝 Starting Paper Trading Engine...")
    print(f"💰 Initial Capital: ₹{config['capital']['initial']:,.0f}")

    # Initialize engine
    fetcher = create_fetcher(args.config)
    engine = PaperTradingEngine(config, fetcher)
    engine._load_state()

    # Load strategies
    swing_strategies = get_all_strategies(config)
    options_strategies = get_all_options_strategies(config)

    print(f"📋 Loaded {len(swing_strategies)} swing strategies, {len(options_strategies)} options strategies")

    if args.once:
        # Single run
        print("\n🔄 Running single cycle...")

        # Fetch current data
        symbols = list(engine.portfolio.positions.keys()) or fetcher.nifty200_symbols[:20]
        df = fetcher.get_price_data(symbols, period="3mo", interval="1d")
        df = ensure_multiindex(df)
        df = add_all_indicators(df, config)

        # Regime
        nifty_df = fetcher.get_nifty_index_data(period="3mo", interval="1d")
        nifty_df = ensure_multiindex(nifty_df)
        nifty_df = add_all_indicators(nifty_df, config)

        detector = RegimeDetector(config)
        regime = detector.detect_nifty_regime(nifty_df)
        regime_info = {
            'regime': regime.regime,
            'size_mult': detector.get_size_multiplier(regime),
            'nifty_trend': regime.nifty_trend,
            'should_trade_swing': detector.should_trade_swing,
            'should_sell_options': detector.should_sell_options
        }

        # Generate signals
        swing_signals = run_all_strategies(df, config, regime_info)

        # Current prices
        prices = {}
        for sym in symbols:
            try:
                prices[sym] = df.xs(sym, level='Symbol')['Close'].iloc[-1]
            except Exception:
                pass

        # Run cycle
        report = engine.run_daily_cycle(swing_signals, [], prices)

        print(f"\n📊 Daily Report:")
        for k, v in report.items():
            if k != 'positions':
                print(f"  {k}: {v}")

        print(f"\n📈 Positions ({len(report['positions'])}):")
        for sym, pos in report['positions'].items():
            print(f"  {sym}: {pos['quantity']} @ {pos['avg_entry_price']:.2f} | "
                  f"Current: {pos['current_price']:.2f} | PnL: ₹{pos['unrealized_pnl']:,.0f}")

    else:
        # Daemon mode (would run continuously)
        print("🔄 Daemon mode not implemented in this version")
        print("   Use --once for single cycle or implement scheduler")


def cmd_dashboard(args):
    """Launch Streamlit dashboard."""
    import subprocess

    print("🚀 Launching Dashboard...")
    print("   Dashboard will open in browser at http://localhost:8501")

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    if not dashboard_path.exists():
        print(f"❌ Dashboard not found at {dashboard_path}")
        return

    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


def cmd_scan(args):
    """Scan symbols for setups."""
    config = load_config(args.config)
    setup_logging(args.log_level)

    fetcher = create_fetcher(args.config)

    symbols = [s.strip() for s in args.symbols.split(',')]
    print(f"🔍 Scanning {len(symbols)} symbols...")

    df = fetcher.get_price_data(symbols, period="6mo", interval="1d")
    df = ensure_multiindex(df)
    df = add_all_indicators(df, config)

    # Check each strategy
    for symbol in symbols:
        try:
            sym_df = df.xs(symbol, level='Symbol')
            latest = sym_df.iloc[-1]

            print(f"\n📊 {symbol}: ₹{latest['Close']:.2f}")
            print(f"   EMA20/50/200: {latest.get('EMA_20', 0):.2f} / {latest.get('EMA_50', 0):.2f} / {latest.get('EMA_200', 0):.2f}")
            print(f"   RSI: {latest.get('RSI_14', 0):.1f} | ADX: {latest.get('ADX_14', 0):.1f}")
            print(f"   ATR: {latest.get('ATR_14', 0):.2f} | Vol Ratio: {latest.get('Volume_Ratio', 0):.2f}")
            print(f"   BB: {latest.get('BBL_20_2.0', 0):.2f} - {latest.get('BBM_20_2.0', 0):.2f} - {latest.get('BBU_20_2.0', 0):.2f}")

            # Trend check
            if latest['Close'] > latest.get('EMA_50', 0) > latest.get('EMA_200', 0):
                print("   ✅ Uptrend (Price > EMA50 > EMA200)")
            elif latest['Close'] < latest.get('EMA_50', 0) < latest.get('EMA_200', 0):
                print("   🔴 Downtrend (Price < EMA50 < EMA200)")
            else:
                print("   🟡 Mixed/Transition")

        except Exception as e:
            print(f"   ❌ Error: {e}")


def cmd_auth(args):
    """Authenticate with Kite Connect."""
    config = load_config(args.config)
    broker_config = config.get('broker', {})

    api_key = broker_config.get('api_key') or os.getenv('KITE_API_KEY')
    api_secret = broker_config.get('api_secret') or os.getenv('KITE_API_SECRET')

    if not api_key or not api_secret:
        print("❌ KITE_API_KEY and KITE_API_SECRET must be set in config.yaml or environment")
        return

    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=api_key)

    if args.auto:
        # Auto login with Selenium (requires chromedriver)
        print("🤖 Auto-login requires Selenium - not implemented in this version")
        print("   Use manual login below")
        return

    # Manual login
    print(f"\n🔐 Login URL: {kite.login_url()}")
    print("   1. Open URL in browser")
    print("   2. Login with Kite credentials")
    print("   3. Copy 'request_token' from redirect URL")
    print("   4. Paste below\n")

    request_token = input("Enter request_token: ").strip()

    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data['access_token']

        print(f"\n✅ Access token generated: {access_token}")

        # Save to config
        config['broker']['access_token'] = access_token
        with open(args.config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"💾 Saved to {args.config}")

    except Exception as e:
        print(f"❌ Auth failed: {e}")


def cmd_regime(args):
    """Check current market regime."""
    config = load_config(args.config)
    setup_logging(args.log_level)

    fetcher = create_fetcher(args.config)

    nifty_df = fetcher.get_nifty_index_data(period="1y", interval="1d")
    nifty_df = ensure_multiindex(nifty_df)
    nifty_df = add_all_indicators(nifty_df, config)

    detector = RegimeDetector(config)
    regime = detector.detect_nifty_regime(nifty_df)

    print(detector.get_regime_summary(regime))

    # Strategy recommendations
    print("\n📋 Strategy Recommendations:")
    for strat in ['trend_pullback', 'breakout_volume', 'mean_reversion_bb']:
        should, reason = detector.should_trade_swing(regime, strat)
        print(f"  {strat}: {'✅' if should else '❌'} {reason}")

    should, reason, params = detector.should_sell_options(regime)
    print(f"\n  Options Selling: {'✅' if should else '❌'} {reason}")
    print(f"  Params: {params}")


def cmd_confluence(args):
    """Analyze multi-timeframe confluence for symbols."""
    config = load_config(args.config)
    setup_logging(args.log_level)

    fetcher = create_fetcher(args.config)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = fetcher.nifty200_symbols[:20]

    print(f"🔍 Analyzing confluence for {len(symbols)} symbols...")

    results = scan_universe_confluence(fetcher, symbols)

    for r in results:
        print(f"\n📊 {r.symbol} @ ₹{r.current_price:.2f}")
        print(f"   Trend: {r.dominant_trend.value.upper()} | Alignment: {r.alignment_score:.0%} | Signal: {r.signal} ({r.signal_strength:.0%})")
        print(f"   Quality: {r.entry_quality} | Est. R:R: {r.risk_reward_estimate:.2f}")

        # Top confluence zones
        for zone in r.confluence_zones[:3]:
            print(f"   🎯 {zone['price']:.2f} | {zone['tf_count']} TFs ({', '.join(zone['unique_timeframes'])}) | Dist: {zone['distance_from_current_pct']:+.2f}%")

        # Notes
        for note in r.notes[:3]:
            print(f"   - {note}")


def cmd_patterns(args):
    """Analyze candlestick patterns for symbols."""
    config = load_config(args.config)
    setup_logging(args.log_level)

    fetcher = create_fetcher(args.config)

    symbols = [s.strip() for s in args.symbols.split(',')]
    print(f"🔍 Analyzing patterns for {len(symbols)} symbols...")

    for symbol in symbols:
        df = fetcher.get_price_data([symbol], period="6mo", interval="1d")
        if df.empty:
            print(f"  ❌ {symbol}: No data")
            continue

        df = ensure_multiindex(df)
        df = add_all_indicators(df, config)

        # Get recent patterns
        patterns = get_latest_patterns(df, n_bars=args.lookback)

        print(f"\n📊 {symbol}:")
        if patterns:
            for p in patterns[:5]:
                print(f"  {p.name:20} | {p.direction.value:8} | Conf: {p.confidence:.0%} | "
                      f"Target: {p.target:.2f if p.target else 'N/A':>7} | Stop: {p.stop:.2f if p.stop else 'N/A':>7}")
        else:
            print("  No recent patterns")


def cmd_volume_profile(args):
    """Analyze volume profile for symbols."""
    config = load_config(args.config)
    setup_logging(args.log_level)

    fetcher = create_fetcher(args.config)

    symbols = [s.strip() for s in args.symbols.split(',')]
    print(f"🔍 Analyzing volume profile for {len(symbols)} symbols...")

    for symbol in symbols:
        df = fetcher.get_price_data([symbol], period="6mo", interval="1d")
        if df.empty:
            print(f"  ❌ {symbol}: No data")
            continue

        df = ensure_multiindex(df)
        df = add_all_indicators(df, config)

        # Get volume profile for latest window
        sym_df = df.xs(symbol, level='Symbol') if isinstance(df.index, pd.MultiIndex) else df
        vp = calculate_volume_profile(sym_df.tail(args.lookback))

        current = sym_df['Close'].iloc[-1]
        signals = get_volume_profile_signals(current, vp)

        print(f"\n📊 {symbol} @ ₹{current:.2f}")
        print(f"   POC: {vp.poc:.2f} | VAH: {vp.vah:.2f} | VAL: {vp.val:.2f}")
        print(f"   HVN: {[f'{h:.2f}' for h in vp.hvn[:3]]}")
        print(f"   LVN: {[f'{l:.2f}' for l in vp.lvn[:3]]}")
        print(f"   Signal: {signals['signal']} | Strength: {signals['strength']:.0%}")
        print(f"   At POC: {signals['at_poc']} | At VAH: {signals['at_vah']} | At VAL: {signals['at_val']}")


def cmd_support_resistance(args):
    """Analyze support/resistance zones for symbols."""
    config = load_config(args.config)
    setup_logging(args.log_level)

    fetcher = create_fetcher(args.config)

    symbols = [s.strip() for s in args.symbols.split(',')]
    print(f"🔍 Analyzing S/R zones for {len(symbols)} symbols...")

    for symbol in symbols:
        df = fetcher.get_price_data([symbol], period="6mo", interval="1d")
        if df.empty:
            print(f"  ❌ {symbol}: No data")
            continue

        df = ensure_multiindex(df)
        df = add_all_indicators(df, config)

        sym_df = df.xs(symbol, level='Symbol') if isinstance(df.index, pd.MultiIndex) else df
        zones = detect_sr_zones(sym_df, lookback=args.lookback)

        current = sym_df['Close'].iloc[-1]
        atr = sym_df.get('ATR_14', pd.Series([30]*len(sym_df))).iloc[-1]
        signals = get_sr_signals(current, zones, atr)

        print(f"\n📊 {symbol} @ ₹{current:.2f}")
        print(f"   Signal: {signals['signal']} | Strength: {signals['strength']:.0%}")
        print(f"   Nearest Support: {signals['nearest_support'].price if signals['nearest_support'] else 'None'}")
        print(f"   Nearest Resistance: {signals['nearest_resistance'].price if signals['nearest_resistance'] else 'None'}")

        for z in zones[:5]:
            zone_type = z.zone_type.upper()
            status = " 💥 BROKEN" if z.is_broken else ""
            print(f"   {zone_type:10} | {z.price:.2f} ({z.price_low:.2f}-{z.price_high:.2f}) | Str: {z.strength:.2f} | Touches: {z.touches}{status}")


def main():
    parser = argparse.ArgumentParser(
        description="Swing Trading + Options Selling Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-c', '--config', default='config.yaml', help='Config file path')
    parser.add_argument('-v', '--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Backtest
    bt = subparsers.add_parser('backtest', help='Run backtest')
    bt.add_argument('-s', '--strategy', default='all', help='Strategy name or "all"')
    bt.add_argument('-u', '--universe', default='nifty200', help='Universe: nifty50, nifty200, or comma-separated symbols')
    bt.add_argument('--start', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    bt.add_argument('--end', default=None, help='End date (YYYY-MM-DD)')
    bt.add_argument('--timeframe', default='1d', help='Timeframe (1d, 1h, 15m)')
    bt.add_argument('-o', '--output', help='Output JSON file for signals')

    # Paper trade
    pt = subparsers.add_parser('papertrade', help='Run paper trading')
    pt.add_argument('--once', action='store_true', help='Run single cycle and exit')
    pt.add_argument('--daemon', action='store_true', help='Run as daemon (not implemented)')

    # Dashboard
    subparsers.add_parser('dashboard', help='Launch Streamlit dashboard')

    # Scan
    sc = subparsers.add_parser('scan', help='Scan symbols for setups')
    sc.add_argument('symbols', help='Comma-separated symbols (e.g., RELIANCE,TCS,HDFCBANK)')

    # Auth
    auth = subparsers.add_parser('auth', help='Authenticate with Kite Connect')
    auth.add_argument('--auto', action='store_true', help='Auto login with Selenium')

    # Regime
    subparsers.add_parser('regime', help='Check current market regime')

    # Confluence
    conf = subparsers.add_parser('confluence', help='Multi-timeframe confluence analysis')
    conf.add_argument('symbols', nargs='?', help='Comma-separated symbols (optional, defaults to top 20)')

    # Patterns
    pat = subparsers.add_parser('patterns', help='Candlestick pattern analysis')
    pat.add_argument('symbols', help='Comma-separated symbols')
    pat.add_argument('--lookback', type=int, default=10, help='Bars to look back for patterns')

    # Volume Profile
    vp = subparsers.add_parser('volume-profile', help='Volume profile analysis')
    vp.add_argument('symbols', help='Comma-separated symbols')
    vp.add_argument('--lookback', type=int, default=50, help='Bars for volume profile')

    # Support/Resistance
    sr = subparsers.add_parser('support-resistance', help='Support/Resistance zone analysis')
    sr.add_argument('symbols', help='Comma-separated symbols')
    sr.add_argument('--lookback', type=int, default=100, help='Bars to analyze')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Route commands
    commands = {
        'backtest': cmd_backtest,
        'papertrade': cmd_papertrade,
        'dashboard': cmd_dashboard,
        'scan': cmd_scan,
        'auth': cmd_auth,
        'regime': cmd_regime,
        'confluence': cmd_confluence,
        'patterns': cmd_patterns,
        'volume-profile': cmd_volume_profile,
        'support-resistance': cmd_support_resistance,
    }

    if args.command in commands:
        try:
            commands[args.command](args)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Unknown command: {args.command}")


if __name__ == "__main__":
    main()