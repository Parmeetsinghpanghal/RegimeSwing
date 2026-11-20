"""
Streamlit Dashboard - Professional UI for swing trading + options selling toolkit.
Cloud-compatible version for Streamlit Cloud deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# Add project root to path - works both locally and on Streamlit Cloud
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# Import after path setup
from data.fetcher import create_fetcher, DataFetcher
from indicators.technical import add_all_indicators, ensure_multiindex
from indicators.regime import RegimeDetector
from indicators.targets import TargetCalculator
from strategies.swing import run_all_strategies, get_all_strategies

st.set_page_config(
    page_title="Swing Toolkit Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .signal-buy { border-left: 4px solid #00cc88; padding-left: 1rem; }
    .signal-sell { border-left: 4px solid #ff4444; padding-left: 1rem; }
    .regime-bull { color: #00cc88; font-weight: bold; }
    .regime-bear { color: #ff4444; font-weight: bold; }
    .regime-sideways { color: #ffaa00; font-weight: bold; }
    .regime-volatile { color: #ff6600; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def get_config_path():
    """Get config.yaml path - works locally and on Streamlit Cloud."""
    # Check Streamlit secrets first (for cloud deployment)
    if hasattr(st, 'secrets') and 'config' in st.secrets:
        import tempfile
        tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(dict(st.secrets['config']), tmp_file)
        tmp_file.close()
        return tmp_file.name

    # Local development - use project config.yaml
    local_path = ROOT_DIR / "config.yaml"
    if local_path.exists():
        return str(local_path)

    return None


@st.cache_data(ttl=300)
def load_config():
    """Load configuration from file or secrets."""
    config_path = get_config_path()
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # Return default config if file not found
    return get_default_config()


def get_default_config():
    """Return minimal default configuration for cloud demo."""
    return {
        'capital': {
            'initial': 500000,
            'max_risk_per_trade': 0.015,
            'max_portfolio_risk': 0.06,
            'max_position_pct': 0.05,
            'max_sector_pct': 0.25,
            'max_positions': 10,
            'correlation_threshold': 0.7,
            'cash_reserve_pct': 0.10
        },
        'drawdown_circuits': {
            'daily_loss_limit': 0.03,
            'daily_stop_limit': 0.05,
            'total_dd_warn': 0.08,
            'total_dd_reduce': 0.12,
            'total_dd_stop': 0.15
        },
        'costs': {
            'brokerage_per_order': 20,
            'stt_equity_delivery': 0.001,
            'stt_equity_intraday': 0.00025,
            'stt_options_sell': 0.000625,
            'exchange_charges': 0.0000345,
            'sebi_charges': 0.000001,
            'stamp_duty': 0.00015,
            'slippage_bps_equity': 15,
            'slippage_bps_options': 50,
            'slippage_bps_illiquid': 100
        },
        'swing': {
            'trend_pullback': {'enabled': True, 'ema_fast': 20, 'ema_medium': 50, 'ema_slow': 200, 'adx_min': 25, 'pullback_zone_atr_mult': 0.5, 'rsi_min': 40, 'rsi_max': 55, 'volume_mult': 1.2, 'stop_atr_mult': 2.0, 'target_r_mult': [1.5, 2.5], 'trail_atr_mult': 3.0, 'max_hold_days': 20},
            'breakout_volume': {'enabled': True, 'lookback_days': 20, 'volume_mult': 1.5, 'min_consolidation_days': 10, 'max_consolidation_range': 0.08, 'stop_atr_mult': 1.5, 'target_r_mult': [2.0, 3.0], 'trail_method': "donchian_20", 'market_regime_filter': True},
            'mean_reversion_bb': {'enabled': True, 'bb_period': 20, 'bb_std': 2.0, 'rsi_max': 30, 'adx_max': 20, 'entry_confirmation': "close_inside_bb", 'stop_atr_mult': 1.0, 'target_levels': ["bb_mid", "bb_upper"], 'max_hold_days': 10, 'quick_exit_days': 3, 'market_regime_filter': True}
        },
        'options': {
            'cash_secured_puts': {'enabled': True, 'iv_rank_min': 60, 'iv_percentile_min': 50, 'delta_target_min': 0.25, 'delta_target_max': 0.30, 'dte_min': 30, 'dte_max': 45, 'premium_per_month_min': 0.015, 'delivery_pct_min': 50, 'trend_filter': True, 'earnings_buffer_days': 5, 'management': {'close_at_profit_pct': 0.50, 'roll_at_dte': 21, 'stop_loss_mult': 2.0, 'assignment_action': "wheel"}},
            'covered_calls': {'enabled': True, 'min_unrealized_gain': 0.10, 'delta_target_min': 0.30, 'delta_target_max': 0.35, 'dte_min': 30, 'dte_max': 45, 'strike_above_cost': True, 'roll_if_itm_dte': 5, 'roll_strikes_up': 1, 'roll_dte_out': 21, 'target_cost_basis_reduction': 0.02},
            'wheel': {'enabled': True, 'max_concurrent_wheels': 3, 'min_capital_per_wheel': 200000, 'sector_diversification': True, 'track_annualized_return': True, 'pause_if_annualized_below': 0.12},
            'credit_spreads': {'enabled': True, 'default_type': 'bull_put', 'dte_min': 21, 'dte_max': 30, 'ic_iv_rank_min': 70},
            'greeks_limits': {'max_portfolio_delta': 0.30, 'max_portfolio_vega': 50000, 'min_portfolio_theta': 0, 'max_gamma_per_trade': 0.02}
        },
        'regime': {
            'nifty_symbol': "NIFTY 50",
            'bull_conditions': ['close > ema50', 'ema50 > ema200', 'adx > 20'],
            'bear_conditions': ['close < ema200', 'adx > 20'],
            'sideways_conditions': ['abs(ema50 - ema200) / ema200 < 0.03', 'adx < 20'],
            'volatile_conditions': ['vix > 22', 'atr_pct > 0.02'],
            'size_multiplier': {'bull': 1.0, 'bear': 0.0, 'sideways': 0.7, 'volatile': 0.5}
        },
        'targets': {
            'atr_period': 14, 'stop_atr_mult': 2.0, 'target_atr_mult': [1.5, 2.5, 3.5],
            'chandelier_period': 22, 'chandelier_mult': 3.0,
            'fib_extensions': [1.272, 1.618, 2.0, 2.618],
            'use_structure_stops': True, 'structure_lookback': 20,
            'force_close_hour': 15.15, 'max_hold_days_default': 20,
            'options_expiry_exit_days': 3, 'use_max_pain_target': True
        },
        'data': {
            'universe': "nifty200", 'lookback_days': 1000, 'timeframe': "1d",
            'cache_dir': "./data/cache", 'use_kite_historical': True, 'fallback_yfinance': True
        },
        'paper_trading': {
            'enabled': True, 'initial_cash': 500000, 'fill_model': "realistic",
            'commission_model': "indian", 'log_file': "./logs/paper_trades.csv",
            'state_file': "./logs/paper_state.db", 'daily_report': True
        }
    }


def is_cloud_deployment():
    """Check if running on Streamlit Cloud."""
    return (
        os.getenv('STREAMLIT_SHARING') == '1' or
        os.getenv('STREAMLIT_CLOUD') == '1' or
        'streamlit.app' in os.getenv('HOSTNAME', '')
    )


def init_session_state():
    """Initialize session state for paper trading (cloud-compatible)."""
    if 'paper_portfolio' not in st.session_state:
        config = load_config()
        initial_cash = config.get('capital', {}).get('initial', 500000)
        st.session_state.paper_portfolio = {
            'cash': initial_cash,
            'initial_cash': initial_cash,
            'positions': {},
            'orders': {},
            'daily_pnl': 0.0,
            'total_realized_pnl': 0.0,
            'peak_value': initial_cash,
            'trades_today': 0
        }
        st.session_state.paper_trades_log = []


def get_portfolio_state():
    """Get current portfolio state from session_state."""
    return st.session_state.get('paper_portfolio', {})


def save_portfolio_state(portfolio):
    """Save portfolio state to session_state."""
    st.session_state.paper_portfolio = portfolio


def add_paper_trade(trade):
    """Add trade to session_state log."""
    if 'paper_trades_log' not in st.session_state:
        st.session_state.paper_trades_log = []
    st.session_state.paper_trades_log.append(trade)


@st.cache_data(ttl=300)
def fetch_market_data(symbols, period="6mo", interval="1d"):
    """Fetch and cache market data with error handling."""
    fetcher = create_fetcher()
    try:
        df = fetcher.get_price_data(symbols, period=period, interval=interval)
        if not df.empty:
            df = ensure_multiindex(df)
            df = add_all_indicators(df, load_config())
        return df
    except Exception as e:
        st.warning(f"Data fetch error: {e}. Using cached data if available.")
        return pd.DataFrame()


def create_candlestick_chart(df, symbol, indicators=True, height=600):
    """Create Plotly candlestick chart with indicators."""
    if isinstance(df.index, pd.MultiIndex):
        sym_df = df.xs(symbol, level='Symbol')
    else:
        sym_df = df

    if sym_df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} Price', 'Volume', 'RSI')
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=sym_df.index,
        open=sym_df['Open'],
        high=sym_df['High'],
        low=sym_df['Low'],
        close=sym_df['Close'],
        name='Price',
        increasing_line_color='#00cc88',
        decreasing_line_color='#ff4444'
    ), row=1, col=1)

    # Indicators
    if indicators:
        # EMAs
        for period in [20, 50, 200]:
            col = f'EMA_{period}'
            if col in sym_df.columns:
                fig.add_trace(go.Scatter(
                    x=sym_df.index, y=sym_df[col],
                    name=f'EMA{period}', line=dict(width=1)
                ), row=1, col=1)

        # Bollinger Bands
        for bb_col, name, color in [
            ('BB_upper', 'BB Upper', 'rgba(173,204,255,0.3)'),
            ('BB_middle', 'BB Mid', 'rgba(173,204,255,0.5)'),
            ('BB_lower', 'BB Lower', 'rgba(173,204,255,0.3)')
        ]:
            if bb_col in sym_df.columns:
                fig.add_trace(go.Scatter(
                    x=sym_df.index, y=sym_df[bb_col],
                    name=name, line=dict(width=1, color=color),
                    fill='tonexty' if name == 'BB Upper' else None
                ), row=1, col=1)

        # Chandelier Exit
        if 'Chandelier_Long' in sym_df.columns:
            fig.add_trace(go.Scatter(
                x=sym_df.index, y=sym_df['Chandelier_Long'],
                name='Chandelier Long', line=dict(color='green', dash='dot')
            ), row=1, col=1)
        if 'Chandelier_Short' in sym_df.columns:
            fig.add_trace(go.Scatter(
                x=sym_df.index, y=sym_df['Chandelier_Short'],
                name='Chandelier Short', line=dict(color='red', dash='dot')
            ), row=1, col=1)

    # Volume
    colors = ['#00cc88' if c >= o else '#ff4444' for c, o in zip(sym_df['Close'], sym_df['Open'])]
    fig.add_trace(go.Bar(
        x=sym_df.index, y=sym_df['Volume'],
        name='Volume', marker_color=colors
    ), row=2, col=1)

    # Volume SMA
    if 'Volume_SMA_20' in sym_df.columns:
        fig.add_trace(go.Scatter(
            x=sym_df.index, y=sym_df['Volume_SMA_20'],
            name='Vol SMA20', line=dict(color='orange', width=1)
        ), row=2, col=1)

    # RSI
    if 'RSI_14' in sym_df.columns:
        fig.add_trace(go.Scatter(
            x=sym_df.index, y=sym_df['RSI_14'],
            name='RSI', line=dict(color='purple')
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)

    fig.update_layout(
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50)
    )

    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    return fig


def render_sidebar():
    """Render sidebar with controls."""
    st.sidebar.title("📈 Swing Toolkit")

    config = load_config()
    cloud_mode = is_cloud_deployment()

    if cloud_mode:
        st.sidebar.caption("☁️ Running on Streamlit Cloud")

    # Mode selection
    mode = st.sidebar.radio(
        "Mode",
        ["Dashboard", "Backtest", "Paper Trading", "Scanner", "Regime Monitor"],
        index=0
    )

    # Universe selection
    universe = st.sidebar.selectbox(
        "Universe",
        ["Nifty 50", "Nifty 100", "Nifty 200", "Custom"],
        index=2
    )

    if universe == "Custom":
        custom_symbols = st.sidebar.text_area(
            "Symbols (comma-separated)",
            value="RELIANCE,TCS,HDFCBANK,ICICIBANK,INFY"
        )
        symbols = [s.strip().upper() for s in custom_symbols.split(',')]
    else:
        fetcher = create_fetcher()
        n = int(universe.split()[1])
        symbols = fetcher.nifty200_symbols[:n]

    # Timeframe
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1d", "1h", "15m", "5m"],
        index=0
    )

    # Date range
    period = st.sidebar.selectbox(
        "Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=2
    )

    return {
        'mode': mode,
        'symbols': symbols,
        'timeframe': timeframe,
        'period': period,
        'config': config
    }


def render_dashboard_page(params):
    """Main dashboard page."""
    config = params['config']
    symbols = params['symbols']
    cloud_mode = is_cloud_deployment()

    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("📈 Swing Trading Dashboard")
        if cloud_mode:
            st.caption("☁️ Running on Streamlit Cloud (Demo Mode)")
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col3:
        st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    # Load data
    with st.spinner("Fetching market data..."):
        df = fetch_market_data(symbols, period=params['period'], interval=params['timeframe'])

    if df.empty:
        st.error("No data available. Please check your internet connection or try different symbols.")
        return

    # ===== REGIME STATUS =====
    st.subheader("📊 Market Regime")
    try:
        fetcher = create_fetcher()
        nifty_df = fetcher.get_nifty_index_data(period="1y", interval="1d")
        nifty_df = ensure_multiindex(nifty_df)
        nifty_df = add_all_indicators(nifty_df, config)

        detector = RegimeDetector(config)
        regime = detector.detect_nifty_regime(nifty_df)

        col1, col2 = st.columns([1, 2])
        with col1:
            regime_class = f"regime-{regime.regime}"
            st.markdown(f'<div class="{regime_class}">Current: {regime.regime.upper()}</div>', unsafe_allow_html=True)
            st.metric("Confidence", f"{regime.confidence:.0%}")
            st.metric("Nifty Trend", regime.nifty_trend.title())
            st.metric("ADX", f"{regime.adx_value:.1f}")
        with col2:
            st.markdown("**Strategy Recommendations:**")
            for strat in ['trend_pullback', 'breakout_volume', 'mean_reversion_bb']:
                should, reason = detector.should_trade_swing(regime, strat)
                icon = "✅" if should else "❌"
                st.write(f"{icon} {strat}: {reason}")

            should_opt, reason_opt, opt_params = detector.should_sell_options(regime)
            icon = "✅" if should_opt else "❌"
            st.write(f"{icon} Options Selling: {reason_opt}")
            if should_opt:
                st.json(opt_params)
    except Exception as e:
        st.warning(f"Regime detection error: {e}")

    # ===== SIGNALS =====
    st.subheader("🎯 Today's Signals")

    with st.spinner("Generating signals..."):
        try:
            regime_info = {
                'regime': regime.regime,
                'size_mult': detector.get_size_multiplier(regime),
                'nifty_trend': regime.nifty_trend,
                'should_trade_swing': detector.should_trade_swing,
                'should_sell_options': detector.should_sell_options
            }
            signals = run_all_strategies(df, config, regime_info)
        except Exception as e:
            st.error(f"Signal generation error: {e}")
            signals = []

    if signals:
        # Convert to DataFrame
        sig_data = []
        for s in signals:
            sig_data.append({
                'Symbol': s.symbol,
                'Strategy': s.strategy,
                'Entry': f"₹{s.entry_price:.2f}",
                'Stop Loss': f"₹{s.stop_price:.2f}",
                'Targets': ', '.join([f"₹{t:.2f}" for t in s.targets]),
                'Size': s.position_size,
                'Risk %': f"{s.risk_pct:.2%}",
                'R-Targets': ', '.join([f"{r:.1f}R" for r in s.r_multiple_targets]),
            })

        sig_df = pd.DataFrame(sig_data)

        # Color code by strategy
        def highlight_strategy(row):
            colors = {
                'trend_pullback': '#e8f5e9',
                'breakout_volume': '#e3f2fd',
                'mean_reversion_bb': '#fff3e0'
            }
            bg = colors.get(row['Strategy'], 'white')
            return [f'background-color: {bg}'] * len(row)

        st.dataframe(sig_df.style.apply(highlight_strategy, axis=1), use_container_width=True)

        # Download signals
        csv = sig_df.to_csv(index=False)
        st.download_button(
            "📥 Download Signals CSV",
            csv,
            f"signals_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    else:
        st.info("No signals generated for current regime")

    # ===== CHARTS =====
    st.subheader("📈 Charts")

    chart_symbol = st.selectbox("Select Symbol", symbols, index=0)

    col1, col2 = st.columns([3, 1])
    with col2:
        show_indicators = st.checkbox("Show Indicators", value=True)
        chart_height = st.slider("Chart Height", 400, 1000, 700)

    with st.spinner(f"Loading chart for {chart_symbol}..."):
        fig = create_candlestick_chart(df, chart_symbol, indicators=show_indicators, height=chart_height)
        st.plotly_chart(fig, use_container_width=True)


def render_backtest_page(params):
    """Backtest page."""
    config = params['config']
    cloud_mode = is_cloud_deployment()

    st.title("🔄 Backtest Engine")

    if cloud_mode:
        st.warning("⚠️ Full walk-forward backtest is resource-intensive and not available on Streamlit Cloud.")
        st.info("""
        **For full backtesting, run locally:**
        ```bash
        python main.py backtest-wf --start 2020-01-01 --end 2024-12-31
        ```
        """)

    st.caption("Quick signal generation for a single period")

    col1, col2, col3 = st.columns(3)
    with col1:
        strategy = st.selectbox("Strategy", ["all", "trend_pullback", "breakout_volume", "mean_reversion_bb"])
    with col2:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
    with col3:
        end_date = st.date_input("End Date", value=datetime.now())

    universe = st.selectbox("Universe", ["Nifty 50", "Nifty 100", "Nifty 200", "Custom"], index=2)
    if universe == "Custom":
        custom_symbols = st.text_area("Symbols (comma-separated)", value="RELIANCE,TCS,HDFCBANK,ICICIBANK,INFY")
        symbols = [s.strip().upper() for s in custom_symbols.split(',')]
    else:
        fetcher = create_fetcher()
        n = int(universe.split()[1])
        symbols = fetcher.nifty200_symbols[:n]

    if st.button("🚀 Generate Signals", type="primary"):
        run_signal_generation(config, symbols, strategy, start_date, end_date)


def run_signal_generation(config, symbols, strategy, start_date, end_date):
    """Generate signals for a single period."""
    from data.fetcher import create_fetcher
    from indicators.technical import add_all_indicators, ensure_multiindex
    from indicators.regime import RegimeDetector
    from strategies.swing import run_all_strategies, get_all_strategies

    with st.spinner("Generating signals..."):
        fetcher = create_fetcher()

        # Fetch data
        df = fetcher.get_price_data(
            symbols,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval="1d"
        )

        if df.empty:
            st.error("No data fetched")
            return

        df = ensure_multiindex(df)
        df = add_all_indicators(df, config)

        # Regime detection
        nifty_df = fetcher.get_nifty_index_data(
            period=f"{(end_date - start_date).days}d",
            interval="1d"
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

        st.info(f"📈 Regime: {regime.regime} (confidence: {regime.confidence:.0%})")

        # Run strategies
        if strategy == "all":
            signals = run_all_strategies(df, config, regime_info)
        else:
            strategies = get_all_strategies(config)
            if strategy not in strategies:
                st.error(f"Unknown strategy: {strategy}")
                return
            signals = strategies[strategy].generate_signals(df, regime_info)

        if not signals:
            st.info("No signals generated")
            return

        # Convert to DataFrame
        sig_data = []
        for s in signals:
            sig_data.append({
                'Symbol': s.symbol,
                'Strategy': s.strategy,
                'Entry': f"₹{s.entry_price:.2f}",
                'Stop Loss': f"₹{s.stop_price:.2f}",
                'Targets': ', '.join([f"₹{t:.2f}" for t in s.targets]),
                'Size': s.position_size,
                'Risk %': f"{s.risk_pct:.2%}",
                'R-Targets': ', '.join([f"{r:.1f}R" for r in s.r_multiple_targets]),
            })

        sig_df = pd.DataFrame(sig_data)
        st.dataframe(sig_df, use_container_width=True)

        # Download
        csv = sig_df.to_csv(index=False)
        st.download_button(
            "📥 Download Signals CSV",
            csv,
            f"signals_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )


def render_paper_trading_page(params):
    """Paper trading page with session_state (cloud-compatible)."""
    config = params['config']
    cloud_mode = is_cloud_deployment()

    st.title("📝 Paper Trading")

    init_session_state()

    if cloud_mode:
        st.info("☁️ Paper trading uses browser session storage (resets on page reload). For persistent trading, run locally.")

    # Initialize engine
    fetcher = create_fetcher()

    # Portfolio summary
    st.subheader("💼 Portfolio Summary")
    portfolio = get_portfolio_state()

    # Update position prices
    for sym, pos in portfolio.get('positions', {}).items():
        try:
            price_data = fetcher.get_price_data([sym], period="5d", interval="1d")
            if not price_data.empty:
                price_data = ensure_multiindex(price_data)
                latest_price = price_data.xs(sym, level='Symbol')['Close'].iloc[-1]
                pos['current_price'] = latest_price
                pos['unrealized_pnl'] = (latest_price - pos['avg_entry_price']) * pos['quantity']
        except Exception:
            pass

    total_value = portfolio['cash'] + sum(p.get('market_value', p['quantity'] * p.get('current_price', p['avg_entry_price'])) for p in portfolio['positions'].values())
    total_unrealized = sum(p.get('unrealized_pnl', 0) for p in portfolio['positions'].values())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", f"₹{total_value:,.0f}")
    with col2:
        st.metric("Cash", f"₹{portfolio['cash']:,.0f}")
    with col3:
        st.metric("Unrealized PnL", f"₹{total_unrealized:,.0f}")
    with col4:
        dd = (portfolio.get('peak_value', total_value) - total_value) / portfolio.get('peak_value', total_value) * 100
        st.metric("Drawdown", f"{dd:.2f}%")

    # Update peak
    if total_value > portfolio.get('peak_value', 0):
        portfolio['peak_value'] = total_value
        save_portfolio_state(portfolio)

    # Positions
    if portfolio['positions']:
        st.subheader("📋 Positions")
        pos_data = []
        for sym, pos in portfolio['positions'].items():
            pos_data.append({
                'Symbol': sym,
                'Qty': pos['quantity'],
                'Avg Entry': f"₹{pos['avg_entry_price']:.2f}",
                'Current': f"₹{pos.get('current_price', pos['avg_entry_price']):.2f}",
                'Unrealized': f"₹{pos.get('unrealized_pnl', 0):,.0f}",
                'Realized': f"₹{pos.get('realized_pnl', 0):,.0f}",
                'Stop': f"₹{pos.get('stop_price', 0):.2f}",
                'Strategy': pos.get('strategy', 'unknown')
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)

    # Run cycle button
    if st.button("🔄 Run Daily Cycle", type="primary"):
        with st.spinner("Running daily cycle..."):
            # Fetch current data
            symbols = list(portfolio['positions'].keys()) or fetcher.nifty200_symbols[:20]
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
                'should_trade_swing': detector.should_trade_swing
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

            # Process signals - simplified execution
            for s in swing_signals:
                if s.symbol in prices:
                    # Check if position already exists
                    if s.symbol not in portfolio['positions']:
                        cost = s.entry_price * s.position_size
                        if cost <= portfolio['cash']:
                            portfolio['cash'] -= cost
                            portfolio['positions'][s.symbol] = {
                                'quantity': s.position_size,
                                'avg_entry_price': s.entry_price,
                                'current_price': s.entry_price,
                                'unrealized_pnl': 0,
                                'realized_pnl': 0,
                                'entry_time': datetime.now().isoformat(),
                                'stop_price': s.stop_price,
                                'targets': s.targets,
                                'strategy': s.strategy
                            }
                            add_paper_trade({
                                'timestamp': datetime.now().isoformat(),
                                'symbol': s.symbol,
                                'side': 'BUY',
                                'quantity': s.position_size,
                                'price': s.entry_price,
                                'strategy': s.strategy
                            })

            # Update existing positions
            for sym, pos in portfolio['positions'].items():
                if sym in prices:
                    pos['current_price'] = prices[sym]
                    pos['unrealized_pnl'] = (prices[sym] - pos['avg_entry_price']) * pos['quantity']

            save_portfolio_state(portfolio)
            st.success("Cycle complete!")
            st.rerun()

    # Trade history
    if st.session_state.get('paper_trades_log'):
        with st.expander("📜 Trade History"):
            trades_df = pd.DataFrame(st.session_state.paper_trades_log)
            st.dataframe(trades_df, use_container_width=True)


def render_scanner_page(params):
    """Scanner page."""
    config = params['config']
    fetcher = create_fetcher()
    symbols = params['symbols']

    st.title("🔍 Market Scanner")

    if st.button("🔍 Scan Now", type="primary"):
        with st.spinner("Scanning..."):
            try:
                df = fetcher.get_price_data(symbols, period="6mo", interval="1d")
                df = ensure_multiindex(df)
                df = add_all_indicators(df, config)

                results = []
                for symbol in symbols:
                    try:
                        sym_df = df.xs(symbol, level='Symbol')
                        latest = sym_df.iloc[-1]

                        # Trend
                        trend = "🟢 Up" if latest['Close'] > latest.get('EMA_50', 0) > latest.get('EMA_200', 0) else \
                                "🔴 Down" if latest['Close'] < latest.get('EMA_50', 0) < latest.get('EMA_200', 0) else "🟡 Mixed"

                        results.append({
                            'Symbol': symbol,
                            'Price': f"₹{latest['Close']:.2f}",
                            'Change': f"{(latest['Close']/sym_df['Close'].iloc[-2]-1)*100:.2f}%",
                            'Trend': trend,
                            'RSI': f"{latest.get('RSI_14', 0):.1f}",
                            'ADX': f"{latest.get('ADX_14', 0):.1f}",
                            'Vol Ratio': f"{latest.get('Volume_Ratio', 0):.2f}",
                            'BB Position': "Lower" if latest['Close'] < latest.get('BB_lower', 0) else
                                          "Upper" if latest['Close'] > latest.get('BB_upper', 0) else "Mid",
                        })
                    except Exception as e:
                        results.append({'Symbol': symbol, 'Error': str(e)})

                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True)
            except Exception as e:
                st.error(f"Scan failed: {e}")


def render_regime_page(params):
    """Regime monitor page."""
    config = params['config']

    st.title("📊 Regime Monitor")

    try:
        fetcher = create_fetcher()
        nifty_df = fetcher.get_nifty_index_data(period="2y", interval="1d")
        nifty_df = ensure_multiindex(nifty_df)
        nifty_df = add_all_indicators(nifty_df, config)

        detector = RegimeDetector(config)

        # Historical regimes
        with st.spinner("Analyzing historical regimes..."):
            regimes_df = detector.analyze_historical_regimes(nifty_df, detector)

        if not regimes_df.empty:
            # Regime distribution
            col1, col2 = st.columns(2)
            with col1:
                regime_counts = regimes_df['regime'].value_counts()
                fig = go.Figure(data=[go.Pie(labels=regime_counts.index, values=regime_counts.values)])
                fig.update_layout(title="Regime Distribution (Historical)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Regime timeline
                fig = go.Figure()
                regime_colors = {'bull': 'green', 'bear': 'red', 'sideways': 'orange', 'volatile': 'purple'}
                for regime in regimes_df['regime'].unique():
                    mask = regimes_df['regime'] == regime
                    fig.add_trace(go.Scatter(
                        x=regimes_df.index[mask],
                        y=[regime] * mask.sum(),
                        mode='markers',
                        name=regime,
                        marker=dict(color=regime_colors.get(regime, 'gray'), size=8)
                    ))
                fig.update_layout(title="Regime Timeline", yaxis_title="Regime")
                st.plotly_chart(fig, use_container_width=True)

            # Current regime details
            st.subheader("Current Regime Details")
            latest_regime = detector.detect_nifty_regime(nifty_df)
            st.markdown(detector.get_regime_summary(latest_regime))
    except Exception as e:
        st.error(f"Regime analysis error: {e}")


def is_cloud_deployment():
    """Check if running on Streamlit Cloud."""
    return (
        os.getenv('STREAMLIT_SHARING') == '1' or
        os.getenv('STREAMLIT_CLOUD') == '1' or
        'streamlit.app' in os.getenv('HOSTNAME', '')
    )


def main():
    params = render_sidebar()

    # Route to appropriate page
    if params['mode'] == "Dashboard":
        render_dashboard_page(params)
    elif params['mode'] == "Backtest":
        render_backtest_page(params)
    elif params['mode'] == "Paper Trading":
        render_paper_trading_page(params)
    elif params['mode'] == "Scanner":
        render_scanner_page(params)
    elif params['mode'] == "Regime Monitor":
        render_regime_page(params)


if __name__ == "__main__":
    main()