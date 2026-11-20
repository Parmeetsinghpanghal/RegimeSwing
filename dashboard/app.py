"""
Streamlit Dashboard - Professional UI for swing trading + options selling toolkit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import create_fetcher, DataFetcher
from indicators.technical import add_all_indicators, ensure_multiindex
from indicators.regime import RegimeDetector
from indicators.targets import TargetCalculator
from strategies.swing import run_all_strategies, get_all_strategies
from papertrade.engine import PaperTradingEngine, Portfolio, Position

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


@st.cache_data(ttl=300)
def load_config():
    """Load configuration."""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=60)
def load_portfolio_state():
    """Load paper trading portfolio from SQLite."""
    db_path = Path("logs/paper_state.db")
    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(db_path)
        # Portfolio
        row = conn.execute("SELECT value FROM portfolio_state WHERE key = 'portfolio'").fetchone()
        if row:
            portfolio = json.loads(row[0])
        else:
            portfolio = {}

        # Positions
        pos_df = pd.read_sql("SELECT * FROM positions", conn)
        # Orders
        ord_df = pd.read_sql("SELECT * FROM orders", conn)

        conn.close()
        return {'portfolio': portfolio, 'positions': pos_df, 'orders': ord_df}
    except Exception as e:
        st.error(f"Failed to load state: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_market_data(symbols, period="6mo", interval="1d"):
    """Fetch and cache market data."""
    fetcher = create_fetcher()
    df = fetcher.get_price_data(symbols, period=period, interval=interval)
    if not df.empty:
        df = ensure_multiindex(df)
        df = add_all_indicators(df, load_config())
    return df


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
            ('BBU_20_2.0', 'BB Upper', 'rgba(173,204,255,0.3)'),
            ('BBM_20_2.0', 'BB Mid', 'rgba(173,204,255,0.5)'),
            ('BBL_20_2.0', 'BB Lower', 'rgba(173,204,255,0.3)')
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

    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("📈 Swing Trading Dashboard")
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
        st.error("No data available")
        return

    # Load paper trading state
    state = load_portfolio_state()

    # ===== PORTFOLIO OVERVIEW =====
    if state and state['portfolio']:
        st.subheader("💼 Paper Trading Portfolio")
        portfolio = state['portfolio']

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Value", f"₹{portfolio.get('total_value', 0):,.0f}")
        with col2:
            st.metric("Cash", f"₹{portfolio.get('cash', 0):,.0f}")
        with col3:
            pnl_color = "normal"
            st.metric("Unrealized PnL", f"₹{portfolio.get('total_unrealized_pnl', 0):,.0f}")
        with col4:
            st.metric("Realized PnL", f"₹{portfolio.get('total_realized_pnl', 0):,.0f}")
        with col5:
            dd = portfolio.get('drawdown_pct', 0)
            st.metric("Drawdown", f"{dd:.2%}", delta_color="inverse")

        # Positions table
        if state['positions'] is not None and not state['positions'].empty:
            st.subheader("📋 Current Positions")
            pos_df = state['positions']
            display_cols = ['symbol', 'quantity', 'avg_entry_price', 'current_price',
                           'unrealized_pnl', 'realized_pnl', 'stop_price', 'strategy']
            available_cols = [c for c in display_cols if c in pos_df.columns]
            st.dataframe(pos_df[available_cols].style.format({
                'avg_entry_price': '₹{:.2f}',
                'current_price': '₹{:.2f}',
                'unrealized_pnl': '₹{:,.0f}',
                'realized_pnl': '₹{:,.0f}',
                'stop_price': '₹{:.2f}'
            }), use_container_width=True)

    # ===== REGIME STATUS =====
    st.subheader("📊 Market Regime")
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

    # ===== SIGNALS =====
    st.subheader("🎯 Today's Signals")

    with st.spinner("Generating signals..."):
        regime_info = {
            'regime': regime.regime,
            'size_mult': detector.get_size_multiplier(regime),
            'nifty_trend': regime.nifty_trend,
            'should_trade_swing': detector.should_trade_swing,
            'should_sell_options': detector.should_sell_options
        }
        signals = run_all_strategies(df, config, regime_info)

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
                'Time': s.timestamp.strftime('%H:%M') if hasattr(s.timestamp, 'strftime') else str(s.timestamp)
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

    # Symbol selector for chart
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

    st.title("🔄 Backtest Engine")

    col1, col2, col3 = st.columns(3)
    with col1:
        strategy = st.selectbox("Strategy", ["all", "trend_pullback", "breakout_volume", "mean_reversion_bb"])
    with col2:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
    with col3:
        end_date = st.date_input("End Date", value=datetime.now())

    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            # This would call the backtest engine
            st.info("Backtest engine integration pending - run via CLI: `python main.py backtest`")


def render_paper_trading_page(params):
    """Paper trading page."""
    st.title("📝 Paper Trading")

    config = params['config']

    # Initialize engine
    fetcher = create_fetcher()
    engine = PaperTradingEngine(config, fetcher)
    engine._load_state()

    # Portfolio summary
    st.subheader("💼 Portfolio Summary")
    report = engine.daily_report()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", f"₹{report['total_value']:,.0f}")
    with col2:
        st.metric("Cash", f"₹{report['cash']:,.0f}")
    with col3:
        st.metric("Total PnL", f"₹{report['daily_pnl']:,.0f}")
    with col4:
        st.metric("Drawdown", f"{report['drawdown_pct']:.2%}")

    # Positions
    if report['positions']:
        st.subheader("📋 Positions")
        pos_data = []
        for sym, pos in report['positions'].items():
            pos_data.append({
                'Symbol': sym,
                'Qty': pos['quantity'],
                'Avg Entry': f"₹{pos['avg_entry_price']:.2f}",
                'Current': f"₹{pos['current_price']:.2f}",
                'Unrealized': f"₹{pos['unrealized_pnl']:,.0f}",
                'Realized': f"₹{pos['realized_pnl']:,.0f}",
                'Stop': f"₹{pos['stop_price']:.2f}",
                'Strategy': pos['strategy']
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)

    # Run cycle button
    if st.button("🔄 Run Daily Cycle", type="primary"):
        with st.spinner("Running daily cycle..."):
            # Fetch data
            symbols = list(report['positions'].keys()) or fetcher.nifty200_symbols[:20]
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

            # Run cycle
            new_report = engine.run_daily_cycle(swing_signals, [], prices)
            st.success("Cycle complete!")
            st.rerun()


def render_scanner_page(params):
    """Scanner page."""
    config = params['config']

    st.title("🔍 Market Scanner")

    # Quick scan
    fetcher = create_fetcher()
    symbols = params['symbols']

    if st.button("🔍 Scan Now", type="primary"):
        with st.spinner("Scanning..."):
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
                        'BB Position': "Lower" if latest['Close'] < latest.get('BBL_20_2.0', 0) else
                                      "Upper" if latest['Close'] > latest.get('BBU_20_2.0', 0) else "Mid",
                        'Setup': ""
                    })
                except Exception as e:
                    results.append({'Symbol': symbol, 'Error': str(e)})

            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)


def render_regime_page(params):
    """Regime monitor page."""
    config = params['config']

    st.title("📊 Regime Monitor")

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