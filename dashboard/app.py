"""
Streamlit Dashboard - Professional UI for swing trading + options selling toolkit.
Upgraded with TradingView-style charts, Sensibull-style options payoff, and multi-factor scoring.
Cloud-compatible version for Streamlit Cloud deployment.
"""

# Add project root to path at the VERY TOP - before any other imports
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# Now import other modules
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
import logging

# QuantStats for tear-sheets (optional)
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False
    qs = None

# Import SwingScore type for type hints
try:
    from strategies.scoring import SwingScore
except ImportError:
    SwingScore = None

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Swing Toolkit Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
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
    .score-high { color: #00cc88; font-weight: bold; font-size: 1.2em; }
    .score-medium { color: #ffaa00; font-weight: bold; font-size: 1.2em; }
    .score-low { color: #ff4444; font-weight: bold; font-size: 1.2em; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00cc88;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def get_config_path():
    """Get config.yaml path - works locally and on Streamlit Cloud."""
    if hasattr(st, 'secrets') and 'config' in st.secrets:
        import tempfile
        tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(dict(st.secrets['config']), tmp_file)
        tmp_file.close()
        return tmp_file.name

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
    return get_default_config()


def get_default_config():
    """Return minimal default configuration for cloud demo."""
    return {
        'capital': {
            'initial': 500000, 'max_risk_per_trade': 0.015, 'max_portfolio_risk': 0.06,
            'max_position_pct': 0.05, 'max_sector_pct': 0.25, 'max_positions': 10,
            'correlation_threshold': 0.7, 'cash_reserve_pct': 0.10
        },
        'drawdown_circuits': {
            'daily_loss_limit': 0.03, 'daily_stop_limit': 0.05,
            'total_dd_warn': 0.08, 'total_dd_reduce': 0.12, 'total_dd_stop': 0.15
        },
        'costs': {
            'brokerage_per_order': 20, 'stt_equity_delivery': 0.001, 'stt_equity_intraday': 0.00025,
            'stt_options_sell': 0.000625, 'exchange_charges': 0.0000345, 'sebi_charges': 0.000001,
            'stamp_duty': 0.00015, 'slippage_bps_equity': 15, 'slippage_bps_options': 50,
            'slippage_bps_illiquid': 100
        },
        'swing': {
            'trend_pullback': {'enabled': True, 'ema_fast': 20, 'ema_medium': 50, 'ema_slow': 200, 'adx_min': 25,
                              'pullback_zone_atr_mult': 0.5, 'rsi_min': 40, 'rsi_max': 55, 'volume_mult': 1.2,
                              'stop_atr_mult': 2.0, 'target_r_mult': [1.5, 2.5], 'trail_atr_mult': 3.0, 'max_hold_days': 20},
            'breakout_volume': {'enabled': True, 'lookback_days': 20, 'volume_mult': 1.5,
                               'min_consolidation_days': 10, 'max_consolidation_range': 0.08, 'stop_atr_mult': 1.5,
                               'target_r_mult': [2.0, 3.0], 'trail_method': "donchian_20", 'market_regime_filter': True},
            'mean_reversion_bb': {'enabled': True, 'bb_period': 20, 'bb_std': 2.0, 'rsi_max': 30, 'adx_max': 20,
                                 'entry_confirmation': "close_inside_bb", 'stop_atr_mult': 1.0,
                                 'target_levels': ["bb_mid", "bb_upper"], 'max_hold_days': 10, 'quick_exit_days': 3,
                                 'market_regime_filter': True}
        },
        'options': {
            'cash_secured_puts': {'enabled': True, 'iv_rank_min': 60, 'iv_percentile_min': 50,
                                 'delta_target_min': 0.25, 'delta_target_max': 0.30, 'dte_min': 30, 'dte_max': 45,
                                 'premium_per_month_min': 0.015, 'delivery_pct_min': 50, 'trend_filter': True,
                                 'earnings_buffer_days': 5, 'management': {'close_at_profit_pct': 0.50,
                                 'roll_at_dte': 21, 'stop_loss_mult': 2.0, 'assignment_action': "wheel"}},
            'covered_calls': {'enabled': True, 'min_unrealized_gain': 0.10, 'delta_target_min': 0.30,
                             'delta_target_max': 0.35, 'dte_min': 30, 'dte_max': 45, 'strike_above_cost': True,
                             'roll_if_itm_dte': 5, 'roll_strikes_up': 1, 'roll_dte_out': 21,
                             'target_cost_basis_reduction': 0.02},
            'wheel': {'enabled': True, 'max_concurrent_wheels': 3, 'min_capital_per_wheel': 200000,
                     'sector_diversification': True, 'track_annualized_return': True, 'pause_if_annualized_below': 0.12},
            'credit_spreads': {'enabled': True, 'default_type': 'bull_put', 'dte_min': 21, 'dte_max': 30, 'ic_iv_rank_min': 70},
            'greeks_limits': {'max_portfolio_delta': 0.30, 'max_portfolio_vega': 50000,
                             'min_portfolio_theta': 0, 'max_gamma_per_trade': 0.02}
        },
        'regime': {
            'nifty_symbol': "NIFTY 50", 'bull_conditions': {'adx_min': 20}, 'bear_conditions': {'adx_min': 20},
            'sideways_conditions': {'ema_tolerance': 0.03, 'adx_max': 20},
            'volatile_conditions': {'vix_threshold': 22, 'atr_pct_threshold': 0.02},
            'size_multiplier': {'bull': 1.0, 'bear': 0.0, 'sideways': 0.7, 'volatile': 0.5}
        },
        'targets': {
            'atr_period': 14, 'stop_atr_mult': 2.0, 'target_atr_mult': [1.5, 2.5, 3.5],
            'chandelier_period': 22, 'chandelier_mult': 3.0, 'fib_extensions': [1.272, 1.618, 2.0, 2.618],
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
        },
        'scoring': {
            'weights': {
                'trend_alignment': 0.25, 'momentum': 0.20, 'volume_surge': 0.15,
                'volatility': 0.15, 'structure': 0.15, 'regime_alignment': 0.10
            },
            'backtest_gate': {
                'min_win_rate': 0.55, 'min_expectancy': 0.3, 'min_profit_factor': 1.0, 'min_trades': 10
            }
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
            'cash': initial_cash, 'initial_cash': initial_cash, 'positions': {},
            'orders': {}, 'daily_pnl': 0.0, 'total_realized_pnl': 0.0,
            'peak_value': initial_cash, 'trades_today': 0
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


def create_tradingview_chart(df, symbol, signal=None, height=700):
    """
    Create TradingView-style candlestick chart with:
    - Candlesticks with volume
    - EMA ribbons (20, 50, 200)
    - Bollinger Bands
    - Chandelier Exit (trailing stops)
    - Entry, Stop Loss, Target lines (if signal provided)
    - Volume profile overlay
    - RSI + MACD subpanels
    """
    if isinstance(df.index, pd.MultiIndex):
        sym_df = df.xs(symbol, level='Symbol')
    else:
        sym_df = df

    if sym_df.empty:
        return go.Figure()

    # Create subplots: Price (60%), Volume (15%), RSI (12.5%), MACD (12.5%)
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=[0.6, 0.15, 0.125, 0.125],
        subplot_titles=(f'{symbol} - TradingView Style', 'Volume', 'RSI (14)', 'MACD (12,26,9)')
    )

    # ===== MAIN PRICE CHART =====
    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=sym_df.index,
        open=sym_df['Open'],
        high=sym_df['High'],
        low=sym_df['Low'],
        close=sym_df['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
        whiskerwidth=0.8
    ), row=1, col=1)

    # EMA Ribbons (TradingView style)
    ema_colors = {20: '#ff9800', 50: '#2196f3', 200: '#9c27b0'}
    for period in [20, 50, 200]:
        col = f'EMA_{period}'
        if col in sym_df.columns:
            fig.add_trace(go.Scatter(
                x=sym_df.index, y=sym_df[col],
                name=f'EMA {period}',
                line=dict(color=ema_colors.get(period, 'gray'), width=1.5),
                opacity=0.85
            ), row=1, col=1)

    # Bollinger Bands (shaded area)
    if all(c in sym_df.columns for c in ['BB_upper', 'BB_middle', 'BB_lower']):
        fig.add_trace(go.Scatter(
            x=sym_df.index, y=sym_df['BB_upper'],
            name='BB Upper', line=dict(color='rgba(33,150,243,0.3)', width=1),
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sym_df.index, y=sym_df['BB_lower'],
            name='BB Lower', line=dict(color='rgba(33,150,243,0.3)', width=1),
            fill='tonexty', fillcolor='rgba(33,150,243,0.08)',
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sym_df.index, y=sym_df['BB_middle'],
            name='BB Mid', line=dict(color='rgba(33,150,243,0.5)', width=1, dash='dot'),
            showlegend=False
        ), row=1, col=1)

    # Chandelier Exit (Trailing Stops)
    if 'Chandelier_Long' in sym_df.columns:
        fig.add_trace(go.Scatter(
            x=sym_df.index, y=sym_df['Chandelier_Long'],
            name='Chandelier Long (Trail)', line=dict(color='#4caf50', width=1.5, dash='dash'),
            opacity=0.7
        ), row=1, col=1)
    if 'Chandelier_Short' in sym_df.columns:
        fig.add_trace(go.Scatter(
            x=sym_df.index, y=sym_df['Chandelier_Short'],
            name='Chandelier Short (Trail)', line=dict(color='#f44336', width=1.5, dash='dash'),
            opacity=0.7
        ), row=1, col=1)

    # Signal markers (Entry, SL, Targets)
    if signal:
        entry = signal['entry_price']
        stop = signal['stop_price']
        targets = signal['targets']
        timestamp = signal.get('timestamp', sym_df.index[-1])

        # Entry line
        fig.add_hline(y=entry, line_dash="solid", line_color="#2196f3", line_width=2,
                      annotation_text=f"ENTRY ₹{entry:.2f}", annotation_position="right",
                      row=1, col=1)
        # Stop Loss line
        fig.add_hline(y=stop, line_dash="dash", line_color="#f44336", line_width=2,
                      annotation_text=f"SL ₹{stop:.2f}", annotation_position="right",
                      row=1, col=1)
        # Target lines
        target_colors = ["#4caf50", "#8bc34a", "#cddc39", "#ffeb3b"]
        for i, target in enumerate(targets):
            fig.add_hline(y=target, line_dash="dot", line_color=target_colors[i % len(target_colors)],
                          line_width=1.5, annotation_text=f"T{i+1} ₹{target:.2f}",
                          annotation_position="right", row=1, col=1)

    # ===== VOLUME =====
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(sym_df['Close'], sym_df['Open'])]
    fig.add_trace(go.Bar(
        x=sym_df.index, y=sym_df['Volume'],
        name='Volume', marker_color=colors, opacity=0.7
    ), row=2, col=1)

    if 'Volume_SMA_20' in sym_df.columns:
        fig.add_trace(go.Scatter(
            x=sym_df.index, y=sym_df['Volume_SMA_20'],
            name='Vol SMA20', line=dict(color='#ff9800', width=1.5)
        ), row=2, col=1)

    # ===== RSI =====
    if 'RSI_14' in sym_df.columns:
        fig.add_trace(go.Scatter(
            x=sym_df.index, y=sym_df['RSI_14'],
            name='RSI', line=dict(color='#9c27b0', width=1.5)
        ), row=3, col=1)
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="#f44336", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#4caf50", line_width=1, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", line_width=1, row=3, col=1)
        # RSI zones
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(244,67,54,0.1)", line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(76,175,80,0.1)", line_width=0, row=3, col=1)

    # ===== MACD =====
    if all(c in sym_df.columns for c in ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']):
        fig.add_trace(go.Scatter(
            x=sym_df.index, y=sym_df['MACD_12_26_9'],
            name='MACD', line=dict(color='#2196f3', width=1.5)
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=sym_df.index, y=sym_df['MACDs_12_26_9'],
            name='Signal', line=dict(color='#ff9800', width=1.5)
        ), row=4, col=1)
        # Histogram
        colors_macd = ['#4caf50' if v >= 0 else '#f44336' for v in sym_df['MACDh_12_26_9']]
        fig.add_trace(go.Bar(
            x=sym_df.index, y=sym_df['MACDh_12_26_9'],
            name='Histogram', marker_color=colors_macd, opacity=0.6
        ), row=4, col=1)

    # Layout
    fig.update_layout(
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
        ),
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=20),
        hovermode='x unified',
        dragmode='zoom'
    )

    fig.update_yaxes(title_text="Price (₹)", row=1, col=1, side='right')
    fig.update_yaxes(title_text="Vol", row=2, col=1, side='right')
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1, side='right')
    fig.update_yaxes(title_text="MACD", row=4, col=1, side='right')

    # Hide x-axis labels for top 3 rows
    for row in [1, 2, 3]:
        fig.update_xaxes(showticklabels=False, row=row, col=1)
    fig.update_xaxes(showticklabels=True, row=4, col=1)

    return fig


def create_options_payoff_chart(signal, config, height=500):
    """
    Create Sensibull-style Options Payoff & Greeks Diagram.
    Shows payoff at expiry for the suggested options strategy.
    """
    if not signal:
        return go.Figure()

    # For swing signals, show CSP/CC payoff based on regime
    entry = signal.get('entry_price', 0)
    stop = signal.get('stop_price', 0)
    targets = signal.get('targets', [])
    symbol = signal.get('symbol', '')

    if entry == 0:
        return go.Figure()

    # Determine options strategy from regime/config
    # This is a simplified version - in production, use the options strategy engine
    spot_range = np.linspace(entry * 0.85, entry * 1.15, 200)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('Options Payoff at Expiry', 'Greeks (Delta, Theta, Vega)')
    )

    # Example: Cash-Secured Put payoff (short put)
    # Strike slightly below entry (delta ~0.30)
    strike = entry * 0.97
    premium = entry * 0.02  # ~2% premium
    dte = 30

    # Payoff at expiry for short put
    payoff = np.where(spot_range >= strike, premium, premium - (strike - spot_range))
    max_profit = premium
    max_loss = strike - premium  # If stock goes to 0
    breakeven = strike - premium

    fig.add_trace(go.Scatter(
        x=spot_range, y=payoff,
        name=f'Short Put {strike:.0f}',
        line=dict(color='#2196f3', width=2),
        fill='tozeroy', fillcolor='rgba(33,150,243,0.15)'
    ), row=1, col=1)

    # Key levels
    fig.add_vline(x=entry, line_dash="solid", line_color="#2196f3", line_width=2,
                  annotation_text=f"Spot ₹{entry:.2f}", row=1, col=1)
    fig.add_vline(x=strike, line_dash="dash", line_color="#ff9800", line_width=1.5,
                  annotation_text=f"Strike ₹{strike:.2f}", row=1, col=1)
    fig.add_vline(x=breakeven, line_dash="dash", line_color="#4caf50", line_width=1.5,
                  annotation_text=f"BE ₹{breakeven:.2f}", row=1, col=1)
    fig.add_vline(x=stop, line_dash="dash", line_color="#f44336", line_width=1.5,
                  annotation_text=f"SL ₹{stop:.2f}", row=1, col=1)
    for i, t in enumerate(targets[:2]):
        fig.add_vline(x=t, line_dash="dot", line_color="#4caf50", line_width=1,
                      annotation_text=f"T{i+1} ₹{t:.2f}", row=1, col=1)

    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=1, col=1)

    # Greeks visualization (simplified)
    # Delta: changes from 0 to -1 as spot goes below strike
    delta = np.where(spot_range >= strike, -0.05, -1 + (strike - spot_range) / (strike * 0.1))
    delta = np.clip(delta, -1, 0)

    # Theta: negative, accelerates near expiry
    theta = -premium / dte * np.ones_like(spot_range)

    # Vega: highest at-the-money
    vega = np.exp(-0.5 * ((spot_range - strike) / (strike * 0.05))**2) * 0.5

    fig.add_trace(go.Scatter(
        x=spot_range, y=delta, name='Delta', line=dict(color='#2196f3', width=1.5)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=spot_range, y=theta, name='Theta (₹/day)', line=dict(color='#ff9800', width=1.5)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=spot_range, y=vega, name='Vega', line=dict(color='#9c27b0', width=1.5)
    ), row=2, col=1)

    fig.update_layout(
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=20),
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="P&L (₹)", row=1, col=1, side='right')
    fig.update_yaxes(title_text="Value", row=2, col=1, side='right')
    fig.update_xaxes(title_text="Spot Price at Expiry (₹)", row=2, col=1)

    return fig


def create_scorecard(score: SwingScore):
    """Create a visual scorecard for a swing score."""
    total = score.total_score
    regime = score.regime
    passed = score.backtest_passed

    # Score color
    if total >= 75:
        score_class = "score-high"
        score_label = "STRONG BUY"
    elif total >= 60:
        score_class = "score-medium"
        score_label = "BUY"
    elif total >= 45:
        score_class = "score-medium"
        score_label = "WATCH"
    else:
        score_class = "score-low"
        score_label = "AVOID"

    return f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; background: #fafafa;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h3 style="margin: 0;">{score.symbol}</h3>
            <span class="{score_class}">{total:.1f}/100</span>
        </div>
        <div style="display: flex; gap: 16px; font-size: 0.9em; color: #666;">
            <span>Regime: <strong>{regime.upper()}</strong> ({score.regime_confidence:.0%})</span>
            <span>Backtest: {'✅ PASSED' if passed else '❌ FAILED'}</span>
            <span>{score_label}</span>
        </div>
        <div style="margin-top: 12px;">
            {''.join([f'<div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #eee;"><span>{f.name.replace("_"," ").title()}</span><span><strong>{f.score:.0f}</strong> (w={f.weight:.0%})</span></div>' for f in score.factor_scores])}
        </div>
    </div>
    """


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
        ["Dashboard", "Scoring & Alerts", "Backtest", "Paper Trading", "Scanner", "Regime Monitor"],
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

    # Scoring settings
    with st.sidebar.expander("⚙️ Scoring Settings"):
        min_score = st.slider("Min Score for Alerts", 0, 100, 60)
        require_backtest = st.checkbox("Require Backtest Pass", value=True)
        top_n = st.slider("Top N Alerts", 1, 20, 5)

    return {
        'mode': mode,
        'symbols': symbols,
        'timeframe': timeframe,
        'period': period,
        'config': config,
        'min_score': min_score,
        'require_backtest': require_backtest,
        'top_n': top_n
    }


def render_dashboard_page(params):
    """Main dashboard page with multi-factor scoring."""
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
        if st.button("🔄 Refresh Data", width='stretch'):
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

        # Robust column handling for Nifty
        if not nifty_df.empty:
            if isinstance(nifty_df.columns, pd.MultiIndex):
                try:
                    nifty_df.columns = nifty_df.columns.get_level_values(1)
                except Exception:
                    nifty_df.columns = nifty_df.columns.get_level_values(0)

            col_mapping = {}
            for col in nifty_df.columns:
                col_str = str(col).lower()
                if col_str in ('open', 'high', 'low', 'close', 'volume', 'adj close', 'adj_close'):
                    col_mapping[col] = col_str.capitalize().replace('_', ' ')
                elif col_str == 'adj close':
                    col_mapping[col] = 'Adj Close'
            if col_mapping:
                nifty_df = nifty_df.rename(columns=col_mapping)
            nifty_df.index = pd.to_datetime(nifty_df.index)

            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in required if c not in nifty_df.columns]
            if missing:
                alt_map = {
                    'Open': ['open', 'OPEN', 'Open Price', 'open price'],
                    'High': ['high', 'HIGH', 'High Price', 'high price'],
                    'Low': ['low', 'LOW', 'Low Price', 'low price'],
                    'Close': ['close', 'CLOSE', 'Close Price', 'Adj Close', 'AdjClose', 'adj close', 'adj_close', 'close price'],
                    'Volume': ['volume', 'VOLUME', 'Vol', 'vol']
                }
                for req in missing[:]:
                    for alt in alt_map.get(req, []):
                        if alt in nifty_df.columns:
                            nifty_df[req] = nifty_df[alt]
                            missing.remove(req)
                            break

                if missing:
                    for req in missing[:]:
                        for col in nifty_df.columns:
                            if col.lower() == req.lower():
                                nifty_df[req] = nifty_df[col]
                                missing.remove(req)
                                break

                if missing and len(nifty_df.columns) >= 5:
                    for i, req in enumerate(missing[:]):
                        if i < len(nifty_df.columns):
                            nifty_df[req] = nifty_df.iloc[:, i]
                            missing.remove(req)

                if missing:
                    st.error(f"Nifty data missing columns: {missing}. Available: {list(nifty_df.columns)}")
                    st.stop()

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
            if should_opt and opt_params:
                with st.expander("📋 Options Parameters"):
                    col1, col2 = st.columns(2)
                    for i, (k, v) in enumerate(opt_params.items()):
                        with col1 if i % 2 == 0 else col2:
                            st.metric(k.replace('_', ' ').title(), v)
    except Exception as e:
        st.warning(f"Regime detection error: {e}")
        regime = None
        detector = None

    if regime is None:
        return

    # ===== MULTI-FACTOR SCORING =====
    st.subheader("🎯 Multi-Factor Swing Scores")

    # Pass regime object for scorer (needs dict with regime string and regime_obj for strategies)
    regime_info = {
        'regime': regime.regime,
        'confidence': regime.confidence,
        'size_mult': detector.get_size_multiplier(regime),
        'nifty_trend': regime.nifty_trend,
        'regime_obj': regime  # Pass actual RegimeState object for strategies
    }

    with st.spinner("Scoring universe..."):
        scorer = create_scorer(config.get('scoring'))
        scores = scorer.score_universe(df, regime_info, 'trend_pullback', min_score=params['min_score'])
        top_picks = scorer.get_top_picks(scores, top_n=params['top_n'], require_backtest=params['require_backtest'])

    if top_picks:
        # Display scorecards
        st.markdown("### Top Picks")
        cols = st.columns(min(len(top_picks), 5))
        for i, score in enumerate(top_picks):
            with cols[i % len(cols)]:
                st.markdown(create_scorecard(score), unsafe_allow_html=True)

        # Detailed table
        st.markdown("### All Scored Symbols")
        score_data = []
        for s in scores:
            score_data.append({
                'Symbol': s.symbol,
                'Score': f"{s.total_score:.1f}",
                'Regime': s.regime,
                'Regime Conf': f"{s.regime_confidence:.0%}",
                'Backtest': "✅" if s.backtest_passed else "❌",
                'Win Rate': f"{s.backtest_metrics.get('win_rate', 0):.1%}" if s.backtest_metrics else "N/A",
                'Expectancy': f"{s.backtest_metrics.get('expectancy', 0):.2f}R" if s.backtest_metrics else "N/A",
                'Profit Factor': f"{s.backtest_metrics.get('profit_factor', 0):.2f}" if s.backtest_metrics else "N/A",
                'Trades': s.backtest_metrics.get('total_trades', 0) if s.backtest_metrics else 0
            })
        score_df = pd.DataFrame(score_data)

        def highlight_score(row):
            score = float(row['Score'])
            if score >= 75:
                return ['background-color: #e8f5e9'] * len(row)
            elif score >= 60:
                return ['background-color: #fff3e0'] * len(row)
            elif score >= 45:
                return ['background-color: #fff8e1'] * len(row)
            return ['background-color: #fbe9e7'] * len(row)

        st.dataframe(score_df.style.apply(highlight_score, axis=1), width='stretch')

        # Download
        csv = score_df.to_csv(index=False)
        st.download_button(
            "📥 Download Scores CSV",
            csv,
            f"swing_scores_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    else:
        st.info("No symbols meet the minimum score threshold")

    # ===== CHARTS TABS =====
    st.subheader("📈 Charts & Analysis")

    if scores:
        chart_symbol = st.selectbox("Select Symbol for Detailed View", [s.symbol for s in scores], index=0)

        # Find signal for this symbol
        signal = None
        for s in scores:
            if s.symbol == chart_symbol:
                signal = {
                    'entry_price': s.factor_scores[0].metadata.get('ema20', 0) if s.factor_scores else 0,
                    'stop_price': s.factor_scores[0].metadata.get('chandelier', 0) if s.factor_scores else 0,
                    'targets': [],
                    'timestamp': s.timestamp,
                    'symbol': s.symbol
                }
                break

        # Try to get actual signal from strategy - pass regime object and wrapper functions
        def should_trade_swing_wrapper(r, strategy):
            return detector.should_trade_swing(regime, strategy)

        def should_sell_options_wrapper(r):
            return detector.should_sell_options(regime)

        regime_info_full = {
            'regime': regime.regime,
            'regime_obj': regime,  # Pass actual RegimeState object
            'size_mult': detector.get_size_multiplier(regime),
            'nifty_trend': regime.nifty_trend,
            'should_trade_swing': should_trade_swing_wrapper,
            'should_sell_options': should_sell_options_wrapper
        }
        all_signals = run_all_strategies(df, config, regime_info_full)
        for sig in all_signals:
            if sig.symbol == chart_symbol:
                signal = {
                    'entry_price': sig.entry_price,
                    'stop_price': sig.stop_price,
                    'targets': sig.targets,
                    'timestamp': sig.timestamp,
                    'symbol': sig.symbol
                }
                break

        tab1, tab2, tab3, tab4 = st.tabs(["📊 TradingView Chart", "📈 Options Payoff", "📋 Signal Details", "📊 QuantStats Tearsheet"])

        with tab1:
            col1, col2 = st.columns([3, 1])
            with col2:
                show_indicators = st.checkbox("Show Indicators", value=True)
                chart_height = st.slider("Chart Height", 500, 1200, 800)
            with st.spinner(f"Loading chart for {chart_symbol}..."):
                fig = create_tradingview_chart(df, chart_symbol, signal, height=chart_height)
                st.plotly_chart(fig, width='stretch')

        with tab2:
            if signal:
                fig = create_options_payoff_chart(signal, config)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No signal available for options payoff")

        with tab3:
            if signal:
                st.json({
                    'symbol': signal['symbol'],
                    'entry': signal['entry_price'],
                    'stop_loss': signal['stop_price'],
                    'targets': signal['targets'],
                    'risk_reward': [(t - signal['entry_price']) / (signal['entry_price'] - signal['stop_price']) for t in signal['targets']] if signal['stop_price'] < signal['entry_price'] else []
                })
            else:
                st.info("No signal details available")

        with tab4:
            st.subheader("📊 QuantStats Tearsheet")
            if not QUANTSTATS_AVAILABLE:
                st.warning("⚠️ QuantStats not installed. Install with: `pip install quantstats`")
                st.info("Run locally: `pip install quantstats` to enable professional tearsheets.")
            else:
                try:
                    if isinstance(df.index, pd.MultiIndex):
                        sym_df = df.xs(chart_symbol, level='Symbol')
                    else:
                        sym_df = df

                    if sym_df.empty:
                        st.warning("No data available for tearsheet")
                    else:
                        # Prepare returns for QuantStats
                        returns = sym_df['Close'].pct_change().dropna()

                        if len(returns) < 2:
                            st.warning("Insufficient data for tearsheet (need at least 2 data points)")
                        else:
                            with st.spinner("Generating QuantStats tearsheet..."):
                                # Generate HTML tearsheet
                                html_report = qs.reports.html(
                                    returns,
                                    benchmark=None,  # Could add Nifty as benchmark
                                    title=f'{chart_symbol} - QuantStats Tearsheet',
                                    output=None  # Return as string
                                )

                                # Display in Streamlit
                                st.components.v1.html(html_report, height=800, scrolling=True)

                except Exception as e:
                    st.error(f"Error generating tearsheet: {e}")
                    st.info("Ensure QuantStats is installed: `pip install quantstats`")


def render_scoring_page(params):
    """Dedicated scoring & alerts page."""
    config = params['config']
    symbols = params['symbols']

    st.title("🎯 Multi-Factor Scoring & Alerts")

    # Load data and regime
    with st.spinner("Loading data..."):
        df = fetch_market_data(symbols, period=params['period'], interval=params['timeframe'])

    if df.empty:
        st.error("No data available")
        return

    fetcher = create_fetcher()
    nifty_df = fetcher.get_nifty_index_data(period="1y", interval="1d")
    nifty_df = ensure_multiindex(nifty_df)
    nifty_df = add_all_indicators(nifty_df, config)

    detector = RegimeDetector(config)
    regime = detector.detect_nifty_regime(nifty_df)

    regime_info = {
        'regime': regime.regime,
        'confidence': regime.confidence,
        'size_mult': detector.get_size_multiplier(regime),
        'nifty_trend': regime.nifty_trend
    }

    st.info(f"📊 Current Regime: **{regime.regime.upper()}** (Confidence: {regime.confidence:.0%}) | Nifty Trend: {regime.nifty_trend}")

    # Run scoring
    if st.button("🚀 Run Full Universe Scan", type="primary"):
        with st.spinner("Scoring all symbols..."):
            scorer = create_scorer(config.get('scoring'))
            scores = scorer.score_universe(df, regime_info, 'trend_pullback', min_score=0)
            top_picks = scorer.get_top_picks(scores, top_n=params['top_n'], require_backtest=params['require_backtest'])

            # Store in session state
            st.session_state['last_scores'] = scores
            st.session_state['last_top_picks'] = top_picks
            st.session_state['last_regime'] = regime_info

    # Display results from session state
    if 'last_scores' in st.session_state:
        scores = st.session_state['last_scores']
        top_picks = st.session_state['last_top_picks']

        st.markdown(f"### 🔔 Top {len(top_picks)} Alerts for Today")

        # Telegram-ready message
        if top_picks:
            alert_msg = f"🎯 *Swing Alerts - {datetime.now().strftime('%d %b %Y')}*\n"
            alert_msg += f"Regime: {st.session_state['last_regime']['regime'].upper()} ({st.session_state['last_regime']['confidence']:.0%})\n\n"
            for i, s in enumerate(top_picks, 1):
                alert_msg += f"{i}. *{s.symbol}* - Score: {s.total_score:.1f}/100\n"
                alert_msg += f"   Entry: ~₹{s.factor_scores[0].metadata.get('ema20', 0):.2f} | "
                alert_msg += f"SL: ~₹{s.factor_scores[0].metadata.get('chandelier', 0):.2f}\n"
            alert_msg += f"\n_Backtest gate: WinRate>55%, Expectancy>0.3R_"

            st.code(alert_msg, language='markdown')
            st.caption("Copy above for Telegram")

            # Scorecards
            cols = st.columns(min(len(top_picks), 5))
            for i, score in enumerate(top_picks):
                with cols[i % len(cols)]:
                    st.markdown(create_scorecard(score), unsafe_allow_html=True)

        # Full table
        with st.expander("📊 All Scored Symbols"):
            score_data = []
            for s in scores:
                score_data.append({
                    'Symbol': s.symbol,
                    'Total Score': f"{s.total_score:.1f}",
                    'Trend': f"{next((f.score for f in s.factor_scores if f.name=='trend_alignment'), 0):.0f}",
                    'Momentum': f"{next((f.score for f in s.factor_scores if f.name=='momentum'), 0):.0f}",
                    'Volume': f"{next((f.score for f in s.factor_scores if f.name=='volume_surge'), 0):.0f}",
                    'Volatility': f"{next((f.score for f in s.factor_scores if f.name=='volatility'), 0):.0f}",
                    'Structure': f"{next((f.score for f in s.factor_scores if f.name=='structure'), 0):.0f}",
                    'Regime': f"{next((f.score for f in s.factor_scores if f.name=='regime_alignment'), 0):.0f}",
                    'Backtest': "✅" if s.backtest_passed else "❌",
                    'Win Rate': f"{s.backtest_metrics.get('win_rate', 0):.1%}" if s.backtest_metrics else "N/A",
                    'Expectancy': f"{s.backtest_metrics.get('expectancy', 0):.2f}R" if s.backtest_metrics else "N/A"
                })
            st.dataframe(pd.DataFrame(score_data), width='stretch')


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

        nifty_df = fetcher.get_nifty_index_data(
            period=f"{(end_date - start_date).days}d",
            interval="1d"
        )
        # Robust Nifty column handling
        if not nifty_df.empty:
            if isinstance(nifty_df.columns, pd.MultiIndex):
                try:
                    nifty_df.columns = nifty_df.columns.get_level_values(1)
                except Exception:
                    nifty_df.columns = nifty_df.columns.get_level_values(0)

            col_mapping = {}
            for col in nifty_df.columns:
                col_str = str(col).lower()
                if col_str in ('open', 'high', 'low', 'close', 'volume', 'adj close', 'adj_close'):
                    col_mapping[col] = col_str.capitalize().replace('_', ' ')
                elif col_str == 'adj close':
                    col_mapping[col] = 'Adj Close'
            if col_mapping:
                nifty_df = nifty_df.rename(columns=col_mapping)
            nifty_df.index = pd.to_datetime(nifty_df.index)

            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in required if c not in nifty_df.columns]
            if missing:
                alt_map = {
                    'Open': ['open', 'OPEN', 'Open Price', 'open price'],
                    'High': ['high', 'HIGH', 'High Price', 'high price'],
                    'Low': ['low', 'LOW', 'Low Price', 'low price'],
                    'Close': ['close', 'CLOSE', 'Close Price', 'Adj Close', 'AdjClose', 'adj close', 'adj_close', 'close price'],
                    'Volume': ['volume', 'VOLUME', 'Vol', 'vol']
                }
                for req in missing[:]:
                    for alt in alt_map.get(req, []):
                        if alt in nifty_df.columns:
                            nifty_df[req] = nifty_df[alt]
                            missing.remove(req)
                            break

                if missing and len(nifty_df.columns) >= 5:
                    for i, req in enumerate(missing[:]):
                        if i < len(nifty_df.columns):
                            nifty_df[req] = nifty_df.iloc[:, i]
                            missing.remove(req)

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
        st.dataframe(sig_df, width='stretch')

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

    fetcher = create_fetcher()

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

    if total_value > portfolio.get('peak_value', 0):
        portfolio['peak_value'] = total_value
        save_portfolio_state(portfolio)

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
        st.dataframe(pd.DataFrame(pos_data), width='stretch')

    if st.button("🔄 Run Daily Cycle", type="primary"):
        with st.spinner("Running daily cycle..."):
            symbols = list(portfolio['positions'].keys()) or fetcher.nifty200_symbols[:20]
            df = fetcher.get_price_data(symbols, period="3mo", interval="1d")
            df = ensure_multiindex(df)
            df = add_all_indicators(df, config)

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

            swing_signals = run_all_strategies(df, config, regime_info)

            prices = {}
            for sym in symbols:
                try:
                    prices[sym] = df.xs(sym, level='Symbol')['Close'].iloc[-1]
                except Exception:
                    pass

            for s in swing_signals:
                if s.symbol in prices:
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

            for sym, pos in portfolio['positions'].items():
                if sym in prices:
                    pos['current_price'] = prices[sym]
                    pos['unrealized_pnl'] = (prices[sym] - pos['avg_entry_price']) * pos['quantity']

            save_portfolio_state(portfolio)
            st.success("Cycle complete!")
            st.rerun()

    if st.session_state.get('paper_trades_log'):
        with st.expander("📜 Trade History"):
            trades_df = pd.DataFrame(st.session_state.paper_trades_log)
            st.dataframe(trades_df, width='stretch')


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
                st.dataframe(res_df, width='stretch')
            except Exception as e:
                st.error(f"Scan failed: {e}")


def render_regime_page(params):
    """Regime monitor page."""
    config = params['config']

    st.title("📊 Regime Monitor")

    try:
        fetcher = create_fetcher()
        nifty_df = fetcher.get_nifty_index_data(period="2y", interval="1d")

        # Robust Nifty column handling
        if not nifty_df.empty:
            if isinstance(nifty_df.columns, pd.MultiIndex):
                try:
                    nifty_df.columns = nifty_df.columns.get_level_values(1)
                except Exception:
                    nifty_df.columns = nifty_df.columns.get_level_values(0)

            col_mapping = {}
            for col in nifty_df.columns:
                col_str = str(col).lower()
                if col_str in ('open', 'high', 'low', 'close', 'volume', 'adj close', 'adj_close'):
                    col_mapping[col] = col_str.capitalize().replace('_', ' ')
                elif col_str == 'adj close':
                    col_mapping[col] = 'Adj Close'
            if col_mapping:
                nifty_df = nifty_df.rename(columns=col_mapping)
            nifty_df.index = pd.to_datetime(nifty_df.index)

            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in required if c not in nifty_df.columns]
            if missing:
                alt_map = {
                    'Open': ['open', 'OPEN', 'Open Price', 'open price'],
                    'High': ['high', 'HIGH', 'High Price', 'high price'],
                    'Low': ['low', 'LOW', 'Low Price', 'low price'],
                    'Close': ['close', 'CLOSE', 'Close Price', 'Adj Close', 'AdjClose', 'adj close', 'adj_close', 'close price'],
                    'Volume': ['volume', 'VOLUME', 'Vol', 'vol']
                }
                for req in missing[:]:
                    for alt in alt_map.get(req, []):
                        if alt in nifty_df.columns:
                            nifty_df[req] = nifty_df[alt]
                            missing.remove(req)
                            break

                if missing and len(nifty_df.columns) >= 5:
                    for i, req in enumerate(missing[:]):
                        if i < len(nifty_df.columns):
                            nifty_df[req] = nifty_df.iloc[:, i]
                            missing.remove(req)

                if missing:
                    st.error(f"Nifty data missing columns: {missing}. Available: {list(nifty_df.columns)}")
                    st.stop()

        nifty_df = ensure_multiindex(nifty_df)
        nifty_df = add_all_indicators(nifty_df, config)

        detector = RegimeDetector(config)

        with st.spinner("Analyzing historical regimes..."):
            from indicators.regime import analyze_historical_regimes
            regimes_df = analyze_historical_regimes(nifty_df, detector)

        if not regimes_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                regime_counts = regimes_df['regime'].value_counts()
                fig = go.Figure(data=[go.Pie(labels=regime_counts.index, values=regime_counts.values,
                                             marker_colors=['#4caf50', '#f44336', '#ff9800', '#9c27b0'])])
                fig.update_layout(title="Regime Distribution (Historical)")
                st.plotly_chart(fig, width='stretch')

            with col2:
                fig = go.Figure()
                regime_colors = {'bull': '#4caf50', 'bear': '#f44336', 'sideways': '#ff9800', 'volatile': '#9c27b0'}
                for regime in regimes_df['regime'].unique():
                    mask = regimes_df['regime'] == regime
                    fig.add_trace(go.Scatter(
                        x=regimes_df.index[mask],
                        y=[regime] * mask.sum(),
                        mode='markers',
                        name=regime,
                        marker=dict(color=regime_colors.get(regime, 'gray'), size=8)
                    ))
                fig.update_layout(title="Regime Timeline", yaxis_title="Regime", showlegend=False)
                st.plotly_chart(fig, width='stretch')

            st.subheader("Current Regime Details")
            latest_regime = detector.detect_nifty_regime(nifty_df)
            st.markdown(detector.get_regime_summary(latest_regime))

            # Strategy recommendations table
            st.subheader("Strategy Suitability by Regime")
            strat_data = []
            for strat in ['trend_pullback', 'breakout_volume', 'mean_reversion_bb']:
                should, reason = detector.should_trade_swing(latest_regime, strat)
                strat_data.append({
                    'Strategy': strat,
                    'Trade?': '✅' if should else '❌',
                    'Reason': reason
                })
            st.dataframe(pd.DataFrame(strat_data), width='stretch')

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

    if params['mode'] == "Dashboard":
        render_dashboard_page(params)
    elif params['mode'] == "Scoring & Alerts":
        render_scoring_page(params)
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