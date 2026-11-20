# Swing Trading + Options Selling Toolkit
**Professional-grade, research-backed toolkit for Indian markets (NSE/BSE)**

Built for systematic swing trading (5-20 day holds) and options selling (CSP, CC, Wheel, Credit Spreads) with regime awareness, realistic risk management, and paper trading.

---

## 🎯 What This Does

| Feature | Description |
|---------|-------------|
| **Swing Strategies** | Trend Pullback, Breakout Volume, Mean Reversion (BB) |
| **Options Strategies** | Cash-Secured Puts, Covered Calls, Wheel, Credit Spreads |
| **Regime Detection** | Bull/Bear/Sideways/Volatile → adjusts sizing & strategy selection |
| **Risk Management** | Portfolio-level risk limits, Greeks limits, drawdown circuits, correlation filters |
| **Targets/Exits** | ATR/Chandelier stops, Fibonacci targets, Max Pain, time-based exits |
| **News/Sentiment** | Free RSS sources + FinBERT sentiment → risk filter |
| **Backtesting** | Walk-forward (3yr train/1yr test), Monte Carlo, regime attribution |
| **Paper Trading** | Realistic fills, SQLite persistence, daily reports, stop/target management |
| **Dashboard** | Streamlit UI: charts, signals, portfolio, regime monitor, scanner |

---

## 📁 Project Structure

```
RegimeSwing_Final_Deploy/
├── streamlit_app.py         # Streamlit Cloud entry point
├── config.yaml              # All parameters (edit this!)
├── main.py                  # CLI entry point
├── requirements.txt         # Dependencies
├── README.md                # This file
├── .streamlit/
│   ├── config.toml          # Streamlit configuration
│   └── secrets.toml.template # Secrets template
├── data/
│   ├── fetcher.py           # Price/option/fundamental data (Kite + yfinance)
│   ├── nse_option_chain.py  # Free NSE option chain
│   └── news_fetcher.py      # RSS + FinBERT sentiment
├── indicators/
│   ├── technical.py         # 20+ indicators (EMA, RSI, ATR, BB, VWAP, etc.)
│   ├── targets.py           # Stops, targets, trailing, position sizing
│   └── regime.py            # Market regime detection
├── strategies/
│   ├── swing.py             # 3 swing strategies
│   └── options.py           # 4 options strategies
├── papertrade/
│   └── engine.py            # Paper trading engine with risk guards
├── backtest/
│   └── engine.py            # Walk-forward backtester + Monte Carlo
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── tests/                   # Unit tests
└── logs/                    # Auto-created: trades.csv, state.db
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd RegimeSwing_Final_Deploy
pip install -r requirements.txt
```

### 2. Configure (IMPORTANT)
Edit `config.yaml` with your settings:
```yaml
capital:
  initial: 500000           # Your capital
  max_risk_per_trade: 0.015 # 1.5% risk per trade

broker:
  api_key: ""               # Your Kite API key
  api_secret: ""            # Your Kite API secret
  # access_token auto-generated daily via `python main.py auth`
```

### 3. Authenticate with Kite (one-time, then daily)
```bash
python main.py auth
# Follow prompts: opens browser → login → copy request_token → paste
```

### 4. Run Backtest
```bash
# Quick signal generation
python main.py backtest --strategy all

# Full walk-forward backtest with Monte Carlo
python main.py backtest-wf --start 2020-01-01 --end 2024-12-31
```

### 5. Paper Trade
```bash
# Single cycle
python main.py papertrade --once

# Continuous daemon mode
python main.py papertrade --daemon --interval 300
```

### 6. Launch Dashboard
```bash
python main.py dashboard
# Opens http://localhost:8501
```

---

## ☁️ Streamlit Cloud Deployment

### One-Click Deploy
[![Deploy to Streamlit](https://streamlit.io/badge.svg)](https://share.streamlit.io/deploy?repository=<your-github-repo>&main-file=streamlit_app.py)

### Manual Deployment Steps

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Main file: `streamlit_app.py`
   - Click "Deploy!"

3. **Configure Secrets (Optional):**
   - In Streamlit Cloud dashboard, go to App Settings → Secrets
   - Add your config from `.streamlit/secrets.toml.template`:
   ```toml
   [broker]
   api_key = "your_kite_api_key"
   api_secret = "your_kite_api_secret"

   [config]
   capital = {initial = 500000, max_risk_per_trade = 0.015}
   ```

### Cloud Limitations
| Feature | Local | Streamlit Cloud |
|---------|-------|-----------------|
| Dashboard/Signals/Regime/Scanner | ✅ | ✅ |
| Walk-Forward Backtest | ✅ | ❌ (resource limits) |
| Paper Trading (persistent) | ✅ | ⚠️ Session only |
| Alerts (Telegram/Email) | ✅ | ✅ (with secrets) |

**Note:** The dashboard runs in "Demo Mode" on Streamlit Cloud with built-in default config. Paper trading and backtesting require local deployment.

---

## 🔧 Commands Reference

```bash
# Backtest
python main.py backtest --strategy all --start 2020-01-01
python main.py backtest-wf --start 2020-01-01 --end 2024-12-31

# Paper Trading
python main.py papertrade --once
python main.py papertrade --daemon --interval 300

# Dashboard
python main.py dashboard

# Authenticate
python main.py auth

# Scan
python main.py scan --symbols RELIANCE,TCS,HDFCBANK

# Regime Check
python main.py regime

# Multi-timeframe Confluence
python main.py confluence --symbols RELIANCE,TCS
```

---

## 📊 Strategies Overview

### Swing Strategies

| Strategy | Regime | Setup | Entry | Stop | Targets |
|----------|--------|-------|-------|-------|---------|
| **Trend Pullback** | Bull | Price > EMA50>EMA200, ADX>25, pullback to EMA20 zone | Close > EMA20 + bullish candle | Chandelier(22,3) | 1.5R, 2.5R, Trail |
| **Breakout Volume** | Bull | 20d consolidation <8%, breakout + 1.5x vol | Close > 20d high | Below breakout low | 2R, 3R, Trail 20d low |
| **Mean Reversion BB** | Sideways/Bull | Price ≤ Lower BB, RSI<30, ADX<20 | Close inside BB + bullish candle | Below candle low | BB Mid → BB Upper |

### Options Strategies

| Strategy | When | Strike | DTE | Management |
|----------|------|--------|-----|------------|
| **CSP** | IV Rank>60, uptrend, delivery>50% | Delta 0.25-0.30 | 30-45 | Close 50%, roll 21 DTE, stop 2x |
| **CC** | Long pos >10% gain | Delta 0.30-0.35 | 30-45 | Roll if ITM at 5 DTE |
| **Wheel** | CSP → Assignment → CC → Repeat | - | - | Track annualized return |
| **Credit Spreads** | Regime-based | 0.20/0.10 delta | 21-30 | Target 50% credit |

---

## 🎯 Realistic Expectations

| Metric | Target |
|--------|--------|
| **Win Rate** | 55-65% (swing), 70-80% (CSP expire worthless) |
| **Avg R:R** | 1.5-2.5 |
| **Expectancy** | 0.4-0.6R per trade |
| **Net CAGR** | 15-20% post-tax |
| **Max Drawdown** | 12-18% |
| **Sharpe** | > 1.0 |

**⚠️ Not 90% win rate** — that's a scam metric. High win rate = tiny targets = blowup risk.

---

## ⚠️ Important Notes

1. **Free Kite Tier Limits**: 3 req/sec, daily historical only, no websocket. Daily auth required.
2. **Costs Matter**: Backtest includes STT, brokerage, slippage, stamp duty. Paper trading simulates same.
3. **Regime Filter**: Strategies auto-disable in unfavorable regimes (e.g., no breakouts in bear).
4. **State Persistence**: Paper trading saves to SQLite — survives crashes/restarts.
5. **No Lookahead Bias**: Walk-forward backtest uses expanding window, no future data.

---

## 🛠️ Extending the Toolkit

### Add New Strategy
```python
# strategies/swing/my_strategy.py
from strategies.swing import BaseSwingStrategy, Signal

class MyStrategy(BaseSwingStrategy):
    def generate_signals(self, df, regime_info):
        # Your logic
        return signals

# Register in strategies/swing/__init__.py
STRATEGY_REGISTRY['my_strategy'] = MyStrategy
```

### Add New Indicator
```python
# indicators/technical.py
def add_my_indicator(df, param=14):
    def _calc(sdf):
        sdf['MY_IND'] = ...  # pandas-ta or custom
        return sdf
    return apply_per_symbol(df, _calc)
```

---

## 📚 Learning Path (Built-in)

| Tool | Purpose |
|------|---------|
| **Bar Replay** | Practice entries on historical data (dashboard) |
| **Strategy Lab** | Instant parameter sensitivity (change param → re-backtest) |
| **AI Coach** | Analyzes your paper trades vs system |
| **Regime Playbook** | Visual cheat sheets for each regime |
| **Graduation Protocol** | 30-day paper→live transition plan |

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Kite auth fails | Check API key/secret, run `python main.py auth` |
| No data fetched | Check internet, try yfinance fallback |
| Dashboard won't start | `pip install streamlit plotly` |
| Backtest slow | Reduce symbols, disable Monte Carlo |

---

## 📄 License & Disclaimer

**Educational/Research purposes only.** Not financial advice.

- Past performance ≠ future results
- Test thoroughly before live trading
- Options involve substantial risk
- Indian market regulations apply (SEBI)

---

## 🤝 Contributing

Issues and PRs welcome! Focus areas:
- More option chain data sources
- Additional patterns (Wyckoff, VCP, Harmonics)
- ML-enhanced targets
- Multi-broker support (Fyers, Upstox)

---

## 🤖 Built with

Python, pandas, pandas-ta, vectorbt, plotly, streamlit, scikit-learn, transformers (FinBERT)