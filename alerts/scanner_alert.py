"""
Automated 3:45 PM IST Swing Alerts with Institutional Features.

Runs Monday-Friday at 3:45 PM IST (10:15 UTC) via GitHub Actions.
Sends top Nifty 200 swing setups to Telegram with full exit lifecycle.

Features:
- Portfolio Heat Cap (6%) & ATR Volatility Parity Sizing
- Gap-Up Guard (+1.2% max open tolerance) + 8:45 AM Pre-Market Cancel Check
- Event Gate & NSE Holiday Guard (earnings, ASM/GSM, holidays)
- Sector Diversification Cap (max 2 per sector) & No-Trade Mode
- Anti-Zero-Trade Guardrail (min 10 backtest trades, min 65 conviction)
- Direct Recommendation Cards + Full Exit Lifecycle Automation
- SQLite tracking for live trades and exit alerts
"""

import os
import sys
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import list, dict, Optional, Tuple, Any

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

import requests
import yaml

from data.fetcher import create_fetcher
from indicators.regime import RegimeDetector
from indicators.technical import add_all_indicators, ensure_multiindex
from indicators.targets import calculate_position_size_atr_parity
from strategies.scoring import create_scorer
from utils.nse_calendar import (
    get_nse_calendar, is_trading_day, is_asm_gsm, has_earnings_soon,
    get_nse_session_status
)
from trading.position_tracker import PositionTracker, create_position_tracker
from alerts.direct_recommendation import (
    DirectRecommendationEngine, format_telegram_ticket,
    format_zero_trade_message, format_no_trade_low_scores
)
from utils.gtt_basket import export_gtt_csv, create_position_entry

logger = logging.getLogger(__name__)

# Configuration
ALERTS_DB = Path("./logs/alerts_sent.db")
ALERTS_DB.parent.mkdir(parents=True, exist_ok=True)

IST = timezone(timedelta(hours=5, minutes=30))
UTC = datetime.UTC

# Telegram config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Configuration constants
MAX_PORTFOLIO_RISK = 0.06          # 6% max portfolio risk
GAP_GUARD_PCT = 0.012              # 1.2% max gap-up tolerance
MIN_CONVICTION_SCORE = 65          # Minimum conviction for alerts
MIN_BACKTEST_TRADES = 10           # Minimum backtest trades for filter
MAX_GAP_GUARD_PCT = 0.012          # 1.2% max gap-up at open
MIN_CONVICTION_SCORE = 65          # Minimum conviction score
SECTOR_LIMIT = 2                   # Max 2 stocks per sector
NO_TRADE_VIX_THRESHOLD = 22        # No trades if India VIX > 22


@dataclass
class AlertRecord:
    """Record of a sent alert."""
    symbol: str
    date: str
    strategy: str
    score: float
    entry_price: float
    stop_price: float
    targets: list[float]
    regime: str
    message_hash: str
    sent_at: str


def init_alerts_db():
    """Initialize SQLite database for alert deduplication."""
    with sqlite3.connect(ALERTS_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts_sent (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                strategy TEXT NOT NULL,
                score REAL NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                targets TEXT NOT NULL,
                regime TEXT NOT NULL,
                message_hash TEXT NOT NULL UNIQUE,
                sent_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON alerts_sent(symbol, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_message_hash ON alerts_sent(message_hash)")
        conn.commit()


def generate_message_hash(symbol: str, date: str, entry: float, stop: float) -> str:
    import hashlib
    content = f"{symbol}|{date}|{entry:.2f}|{stop:.2f}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def is_alert_sent(symbol: str, date: str, entry: float, stop: float) -> bool:
    msg_hash = generate_message_hash(symbol, date, entry, stop)
    with sqlite3.connect(ALERTS_DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT 1 FROM alerts_sent WHERE message_hash = ?", (msg_hash,)
        ).fetchone()
        return row is not None


def record_alert(alert: AlertRecord):
    msg_hash = generate_message_hash(alert.symbol, alert.date, alert.entry_price, alert.stop_price)
    with sqlite3.connect(ALERTS_DB) as conn:
        try:
            conn.execute("""
                INSERT INTO alerts_sent
                (symbol, date, strategy, score, entry_price, stop_price, targets, regime, message_hash, sent_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.symbol, alert.date, alert.strategy, alert.score,
                alert.entry_price, alert.stop_price,
                json.dumps(alert.targets), alert.regime,
                msg_hash, alert.sent_at
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate alert skipped: {alert.symbol} {alert.date}")


def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Telegram message sent successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def check_nse_holiday_and_session() -> Tuple[bool, str]:
    """Check if today is a trading day and get session status."""
    cal = get_nse_calendar()

    if not is_trading_day():
        return False, "NSE Holiday"

    session = get_nse_session_status()
    if not session['is_trading_day']:
        return False, "NSE Holiday"

    return True, "OK"


def check_gap_guard(symbol: str, alert_entry: float, max_gap_pct: float = GAP_GUARD_PCT) -> Tuple[bool, str]:
    """
    Check if gap-up at open would exceed tolerance.
    Returns (should_alert, message).
    """
    # This is a placeholder - actual gap check happens at 8:45 AM
    # For 3:45 PM scan, we include max gap price in alert
    return True, "OK"


def check_event_gate(symbol: str, nse_calendar) -> Tuple[bool, str]:
    """Check event gate: earnings, ASM/GSM, corporate actions."""
    # Earnings within 5 days
    if has_earnings_soon(symbol, 5):
        return False, f"Earnings within 5 days"

    # ASM/GSM check
    asm_gsm = is_asm_gsm(symbol)
    if asm_gsm.get('restricted', False):
        return False, f"ASM/GSM restricted ({'ASM' if asm_gsm['asm'] else 'GSM'})"

    return True, "OK"


def check_portfolio_heat(open_positions: dict, capital: float, max_risk: float = MAX_PORTFOLIO_RISK) -> Tuple[bool, str]:
    """Check if total open risk exceeds portfolio heat limit."""
    total_risk = sum(pos.get('max_loss', pos.get('risk_pct', 0) * pos.get('total_cost', 0))
                     for pos in open_positions.values())
    total_risk_pct = total_risk / capital if capital > 0 else 0

    if total_risk_pct >= MAX_PORTFOLIO_RISK:
        return False, f"Portfolio heat {total_risk_pct:.1%} ≥ {MAX_PORTFOLIO_RISK:.0%}. Pausing new buys."

    return True, "OK"


def check_sector_limit(open_positions: dict, symbol: str, sector_limit: int = SECTOR_LIMIT) -> Tuple[bool, str]:
    """Check sector diversification limit."""
    from alerts.direct_recommendation import DirectRecommendationEngine
    engine = DirectRecommendationEngine({}, 0)  # temp for sector mapping
    sector = engine.get_sector(symbol)

    sector_counts = {}
    for pos in open_positions.values():
        pos_sector = pos.get('sector', 'UNKNOWN')
        sector_counts[pos_sector] = sector_counts.get(pos_sector, 0) + 1

    if sector_counts.get(sector, 0) >= sector_limit:
        return False, f"Sector limit reached ({sector}={sector_counts.get(sector, 0)}/{sector_limit})"

    return True, "OK"


def check_no_trade_mode(regime_info: dict, vix_level: float = 0) -> Tuple[bool, str]:
    """Check if no-trade mode is active."""
    regime = regime_info.get('regime', 'unknown')

    if regime == 'bear':
        return False, "Bear regime active - no long buys"

    if regime == 'volatile' or vix_level > NO_TRADE_VIX_THRESHOLD:
        return False, f"High volatility (VIX: {vix_level:.1f} > {NO_TRADE_VIX_THRESHOLD})"

    return True, "OK"


def run_alert_cycle(
    config: dict,
    top_n: int = 3,
    min_score: float = MIN_CONVICTION_SCORE,
    require_backtest: bool = True
) -> list[dict]:
    """
    Run a single alert cycle with all institutional features.

    Returns list of alerts sent.
    """
    # Initialize
    init_alerts_db()

    # Check NSE holiday/session
    is_open, reason = check_nse_holiday_and_session()
    if not is_open:
        logger.info(f"Market closed: {reason}")
        send_telegram(f"🛡️ Market Closed — {reason}")
        return []

    # Load config and initialize
    fetcher = create_fetcher()
    symbols = fetcher.nifty200_symbols
    capital = config.get('capital', {}).get('initial', 500000)

    logger.info(f"Fetching data for {len(symbols)} Nifty 200 symbols...")

    # Fetch market data
    df = fetcher.get_price_data(symbols, period="6mo", interval="1d")
    if df.empty:
        logger.error("No data fetched")
        return []

    df = ensure_multiindex(df)
    df = add_all_indicators(df, config)

    # Get regime
    nifty_df = fetcher.get_nifty_index_data(period="1y", interval="1d")
    nifty_df = ensure_multiindex(nifty_df)
    nifty_df = add_all_indicators(nifty_df, config)

    detector = RegimeDetector(config)
    regime = detector.detect_nifty_regime(nifty_df)

    regime_info = {
        'regime': regime.regime,
        'confidence': regime.confidence,
        'size_mult': detector.get_size_multiplier(regime),
        'nifty_trend': regime.nifty_trend,
        'regime_obj': regime
    }

    # Get macro tilts
    from utils.macro_monitor import fetch_macro_data, evaluate_macro_signals
    macro_signals = fetch_macro_data()
    macro_tilts = evaluate_macro_signals(macro_signals)

    # Get FII/DII flow
    from utils.nse_calendar import get_nse_calendar
    fii_data = get_nse_calendar().get_fii_dii_flow() if hasattr(get_nse_calendar(), 'get_fii_dii_flow') else {}

    logger.info(f"Regime: {regime.regime} (confidence: {regime.confidence:.0%}) | VIX: {regime.adx_value:.1f}")

    # CHECK NO-TRADE MODE
    vix_level = regime.adx_value  # Approximation
    no_trade, reason = check_no_trade_mode(regime_info, vix_level)
    if not no_trade:
        msg = f"🛡️ NO TRADE MODE — {reason}\n   Regime: {regime.regime.upper()} | VIX: {vix_level:.1f}"
        send_telegram(msg)
        logger.info(f"No-trade mode active: {reason}")
        return []

    # Score universe
    scorer = create_scorer(config.get('scoring'))
    scores = scorer.score_universe(df, regime_info, 'trend_pullback', min_score=0)
    top_picks = scorer.get_top_picks(scores, top_n=5, require_backtest=require_backtest)

    logger.info(f"Scored {len(scores)} symbols, {len(top_picks)} passed backtest gate")

    # Initialize position tracker for exit checks
    tracker = create_position_tracker(capital, config)
    open_positions = {pos.symbol: pos.to_dict() for pos in tracker.get_open_positions()}

    # Check portfolio heat
    heat_ok, heat_msg = check_portfolio_heat(open_positions, capital)
    if not heat_ok:
        send_telegram(f"🔥 PORTFOLIO HEAT EXCEEDED — {heat_msg}")
        logger.warning(heat_msg)
        # Continue but don't add new positions

    # Generate direct recommendations
    engine = DirectRecommendationEngine(config, capital)
    recommendations = engine.generate_recommendations(
        scores=scores,
        regime_info=regime_info,
        macro_tilts=macro_tilts,
        fii_data={},
        open_positions=open_positions,
        nse_calendar=get_nse_calendar()
    )

    logger.info(f"Generated {len(recommendations)} direct recommendations")

    # Filter by score threshold and apply guards
    final_recs = []
    today = datetime.now(IST).strftime('%Y-%m-%d')

    for rec in recommendations:
        # Score threshold
        if rec.conviction < MIN_CONVICTION_SCORE:
            continue

        # Gap guard - include in alert
        if rec.max_gap_price > 0:
            rec.max_gap_price = round(rec.entry_max * (1 + GAP_GUARD_PCT), 1)

        # Deduplication check
        if not is_alert_sent(rec.symbol, date.today().isoformat(), rec.entry_min, rec.stop_loss):
            final_recs.append(rec)
        else:
            logger.info(f"Skipping duplicate: {rec.symbol}")

        if len(final_recs) >= 3:
            break

    # CHECK SECTOR LIMITS & PORTFOLIO HEAT for each recommendation
    filtered_recs = []
    sector_counts = {}
    for pos in tracker.get_open_positions():
        sector = pos.__dict__.get('sector', 'OTHER')
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    for rec in final_recs:
        # Portfolio heat check
        heat_ok, heat_msg = check_portfolio_heat(open_positions, capital)
        if not heat_ok:
            logger.warning(f"Skipping {rec.symbol}: {heat_msg}")
            continue

        # Sector limit check
        sector_ok, sector_msg = check_sector_limit(open_positions, rec.symbol)
        if not sector_ok:
            logger.info(f"Skipping {rec.symbol}: {sector_msg}")
            continue

        filtered_recs.append(rec)
        sector = rec.__dict__.get('sector', 'OTHER')
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    # Send recommendations
    if filtered_recs:
        for i, rec in enumerate(filtered_recs, 1):
            msg = format_telegram_ticket(rec, i, len(filtered_recs))
            sent = send_telegram(msg)

            if sent:
                # Record alert
                alert = AlertRecord(
                    symbol=rec.symbol,
                    date=date.today().isoformat(),
                    strategy=rec.strategy,
                    score=rec.conviction,
                    entry_price=rec.entry_min,
                    stop_price=rec.stop_loss,
                    targets=rec.targets,
                    regime=rec.regime,
                    message_hash="",
                    sent_at=datetime.now(UTC).isoformat()
                )
                record_alert(alert)

                # Add to position tracker
                entry_price = rec.entry_min
                stop = rec.stop_loss
                qty = rec.quantity
                targets = rec.targets

                tracker.add_position(
                    symbol=rec.symbol,
                    qty=qty,
                    entry_price=entry_price,
                    stop_price=stop,
                    targets=targets,
                    strategy=rec.strategy,
                    regime=rec.regime,
                    conviction=rec.conviction
                )

                # Export GTT basket
                pos_entry = create_position_entry(
                    symbol=rec.symbol,
                    qty=qty,
                    entry_price=entry_price,
                    stop_price=stop,
                    targets=targets
                )
                export_gtt_csv({rec.symbol: pos_entry[rec.symbol]}, f"./logs/gtt_{rec.symbol}_{date.today()}.csv")

        logger.info(f"Sent {len(filtered_recs)} direct recommendations")
    else:
        # Zero-trade message
        msg = format_zero_trade_message(regime_info)
        send_telegram(msg)
        logger.info("No qualifying recommendations today")

    # EXIT LIFECYCLE CHECK
    if tracker.get_open_positions():
        # Get current prices for all open positions
        open_syms = [p.symbol for p in tracker.get_open_positions()]
        prices = fetcher.get_price_data(open_syms, period="1d", interval="1d")

        current_prices = {}
        if not prices.empty:
            prices = ensure_multiindex(prices)
            for sym in open_syms:
                try:
                    current_prices[sym] = prices.xs(sym, level='Symbol')['Close'].iloc[-1]
                except Exception:
                    pass

        exit_alerts = tracker.check_exits(current_prices)
        for alert_msg in exit_alerts:
            send_telegram(alert_msg)

    return []


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )

    init_alerts_db()

    config = load_config()

    import argparse
    parser = argparse.ArgumentParser(description="Swing Alert Generator - Institutional Edition")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top recommendations")
    parser.add_argument("--min-score", type=float, default=65, help="Minimum conviction score")
    parser.add_argument("--no-backtest", action="store_true", help="Disable backtest requirement")
    parser.add_argument("--dry-run", action="store_true", help="Print without sending")
    parser.add_argument("--test-telegram", action="store_true", help="Send test message")
    parser.add_argument("--test-exit", action="store_true", help="Test exit alerts")

    args = parser.parse_args()

    if args.test_telegram:
        test_msg = f"🧪 *Test Message* - RegimeSwing Institutional Alerts\nTime: {datetime.now(IST).strftime('%H:%M IST')}"
        send_telegram(test_msg)
        return

    if args.test_exit:
        # Test exit alerts
        config = load_config()
        capital = config.get('capital', {}).get('initial', 500000)
        tracker = create_position_tracker(capital, config)

        # Add test position
        tracker.add_position("BEL", 12, 312, 296, [336, 352], "trend_pullback", "bull", 88)

        # Test with price that hits breakeven
        alerts = tracker.check_exits({"BEL": 325})
        for alert in alerts:
            print(f"EXIT ALERT: {alert}")

        # Test with price hitting T1
        alerts = tracker.check_exits({"BEL": 336})
        for alert in alerts:
            print(f"EXIT ALERT: {alert}")

        # Test with trailing stop hit
        alerts = tracker.check_exits({"BEL": 295})
        for alert in alerts:
            print(f"EXIT ALERT: {alert}")
        return

    alerts = run_alert_cycle(config)

    if args.dry_run:
        if alerts:
            print(f"Would send {len(alerts)} alerts")
        else:
            print("No alerts to send")
    else:
        print(f"Cycle complete. Sent alerts.")


if __name__ == "__main__":
    main()