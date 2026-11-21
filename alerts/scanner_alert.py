"""
Automated 3:45 PM IST Swing Alerts with SQLite Deduplication.

Runs Monday-Friday at 3:45 PM IST (10:15 UTC) via GitHub Actions.
Sends top 5 Nifty 200 swing setups to Telegram.
Tracks sent alerts in logs/alerts_sent.db to avoid duplicates.
"""

import os
import sys
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import list, dict

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

import requests
import yaml

from data.fetcher import create_fetcher
from indicators.regime import RegimeDetector
from indicators.technical import add_all_indicators, ensure_multiindex
from strategies.scoring import create_scorer

logger = logging.getLogger(__name__)

# Alert configuration
ALERTS_DB = Path("./logs/alerts_sent.db")
ALERTS_DB.parent.mkdir(parents=True, exist_ok=True)

IST = timezone(timedelta(hours=5, minutes=30))
UTC = datetime.UTC

# Telegram config (set via environment variables or GitHub secrets)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


@dataclass
class AlertRecord:
    """Record of a sent alert."""
    symbol: str
    date: str          # YYYY-MM-DD
    strategy: str
    score: float
    entry_price: float
    stop_price: float
    targets: list[float]
    regime: str
    message_hash: str  # For deduplication
    sent_at: str       # ISO timestamp


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
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_date
            ON alerts_sent(symbol, date)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_message_hash
            ON alerts_sent(message_hash)
        """)
        conn.commit()


def generate_message_hash(symbol: str, date: str, entry: float, stop: float) -> str:
    """Generate unique hash for alert deduplication."""
    import hashlib
    content = f"{symbol}|{date}|{entry:.2f}|{stop:.2f}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def is_alert_sent(symbol: str, date: str, entry: float, stop: float) -> bool:
    """Check if alert was already sent for this symbol/date/level."""
    msg_hash = generate_message_hash(symbol, date, entry, stop)
    with sqlite3.connect(ALERTS_DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT 1 FROM alerts_sent WHERE message_hash = ?",
            (msg_hash,)
        ).fetchone()
        return row is not None


def record_alert(alert: AlertRecord):
    """Record sent alert in database."""
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


def get_recent_alerts(days: int = 7) -> list[dict]:
    """Get recently sent alerts."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    with sqlite3.connect(ALERTS_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT symbol, date, strategy, score, entry_price, stop_price, targets, regime, sent_at
            FROM alerts_sent
            WHERE date >= ?
            ORDER BY sent_at DESC
        """, (cutoff,)).fetchall()
        return [dict(r) for r in rows]


def format_telegram_message(scores, regime_info: dict, top_n: int = 5) -> str:
    """Format alerts as Telegram message (Markdown)."""
    now_ist = datetime.now(IST)
    date_str = now_ist.strftime('%d %b %Y')
    time_str = now_ist.strftime('%H:%M IST')

    regime = regime_info.get('regime', 'unknown').upper()
    confidence = regime_info.get('confidence', 0)

    # Regime emoji
    regime_emoji = {'BULL': '🟢', 'BEAR': '🔴', 'SIDEWAYS': '🟡', 'VOLATILE': '⚡'}.get(regime, '⚪')

    msg = f"*🎯 Swing Alerts - {date_str} ({time_str})*\n"
    msg += f"Regime: {regime_emoji} *{regime}* ({confidence:.0%}) | Nifty Trend: {regime_info.get('nifty_trend', 'unknown').title()}\n\n"

    if not scores:
        msg += "_No qualifying setups today_"
        return msg

    for i, s in enumerate(scores[:top_n], 1):
        entry = s.factor_scores[0].metadata.get('ema20', 0) if s.factor_scores else 0
        stop = s.factor_scores[0].metadata.get('chandelier', 0) if s.factor_scores else 0

        # Get actual signal details
        risk_r = (entry - stop) / entry * 100 if entry > 0 and stop > 0 else 0
        r1 = ((s.factor_scores[0].metadata.get('ema20', 0) * 1.02) - entry) / (entry - stop) if entry > stop else 0

        msg += f"*{i}. {s.symbol}* — Score: `{s.total_score:.1f}/100`\n"
        msg += f"   📍 Entry: ~₹{entry:.2f} | 🛑 SL: ₹{stop:.2f} ({risk_r:.1f}% risk)\n"
        if s.backtest_passed:
            bt = s.backtest_metrics
            msg += f"   ✅ Backtest: WR={bt.get('win_rate',0):.0%} Exp={bt.get('expectancy',0):.2f}R PF={bt.get('profit_factor',0):.2f}\n"
        else:
            msg += f"   ⚠️ Backtest: *FAILED* (below threshold)\n"
        msg += f"   📊 Regime: {s.regime} | Confidence: {s.regime_confidence:.0%}\n\n"

    msg += f"_Backtest Gate: WinRate >55%, Expectancy >0.3R, PF >1.0, Min 10 trades_\n"
    msg += f"_Auto-generated by RegimeSwing • [GitHub](https://github.com)_"

    return msg


def send_telegram(message: str) -> bool:
    """Send message to Telegram."""
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


def run_alert_cycle(config: dict, top_n: int = 5, min_score: float = 60,
                    require_backtest: bool = True) -> list[dict]:
    """
    Run a single alert cycle: fetch data, score, filter, send alerts.

    Returns list of alerts sent.
    """
    fetcher = create_fetcher()
    symbols = fetcher.nifty200_symbols

    logger.info(f"Fetching data for {len(symbols)} symbols...")

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
        'nifty_trend': regime.nifty_trend
    }

    logger.info(f"Regime: {regime.regime} (confidence: {regime.confidence:.0%})")

    # Score universe
    scorer = create_scorer(config.get('scoring'))
    scores = scorer.score_universe(df, regime_info, 'trend_pullback', min_score=min_score)
    top_picks = scorer.get_top_picks(scores, top_n=top_n, require_backtest=require_backtest)

    if not top_picks:
        logger.info("No qualifying alerts today")
        return []

    # Filter out already sent alerts
    today = datetime.now(IST).strftime('%Y-%m-%d')
    new_alerts = []
    for s in top_picks:
        entry = s.factor_scores[0].metadata.get('ema20', 0) if s.factor_scores else 0
        stop = s.factor_scores[0].metadata.get('chandelier', 0) if s.factor_scores else 0

        if not is_alert_sent(s.symbol, today, entry, stop):
            new_alerts.append(s)
        else:
            logger.info(f"Skipping duplicate alert for {s.symbol}")

    if not new_alerts:
        logger.info("All alerts already sent today")
        return []

    # Send Telegram alert
    message = format_telegram_message(new_alerts, regime_info, top_n)
    sent = send_telegram(message)

    if sent:
        # Record alerts
        for s in new_alerts:
            entry = s.factor_scores[0].metadata.get('ema20', 0) if s.factor_scores else 0
            stop = s.factor_scores[0].metadata.get('chandelier', 0) if s.factor_scores else 0
            targets = []

            alert = AlertRecord(
                symbol=s.symbol,
                date=today,
                strategy='trend_pullback',
                score=s.total_score,
                entry_price=entry,
                stop_price=stop,
                targets=targets,
                regime=s.regime,
                message_hash="",  # Will be generated in record_alert
                sent_at=datetime.now(UTC).isoformat()
            )
            record_alert(alert)

        logger.info(f"Sent {len(new_alerts)} new alerts")
        return [asdict(a) for a in new_alerts]

    return []


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for alert script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )

    init_alerts_db()

    config = load_config()

    # Parse command line args
    import argparse
    parser = argparse.ArgumentParser(description="Swing Alert Generator")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top alerts")
    parser.add_argument("--min-score", type=float, default=60, help="Minimum score threshold")
    parser.add_argument("--no-backtest", action="store_true", help="Disable backtest requirement")
    parser.add_argument("--dry-run", action="store_true", help="Print message without sending")
    parser.add_argument("--test-telegram", action="store_true", help="Send test message")

    args = parser.parse_args()

    if args.test_telegram:
        test_msg = f"🧪 *Test Message* - RegimeSwing alerts working!\nTime: {datetime.now(IST).strftime('%H:%M IST')}"
        send_telegram(test_msg)
        return

    alerts = run_alert_cycle(
        config,
        top_n=args.top_n,
        min_score=args.min_score,
        require_backtest=not args.no_backtest
    )

    if args.dry_run:
        # Print what would be sent
        if alerts:
            fetcher = create_fetcher()
            nifty_df = fetcher.get_nifty_index_data(period="1y", interval="1d")
            nifty_df = ensure_multiindex(nifty_df)
            nifty_df = add_all_indicators(nifty_df, config)
            detector = RegimeDetector(config)
            regime = detector.detect_nifty_regime(nifty_df)
            regime_info = {
                'regime': regime.regime,
                'confidence': regime.confidence,
                'nifty_trend': regime.nifty_trend
            }
            # We need scores for formatting - fetch them again
            df = fetcher.get_price_data(fetcher.nifty200_symbols, period="6mo", interval="1d")
            df = ensure_multiindex(df)
            df = add_all_indicators(df, config)
            scorer = create_scorer(config.get('scoring'))
            scores = scorer.score_universe(df, regime_info, 'trend_pullback', min_score=args.min_score)
            top_picks = scorer.get_top_picks(scores, top_n=args.top_n, require_backtest=not args.no_backtest)
            print(format_telegram_message(top_picks, regime_info, args.top_n))
        else:
            print("No alerts to send")
    else:
        print(f"Cycle complete. Sent {len(alerts)} alerts.")


if __name__ == "__main__":
    main()