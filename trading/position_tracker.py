"""
Position Tracker & Exit Lifecycle Automation.
Tracks live trades in SQLite, sends automatic exit/trailing updates via Telegram.
"""

import sqlite3
import logging
import json
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from typing import Union
import threading

logger = logging.getLogger(__name__)

# Database path
DB_DIR = Path("./logs")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "live_trades.db"


@dataclass
class LivePosition:
    """Live trade position with exit tracking."""
    id: int = 0
    symbol: str = ""
    qty: int = 0
    entry_price: float = 0.0
    stop_price: float = 0.0
    targets: List[float] = None
    entry_date: str = ""
    entry_time: str = ""
    product_type: str = "CNC"
    strategy: str = ""
    regime: str = ""
    conviction: int = 0

    # Exit tracking
    t1_booked: bool = False
    t2_booked: bool = False
    breakeven_moved: bool = False
    trailing_stop: float = 0.0
    highest_since_entry: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    status: str = "OPEN"  # OPEN, PARTIAL, CLOSED, STOPPED
    exit_reason: str = ""
    exit_price: float = 0.0
    exit_date: str = ""
    exit_time: str = ""

    # Risk metrics
    risk_per_share: float = 0.0
    max_loss: float = 0.0
    total_cost: float = 0.0

    def __post_init__(self):
        if self.targets is None:
            self.targets = []
        if not self.entry_date:
            self.entry_date = date.today().isoformat()
        if not self.entry_time:
            self.entry_time = datetime.now().strftime("%H:%M:%S")
        self.risk_per_share = abs(self.entry_price - self.stop_price)
        self.max_loss = self.risk_per_share * self.qty
        self.total_cost = self.entry_price * self.qty
        self.highest_since_entry = self.entry_price

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'LivePosition':
        return cls(**data)


class PositionTracker:
    """
    Tracks live trades in SQLite, monitors exits, sends Telegram alerts.
    """

    def __init__(
        self,
        capital: float,
        config: Dict,
        telegram_sender: Optional[callable] = None
    ):
        self.capital = capital
        self.config = config
        self.telegram_sender = telegram_sender
        self.db_path = DB_PATH
        self.lock = threading.Lock()
        self._init_db()

        # Config values
        self.max_risk_per_trade = config.get('capital', {}).get('max_risk_per_trade', 0.015)
        self.max_portfolio_risk = config.get('capital', {}).get('max_portfolio_risk', 0.06)
        self.max_position_pct = config.get('capital', {}).get('max_position_pct', 0.05)
        self.max_positions = config.get('capital', {}).get('max_positions', 10)
        self.sector_limit = config.get('capital', {}).get('max_sector_pct', 0.25)

        # Exit parameters
        self.trail_method = config.get('targets', {}).get('trail_method', 'chandelier')
        self.chandelier_mult = config.get('targets', {}).get('chandelier_mult', 3.0)
        self.force_close_hour = config.get('targets', {}).get('force_close_hour', 15.15)
        self.max_hold_days = config.get('targets', {}).get('max_hold_days_default', 20)
        self.time_stop_days = config.get('targets', {}).get('max_hold_days_default', 7)
        self.min_move_pct = 0.02
        self.max_days_no_move = 3

    def _init_db(self):
        """Initialize SQLite database with positions and alerts tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    qty INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    targets TEXT NOT NULL,  -- JSON array
                    entry_date TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    product_type TEXT DEFAULT 'CNC',
                    strategy TEXT DEFAULT '',
                    regime TEXT DEFAULT '',
                    conviction INTEGER DEFAULT 0,

                    t1_booked BOOLEAN DEFAULT 0,
                    t2_booked BOOLEAN DEFAULT 0,
                    breakeven_moved BOOLEAN DEFAULT 0,
                    trailing_stop REAL DEFAULT 0,
                    highest_since_entry REAL DEFAULT 0,
                    current_price REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'OPEN',
                    exit_reason TEXT DEFAULT '',
                    exit_price REAL DEFAULT 0,
                    exit_date TEXT DEFAULT '',
                    exit_time TEXT DEFAULT '',

                    risk_per_share REAL DEFAULT 0,
                    max_loss REAL DEFAULT 0,
                    total_cost REAL DEFAULT 0,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id INTEGER,
                    alert_type TEXT NOT NULL,  -- ENTRY, BREAKEVEN, T1, TRAILING, EXIT, TIME_STOP
                    message TEXT NOT NULL,
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    telegram_sent BOOLEAN DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_status
                ON live_positions(symbol, status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON live_positions(status)
            """)
            conn.commit()

    def add_position(
        self,
        symbol: str,
        qty: int,
        entry_price: float,
        stop_price: float,
        targets: List[float],
        strategy: str = "",
        regime: str = "",
        conviction: int = 0,
        product_type: str = "CNC"
    ) -> int:
        """Add a new live position."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO live_positions (
                        symbol, qty, entry_price, stop_price, targets,
                        entry_date, entry_time, product_type, strategy,
                        regime, conviction
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, qty, entry_price, stop_price,
                    json.dumps(targets),
                    date.today().isoformat(),
                    datetime.now().strftime("%H:%M:%S"),
                    product_type, strategy, regime, conviction
                ))
                conn.commit()
                return cursor.lastrowid

    def get_open_positions(self) -> List[LivePosition]:
        """Get all open positions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM live_positions WHERE status IN ('OPEN', 'PARTIAL')
                ORDER BY entry_date DESC
            """)
            rows = cursor.fetchall()

        positions = []
        for row in rows:
            pos = LivePosition(**dict(row))
            pos.targets = json.loads(pos.targets) if isinstance(pos.targets, str) else pos.targets
            positions.append(pos)
        return positions

    def get_position(self, position_id: int) -> Optional[LivePosition]:
        """Get a single position by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM live_positions WHERE id = ?", (position_id,)
            ).fetchone()

        if row:
            pos = LivePosition(**dict(row))
            pos.targets = json.loads(pos.targets) if isinstance(pos.targets, str) else pos.targets
            return pos
        return None

    def update_position(self, pos: LivePosition):
        """Update position in database."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE live_positions SET
                        qty = ?, entry_price = ?, stop_price = ?, targets = ?,
                        t1_booked = ?, t2_booked = ?, breakeven_moved = ?,
                        trailing_stop = ?, highest_since_entry = ?,
                        current_price = ?, unrealized_pnl = ?, realized_pnl = ?,
                        status = ?, exit_reason = ?, exit_price = ?,
                        exit_date = ?, exit_time = ?,
                        unrealized_pnl = ?, realized_pnl = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    pos.qty, pos.entry_price, pos.stop_price, json.dumps(pos.targets),
                    pos.t1_booked, pos.t2_booked, pos.breakeven_moved,
                    pos.trailing_stop, pos.highest_since_entry,
                    pos.current_price, pos.unrealized_pnl, pos.realized_pnl,
                    pos.status, pos.exit_reason, pos.exit_price,
                    pos.exit_date, pos.exit_time,
                    pos.unrealized_pnl, pos.realized_pnl,
                    pos.id
                ))
                conn.commit()

    def log_alert(self, position_id: int, alert_type: str, message: str, telegram_sent: bool = False):
        """Log alert to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO alert_log (position_id, alert_type, message, telegram_sent)
                VALUES (?, ?, ?, ?)
            """, (position_id, alert_type, message, telegram_sent))
            conn.commit()

    def send_telegram(self, message: str) -> bool:
        """Send Telegram message if sender configured."""
        if self.telegram_sender:
            try:
                self.telegram_sender(message)
                return True
            except Exception as e:
                logger.error(f"Telegram send failed: {e}")
        return False

    def check_exits(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Check all open positions for exit conditions.
        Returns list of alert messages to send.
        """
        alerts = []
        positions = self.get_open_positions()

        if not positions:
            return alerts

        for pos in positions:
            price = current_prices.get(pos.symbol)
            if price is None:
                continue

            pos.current_price = price
            pos.unrealized_pnl = (price - pos.entry_price) * pos.qty

            # Update highest since entry
            if price > pos.highest_since_entry:
                pos.highest_since_entry = price

            # Calculate trailing stop
            if pos.trailing_stop == 0:
                pos.trailing_stop = pos.highest_since_entry - pos.risk_per_share

            # Check exit conditions
            exit_triggered = False
            exit_reason = ""
            exit_price = price

            # 1. Stop-Loss hit
            if price <= pos.stop_price:
                exit_triggered = True
                exit_reason = "STOP_LOSS"
                exit_price = pos.stop_price

            # 2. Trailing stop hit (after T1 or breakeven)
            elif pos.trailing_stop > 0 and price <= pos.trailing_stop:
                exit_triggered = True
                exit_reason = "TRAILING_STOP"
                exit_price = pos.trailing_stop

            # 3. Time-stop (7+ days flat)
            elif (datetime.now().date() - datetime.fromisoformat(pos.entry_date).date()).days >= self.time_stop_days:
                if price < pos.entry_price * 1.02:  # Less than 2% gain
                    exit_triggered = True
                    exit_reason = "TIME_STOP_7D_FLAT"
                    exit_price = price

            # 4. Force close (3:15 PM)
            now = datetime.now()
            if now.hour + now.minute / 60 >= self.force_close_hour:
                exit_triggered = True
                exit_reason = "FORCE_CLOSE_EOD"
                exit_price = price

            # Generate exit alerts
            if exit_triggered:
                self._close_position(pos, exit_price, exit_reason)
                alerts.append(self._format_exit_alert(pos, exit_reason, exit_price))
                continue

            # Check profit-taking alerts
            alerts.extend(self._check_profit_alerts(pos, price))

        return alerts

    def _check_profit_alerts(self, pos: LivePosition, price: float) -> List[str]:
        """Check for breakeven, T1, T2 alerts."""
        alerts = []

        # 1. Breakeven shield (+1R)
        breakeven_price = pos.entry_price + pos.risk_per_share
        if not pos.breakeven_moved and price >= breakeven_price:
            pos.breakeven_moved = True
            pos.stop_price = pos.entry_price  # Move to breakeven
            self.update_position(pos)
            alerts.append(
                f"🔒 {pos.symbol}: Move SL to BREAKEVEN (₹{pos.entry_price:.1f}) | +1R hit"
            )
            self.log_alert(pos.id, "BREAKEVEN", alerts[-1])

        # 2. Target 1 hit (50% book)
        if not pos.t1_booked and pos.targets and price >= pos.targets[0]:
            pos.t1_booked = True
            # Move stop to breakeven if not already
            if not pos.breakeven_moved:
                pos.stop_price = pos.entry_price
                pos.breakeven_moved = True
            self.update_position(pos)
            alerts.append(
                f"💰 {pos.symbol}: T1 HIT (₹{pos.targets[0]:.1f}) | Book 50% | Trail remaining with SL @ ₹{pos.stop_price:.1f}"
            )
            self.log_alert(pos.id, "T1", alerts[-1])

        # 3. Target 2 hit (trail remaining)
        if not pos.t2_booked and len(pos.targets) > 1 and price >= pos.targets[1]:
            pos.t2_booked = True
            self.update_position(pos)
            alerts.append(
                f"🏁 {pos.symbol}: T2 HIT (₹{pos.targets[1]:.1f}) | Trail remaining with Chandelier"
            )
            self.log_alert(pos.id, "T2", alerts[-1])

        return alerts

    def _close_position(self, pos: LivePosition, exit_price: float, reason: str):
        """Close position and record realized P&L."""
        pos.status = "CLOSED"
        pos.exit_price = exit_price
        pos.exit_reason = reason
        pos.exit_date = date.today().isoformat()
        pos.exit_time = datetime.now().strftime("%H:%M:%S")
        pos.realized_pnl = (exit_price - pos.entry_price) * pos.qty
        pos.unrealized_pnl = 0
        self.update_position(pos)

    def _format_exit_alert(self, pos: LivePosition, reason: str, exit_price: float) -> str:
        """Format exit alert message."""
        icons = {
            "STOP_LOSS": "🛑",
            "TRAILING_STOP": "🏁",
            "TIME_STOP_7D_FLAT": "⏰",
            "FORCE_CLOSE_EOD": "🔔",
            "TARGET_2": "🎯"
        }
        icon = icons.get(reason, "📉")

        pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100
        total_pnl = (exit_price - pos.entry_price) * pos.qty

        return (
            f"{icon} {pos.symbol}: {reason.replace('_', ' ').title()} @ ₹{exit_price:.1f}\n"
            f"   P&L: {pnl_pct:+.1f}% (₹{total_pnl:+,.0f}) | Exit: {pos.exit_time}\n"
            f"   Reason: {reason.replace('_', ' ')}"
        )

    def check_portfolio_heat(self) -> Tuple[bool, str]:
        """Check if total portfolio risk exceeds max."""
        positions = self.get_open_positions()
        total_risk = sum(pos.max_loss for pos in positions)
        total_risk_pct = total_risk / self.capital if self.capital > 0 else 0

        if total_risk_pct >= self.max_portfolio_risk:
            return False, f"Portfolio heat {total_risk_pct:.1%} ≥ {self.max_portfolio_risk:.0%}. Pausing new buys."

        return True, None

    def get_sector_counts(self) -> Dict[str, int]:
        """Get count of open positions per sector."""
        positions = self.get_open_positions()
        # This would need sector mapping - placeholder for now
        return {}

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary for dashboard."""
        positions = self.get_open_positions()
        total_unrealized = sum(p.unrealized_pnl for p in positions)
        total_realized = sum(p.realized_pnl for p in positions)
        total_invested = sum(p.total_cost for p in positions)

        return {
            'open_positions': len(positions),
            'total_invested': total_invested,
            'total_unrealized': total_unrealized,
            'total_realized': total_realized,
            'total_pnl': total_unrealized + total_realized,
            'portfolio_heat': sum(p.max_loss for p in positions) / self.capital if self.capital > 0 else 0,
            'cash_available': self.capital - total_invested
        }

    def close_position_manually(self, position_id: int, exit_price: float) -> bool:
        """Manually close a position."""
        pos = self.get_position(position_id)
        if not pos or pos.status != 'OPEN':
            return False

        self._close_position(pos, exit_price, "MANUAL")
        alert = self._format_exit_alert(pos, "MANUAL", exit_price)
        self.send_telegram(alert)
        return True


def create_position_tracker(capital: float, config: Dict, telegram_sender=None) -> PositionTracker:
    """Factory function to create PositionTracker."""
    return PositionTracker(capital, config, telegram_sender)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test
    config = {
        'capital': {
            'initial': 500000,
            'max_risk_per_trade': 0.015,
            'max_portfolio_risk': 0.06,
            'max_position_pct': 0.05,
            'max_positions': 10
        },
        'targets': {
            'force_close_hour': 15.15,
            'max_hold_days_default': 20,
            'trail_method': 'chandelier',
            'chandelier_mult': 3.0
        }
    }

    tracker = create_position_tracker(500000, config)

    # Add test position
    pid = tracker.add_position(
        symbol="BEL",
        qty=12,
        entry_price=312,
        stop_price=296,
        targets=[336, 352],
        strategy="trend_pullback",
        regime="bull",
        conviction=88
    )
    print(f"Added position ID: {pid}")

    # Check exits with test prices
    prices = {"BEL": 325}  # +1R hit
    alerts = tracker.check_exits(prices)
    print(f"Alerts: {alerts}")

    # Check portfolio summary
    print(f"Portfolio: {tracker.get_portfolio_summary()}")