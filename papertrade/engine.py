"""
Paper Trading Engine - Simulates trades with realistic fills, risk management,
and portfolio tracking. Error-proof with state persistence.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
import logging
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"          # Stop Loss
    SLM = "SLM"        # Stop Loss Market


@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float = 0.0          # Limit price
    trigger_price: float = 0.0  # For SL/SLM
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    strategy: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'trigger_price': self.trigger_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'timestamp': self.timestamp.isoformat(),
            'strategy': self.strategy,
            'metadata': self.metadata
        }


@dataclass
class Position:
    """Position tracking."""
    symbol: str
    quantity: int
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    stop_price: float = 0.0
    targets: List[float] = field(default_factory=list)
    strategy: str = ""
    # Options Greeks (for options positions)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price

    def update_price(self, price: float):
        self.current_price = price
        self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_entry_price': self.avg_entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'entry_time': self.entry_time.isoformat(),
            'stop_price': self.stop_price,
            'targets': self.targets,
            'strategy': self.strategy,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega
        }


@dataclass
class Portfolio:
    """Portfolio state."""
    cash: float
    initial_cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: Dict[str, Order] = field(default_factory=dict)
    daily_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    peak_value: float = 0.0
    max_drawdown: float = 0.0
    trades_today: int = 0
    last_reset_date: datetime = field(default_factory=datetime.now)

    @property
    def total_value(self) -> float:
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def drawdown_pct(self) -> float:
        if self.peak_value == 0:
            return 0.0
        return (self.peak_value - self.total_value) / self.peak_value

    def update_peak(self):
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value

    def to_dict(self) -> Dict:
        return {
            'cash': self.cash,
            'initial_cash': self.initial_cash,
            'total_value': self.total_value,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'total_realized_pnl': self.total_realized_pnl,
            'drawdown_pct': self.drawdown_pct,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'orders': {k: v.to_dict() for k, v in self.orders.items()}
        }


class FillModel:
    """Realistic fill simulation."""

    def __init__(self, config: dict):
        self.config = config
        self.costs = config.get('costs', {})
        self.slippage_equity = self.costs.get('slippage_bps_equity', 15) / 10000
        self.slippage_options = self.costs.get('slippage_bps_options', 50) / 10000

    def simulate_market_order(
        self,
        side: OrderSide,
        price: float,
        quantity: int,
        is_option: bool = False,
        volume: int = 0,
        avg_volume: int = 0,
        spread_pct: float = 0.001
    ) -> Tuple[float, int]:
        """
        Simulate market order fill.

        Returns:
            (fill_price, filled_quantity)
        """
        # Base slippage
        slippage = self.slippage_options if is_option else self.slippage_equity

        # Adjust for volume participation
        if avg_volume > 0 and volume > 0:
            participation = quantity / avg_volume
            if participation > 0.01:  # > 1% of avg volume
                slippage *= (1 + participation * 10)

        # Adjust for spread
        slippage += spread_pct / 2

        # Random component (±20%)
        import random
        slippage *= random.uniform(0.8, 1.2)

        if side == OrderSide.BUY:
            fill_price = price * (1 + slippage)
        else:
            fill_price = price * (1 - slippage)

        # Assume full fill for liquid, partial for illiquid
        if is_option and volume < 100:
            fill_qty = int(quantity * random.uniform(0.5, 1.0))
        else:
            fill_qty = quantity

        return round(fill_price, 2), fill_qty

    def simulate_limit_order(
        self,
        side: OrderSide,
        limit_price: float,
        high: float,
        low: float,
        quantity: int
    ) -> Tuple[Optional[float], int]:
        """
        Simulate limit order fill based on day's high/low.

        Returns:
            (fill_price or None, filled_quantity)
        """
        if side == OrderSide.BUY and low <= limit_price:
            # Filled at min(limit, open) typically
            fill_price = min(limit_price, high)  # Simplified
            return fill_price, quantity
        elif side == OrderSide.SELL and high >= limit_price:
            fill_price = max(limit_price, low)
            return fill_price, quantity
        return None, 0


class RiskGuards:
    """Hard risk enforcement - prevents catastrophic losses."""

    def __init__(self, config: dict):
        self.config = config
        self.capital = config.get('capital', {})
        self.drawdown_circuits = config.get('drawdown_circuits', {})
        self.greeks_limits = config.get('options', {}).get('greeks_limits', {})

        # State
        self.daily_start_value = 0.0
        self.daily_pnl = 0.0
        self.total_dd_triggered = False

    def check_pre_trade(
        self,
        signal: Any,
        portfolio: Portfolio,
        current_prices: Dict[str, float]
    ) -> Tuple[bool, str, Optional[int]]:
        """
        Check if trade should be allowed.

        Returns:
            (allowed, reason, adjusted_quantity)
        """
        # 1. Daily loss circuit
        daily_loss_limit = self.drawdown_circuits.get('daily_loss_limit', 0.03)
        daily_stop_limit = self.drawdown_circuits.get('daily_stop_limit', 0.05)

        if self.daily_start_value > 0:
            daily_dd = (self.daily_start_value - portfolio.total_value) / self.daily_start_value
            if daily_dd >= daily_stop_limit:
                return False, f"Daily stop limit hit: {daily_dd:.2%}", 0
            elif daily_dd >= daily_loss_limit:
                # Halve position size
                return True, f"Daily loss limit: halving size", signal.position_size // 2

        # 2. Total drawdown circuit
        total_dd_warn = self.drawdown_circuits.get('total_dd_warn', 0.08)
        total_dd_reduce = self.drawdown_circuits.get('total_dd_reduce', 0.12)
        total_dd_stop = self.drawdown_circuits.get('total_dd_stop', 0.15)

        if portfolio.drawdown_pct >= total_dd_stop:
            return False, f"Total DD stop: {portfolio.drawdown_pct:.2%}", 0
        elif portfolio.drawdown_pct >= total_dd_reduce:
            return True, f"Total DD reduce: halving size", signal.position_size // 2
        elif portfolio.drawdown_pct >= total_dd_warn:
            return True, f"Total DD warning", signal.position_size

        # 3. Position size limits
        max_pos_pct = self.capital.get('max_position_pct', 0.05)
        position_value = signal.position_size * signal.entry_price
        if position_value > portfolio.total_value * max_pos_pct:
            adjusted = int(portfolio.total_value * max_pos_pct / signal.entry_price)
            return True, f"Position size cap: {adjusted}", adjusted

        # 4. Portfolio risk limit
        max_portfolio_risk = self.capital.get('max_portfolio_risk', 0.06)
        current_risk = sum(
            abs(p.stop_price - p.current_price) * p.quantity
            for p in portfolio.positions.values()
            if p.stop_price > 0
        )
        new_risk = abs(signal.entry_price - signal.stop_price) * signal.position_size
        if current_risk + new_risk > portfolio.total_value * max_portfolio_risk:
            return False, "Portfolio risk limit exceeded", 0

        # 5. Sector exposure (simplified - would need sector mapping)
        max_sector_pct = self.capital.get('max_sector_pct', 0.25)
        # Skip for now

        # 6. Max positions
        max_positions = self.capital.get('max_positions', 10)
        if len(portfolio.positions) >= max_positions:
            return False, f"Max positions ({max_positions}) reached", 0

        # 7. Cash reserve
        cash_reserve = self.capital.get('cash_reserve_pct', 0.10)
        if portfolio.cash - position_value < portfolio.total_value * cash_reserve:
            return False, "Insufficient cash reserve", 0

        return True, "OK", signal.position_size

    def check_intraday(self, portfolio: Portfolio) -> Tuple[bool, str]:
        """Check intraday risk limits."""
        daily_loss_limit = self.drawdown_circuits.get('daily_loss_limit', 0.03)
        daily_stop_limit = self.drawdown_circuits.get('daily_stop_limit', 0.05)

        if self.daily_start_value > 0:
            daily_dd = (self.daily_start_value - portfolio.total_value) / self.daily_start_value
            if daily_dd >= daily_stop_limit:
                return False, f"DAILY STOP: {daily_dd:.2%} - LIQUIDATE ALL"
            elif daily_dd >= daily_loss_limit:
                return True, f"DAILY WARNING: {daily_dd:.2%} - Reduce exposure"

        return True, "OK"

    def check_options_greeks(self, portfolio: Portfolio) -> Tuple[bool, str]:
        """Check portfolio Greeks limits."""
        max_delta = self.greeks_limits.get('max_portfolio_delta', 0.3)
        max_vega = self.greeks_limits.get('max_portfolio_vega', 50000)
        min_theta = self.greeks_limits.get('min_portfolio_theta', 0)

        net_delta = sum(p.delta * p.quantity for p in portfolio.positions.values())
        net_vega = sum(p.vega * p.quantity for p in portfolio.positions.values())
        net_theta = sum(p.theta * p.quantity for p in portfolio.positions.values())

        if abs(net_delta) > max_delta:
            return False, f"Net Delta {net_delta:.2f} > {max_delta}"
        if net_vega > max_vega:
            return False, f"Net Vega ₹{net_vega:,.0f} > ₹{max_vega:,.0f}"
        if net_theta < min_theta:
            return False, f"Net Theta {net_theta:.2f} < {min_theta} (not net seller)"

        return True, "OK"

    def force_close_all(self, portfolio: Portfolio, reason: str) -> List[Order]:
        """Generate orders to close all positions."""
        orders = []
        for symbol, position in portfolio.positions.items():
            if position.quantity > 0:
                order = Order(
                    order_id=str(uuid.uuid4())[:8],
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity,
                    strategy="RISK_GUARD",
                    metadata={'reason': reason}
                )
                orders.append(order)
        return orders

    def reset_daily(self, portfolio_value: float):
        """Reset daily tracking at market open."""
        self.daily_start_value = portfolio_value
        self.daily_pnl = 0.0


class PaperTradingEngine:
    """
    Main paper trading engine.
    Handles signal processing, order management, portfolio tracking.
    """

    def __init__(self, config: dict, data_fetcher=None):
        self.config = config
        self.data_fetcher = data_fetcher
        self.fill_model = FillModel(config)
        self.risk_guards = RiskGuards(config)

        # Portfolio
        initial_cash = config.get('capital', {}).get('initial', 500000)
        self.portfolio = Portfolio(cash=initial_cash, initial_cash=initial_cash)

        # State persistence
        self.state_file = Path(config.get('paper_trading', {}).get('state_file', './logs/paper_state.db'))
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # Trade log
        self.trade_log_file = Path(config.get('paper_trading', {}).get('log_file', './logs/paper_trades.csv'))
        self.trade_log_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_trade_log()

        # Strategies (injected)
        self.strategies = {}
        self.options_strategies = {}

        # Performance tracking
        self.daily_returns = []
        self.equity_curve = []

    def _init_db(self):
        """Initialize SQLite database for state persistence."""
        with sqlite3.connect(self.state_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity INTEGER,
                    avg_entry_price REAL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    entry_time TEXT,
                    stop_price REAL,
                    targets TEXT,
                    strategy TEXT,
                    delta REAL,
                    gamma REAL,
                    theta REAL,
                    vega REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    order_type TEXT,
                    quantity INTEGER,
                    price REAL,
                    trigger_price REAL,
                    status TEXT,
                    filled_quantity INTEGER,
                    avg_fill_price REAL,
                    timestamp TEXT,
                    strategy TEXT,
                    metadata TEXT
                )
            """)

    def _init_trade_log(self):
        """Initialize CSV trade log."""
        if not self.trade_log_file.exists():
            import csv
            with open(self.trade_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'price',
                    'strategy', 'order_id', 'pnl', 'commission', 'slippage'
                ])

    def _save_state(self):
        """Persist portfolio state to SQLite."""
        try:
            with sqlite3.connect(self.state_file) as conn:
                # Portfolio
                conn.execute(
                    "INSERT OR REPLACE INTO portfolio_state (key, value) VALUES (?, ?)",
                    ('portfolio', json.dumps(self.portfolio.to_dict()))
                )

                # Positions
                for symbol, pos in self.portfolio.positions.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO positions
                        (symbol, quantity, avg_entry_price, current_price, unrealized_pnl,
                         realized_pnl, entry_time, stop_price, targets, strategy,
                         delta, gamma, theta, vega)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, pos.quantity, pos.avg_entry_price, pos.current_price,
                        pos.unrealized_pnl, pos.realized_pnl, pos.entry_time.isoformat(),
                        pos.stop_price, json.dumps(pos.targets), pos.strategy,
                        pos.delta, pos.gamma, pos.theta, pos.vega
                    ))

                # Orders
                for order_id, order in self.portfolio.orders.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO orders
                        (order_id, symbol, side, order_type, quantity, price, trigger_price,
                         status, filled_quantity, avg_fill_price, timestamp, strategy, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        order_id, order.symbol, order.side.value, order.order_type.value,
                        order.quantity, order.price, order.trigger_price, order.status.value,
                        order.filled_quantity, order.avg_fill_price, order.timestamp.isoformat(),
                        order.strategy, json.dumps(order.metadata)
                    ))

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self):
        """Load portfolio state from SQLite."""
        try:
            with sqlite3.connect(self.state_file) as conn:
                # Load portfolio
                row = conn.execute("SELECT value FROM portfolio_state WHERE key = 'portfolio'").fetchone()
                if row:
                    data = json.loads(row[0])
                    self.portfolio.cash = data['cash']
                    self.portfolio.initial_cash = data['initial_cash']
                    self.portfolio.total_realized_pnl = data['total_realized_pnl']
                    self.portfolio.peak_value = data.get('total_value', self.portfolio.initial_cash)

                # Load positions
                rows = conn.execute("SELECT * FROM positions").fetchall()
                for row in rows:
                    pos = Position(
                        symbol=row[0],
                        quantity=row[1],
                        avg_entry_price=row[2],
                        current_price=row[3],
                        unrealized_pnl=row[4],
                        realized_pnl=row[5],
                        entry_time=datetime.fromisoformat(row[6]),
                        stop_price=row[7],
                        targets=json.loads(row[8]),
                        strategy=row[9],
                        delta=row[10],
                        gamma=row[11],
                        theta=row[12],
                        vega=row[13]
                    )
                    self.portfolio.positions[row[0]] = pos

                # Load orders
                rows = conn.execute("SELECT * FROM orders").fetchall()
                for row in rows:
                    order = Order(
                        order_id=row[0],
                        symbol=row[1],
                        side=OrderSide(row[2]),
                        order_type=OrderType(row[3]),
                        quantity=row[4],
                        price=row[5],
                        trigger_price=row[6],
                        status=OrderStatus(row[7]),
                        filled_quantity=row[8],
                        avg_fill_price=row[9],
                        timestamp=datetime.fromisoformat(row[10]),
                        strategy=row[11],
                        metadata=json.loads(row[12])
                    )
                    self.portfolio.orders[row[0]] = order

            logger.info(f"Loaded state: {len(self.portfolio.positions)} positions, {len(self.portfolio.orders)} orders")

        except Exception as e:
            logger.warning(f"Could not load state: {e}")

    def _log_trade(self, order: Order, fill_price: float, commission: float, slippage: float):
        """Log filled trade to CSV."""
        import csv
        pnl = 0.0
        if order.side == OrderSide.SELL and order.symbol in self.portfolio.positions:
            pos = self.portfolio.positions[order.symbol]
            pnl = (fill_price - pos.avg_entry_price) * order.filled_quantity

        with open(self.trade_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                order.symbol,
                order.side.value,
                order.filled_quantity,
                fill_price,
                order.strategy,
                order.order_id,
                pnl,
                commission,
                slippage
            ])

    def process_signals(self, signals: List, current_prices: Dict[str, float]) -> List[Order]:
        """
        Process entry signals through risk guards and create orders.

        Args:
            signals: List of Signal objects from strategies
            current_prices: Current market prices {symbol: price}

        Returns:
            List of created orders
        """
        orders = []

        for signal in signals:
            symbol = signal.symbol
            if symbol not in current_prices:
                logger.warning(f"No price for {symbol}, skipping")
                continue

            # Risk guard check
            allowed, reason, adj_qty = self.risk_guards.check_pre_trade(
                signal, self.portfolio, current_prices
            )

            if not allowed:
                logger.info(f"Risk guard rejected {symbol}: {reason}")
                continue

            if adj_qty != signal.position_size:
                logger.info(f"Risk guard adjusted {symbol} qty: {signal.position_size} -> {adj_qty}")
                signal.position_size = adj_qty

            # Create market order
            order = Order(
                order_id=str(uuid.uuid4())[:8],
                symbol=symbol,
                side=OrderSide.BUY if signal.side == 'long' else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=signal.position_size,
                strategy=signal.strategy,
                metadata={
                    'stop_price': signal.stop_price,
                    'targets': signal.targets,
                    'risk_pct': signal.risk_pct,
                    'r_targets': signal.r_multiple_targets
                }
            )

            # Simulate fill
            price = current_prices[symbol]
            is_option = 'option' in signal.strategy.lower() if hasattr(signal, 'strategy') else False

            fill_price, fill_qty = self.fill_model.simulate_market_order(
                order.side, price, order.quantity, is_option
            )

            if fill_qty == 0:
                logger.warning(f"Fill failed for {symbol}")
                continue

            # Execute fill
            order.status = OrderStatus.FILLED
            order.filled_quantity = fill_qty
            order.avg_fill_price = fill_price

            # Update portfolio
            self._execute_fill(order, fill_price, fill_qty)

            # Log trade
            commission = self._calculate_commission(fill_price, fill_qty, is_option)
            slippage = abs(fill_price - price) / price
            self._log_trade(order, fill_price, commission, slippage)

            # Save state
            self._save_state()

            orders.append(order)
            logger.info(f"Filled {order.side.value} {fill_qty} {symbol} @ {fill_price:.2f} [{order.strategy}]")

        return orders

    def _execute_fill(self, order: Order, fill_price: float, fill_qty: int):
        """Update portfolio after fill."""
        symbol = order.symbol
        commission = self._calculate_commission(fill_price, fill_qty, False)

        if order.side == OrderSide.BUY:
            # New or add to position
            cost = fill_price * fill_qty + commission
            if cost > self.portfolio.cash:
                logger.warning(f"Insufficient cash for {symbol}")
                return

            self.portfolio.cash -= cost

            if symbol in self.portfolio.positions:
                pos = self.portfolio.positions[symbol]
                # Average up
                total_qty = pos.quantity + fill_qty
                pos.avg_entry_price = (pos.avg_entry_price * pos.quantity + fill_price * fill_qty) / total_qty
                pos.quantity = total_qty
            else:
                # New position
                pos = Position(
                    symbol=symbol,
                    quantity=fill_qty,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    entry_time=datetime.now(),
                    stop_price=order.metadata.get('stop_price', 0),
                    targets=order.metadata.get('targets', []),
                    strategy=order.strategy
                )
                self.portfolio.positions[symbol] = pos

        else:  # SELL
            if symbol not in self.portfolio.positions:
                logger.warning(f"No position to sell for {symbol}")
                return

            pos = self.portfolio.positions[symbol]
            sell_qty = min(fill_qty, pos.quantity)
            proceeds = fill_price * sell_qty - commission

            self.portfolio.cash += proceeds
            pos.quantity -= sell_qty
            pos.realized_pnl += (fill_price - pos.avg_entry_price) * sell_qty
            self.portfolio.total_realized_pnl += (fill_price - pos.avg_entry_price) * sell_qty

            if pos.quantity == 0:
                del self.portfolio.positions[symbol]

    def _calculate_commission(self, price: float, qty: int, is_option: bool) -> float:
        """Calculate commission per Indian cost structure."""
        costs = self.config.get('costs', {})
        turnover = price * qty

        brokerage = costs.get('brokerage_per_order', 20)
        stt = turnover * (costs.get('stt_options_sell', 0.000625) if is_option else costs.get('stt_equity_delivery', 0.001))
        exchange = turnover * costs.get('exchange_charges', 0.0000345)
        sebi = turnover * costs.get('sebi_charges', 0.000001)
        stamp = turnover * costs.get('stamp_duty', 0.00015)

        return brokerage + stt + exchange + sebi + stamp

    def update_positions(self, current_prices: Dict[str, float]):
        """Update all positions with current prices."""
        for symbol, position in self.portfolio.positions.items():
            if symbol in current_prices:
                position.update_price(current_prices[symbol])

        # Update portfolio metrics
        self.portfolio.update_peak()
        self.risk_guards.check_options_greeks(self.portfolio)

    def check_stop_targets(self, current_prices: Dict[str, float]) -> List[Order]:
        """Check and execute stop loss / target orders."""
        orders = []

        for symbol, position in list(self.portfolio.positions.items()):
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]

            # Check stop loss
            if position.stop_price > 0 and price <= position.stop_price:
                order = Order(
                    order_id=str(uuid.uuid4())[:8],
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity,
                    strategy=f"{position.strategy}_SL",
                    metadata={'exit_reason': 'stop_loss'}
                )
                fill_price, fill_qty = self.fill_model.simulate_market_order(OrderSide.SELL, price, position.quantity)
                order.status = OrderStatus.FILLED
                order.filled_quantity = fill_qty
                order.avg_fill_price = fill_price
                self._execute_fill(order, fill_price, fill_qty)
                self._log_trade(order, fill_price,
                    self._calculate_commission(fill_price, fill_qty, False),
                    abs(fill_price - price) / price)
                orders.append(order)
                logger.info(f"STOP LOSS hit for {symbol} @ {fill_price:.2f}")

            # Check targets (partial exits)
            elif position.targets:
                for i, target in enumerate(position.targets):
                    if price >= target:
                        # Exit portion (e.g., 50% at T1, 30% at T2, 20% at T3)
                        exit_pct = [0.5, 0.3, 0.2][i] if i < 3 else 0.2
                        exit_qty = int(position.quantity * exit_pct)
                        if exit_qty > 0:
                            order = Order(
                                order_id=str(uuid.uuid4())[:8],
                                symbol=symbol,
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET,
                                quantity=exit_qty,
                                strategy=f"{position.strategy}_T{i+1}",
                                metadata={'exit_reason': f'target_{i+1}'}
                            )
                            fill_price, fill_qty = self.fill_model.simulate_market_order(OrderSide.SELL, price, exit_qty)
                            order.status = OrderStatus.FILLED
                            order.filled_quantity = fill_qty
                            order.avg_fill_price = fill_price
                            self._execute_fill(order, fill_price, fill_qty)
                            self._log_trade(order, fill_price,
                                self._calculate_commission(fill_price, fill_qty, False),
                                abs(fill_price - price) / price)
                            orders.append(order)
                            logger.info(f"TARGET {i+1} hit for {symbol} @ {fill_price:.2f}")
                            break  # Only one target per bar

        return orders

    def check_time_exits(self, current_time: datetime) -> List[Order]:
        """Check time-based exits (EOD, max hold days)."""
        orders = []
        force_hour = self.config.get('targets', {}).get('force_close_hour', 15.15)

        # EOD exit
        if current_time.hour + current_time.minute / 60 >= force_hour:
            for symbol, position in list(self.portfolio.positions.items()):
                # Only exit if configured for intraday
                # For swing, skip EOD exit
                pass

        # Max hold days
        max_days = self.config.get('targets', {}).get('max_hold_days_default', 20)
        for symbol, position in list(self.portfolio.positions.items()):
            if (current_time - position.entry_time).days >= max_days:
                order = Order(
                    order_id=str(uuid.uuid4())[:8],
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity,
                    strategy=f"{position.strategy}_TIME",
                    metadata={'exit_reason': 'max_hold_days'}
                )
                # Would need current price - simplified
                orders.append(order)

        return orders

    def daily_report(self) -> Dict:
        """Generate daily performance report."""
        self.portfolio.update_peak()

        return {
            'date': datetime.now().date().isoformat(),
            'total_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'positions_value': self.portfolio.total_value - self.portfolio.cash,
            'daily_pnl': self.portfolio.total_value - self.portfolio.initial_cash - self.portfolio.total_realized_pnl,
            'total_realized_pnl': self.portfolio.total_realized_pnl,
            'total_unrealized_pnl': self.portfolio.total_unrealized_pnl,
            'drawdown_pct': self.portfolio.drawdown_pct,
            'num_positions': len(self.portfolio.positions),
            'positions': {k: v.to_dict() for k, v in self.portfolio.positions.items()},
            'returns_pct': (self.portfolio.total_value / self.portfolio.initial_cash - 1) * 100
        }

    def run_daily_cycle(
        self,
        swing_signals: List,
        options_signals: List,
        current_prices: Dict[str, float],
        current_time: datetime = None
    ) -> Dict:
        """
        Run complete daily paper trading cycle.

        Returns:
            Daily report dict
        """
        if current_time is None:
            current_time = datetime.now()

        # Reset daily tracking at market open (9:15)
        if current_time.hour == 9 and current_time.minute <= 20:
            self.risk_guards.reset_daily(self.portfolio.total_value)

        # 1. Update positions with current prices
        self.update_positions(current_prices)

        # 2. Check stops and targets
        self.check_stop_targets(current_prices)

        # 3. Process new swing signals
        if swing_signals:
            self.process_signals(swing_signals, current_prices)

        # 4. Process options signals (simplified - would need option chains)
        # For now, skip options execution in paper trading

        # 5. Check time exits
        self.check_time_exits(current_time)

        # 6. Generate report
        report = self.daily_report()

        # 7. Save state
        self._save_state()

        return report

    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics from trade log."""
        if not self.trade_log_file.exists():
            return {}

        try:
            trades = pd.read_csv(self.trade_log_file)
            if trades.empty:
                return {}

            # Filter filled trades
            trades = trades[trades['side'] == 'SELL']  # Only closed trades

            if trades.empty:
                return {}

            wins = trades[trades['pnl'] > 0]
            losses = trades[trades['pnl'] < 0]

            return {
                'total_trades': len(trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(trades) if len(trades) > 0 else 0,
                'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
                'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
                'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else float('inf'),
                'total_pnl': trades['pnl'].sum(),
                'expectancy': (len(wins)/len(trades) * wins['pnl'].mean()) + (len(losses)/len(trades) * losses['pnl'].mean()) if len(trades) > 0 else 0
            }

        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return {}


if __name__ == "__main__":
    import yaml
    from data.fetcher import create_fetcher

    logging.basicConfig(level=logging.INFO)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create engine
    fetcher = create_fetcher()
    engine = PaperTradingEngine(config, fetcher)

    # Load any existing state
    engine._load_state()

    # Test with sample signals
    from strategies.swing import Signal, run_all_strategies
    from indicators.technical import add_all_indicators, ensure_multiindex

    symbols = ["RELIANCE", "TCS", "HDFCBANK"]
    dfs = []
    for sym in symbols:
        df = fetcher.get_price_data(sym, period="3mo")
        if not df.empty:
            dfs.append(df)

    if dfs:
        combined = pd.concat(dfs)
        combined = ensure_multiindex(combined)
        combined = add_all_indicators(combined)

        # Run strategies
        signals = run_all_strategies(combined, config)

        # Current prices
        prices = {s.symbol: s.entry_price for s in signals}

        # Run daily cycle
        report = engine.run_daily_cycle(signals, [], prices)

        print("\n=== DAILY REPORT ===")
        for k, v in report.items():
            if k != 'positions':
                print(f"  {k}: {v}")

        print(f"\nPositions: {len(report['positions'])}")
        for sym, pos in report['positions'].items():
            print(f"  {sym}: {pos['quantity']} @ {pos['avg_entry_price']:.2f} | PnL: {pos['unrealized_pnl']:,.0f}")

        # Performance
        perf = engine.get_performance_metrics()
        print(f"\nPerformance: {perf}")