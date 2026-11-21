"""
Direct Recommendation Engine - Institutional Call Cards.
Synthesizes all research into crisp BUY/SL/T1/T2/EXIT Telegram cards.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class DirectRecommendation:
    """Institutional-grade execution ticket."""
    symbol: str
    action: str = "BUY"
    conviction: int = 0
    regime: str = ""
    entry_min: float = 0.0
    entry_max: float = 0.0
    max_gap_price: float = 0.0
    quantity: int = 0
    total_cost: float = 0.0
    stop_loss: float = 0.0
    max_loss: float = 0.0
    targets: List[float] = field(default_factory=list)
    time_horizon_days: int = 10
    rationale_bullets: List[str] = field(default_factory=list)
    risk_pct: float = 0.0
    strategy: str = ""
    gap_guard_pct: float = 0.012  # 1.2% max gap


class DirectRecommendationEngine:
    """
    Synthesizes all research into direct institutional call cards.
    """

    def __init__(self, config: Dict, capital: float):
        self.config = config
        self.capital = capital

        cap_cfg = config.get('capital', {})
        self.max_risk_per_trade = cap_cfg.get('max_risk_per_trade', 0.015)
        self.max_portfolio_risk = cap_cfg.get('max_portfolio_risk', 0.06)
        self.max_position_pct = cap_cfg.get('max_position_pct', 0.05)
        self.max_positions = cap_cfg.get('max_positions', 10)
        self.sector_limit = 2  # max 2 per sector

        self.min_conviction = 65  # minimum score for alert
        self.min_backtest_trades = 10  # minimum trades for backtest filter
        self.gap_guard_pct = 0.012  # 1.2% max gap

        # Sector mapping for diversification
        self.sector_map = {
            'BANKING': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'FEDERALBNK', 'BANDHANBNK'],
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'PERSISTENT', 'COFORGE', 'MPHASIS'],
            'PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'TORNTPHARM', 'ALKEM', 'LUPIN', 'AUROPHARMA'],
            'AUTO': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT', 'TVSMOTOR'],
            'ENERGY': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL', 'OIL'],
            'METALS': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'SAIL', 'NMDC'],
            'CEMENT': ['ULTRACEMCO', 'SHREECEM', 'ACC', 'AMBUJACEM', 'DALMIACEM'],
            'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO', 'GODREJCP'],
            'CONSUMER': ['TITAN', 'TRENT', 'AVENUE', 'ZOMATO', 'JUBLFOOD'],
            'INFRA': ['LT', 'BEL', 'HAL', 'IRCON', 'RVNL', 'IRFC'],
            'CHEMICALS': ['PIDILITIND', 'SRF', 'NAVINFLUOR', 'ATUL', 'AARTIIND', 'DEEPAKNTR'],
            'FINANCE': ['BAJFINANCE', 'BAJAJFINSV', 'HDFCLIFE', 'SBILIFE', 'ICICIPRULI', 'MUTHOOTFIN', 'MANAPPURAM'],
        }

        # Reverse mapping for quick lookup
        self.symbol_to_sector = {}
        for sector, symbols in self.sector_map.items():
            for sym in symbols:
                self.symbol_to_sector[sym] = sector

    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self.symbol_to_sector.get(symbol.upper(), 'OTHER')

    def generate_recommendations(
        self,
        scores: List,  # List of SwingScore objects
        regime_info: Dict,
        macro_tilts: Dict,
        fii_data: Dict,
        open_positions: Dict,
        nse_calendar: Any = None
    ) -> List['DirectRecommendation']:
        """
        Generate direct recommendation cards from scored symbols.
        """

        # 1. Filter by conviction threshold
        candidates = [s for s in scores if s.total_score >= self.min_conviction and s.backtest_passed]

        logger.info(f"DirectRecommendation: {len(scores)} scored, {len(candidates)} passed conviction ≥ {self.min_conviction}")

        # 2. Apply vetoes
        candidates = self._apply_vetoes(candidates, regime_info, macro_tilts, nse_calendar)

        # 3. Apply portfolio filters
        candidates = self._apply_portfolio_filters(candidates, open_positions)

        # 4. Rank and take top 3
        candidates.sort(key=lambda x: x.total_score, reverse=True)
        top_picks = candidates[:3]

        logger.info(f"DirectRecommendation: Top {len(top_picks)} picks after filters")

        # 5. Build execution tickets
        recommendations = []
        for s in top_picks:
            rec = self._build_ticket(s, regime_info)
            recommendations.append(rec)

        return recommendations

    def _apply_vetoes(self, candidates, regime_info, macro_tilts, nse_calendar) -> List:
        """Apply hard vetoes: red-flag news, earnings, sector avoid, regime mismatch."""
        filtered = []
        for s in candidates:
            # Red-flag news veto
            if s.factor_scores and any(f.name == 'news' and f.metadata.get('red_flag', False) for f in s.factor_scores):
                logger.info(f"VETO {s.symbol}: Red-flag news")
                continue

            # Earnings within 5 days (check metadata or nse_calendar)
            if s.metadata.get('earnings_within_5d', False):
                logger.info(f"VETO {s.symbol}: Earnings within 5 days")
                continue

            if nse_calendar and nse_calendar.has_earnings_soon(s.symbol, 5):
                logger.info(f"VETO {s.symbol}: NSE calendar shows earnings soon")
                continue

            # Macro sector avoid
            if s.symbol in macro_tilts.get('avoid', set()):
                logger.info(f"VETO {s.symbol}: Macro sector avoid")
                continue

            # ASM/GSM check
            try:
                from utils.nse_calendar import is_asm_gsm
                asm_gsm = is_asm_gsm(s.symbol)
                if asm_gsm.get('restricted', False):
                    logger.info(f"VETO {s.symbol}: ASM/GSM restricted")
                    continue
            except Exception:
                pass

            # Regime mismatch - check if strategy should trade in current regime
            regime_obj = regime_info.get('regime_obj')
            if regime_obj:
                from indicators.regime import RegimeDetector
                detector = RegimeDetector({})
                should, _ = detector.should_trade_swing(regime_obj, 'trend_pullback')
                if not should:
                    logger.info(f"VETO {s.symbol}: Regime mismatch")
                    continue

            filtered.append(s)
        return filtered

    def _apply_portfolio_filters(self, candidates, open_positions) -> List:
        """Enforce sector limit and portfolio heat."""
        # Count current sectors
        sector_counts = {}
        for pos in open_positions.values():
            sector = pos.get('sector', self.get_sector(pos.get('symbol', '')))
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        filtered = []
        for s in candidates:
            sector = self.get_sector(s.symbol)
            if sector_counts.get(sector, 0) >= self.sector_limit:
                logger.info(f"FILTER {s.symbol}: Sector limit reached ({sector}={sector_counts.get(sector, 0)})")
                continue

            filtered.append(s)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        return filtered

    def _build_ticket(self, score, regime_info: Dict) -> 'DirectRecommendation':
        """Build executable institutional call card."""

        # Extract key data from score
        ema20 = score.factor_scores[0].metadata.get('ema20', 0) if score.factor_scores else 0
        chandelier = score.factor_scores[0].metadata.get('chandelier', 0) if score.factor_scores else 0
        atr = score.factor_scores[0].metadata.get('atr', 0) if score.factor_scores else 0

        # Use first factor's metadata for entry/stop estimates
        entry = ema20 if ema20 > 0 else score.factor_scores[0].metadata.get('close', 0) if score.factor_scores else 0
        stop = chandelier if chandelier > 0 else score.factor_scores[0].metadata.get('stop', 0) if score.factor_scores else entry * 0.95

        # If no stop, use ATR-based
        if stop <= 0 or stop >= entry:
            atr_val = atr if atr > 0 else entry * 0.02
            stop = entry - 2 * atr_val

        # Targets
        targets = score.metadata.get('targets', [])
        if not targets:
            risk = entry - stop
            targets = [round(entry + 1.5 * risk, 1), round(entry + 2.5 * risk, 1)]

        # Position sizing
        risk_per_share = entry - stop
        if risk_per_share <= 0:
            risk_per_share = entry * 0.03

        risk_per_trade = self.capital * self.max_risk_per_trade
        qty = max(1, int(risk_per_trade / risk_per_share))

        # Cap at max position %
        max_qty = int(self.capital * 0.05 / entry)
        qty = min(qty, max_qty)

        # Gap guard
        max_gap = entry * (1 + self.gap_guard_pct)

        # Build rationale bullets
        reasons = []
        for f in score.factor_scores:
            if f.score > 60:
                reasons.append(f"{f.name.replace('_', ' ').title()}: {f.description}")
        reasons = reasons[:2]  # Max 2 bullets

        # Time horizon
        time_horizon = 10  # default swing days

        return DirectRecommendation(
            symbol=score.symbol,
            action="BUY",
            conviction=int(score.total_score),
            regime=regime_info.get('regime', 'unknown'),
            entry_min=round(entry * 0.99, 1),
            entry_max=round(entry * 1.01, 1),
            max_gap_price=round(max_gap, 1),
            quantity=qty,
            total_cost=round(qty * entry, 0),
            stop_loss=round(stop, 1),
            max_loss=round(qty * risk_per_share, 0),
            targets=[round(t, 1) for t in targets[:2]],
            time_horizon_days=10,
            rationale_bullets=reasons[:2],
            risk_pct=round(risk_per_share / entry * 100, 1) if entry > 0 else 0,
            strategy=score.metadata.get('strategy', 'trend_pullback'),
            gap_guard_pct=self.gap_guard_pct
        )


def format_telegram_ticket(rec: DirectRecommendation, index: int, total: int) -> str:
    """Format as exact institutional Telegram call card."""

    lines = [
        f"✅ DIRECT BUY RECOMMENDATION ({index} of {total})",
        f"Action: BUY `{rec.symbol}` ({rec.symbol.replace('.NS', '')})",
        f"Conviction: {rec.conviction}/100 ({rec.regime.title()} Regime)",
        f"💵 ORDER DETAILS (Capital: ₹{rec.total_cost:,.0f}):",
        f"• Entry Zone: ₹{rec.entry_min} – ₹{rec.entry_max} (Max Gap: ₹{rec.max_gap_price})",
        f"• Quantity: {rec.quantity} Shares | Total: ₹{rec.total_cost:,.0f}",
        f"• Stop-Loss (GTT): ₹{rec.stop_loss} | Max Loss: ₹{rec.max_loss:,.0f} ({rec.risk_pct}%)",
    ]

    for i, t in enumerate(rec.targets, 1):
        pct_label = "50% Exit" if i == 1 else "Final Exit"
        lines.append(f"• Target {i} ({pct_label}): ₹{t}")

    lines.append(f"⏱ Time Horizon: {rec.time_horizon_days} trading days")
    lines.append(f"🧠 RATIONALE:")
    for r in rec.rationale_bullets:
        lines.append(f"• {r}")
    lines.append(f"• Gap Guard: Do NOT buy if 9:15 AM open > ₹{rec.max_gap_price}")

    return "\n".join(lines)


def format_zero_trade_message(regime_info: Dict) -> str:
    """Format no-trade day message."""
    regime = regime_info.get('regime', 'unknown').upper()
    vix = regime_info.get('vix_level', 0)

    reasons = []
    if regime == 'BEAR':
        reasons.append("Bear regime active")
    if regime == 'VOLATILE':
        reasons.append("High volatility (VIX > 22)")
    if regime_info.get('vix_level', 0) > 22:
        reasons.append(f"India VIX elevated at {vix:.1f}")
    if regime == 'SIDEWAYS':
        reasons.append("Sideways/choppy market")

    reason_str = "; ".join(reasons) if reasons else "No high-conviction setups"

    return f"🛡️ NO TRADE TODAY — {reason_str}\n   Regime: {regime} | VIX: {vix:.1f}\n   Capital preserved for better opportunities."


def format_no_trade_low_scores() -> str:
    return "🛡️ NO TRADE TODAY — No setups meet conviction threshold (≥65)\n   Market conditions not favorable for swing entries."


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test
    config = {
        'capital': {
            'initial': 500000,
            'max_risk_per_trade': 0.015,
            'max_position_pct': 0.05
        }
    }

    engine = DirectRecommendationEngine(config, 500000)

    # Mock score object
    class MockFactor:
        def __init__(self, name, score, metadata, desc):
            self.name = name
            self.score = score
            self.metadata = metadata
            self.description = desc

    class MockScore:
        def __init__(self):
            self.symbol = "BEL"
            self.total_score = 88
            self.backtest_passed = True
            self.metadata = {
                'targets': [336, 352],
                'strategy': 'trend_pullback'
            }
            self.factor_scores = [
                MockFactor('trend_alignment', 85, {'ema20': 312, 'chandelier': 296}, 'Strong uptrend pullback to 20 EMA with Bullish Engulfing + 1.8x volume'),
                MockFactor('momentum', 75, {}, 'RSI 55, MACD bullish'),
                MockFactor('news', 80, {'red_flag': False}, '₹1,155 Cr defence order from Navy')
            ]

    mock_score = MockScore()
    regime_info = {'regime': 'bull', 'size_mult': 1.0}
    macro_tilts = {'avoid': set(), 'favor': {'BEL'}}

    recs = engine.generate_recommendations([mock_score], regime_info, macro_tilts, {}, {})
    for i, rec in enumerate(recs, 1):
        print(format_telegram_ticket(rec, i, len(recs)))
        print("\n---\n")

    # Test zero trade
    print(format_zero_trade_message({'regime': 'bear', 'vix_level': 25}))