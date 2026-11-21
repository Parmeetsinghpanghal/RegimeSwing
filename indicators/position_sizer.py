"""
Capital-Tiered Position Sizer - Institutional-grade position sizing.
Automatically adjusts position sizes based on capital tier, stock price, and volatility.
Protects small accounts from oversizing on expensive stocks.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CapitalTier(Enum):
    """Capital tier classification."""
    MICRO = "micro"       # < 50,000
    SMALL = "small"       # 50,000 - 2,00,000
    MEDIUM = "medium"     # 2,00,000 - 5,00,000
    LARGE = "large"       # 5,00,000 - 10,00,000
    HNI = "hni"           # 10,00,000+


@dataclass
class CapitalTierConfig:
    """Configuration for a capital tier."""
    name: CapitalTier
    min_capital: float
    max_capital: float
    max_stock_price_pct: float  # Max single stock price as % of capital
    max_position_pct: float     # Max position as % of capital
    max_sector_pct: float       # Max sector exposure
    max_positions: int          # Max concurrent positions
    min_turnover_cr: float      # Minimum daily turnover (crores)
    lot_size_multiplier: float  # Multiplier for lot sizes (options)


# Default tier configurations
TIER_CONFIGS = {
    CapitalTier.MICRO: CapitalTierConfig(
        name=CapitalTier.MICRO,
        min_capital=0,
        max_capital=50_000,
        max_stock_price_pct=0.10,      # Stock price <= 10% of capital
        max_position_pct=0.08,         # 8% max position
        max_sector_pct=0.25,           # 25% max sector
        max_positions=3,
        min_turnover_cr=10,            # 10 Cr minimum daily turnover
        lot_size_multiplier=1.0
    ),
    CapitalTier.SMALL: CapitalTierConfig(
        name=CapitalTier.SMALL,
        min_capital=50_000,
        max_capital=2_00_000,
        max_stock_price_pct=0.12,
        max_position_pct=0.10,
        max_sector_pct=0.25,
        max_positions=5,
        min_turnover_cr=25,
        lot_size_multiplier=1.0
    ),
    CapitalTier.MEDIUM: CapitalTierConfig(
        name=CapitalTier.MEDIUM,
        min_capital=2_00_000,
        max_capital=5_00_000,
        max_stock_price_pct=0.15,
        max_position_pct=0.12,
        max_sector_pct=0.30,
        max_positions=8,
        min_turnover_cr=50,
        lot_size_multiplier=1.0
    ),
    CapitalTier.LARGE: CapitalTierConfig(
        name=CapitalTier.LARGE,
        min_capital=5_00_000,
        max_capital=10_00_000,
        max_stock_price_pct=0.20,
        max_position_pct=0.15,
        max_sector_pct=0.30,
        max_positions=12,
        min_turnover_cr=100,
        lot_size_multiplier=1.0
    ),
    CapitalTier.HNI: CapitalTierConfig(
        name=CapitalTier.HNI,
        min_capital=10_00_000,
        max_capital=float('inf'),
        max_stock_price_pct=0.25,
        max_position_pct=0.20,
        max_sector_pct=0.35,
        max_positions=20,
        min_turnover_cr=200,
        lot_size_multiplier=1.0
    ),
}


@dataclass
class PositionSizingResult:
    """Result of position sizing calculation."""
    symbol: str
    recommended_qty: int
    max_allowed_qty: int
    entry_price: float
    stop_price: float
    risk_per_share: float
    position_value: float
    risk_amount: float
    risk_pct_of_capital: float
    tier: CapitalTier
    warnings: List[str]
    rejected: bool = False
    rejection_reason: str = ""


class CapitalTieredPositionSizer:
    """
    Institutional-grade position sizer that adapts to capital tier.
    Automatically skips expensive stocks for small accounts.
    """

    def __init__(self, capital: float, config: dict = None):
        self.capital = capital
        self.config = config or {}
        self.tier = self._determine_tier(capital)
        self.tier_config = TIER_CONFIGS[self.tier]
        self.capital_cfg = config.get('capital', {}) if config else {}

        logger.info(f"Position sizer initialized: Capital=₹{capital:,.0f}, Tier={self.tier.value}")

    def _determine_tier(self, capital: float) -> 'CapitalTier':
        """Determine capital tier based on available capital."""
        for tier in [CapitalTier.HNI, CapitalTier.LARGE, CapitalTier.MEDIUM, CapitalTier.SMALL, CapitalTier.MICRO]:
            if capital >= TIER_CONFIGS[tier].min_capital:
                return tier
        return CapitalTier.MICRO

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        atr_pct: float = 0.02,
        atr_target_pct: float = 0.02,
        max_position_pct: float = None,
        sector: str = "UNKNOWN",
        current_sector_exposure: float = 0.0,
        portfolio_positions: int = 0
    ) -> PositionSizingResult:
        """
        Calculate position size with all tier-based constraints.

        Args:
            symbol: Stock symbol
            entry_price: Expected entry price
            stop_price: Stop loss price
            atr_pct: Current ATR% of the stock
            atr_target_pct: Target ATR% for parity sizing (default 2%)
            max_position_pct: Override max position % (defaults to tier config)
            sector: Stock sector
            current_sector_exposure: Current exposure to this sector (as fraction of capital)
            portfolio_positions: Current number of open positions

        Returns:
            PositionSizingResult with all sizing details
        """
        warnings = []

        # Use tier config if not overridden
        max_pos_pct = max_position_pct or self.tier_config.max_position_pct
        max_stock_price_pct = self.tier_config.max_stock_price_pct
        max_sector_pct = self.tier_config.max_sector_pct
        max_positions = self.tier_config.max_positions
        min_turnover = self.tier_config.min_turnover_cr

        # Basic validation
        if entry_price <= 0 or stop_price <= 0:
            return PositionSizingResult(
                symbol=symbol,
                recommended_qty=0,
                max_allowed_qty=0,
                entry_price=entry_price,
                stop_price=stop_price,
                risk_per_share=0,
                position_value=0,
                risk_amount=0,
                risk_pct_of_capital=0,
                tier=self.tier,
                warnings=["Invalid entry/stop price"],
                rejected=True,
                rejection_reason="Invalid entry/stop price"
            )

        if stop_price >= entry_price:
            return PositionSizingResult(
                symbol=symbol,
                recommended_qty=0,
                max_allowed_qty=0,
                entry_price=entry_price,
                stop_price=stop_price,
                risk_per_share=0,
                position_value=0,
                risk_amount=0,
                risk_pct_of_capital=0,
                tier=self.tier,
                warnings=["Stop price must be below entry price"],
                rejected=True,
                rejection_reason="Invalid stop price"
            )

        # Risk per share
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return PositionSizingResult(
                symbol=symbol,
                recommended_qty=0,
                max_allowed_qty=0,
                entry_price=entry_price,
                stop_price=stop_price,
                risk_per_share=0,
                position_value=0,
                risk_amount=0,
                risk_pct_of_capital=0,
                tier=self.tier,
                warnings=["Invalid risk per share"],
                rejected=True,
                rejection_reason="Invalid risk calculation"
            )

        # Capital for this trade
        risk_amount = self.capital * self.capital_cfg.get('max_risk_per_trade', 0.015)

        # Base position size from risk
        base_qty = int(risk_amount / (entry_price - stop_price))

        # Cap at max position %
        max_qty_by_pct = int(self.capital * max_pos_pct / entry_price)
        qty = min(base_qty, max_qty_by_pct)

        # Check stock price vs capital (tier constraint)
        max_stock_price = self.capital * self.tier_config.max_stock_price_pct
        if entry_price > max_stock_price:
            warnings.append(f"Stock price ₹{entry_price:.0f} exceeds tier limit ₹{max_stock_price:,.0f} ({self.tier_config.max_stock_price_pct*100:.0f}% of capital)")
            return PositionSizingResult(
                symbol=symbol,
                recommended_qty=0,
                max_allowed_qty=0,
                entry_price=entry_price,
                stop_price=stop_price,
                risk_per_share=risk_per_share,
                position_value=0,
                risk_amount=0,
                risk_pct_of_capital=0,
                tier=self.tier,
                warnings=warnings,
                rejected=True,
                rejection_reason=f"Stock price ₹{entry_price:.0f} exceeds tier limit ₹{max_stock_price:,.0f}"
            )

        # ATR Parity Sizing - adjust for volatility
        if atr_pct > 0:
            atr_multiplier = min(2.0, max(0.5, 0.02 / max(atr_pct, 0.005)))
            qty = int(qty * atr_multiplier)
            if atr_multiplier != 1.0:
                warnings.append(f"ATR parity adjustment: {atr_multiplier:.2f}x (stock ATR%: {atr_pct:.2%})")

        # Re-apply caps after ATR adjustment
        qty = min(qty, max_qty_by_pct)
        qty = max(1, qty)

        # Sector exposure check
        if sector != "UNKNOWN":
            sector_exposure_pct = current_sector_exposure / self.capital if self.capital > 0 else 0
            if sector_exposure_pct >= self.tier_config.max_sector_pct:
                return PositionSizingResult(
                    symbol=symbol,
                    recommended_qty=0,
                    max_allowed_qty=0,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    risk_per_share=entry_price - stop_price,
                    position_value=0,
                    risk_amount=0,
                    risk_pct_of_capital=0,
                    tier=self.tier,
                    warnings=[f"Sector limit reached: {sector} at {sector_exposure_pct:.1%} (max {self.tier_config.max_sector_pct*100:.0f}%)"],
                    rejected=True,
                    rejection_reason=f"Sector limit reached for {sector}"
                )

        # Position count check
        if portfolio_positions >= self.tier_config.max_positions:
            return PositionSizingResult(
                symbol=symbol,
                recommended_qty=0,
                max_allowed_qty=0,
                entry_price=entry_price,
                stop_price=stop_price,
                risk_per_share=entry_price - stop_price,
                position_value=0,
                risk_amount=0,
                risk_pct_of_capital=0,
                tier=self.tier,
                warnings=[f"Max positions ({self.tier_config.max_positions}) reached"],
                rejected=True,
                rejection_reason="Max positions limit reached"
            )

        # Final calculations
        position_value = qty * entry_price
        risk_amount = (entry_price - stop_price) * qty
        risk_pct_of_capital = risk_amount / self.capital if self.capital > 0 else 0

        warnings.append(f"Tier: {self.tier.value.upper()} | Capital: ₹{self.capital:,.0f} | Risk: {risk_pct_of_capital:.2%}")

        return PositionSizingResult(
            symbol=symbol,
            recommended_qty=qty,
            max_allowed_qty=max_qty_by_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            risk_per_share=entry_price - stop_price,
            position_value=qty * entry_price,
            risk_amount=risk_amount,
            risk_pct_of_capital=risk_pct_of_capital,
            tier=self.tier,
            warnings=warnings,
            rejected=False
        )

    def filter_universe(self, symbols: list, price_data: dict, min_turnover_cr: float = None) -> list:
        """
        Filter universe based on tier constraints.
        Removes stocks that are too expensive or illiquid for the tier.
        """
        min_turnover = min_turnover_cr or self.tier_config.min_turnover_cr
        max_price = self.capital * self.tier_config.max_stock_price_pct

        filtered = []
        for symbol in symbols:
            price = price_data.get(symbol, {}).get('close', 0)
            turnover = price_data.get(symbol, {}).get('turnover_cr', 0)

            if price > 0 and price <= self.capital * self.tier_config.max_stock_price_pct:
                if turnover >= min_turnover:
                    filtered.append(symbol)
                else:
                    logger.debug(f"Filtered {symbol}: turnover {turnover:.1f}Cr < {min_turnover}Cr")
            else:
                logger.debug(f"Filtered {symbol}: price {price} > max {self.capital * self.tier_config.max_stock_price_pct:.0f}")

        return filtered


def create_position_sizer(capital: float, config: dict = None) -> CapitalTieredPositionSizer:
    """Factory function to create position sizer."""
    return CapitalTieredPositionSizer(capital, config)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test different capital tiers
    for capital in [25_000, 50_000, 1_00_000, 5_00_000, 20_00_000]:
        sizer = create_position_sizer(capital)
        print(f"\nCapital: ₹{capital:,.0f} -> Tier: {sizer.tier.value}")
        print(f"  Max stock price: ₹{sizer.tier_config.max_stock_price_pct*100:.0f}% = ₹{sizer.capital * sizer.tier_config.max_stock_price_pct:,.0f}")
        print(f"  Max position: {sizer.tier_config.max_position_pct*100:.0f}% = ₹{sizer.capital * sizer.tier_config.max_position_pct:,.0f}")
        print(f"  Max positions: {sizer.tier_config.max_positions}")
        print(f"  Min turnover: ₹{sizer.tier_config.min_turnover_cr} Cr")

        # Test sizing
        result = sizer.calculate_position_size(
            symbol="TEST",
            entry_price=500,
            stop_price=475,
            atr_pct=0.025
        )
        print(f"  Test sizing (₹500 entry, ₹475 SL): Qty={result.recommended_qty}, Risk={result.risk_pct_of_capital:.2%}")