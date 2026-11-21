"""
GTT (Good Till Triggered) Order Basket Generator.
Exports broker-compatible CSV for Zerodha Kite, Dhan, Fyers, Angel One, Groww.
"""

import pandas as pd
from typing import List, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class GTTOrder:
    """Single GTT order leg."""
    symbol: str
    transaction_type: str  # "BUY" or "SELL"
    quantity: int
    price: float           # Limit price
    trigger_price: float   # Trigger price
    product_type: str      # "CNC" (delivery), "MIS" (intraday), "NRML" (futures)
    order_type: str = "LIMIT"  # "LIMIT" or "MARKET"
    validity: str = "DAY"      # "DAY" or "GTT"
    disclosed_quantity: int = 0
    exchange: str = "NSE"


def generate_gtt_stop_loss_order(
    symbol: str,
    quantity: int,
    stop_price: float,
    trigger_buffer_pct: float = 0.005,  # 0.5% buffer
    product_type: str = "CNC",
    exchange: str = "NSE"
) -> GTTOrder:
    """
    Generate a GTT stop-loss order.

    For SELL stop-loss:
    - Trigger price = stop_price * (1 - buffer)
    - Limit price = stop_price
    """
    trigger = round(stop_price * (1 - trigger_buffer_pct), 1)
    limit = round(stop_price, 1)

    return GTTOrder(
        symbol=symbol,
        transaction_type="SELL",
        quantity=quantity,
        price=limit,
        trigger_price=trigger,
        product_type=product_type,
        exchange=exchange
    )


def generate_gtt_target_order(
    symbol: str,
    quantity: int,
    target_price: float,
    trigger_buffer_pct: float = 0.002,  # 0.2% buffer
    product_type: str = "CNC",
    exchange: str = "NSE"
) -> GTTOrder:
    """
    Generate a GTT target order (limit sell).

    For SELL target:
    - Trigger price = target_price * (1 + buffer)
    - Limit price = target_price
    """
    trigger = round(target_price * (1 + trigger_buffer_pct), 1)
    limit = round(target_price, 1)

    return GTTOrder(
        symbol=symbol,
        transaction_type="SELL",
        quantity=quantity,
        price=limit,
        trigger_price=trigger,
        product_type=product_type,
        exchange=exchange
    )


def generate_gtt_basket(
    positions: Dict,
    trigger_buffer_pct: float = 0.005,
    target_buffer_pct: float = 0.002,
    product_type: str = "CNC",
    exchange: str = "NSE"
) -> pd.DataFrame:
    """
    Generate complete GTT basket for all open positions.

    Args:
        positions: Dict of symbol -> position dict with keys:
            - qty: int
            - stop_price: float
            - targets: List[float] (target prices)
            - product_type: str (optional, default "CNC")
        trigger_buffer_pct: Buffer below stop for trigger
        target_buffer_pct: Buffer above target for trigger
        product_type: "CNC" or "MIS"
        exchange: "NSE" or "BSE"

    Returns:
        DataFrame with GTT orders ready for broker import
    """
    orders = []

    for symbol, pos in positions.items():
        qty = pos.get('qty', pos.get('quantity', 0))
        if qty <= 0:
            continue

        prod_type = pos.get('product_type', product_type)

        # Stop-loss order
        sl = pos.get('stop_price', pos.get('sl'))
        if sl:
            orders.append(generate_gtt_stop_loss_order(
                symbol=symbol,
                quantity=qty,
                stop_price=sl,
                trigger_buffer_pct=trigger_buffer_pct,
                product_type=prod_type,
                exchange=exchange
            ))

        # Target orders (partial quantities)
        targets = pos.get('targets', [])
        if targets:
            # Split quantity: 50% at T1, 30% at T2, 20% at T3+
            remaining_qty = qty
            for i, target in enumerate(targets):
                if i == 0:
                    # T1: 50%
                    tgt_qty = qty // 2
                elif i == 1:
                    # T2: 30% of remaining
                    tgt_qty = int(qty * 0.3)
                else:
                    # T3+: remaining
                    tgt_qty = remaining_qty

                if tgt_qty <= 0:
                    break

                remaining_qty -= tgt_qty

                orders.append(generate_gtt_target_order(
                    symbol=symbol,
                    quantity=tgt_qty,
                    target_price=target,
                    trigger_buffer_pct=target_buffer_pct,
                    product_type=prod_type,
                    exchange=exchange
                ))

    if not orders:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame([asdict(o) for o in orders])

    # Reorder columns for broker compatibility
    cols_order = [
        'symbol', 'transaction_type', 'quantity', 'price',
        'trigger_price', 'product_type', 'order_type',
        'validity', 'disclosed_quantity', 'exchange'
    ]
    df = df[cols_order]

    return df


def export_gtt_csv(
    positions: Dict,
    filepath: Union[str, Path],
    **kwargs
) -> str:
    """
    Export GTT basket to CSV file.

    Returns the filepath.
    """
    df = generate_gtt_basket(positions, **kwargs)

    if df.empty:
        logger.warning("No orders to export")
        return ""

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=False)
    logger.info(f"Exported {len(df)} GTT orders to {filepath}")

    return str(filepath)


# Broker-specific formatters
def format_for_zerodha_kite(df: pd.DataFrame) -> pd.DataFrame:
    """Format for Zerodha Kite GTT CSV import."""
    # Zerodha expects specific column names
    return df.rename(columns={
        'symbol': 'tradingsymbol',
        'transaction_type': 'transaction_type',
        'quantity': 'quantity',
        'price': 'price',
        'trigger_price': 'trigger_price',
        'product_type': 'product',
        'order_type': 'order_type',
        'validity': 'validity',
        'disclosed_quantity': 'disclosed_quantity',
        'exchange': 'exchange'
    })


def format_for_dhan(df: pd.DataFrame) -> pd.DataFrame:
    """Format for Dhan GTT CSV import."""
    return df.rename(columns={
        'symbol': 'security_id',
        'transaction_type': 'transaction_type',
        'quantity': 'quantity',
        'price': 'price',
        'trigger_price': 'trigger_price',
        'product_type': 'product_type',
        'order_type': 'order_type',
        'validity': 'validity',
        'disclosed_quantity': 'disclosed_quantity',
        'exchange': 'exchange'
    })


def format_for_fyers(df: pd.DataFrame) -> pd.DataFrame:
    """Format for Fyers GTT CSV import."""
    return df.rename(columns={
        'symbol': 'symbol',
        'transaction_type': 'side',
        'quantity': 'qty',
        'price': 'limitPrice',
        'trigger_price': 'stopPrice',
        'product_type': 'productType',
        'order_type': 'orderType',
        'validity': 'validity',
        'disclosed_quantity': 'disclosedQty',
        'exchange': 'exchange'
    })


def format_for_angel_one(df: pd.DataFrame) -> pd.DataFrame:
    """Format for Angel One GTT CSV import."""
    return df.rename(columns={
        'symbol': 'symbol',
        'transaction_type': 'transactiontype',
        'quantity': 'quantity',
        'price': 'price',
        'trigger_price': 'triggerprice',
        'product_type': 'producttype',
        'order_type': 'ordertype',
        'validity': 'validity',
        'disclosed_quantity': 'disclosedqty',
        'exchange': 'exchange'
    })


def format_for_groww(df: pd.DataFrame) -> pd.DataFrame:
    """Format for Groww GTT CSV import."""
    return df.rename(columns={
        'symbol': 'symbol',
        'transaction_type': 'transaction_type',
        'quantity': 'quantity',
        'price': 'price',
        'trigger_price': 'trigger_price',
        'product_type': 'product_type',
        'order_type': 'order_type',
        'validity': 'validity',
        'disclosed_quantity': 'disclosed_quantity',
        'exchange': 'exchange'
    })


BROKER_FORMATTERS = {
    'zerodha': format_for_zerodha_kite,
    'dhan': format_for_dhan,
    'fyers': format_for_fyers,
    'angel_one': format_for_angel_one,
    'groww': format_for_groww,
}


def export_gtt_for_broker(
    positions: Dict,
    broker: str,
    filepath: Union[str, Path],
    **kwargs
) -> str:
    """
    Export GTT basket formatted for specific broker.

    Args:
        positions: Position dict
        broker: "zerodha", "dhan", "fyers", "angel_one", "groww"
        filepath: Output path
    """
    df = generate_gtt_basket(positions, **kwargs)

    if df.empty:
        return ""

    formatter = BROKER_FORMATTERS.get(broker.lower())
    if formatter:
        df = formatter(df)

    return export_gtt_csv(df, filepath, **kwargs)


# Convenience function for manual position entry
def create_position_entry(
    symbol: str,
    qty: int,
    entry_price: float,
    stop_price: float,
    targets: List[float],
    product_type: str = "CNC"
) -> Dict:
    """Create a position dict for GTT basket generation."""
    return {
        symbol: {
            'qty': qty,
            'quantity': qty,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'sl': stop_price,
            'targets': targets,
            'product_type': product_type
        }
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test with sample positions
    positions = {
        "BEL": {
            'qty': 12,
            'quantity': 12,
            'entry_price': 312,
            'stop_price': 296,
            'sl': 296,
            'targets': [336, 352],
            'product_type': 'CNC'
        },
        "DIXON": {
            'qty': 5,
            'quantity': 5,
            'entry_price': 4200,
            'stop_price': 3950,
            'sl': 3950,
            'targets': [4600, 4850],
            'product_type': 'CNC'
        }
    }

    # Generate standard format
    df = generate_gtt_basket(positions)
    print("Standard GTT Basket:")
    print(df.to_string(index=False))

    # Test Zerodha format
    df_zerodha = format_for_zerodha_kite(df)
    print("\nZerodha Kite Format:")
    print(df_zerodha.to_string(index=False))

    # Export to file
    export_gtt_csv(positions, "./logs/gtt_basket_test.csv")
    print("\nExported to logs/gtt_basket_test.csv")