from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd


def D(value: object) -> Decimal:
    return Decimal(str(value))


@dataclass
class Lot:
    quantity: Decimal
    price: Decimal


def calculate_fifo(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lots: dict[str, deque[Lot]] = defaultdict(deque)
    realized: list[dict[str, object]] = []
    realized_totals: dict[str, Decimal] = defaultdict(Decimal)

    for row in transactions.itertuples(index=False):
        ticker, qty, price = row.ticker, D(row.quantity), D(row.price)
        if row.transaction_type == "Buy":
            lots[ticker].append(Lot(qty, price))
            continue
        remaining = qty
        cost = Decimal("0")
        while remaining > 0:
            lot = lots[ticker][0]
            used = min(remaining, lot.quantity)
            cost += used * lot.price
            lot.quantity -= used
            remaining -= used
            if lot.quantity == 0:
                lots[ticker].popleft()
        proceeds = qty * price
        gain = proceeds - cost
        realized_totals[ticker] += gain
        realized.append({"ticker": ticker, "date": row.date, "quantity": float(qty), "proceeds": float(proceeds), "cost_basis": float(cost), "realized_gain": float(gain)})

    holdings = []
    for ticker, queue in lots.items():
        quantity = sum((lot.quantity for lot in queue), Decimal("0"))
        if quantity <= 0:
            continue
        basis = sum((lot.quantity * lot.price for lot in queue), Decimal("0"))
        holdings.append({"ticker": ticker, "current_quantity": float(quantity), "total_cost_basis": float(basis), "average_cost_basis": float(basis / quantity), "realized_gain": float(realized_totals[ticker])})
    return pd.DataFrame(holdings), pd.DataFrame(realized)


def enrich_holdings(holdings: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty:
        return holdings.copy()
    out = holdings.merge(quotes, on="ticker", how="left")
    out["market_value"] = out["current_quantity"] * out["current_price"]
    out["unrealized_gain"] = out["market_value"] - out["total_cost_basis"]
    out["unrealized_gain_pct"] = out["unrealized_gain"] / out["total_cost_basis"] * 100
    out["daily_change_pct"] = (out["current_price"] / out["previous_close"] - 1) * 100
    total = out["market_value"].sum(min_count=1)
    out["allocation_pct"] = out["market_value"] / total * 100 if total and total > 0 else float("nan")
    return out.sort_values("market_value", ascending=False, na_position="last")

