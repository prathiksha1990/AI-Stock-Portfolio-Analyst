from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def xnpv(rate: float, cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    base = cashflows[0][0]
    return sum(value / (1 + rate) ** ((when - base).days / 365.0) for when, value in cashflows)


def xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float | None:
    if not cashflows or not any(v < 0 for _, v in cashflows) or not any(v > 0 for _, v in cashflows):
        return None
    try:
        return brentq(lambda r: xnpv(r, cashflows), -0.9999, 1000.0)
    except ValueError:
        return None


def build_performance(transactions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or prices.empty:
        return pd.DataFrame()
    dates = prices.index
    quantities = {ticker: 0.0 for ticker in prices.columns}
    invested = 0.0
    rows = []
    grouped = {d: g for d, g in transactions.groupby(transactions["date"].dt.normalize())}
    for dt in dates:
        normalized = pd.Timestamp(dt).tz_localize(None).normalize()
        for _, tx in grouped.get(normalized, pd.DataFrame()).iterrows():
            sign = 1 if tx["transaction_type"] == "Buy" else -1
            quantities[tx["ticker"]] = quantities.get(tx["ticker"], 0.0) + sign * tx["quantity"]
            invested += sign * tx["quantity"] * tx["price"]
        value = sum(quantities.get(t, 0.0) * prices.at[dt, t] for t in prices.columns if pd.notna(prices.at[dt, t]))
        rows.append({"date": dt, "portfolio_value": value, "net_invested": invested, "gain": value - invested})
    return pd.DataFrame(rows).set_index("date")


def portfolio_xirr(transactions: pd.DataFrame, current_value: float, as_of: pd.Timestamp | None = None) -> float | None:
    flows = []
    for row in transactions.itertuples(index=False):
        value = float(row.quantity * row.price)
        flows.append((pd.Timestamp(row.date), -value if row.transaction_type == "Buy" else value))
    flows.append((as_of or pd.Timestamp.today().normalize(), current_value))
    return xirr(sorted(flows, key=lambda x: x[0]))

