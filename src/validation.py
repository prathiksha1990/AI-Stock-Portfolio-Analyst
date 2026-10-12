from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO

import pandas as pd

REQUIRED_COLUMNS = ["ticker", "date", "transaction_type", "quantity", "price"]


@dataclass
class ValidationResult:
    data: pd.DataFrame | None
    errors: pd.DataFrame
    duplicate_rows: pd.DataFrame

    @property
    def is_valid(self) -> bool:
        return self.data is not None and self.errors.empty


def read_csv_strict(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(payload), dtype=str, keep_default_na=False)


def validate_transactions(raw: pd.DataFrame, today: date | None = None) -> ValidationResult:
    today = today or date.today()
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        errors = pd.DataFrame([{"row": "File", "field": "columns", "message": f"Missing required columns: {', '.join(missing)}"}])
        return ValidationResult(None, errors, pd.DataFrame())

    df = raw.copy()
    df["_original_row"] = range(2, len(df) + 2)
    issues: list[dict[str, object]] = []

    def issue(i: int, field: str, message: str) -> None:
        issues.append({"row": int(df.at[i, "_original_row"]), "field": field, "message": message})

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["transaction_type"] = df["transaction_type"].astype(str).str.strip().str.title()
    parsed_dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    quantities = pd.to_numeric(df["quantity"], errors="coerce")
    prices = pd.to_numeric(df["price"], errors="coerce")

    for i in df.index:
        if not df.at[i, "ticker"] or not df.at[i, "ticker"].replace(".", "").replace("-", "").isalnum():
            issue(i, "ticker", "Ticker must be a non-empty symbol containing letters, numbers, '.' or '-'.")
        if pd.isna(parsed_dates.at[i]):
            issue(i, "date", "Date must use the unambiguous YYYY-MM-DD format and be valid.")
        elif parsed_dates.at[i].date() > today:
            issue(i, "date", "Transaction date cannot be in the future.")
        if df.at[i, "transaction_type"] not in {"Buy", "Sell"}:
            issue(i, "transaction_type", "Transaction type must be Buy or Sell.")
        if pd.isna(quantities.at[i]) or quantities.at[i] <= 0:
            issue(i, "quantity", "Quantity must be a positive number.")
        if pd.isna(prices.at[i]) or prices.at[i] <= 0:
            issue(i, "price", "Price must be a positive number.")

    if issues:
        return ValidationResult(None, pd.DataFrame(issues), pd.DataFrame())

    df["date"] = parsed_dates.dt.normalize()
    df["quantity"] = quantities.astype(float)
    df["price"] = prices.astype(float)
    duplicates = df[df.duplicated(REQUIRED_COLUMNS, keep=False)].copy()
    df = df.sort_values(["date", "_original_row"], kind="stable").reset_index(drop=True)

    balances: dict[str, float] = {}
    for i, row in df.iterrows():
        ticker = row["ticker"]
        available = balances.get(ticker, 0.0)
        if row["transaction_type"] == "Buy":
            balances[ticker] = available + row["quantity"]
        else:
            if row["quantity"] > available + 1e-9:
                issues.append({"row": int(row["_original_row"]), "field": "quantity", "message": f"Sell of {row['quantity']:g} {ticker} exceeds {available:g} available shares."})
            else:
                balances[ticker] = available - row["quantity"]

    if issues:
        return ValidationResult(None, pd.DataFrame(issues), duplicates)
    return ValidationResult(df, pd.DataFrame(columns=["row", "field", "message"]), duplicates)

