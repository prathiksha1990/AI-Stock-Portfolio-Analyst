from datetime import date
import pandas as pd
from src.validation import validate_transactions


def frame(rows):
    return pd.DataFrame(rows, columns=["ticker", "date", "transaction_type", "quantity", "price"])


def test_normalizes_and_preserves_same_day_order():
    result = validate_transactions(frame([[" aapl ", "2024-01-01", "buy", "2.5", "100"], ["AAPL", "2024-01-01", "SELL", "1", "120"]]), today=date(2025, 1, 1))
    assert result.is_valid
    assert result.data["transaction_type"].tolist() == ["Buy", "Sell"]
    assert result.data.iloc[0].ticker == "AAPL"


def test_rejects_oversell_and_bad_date():
    oversell = validate_transactions(frame([["AAPL", "2024-01-01", "Sell", 1, 100]]), today=date(2025, 1, 1))
    assert not oversell.is_valid
    bad = validate_transactions(frame([["AAPL", "01/02/2024", "Buy", 1, 100]]), today=date(2025, 1, 1))
    assert not bad.is_valid


def test_reports_duplicates_without_rejecting():
    df = frame([["AAPL", "2024-01-01", "Buy", 1, 100], ["AAPL", "2024-01-01", "Buy", 1, 100]])
    result = validate_transactions(df, today=date(2025, 1, 1))
    assert result.is_valid
    assert len(result.duplicate_rows) == 2

