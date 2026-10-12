import pandas as pd
from src.performance import xirr


def test_xirr_simple_one_year_double():
    result = xirr([(pd.Timestamp("2024-01-01"), -100), (pd.Timestamp("2025-01-01"), 200)])
    assert result is not None
    assert abs(result - 0.9962) < 0.01


def test_xirr_requires_both_signs():
    assert xirr([(pd.Timestamp("2024-01-01"), -100)]) is None

