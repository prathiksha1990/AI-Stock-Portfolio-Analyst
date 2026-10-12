import pandas as pd
from src.portfolio import calculate_fifo


def test_fifo_partial_lot_and_realized_gain():
    tx = pd.DataFrame([
        {"ticker":"AAPL", "date":pd.Timestamp("2024-01-01"), "transaction_type":"Buy", "quantity":10, "price":100},
        {"ticker":"AAPL", "date":pd.Timestamp("2024-02-01"), "transaction_type":"Buy", "quantity":10, "price":150},
        {"ticker":"AAPL", "date":pd.Timestamp("2024-03-01"), "transaction_type":"Sell", "quantity":12, "price":180},
    ])
    holdings, realized = calculate_fifo(tx)
    assert holdings.iloc[0].current_quantity == 8
    assert holdings.iloc[0].total_cost_basis == 1200
    assert holdings.iloc[0].average_cost_basis == 150
    # Proceeds: 12 × $180 = $2,160; FIFO basis: 10 × $100 + 2 × $150 = $1,300.
    assert realized.iloc[0].realized_gain == 860


def test_fully_sold_position_removed():
    tx = pd.DataFrame([
        {"ticker":"MSFT", "date":pd.Timestamp("2024-01-01"), "transaction_type":"Buy", "quantity":2, "price":100},
        {"ticker":"MSFT", "date":pd.Timestamp("2024-02-01"), "transaction_type":"Sell", "quantity":2, "price":120},
    ])
    holdings, _ = calculate_fifo(tx)
    assert holdings.empty

