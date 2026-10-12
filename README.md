# AI Stock Analyst

A modular Streamlit application that validates a CSV transaction ledger, reconstructs open positions with FIFO accounting, retrieves latest-available and historical prices, charts portfolio performance, computes XIRR, and provides optional Groq-powered portfolio chat.

## Run locally

Python 3.12+ and `uv` are required. This project intentionally uses no other package manager.

```bash
uv sync
cp .env.example .env
# Optionally add GROQ_API_KEY to .env
uv run streamlit run app.py
```

Run tests:

```bash
uv run pytest
```

## CSV schema

```csv
ticker,date,transaction_type,quantity,price
AAPL,2024-01-15,Buy,10,185.50
MSFT,2024-02-10,Buy,5,410.00
AAPL,2024-09-12,Sell,8,220.00
```

Dates must be `YYYY-MM-DD`. Quantity and price must be positive. Sells cannot exceed shares available at that point in chronological/file order. Exact duplicates produce a warning and remain in the ledger.

## Architecture

- `src/validation.py`: strict CSV and row-level validation
- `src/portfolio.py`: deterministic Decimal-backed FIFO accounting
- `src/market_data.py`: yfinance quote and price-history service
- `src/performance.py`: historical value reconstruction and XIRR
- `src/ai.py`: Groq client and bounded portfolio context
- `app.py`: Streamlit state, UI, caching, and visualization

Uploaded data stays in Streamlit session memory and is not persisted. Market quotes are latest available and may be delayed. This software is for informational purposes only.
