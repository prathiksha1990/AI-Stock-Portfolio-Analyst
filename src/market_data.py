from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


def fetch_quotes(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", "current_price", "previous_close", "quote_timestamp", "is_stale"])
    data = yf.download(tickers=tickers, period="10d", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
    rows = []
    now = datetime.now(timezone.utc)
    for ticker in tickers:
        try:
            frame = data[ticker] if len(tickers) > 1 else data
            closes = frame["Close"].dropna()
            latest = float(closes.iloc[-1])
            previous = float(closes.iloc[-2]) if len(closes) > 1 else float("nan")
            observed = pd.Timestamp(closes.index[-1])
            observed = observed.tz_localize("UTC") if observed.tzinfo is None else observed.tz_convert("UTC")
            rows.append({"ticker": ticker, "current_price": latest, "previous_close": previous, "quote_timestamp": observed, "is_stale": (now - observed.to_pydatetime()).total_seconds() > 4 * 86400})
        except (KeyError, IndexError, TypeError, ValueError):
            rows.append({"ticker": ticker, "current_price": float("nan"), "previous_close": float("nan"), "quote_timestamp": pd.NaT, "is_stale": True})
    return pd.DataFrame(rows)


def fetch_history(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp | None = None) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(tickers=tickers, start=start.date(), end=(end or pd.Timestamp.today()).date() + pd.Timedelta(days=1), interval="1d", auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.sort_index().ffill()

