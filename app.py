from __future__ import annotations

import hashlib
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ai import chat, portfolio_context
from src.config import settings
from src.market_data import fetch_history, fetch_quotes
from src.performance import build_performance, portfolio_xirr
from src.portfolio import calculate_fifo, enrich_holdings
from src.validation import read_csv_strict, validate_transactions

st.set_page_config(page_title="AI Stock Analyst", page_icon="📈", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1450px}
  [data-testid="stMetric"] {background:#fff; border:1px solid #e8edf3; border-radius:14px; padding:16px; box-shadow:0 2px 10px rgba(15,23,42,.04)}
  [data-testid="stMetricValue"] {font-weight:750}
  .positive {color:#15803d}.negative {color:#b91c1c}
  .eyebrow {color:#64748b; letter-spacing:.08em; text-transform:uppercase; font-size:.75rem; font-weight:700}
  .empty {padding:2.5rem; text-align:center; border:1px dashed #cbd5e1; border-radius:16px; color:#64748b; background:#f8fafc}
</style>
""", unsafe_allow_html=True)


def initialize_state() -> None:
    defaults = {"transactions": None, "holdings_base": None, "holdings": None, "realized": None, "quotes": None, "refresh_nonce": 0, "refresh_time": None, "chat_history": [], "upload_hash": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_data(ttl=settings.market_data_ttl_seconds, show_spinner=False)
def cached_quotes(tickers: tuple[str, ...], nonce: int) -> pd.DataFrame:
    return fetch_quotes(list(tickers))


@st.cache_data(ttl=settings.market_data_ttl_seconds, show_spinner=False)
def cached_history(tickers: tuple[str, ...], start: pd.Timestamp, nonce: int) -> pd.DataFrame:
    return fetch_history(list(tickers), start)


def refresh_market_data() -> None:
    base = st.session_state.holdings_base
    if base is None or base.empty:
        return
    tickers = tuple(sorted(base["ticker"].tolist()))
    with st.spinner("Fetching latest available market data…"):
        quotes = cached_quotes(tickers, st.session_state.refresh_nonce)
    st.session_state.quotes = quotes
    st.session_state.holdings = enrich_holdings(base, quotes)
    st.session_state.refresh_time = datetime.now().astimezone()


def money(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"${value:,.2f}"


def pct(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.2f}%"


def empty_state(title: str, detail: str) -> None:
    st.markdown(f'<div class="empty"><h3>{title}</h3><p>{detail}</p></div>', unsafe_allow_html=True)


initialize_state()

with st.sidebar:
    st.title("📈 AI Stock Analyst")
    st.caption("FIFO portfolio accounting, market analytics, and portfolio-aware AI.")
    st.divider()
    if st.session_state.transactions is None:
        st.info("No portfolio uploaded")
        held = 0
    else:
        st.success("Portfolio loaded")
        held = 0 if st.session_state.holdings_base is None else len(st.session_state.holdings_base)
    st.metric("Currently held stocks", held)
    refresh_text = st.session_state.refresh_time.strftime("%b %d, %Y · %I:%M %p %Z") if st.session_state.refresh_time else "Not refreshed"
    st.caption(f"Market data: {refresh_text}")
    if st.button("Refresh Market Data", use_container_width=True, disabled=held == 0):
        st.session_state.refresh_nonce += 1
        refresh_market_data()
        st.rerun()
    st.divider()
    st.caption("For informational purposes only. Quotes may be delayed and results are not tax, legal, or investment advice.")

st.markdown('<div class="eyebrow">Personal portfolio intelligence</div>', unsafe_allow_html=True)
st.title("AI Stock Analyst")
st.caption("Understand what you own, how it performed, and where your portfolio risk is concentrated.")

upload_tab, portfolio_tab, history_tab, ai_tab = st.tabs(["Data Upload", "Consolidated Portfolio", "Historical Performance", "AI Analyst"])

with upload_tab:
    st.subheader("Upload transaction history")
    st.caption("CSV only · dates must be YYYY-MM-DD · transactions are processed in file order within each day")
    uploaded = st.file_uploader("Choose a complete transaction-history CSV", type=["csv"])
    with st.expander("Required CSV format", expanded=uploaded is None):
        st.code("ticker,date,transaction_type,quantity,price\nAAPL,2024-01-15,Buy,10,185.50\nAAPL,2024-09-12,Sell,8,220.00", language="csv")
    if uploaded is not None:
        payload = uploaded.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        try:
            raw = read_csv_strict(payload)
            st.markdown("#### Original uploaded data")
            st.dataframe(raw, use_container_width=True, hide_index=True)
            result = validate_transactions(raw)
            if not result.errors.empty:
                st.error("Validation failed. Fix the rows below before portfolio calculations can proceed.")
                st.dataframe(result.errors, use_container_width=True, hide_index=True)
            else:
                if not result.duplicate_rows.empty:
                    st.warning(f"Found {len(result.duplicate_rows)} rows involved in exact duplicates. They are retained; confirm they are intentional.")
                    st.dataframe(result.duplicate_rows.drop(columns="_original_row"), use_container_width=True, hide_index=True)
                if digest != st.session_state.upload_hash:
                    base, realized = calculate_fifo(result.data)
                    st.session_state.update(transactions=result.data, holdings_base=base, realized=realized, quotes=None, holdings=None, chat_history=[], upload_hash=digest, refresh_time=None)
                    if not base.empty:
                        refresh_market_data()
                st.success("Validation passed. Portfolio calculations are ready.")
                tx = result.data
                cols = st.columns(6)
                values = [("Transactions", len(tx)), ("Unique tickers", tx.ticker.nunique()), ("Buys", int((tx.transaction_type == "Buy").sum())), ("Sells", int((tx.transaction_type == "Sell").sum())), ("Earliest", tx.date.min().date().isoformat()), ("Latest", tx.date.max().date().isoformat())]
                for col, (label, value) in zip(cols, values):
                    col.metric(label, value)
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    elif st.session_state.transactions is None:
        empty_state("Your portfolio starts with a CSV", "Upload your complete Buy and Sell history to validate and reconstruct current holdings.")

with portfolio_tab:
    if st.session_state.transactions is None:
        empty_state("No portfolio yet", "Upload a valid transaction CSV in Data Upload to see current holdings.")
    elif st.session_state.holdings_base is not None and st.session_state.holdings_base.empty:
        empty_state("No open positions", "Every purchased share has been sold. Realized transactions remain available in performance analysis.")
    else:
        holdings = st.session_state.holdings
        if holdings is None:
            refresh_market_data()
            holdings = st.session_state.holdings
        unavailable = holdings[holdings.current_price.isna()]["ticker"].tolist()
        if unavailable:
            st.warning(f"No market quote was available for: {', '.join(unavailable)}. Totals exclude unavailable values; prices are never replaced with zero.")
        stale = holdings[holdings.is_stale.fillna(True)]["ticker"].tolist()
        if stale:
            st.warning(f"Latest available quote may be stale for: {', '.join(stale)}.")
        valid = holdings.dropna(subset=["market_value"])
        total_value = valid.market_value.sum()
        total_basis = valid.total_cost_basis.sum()
        unrealized = total_value - total_basis
        unrealized_pct = unrealized / total_basis * 100 if total_basis else float("nan")
        daily = ((valid.current_price - valid.previous_close) * valid.current_quantity).sum(min_count=1)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Current Portfolio Value", money(total_value))
        metric_cols[1].metric("Unrealized Gain / Loss", money(unrealized), pct(unrealized_pct))
        metric_cols[2].metric("Unrealized Return", pct(unrealized_pct))
        metric_cols[3].metric("Today’s Price Change", money(daily), help="Price movement versus previous session close. This is not actual daily investment return and excludes transaction timing effects.")
        left, right = st.columns([1, 1.7])
        with left:
            st.markdown("#### Portfolio allocation")
            if total_value > 0:
                fig = px.pie(valid.sort_values("market_value", ascending=False), values="market_value", names="ticker", hole=.62, color_discrete_sequence=px.colors.qualitative.Safe)
                fig.update_traces(textinfo="label+percent", hovertemplate="<b>%{label}</b><br>Market value: $%{value:,.2f}<br>Allocation: %{percent}<extra></extra>")
                fig.update_layout(margin=dict(l=10, r=10, t=15, b=10), legend_orientation="h", height=390)
                st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("#### Current holdings")
            display = holdings.rename(columns={"ticker":"Ticker", "current_quantity":"Current Quantity", "average_cost_basis":"Average Cost Basis", "total_cost_basis":"Total Cost Basis", "current_price":"Current Price", "market_value":"Market Value", "unrealized_gain":"Unrealized Gain ($)", "unrealized_gain_pct":"Unrealized Gain (%)", "allocation_pct":"Allocation (%)", "daily_change_pct":"Daily Change (%)"})
            columns = ["Ticker", "Current Quantity", "Average Cost Basis", "Total Cost Basis", "Current Price", "Market Value", "Unrealized Gain ($)", "Unrealized Gain (%)", "Allocation (%)", "Daily Change (%)"]
            st.dataframe(display[columns].style.format({"Current Quantity":"{:,.4f}", "Average Cost Basis":"${:,.2f}", "Total Cost Basis":"${:,.2f}", "Current Price":"${:,.2f}", "Market Value":"${:,.2f}", "Unrealized Gain ($)":"${:,.2f}", "Unrealized Gain (%)":"{:,.2f}%", "Allocation (%)":"{:,.2f}%", "Daily Change (%)":"{:,.2f}%"}, na_rep="—").map(lambda v: "color:#15803d" if isinstance(v, (int, float)) and v > 0 else ("color:#b91c1c" if isinstance(v, (int, float)) and v < 0 else ""), subset=["Unrealized Gain ($)", "Unrealized Gain (%)", "Daily Change (%)"]), use_container_width=True, hide_index=True, height=390)

with history_tab:
    tx = st.session_state.transactions
    if tx is None:
        empty_state("Performance awaits your data", "Upload a valid CSV to build the historical portfolio value series.")
    else:
        tickers = tuple(sorted(tx.ticker.unique()))
        with st.spinner("Loading historical adjusted prices…"):
            prices = cached_history(tickers, tx.date.min(), st.session_state.refresh_nonce)
        performance = build_performance(tx, prices)
        if performance.empty:
            st.warning("Historical prices are unavailable for this date range.")
        else:
            current_value = float(performance.portfolio_value.iloc[-1])
            irr = portfolio_xirr(tx, current_value, pd.Timestamp(performance.index[-1]))
            realized_total = 0.0 if st.session_state.realized is None or st.session_state.realized.empty else st.session_state.realized.realized_gain.sum()
            cols = st.columns(3)
            cols[0].metric("Latest Historical Value", money(current_value))
            cols[1].metric("Realized Gain / Loss", money(realized_total))
            cols[2].metric("Money-Weighted Return (XIRR)", pct(irr * 100) if irr is not None else "Not available", help="Annualized return accounting for the timing and size of cash flows.")
            plot = performance.reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot.date, y=plot.portfolio_value, name="Portfolio value", line=dict(color="#2563eb", width=3), fill="tozeroy", fillcolor="rgba(37,99,235,.08)"))
            fig.add_trace(go.Scatter(x=plot.date, y=plot.net_invested, name="Net invested", line=dict(color="#64748b", dash="dash")))
            fig.update_layout(title="Portfolio value vs. net invested", yaxis_tickprefix="$", hovermode="x unified", legend_orientation="h", margin=dict(l=20,r=20,t=60,b=20), height=460)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Realized FIFO gains by sale"):
                realized = st.session_state.realized
                if realized is None or realized.empty:
                    st.caption("No sell transactions.")
                else:
                    st.dataframe(realized.style.format({"quantity":"{:,.4f}", "proceeds":"${:,.2f}", "cost_basis":"${:,.2f}", "realized_gain":"${:,.2f}"}), use_container_width=True, hide_index=True)

with ai_tab:
    if st.session_state.transactions is None:
        empty_state("Your analyst needs portfolio context", "Upload a valid CSV first. AI chat is optional and all accounting remains deterministic Python.")
    elif not settings.groq_api_key:
        st.info("AI chat is disabled because GROQ_API_KEY is not configured. All portfolio and performance features remain available.")
        st.code("cp .env.example .env\n# Add GROQ_API_KEY to .env\nuv run streamlit run app.py", language="bash")
    else:
        st.subheader("Portfolio-aware AI analyst")
        st.caption("Ask about concentration, risk, performance, or holdings. Responses are educational, not financial advice.")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        prompt = st.chat_input("What would you like to understand about your portfolio?")
        if prompt:
            st.session_state.chat_history.append({"role":"user", "content":prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            current = st.session_state.holdings if st.session_state.holdings is not None else st.session_state.holdings_base
            metrics = {"as_of": str(st.session_state.refresh_time), "note": "Quotes are latest available and may be delayed."}
            context = portfolio_context(current, metrics)
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing portfolio…"):
                        answer = chat(settings.groq_api_key, settings.groq_model, st.session_state.chat_history, context)
                    st.markdown(answer)
                st.session_state.chat_history.append({"role":"assistant", "content":answer})
            except Exception as exc:
                st.error(f"Groq request failed: {exc}")

