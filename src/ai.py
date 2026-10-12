from __future__ import annotations

import json

import pandas as pd
from groq import Groq


SYSTEM_PROMPT = """You are a cautious personal portfolio analyst. Use only the supplied portfolio facts for numeric claims. Never invent quotes or holdings. Explain risks, concentration, performance, and diversification clearly. Do not present personalized investment instructions; distinguish education from advice. Keep answers concise and mention that market data may be delayed."""


def portfolio_context(holdings: pd.DataFrame, metrics: dict[str, object]) -> str:
    safe = holdings.drop(columns=[c for c in ["quote_timestamp"] if c in holdings], errors="ignore").round(4)
    return json.dumps({"metrics": metrics, "holdings": safe.where(pd.notna(safe), None).to_dict("records")}, default=str)


def chat(api_key: str, model: str, messages: list[dict[str, str]], context: str) -> str:
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(model=model, temperature=0.2, messages=[{"role": "system", "content": SYSTEM_PROMPT + "\nPortfolio data:\n" + context}, *messages])
    return completion.choices[0].message.content or "No response was returned."

