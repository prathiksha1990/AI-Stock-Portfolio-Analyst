from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None = os.getenv("GROQ_API_KEY") or None
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    market_data_ttl_seconds: int = int(os.getenv("MARKET_DATA_TTL_SECONDS", "900"))


settings = Settings()

