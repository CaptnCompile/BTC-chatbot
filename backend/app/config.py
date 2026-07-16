"""Application settings, loaded from the environment / .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---------------------------------------------------------------
    anthropic_api_key: str = ""

    # Opus 4.8 is the default. `claude-sonnet-5` is a cheaper/faster swap that
    # needs no code change — both take the same request surface (adaptive
    # thinking, no sampling params).
    anthropic_model: str = "claude-opus-4-8"

    # Thinking depth / token spend. "low" keeps the chat responsive; the market
    # snapshot is precomputed, so the model interprets rather than derives.
    # Raise to "medium"/"high" for more thorough reasoning at higher latency.
    llm_effort: str = "low"
    llm_max_tokens: int = 2048

    # --- Market data -------------------------------------------------------
    symbol: str = "BTC"
    quote: str = "USD"

    # Cache TTLs, in seconds. The feed is the rate-limited/latency-bound
    # resource, so every read goes through the cache.
    price_ttl_seconds: float = 10.0
    candles_ttl_seconds: float = 60.0

    feed_timeout_seconds: float = 8.0

    # --- Server ------------------------------------------------------------
    # Vite dev server origins allowed to call this API.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
