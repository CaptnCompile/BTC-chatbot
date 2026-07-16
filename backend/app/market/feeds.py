"""Market data providers, normalised behind one interface.

Two providers are implemented and tried in order. Binance is primary: free, no
API key, deep history, generous rate limits. Coinbase is the fallback, and it
earns its place — Binance geo-blocks some regions (notably the US), so a
single-provider setup that works on a dev laptop can fail outright once
deployed. Both are public, keyless endpoints.

Every provider returns the same `Quote` / `Candle` shapes, so the rest of the
app never learns which one answered.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Protocol

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Canonical intervals. Deliberately the *intersection* of what Binance and
# Coinbase both support, so a fallback can always serve the same request.
INTERVALS: dict[str, int] = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "1d": 86400,
}


class FeedError(RuntimeError):
    """No provider could satisfy the request."""


class Candle(BaseModel):
    ts: datetime  # candle open time, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


class Quote(BaseModel):
    price: float
    change_24h_pct: float | None = None
    high_24h: float | None = None
    low_24h: float | None = None
    volume_24h: float | None = None  # base units (BTC)
    source: str = "unknown"
    as_of: datetime


class Provider(Protocol):
    name: str

    async def fetch_quote(self, client: httpx.AsyncClient) -> Quote: ...

    async def fetch_candles(
        self, client: httpx.AsyncClient, interval: str, limit: int
    ) -> list[Candle]: ...


def _utc(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


class BinanceProvider:
    """https://api.binance.com — public, keyless."""

    name = "binance"
    base = "https://api.binance.com/api/v3"
    symbol = "BTCUSDT"

    async def fetch_quote(self, client: httpx.AsyncClient) -> Quote:
        r = await client.get(f"{self.base}/ticker/24hr", params={"symbol": self.symbol})
        r.raise_for_status()
        d = r.json()
        return Quote(
            price=float(d["lastPrice"]),
            change_24h_pct=float(d["priceChangePercent"]),
            high_24h=float(d["highPrice"]),
            low_24h=float(d["lowPrice"]),
            volume_24h=float(d["volume"]),
            source=self.name,
            as_of=datetime.now(timezone.utc),
        )

    async def fetch_candles(
        self, client: httpx.AsyncClient, interval: str, limit: int
    ) -> list[Candle]:
        r = await client.get(
            f"{self.base}/klines",
            params={"symbol": self.symbol, "interval": interval, "limit": limit},
        )
        r.raise_for_status()
        # [openTime_ms, open, high, low, close, volume, closeTime, ...], oldest first.
        return [
            Candle(
                ts=_utc(row[0] / 1000),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in r.json()
        ]


class CoinbaseProvider:
    """https://api.exchange.coinbase.com — public, keyless."""

    name = "coinbase"
    base = "https://api.exchange.coinbase.com"
    product = "BTC-USD"

    async def fetch_quote(self, client: httpx.AsyncClient) -> Quote:
        # Coinbase splits this across two endpoints; /stats carries the 24h
        # window, /ticker the freshest print.
        stats_r, tick_r = await asyncio.gather(
            client.get(f"{self.base}/products/{self.product}/stats"),
            client.get(f"{self.base}/products/{self.product}/ticker"),
        )
        stats_r.raise_for_status()
        tick_r.raise_for_status()
        stats, tick = stats_r.json(), tick_r.json()

        price = float(tick.get("price") or stats["last"])
        open_24h = float(stats["open"]) if stats.get("open") else None
        change = ((price - open_24h) / open_24h * 100) if open_24h else None

        return Quote(
            price=price,
            change_24h_pct=change,
            high_24h=float(stats["high"]) if stats.get("high") else None,
            low_24h=float(stats["low"]) if stats.get("low") else None,
            volume_24h=float(stats["volume"]) if stats.get("volume") else None,
            source=self.name,
            as_of=datetime.now(timezone.utc),
        )

    async def fetch_candles(
        self, client: httpx.AsyncClient, interval: str, limit: int
    ) -> list[Candle]:
        granularity = INTERVALS[interval]
        r = await client.get(
            f"{self.base}/products/{self.product}/candles",
            params={"granularity": granularity},
        )
        r.raise_for_status()
        # Coinbase returns [time_s, low, high, open, close, volume] — note the
        # column order differs from Binance — and sorts newest-first, capped at
        # 300 rows. Re-sort ascending and take the most recent `limit`.
        rows = sorted(r.json(), key=lambda row: row[0])
        candles = [
            Candle(
                ts=_utc(row[0]),
                open=float(row[3]),
                high=float(row[2]),
                low=float(row[1]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in rows
        ]
        return candles[-limit:]


PROVIDERS: list[Provider] = [BinanceProvider(), CoinbaseProvider()]


async def fetch_quote(timeout: float) -> Quote:
    """First provider to answer wins."""
    errors: list[str] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for provider in PROVIDERS:
            try:
                return await provider.fetch_quote(client)
            except Exception as exc:
                errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                logger.warning("quote fetch failed via %s: %s", provider.name, exc)
    raise FeedError(f"all providers failed for quote — {'; '.join(errors)}")


async def fetch_candles(interval: str, limit: int, timeout: float) -> list[Candle]:
    if interval not in INTERVALS:
        raise ValueError(
            f"unsupported interval {interval!r}; expected one of {sorted(INTERVALS)}"
        )

    errors: list[str] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for provider in PROVIDERS:
            try:
                candles = await provider.fetch_candles(client, interval, limit)
                if candles:
                    return candles
                errors.append(f"{provider.name}: returned no candles")
            except Exception as exc:
                errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                logger.warning("candle fetch failed via %s: %s", provider.name, exc)
    raise FeedError(
        f"all providers failed for {interval} candles — {'; '.join(errors)}"
    )
