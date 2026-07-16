"""Assembles the market snapshot that grounds every answer.

All feed access goes through the cache, so a burst of chat messages costs at
most one upstream request per key per TTL window.

The snapshot deliberately spans two timeframes. "Is BTC volatile today?" is an
intraday question and "what's the trend?" usually isn't, so answering both well
needs hourly *and* daily context in front of the model at once.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from ..config import get_settings
from .cache import market_cache
from .feeds import Candle, Quote, fetch_candles, fetch_quote
from .indicators import TimeframeAnalysis, analyze

# 7 days of hourly, 90 days of daily. Both must exceed 50 candles or SMA50 is
# None and the trend classifier can only answer "unknown" — 30 daily candles
# looks like "a month of history" but silently disables the daily trend read
# and the 30d change. Depth is free here: it is the same single request.
HOURLY_LOOKBACK = 168
DAILY_LOOKBACK = 90


@dataclass
class MarketSnapshot:
    symbol: str
    quote: Quote
    timeframes: dict[str, TimeframeAnalysis]
    recent_hourly: list[Candle]
    generated_at: datetime


async def get_quote() -> Quote:
    s = get_settings()
    return await market_cache.get_or_fetch(
        "quote",
        s.price_ttl_seconds,
        lambda: fetch_quote(s.feed_timeout_seconds),
    )


async def get_candles(interval: str, limit: int) -> list[Candle]:
    s = get_settings()
    return await market_cache.get_or_fetch(
        f"candles:{interval}:{limit}",
        s.candles_ttl_seconds,
        lambda: fetch_candles(interval, limit, s.feed_timeout_seconds),
    )


async def get_snapshot() -> MarketSnapshot:
    """Live price + hourly and daily analysis, all cache-backed."""
    s = get_settings()

    quote, hourly, daily = await asyncio.gather(
        get_quote(),
        get_candles("1h", HOURLY_LOOKBACK),
        get_candles("1d", DAILY_LOOKBACK),
    )

    return MarketSnapshot(
        symbol=f"{s.symbol}/{s.quote}",
        quote=quote,
        timeframes={
            "1h": analyze(hourly, "1h"),
            "1d": analyze(daily, "1d"),
        },
        recent_hourly=hourly[-24:],
        generated_at=datetime.now(timezone.utc),
    )


def _fmt(value: float | None, spec: str = ",.2f", suffix: str = "") -> str:
    return f"{value:{spec}}{suffix}" if value is not None else "n/a"


def _render_timeframe(a: TimeframeAnalysis) -> str:
    label = {"1h": "HOURLY (last 7 days)", "1d": "DAILY (last 90 days)"}.get(
        a.interval, a.interval
    )
    changes = ", ".join(f"{k} {v:+.2f}%" for k, v in a.changes.items()) or "n/a"

    return "\n".join(
        [
            f"{label} — {a.candles_analyzed} candles",
            f"  Trend: {a.trend} ({a.trend_rationale})",
            f"  Changes: {changes}",
            f"  SMA20: ${_fmt(a.sma_20)}   SMA50: ${_fmt(a.sma_50)}",
            f"  RSI(14): {_fmt(a.rsi_14, '.1f')}",
            f"  ATR(14): ${_fmt(a.atr_14)} ({_fmt(a.atr_pct, '.2f', '%')} of price)",
            f"  Realised volatility (annualised): {_fmt(a.realized_vol_annual_pct, '.1f', '%')}",
            f"  Bollinger width: {_fmt(a.bollinger_width_pct, '.2f', '%')} of price",
            f"  Range: ${_fmt(a.range_low)} – ${_fmt(a.range_high)}"
            f"  (now {_fmt(a.pct_from_range_high, '+.2f', '%')} from high,"
            f" {_fmt(a.pct_from_range_low, '+.2f', '%')} from low)",
        ]
    )


def render_snapshot(snap: MarketSnapshot) -> str:
    """Render the snapshot as the text block handed to the model.

    Plain labelled text rather than JSON: it reads unambiguously, costs fewer
    tokens than pretty-printed JSON, and there is no schema for the model to
    misread.
    """
    q = snap.quote
    lines = [
        f"=== LIVE MARKET DATA: {snap.symbol} ===",
        f"Retrieved: {snap.generated_at:%Y-%m-%d %H:%M:%S} UTC (source: {q.source})",
        "",
        f"Price: ${q.price:,.2f}",
        f"24h change: {_fmt(q.change_24h_pct, '+.2f', '%')}",
        f"24h range: ${_fmt(q.low_24h)} – ${_fmt(q.high_24h)}",
        f"24h volume: {_fmt(q.volume_24h, ',.0f')} BTC",
        "",
        _render_timeframe(snap.timeframes["1h"]),
        "",
        _render_timeframe(snap.timeframes["1d"]),
        "",
        "RECENT HOURLY CLOSES (oldest to newest, UTC):",
    ]

    lines.extend(
        f"  {c.ts:%m-%d %H:%M}  O={c.open:>9,.2f}  H={c.high:>9,.2f}"
        f"  L={c.low:>9,.2f}  C={c.close:>9,.2f}"
        for c in snap.recent_hourly
    )
    return "\n".join(lines)
