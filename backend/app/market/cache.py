"""Async TTL cache sitting in front of the market data feed.

Three behaviours matter here, and they map to the three reasons the cache
exists at all:

* **TTL** — a fresh value is reused until it expires, so a burst of chat
  messages costs one upstream request instead of one per message.
* **Single-flight** — concurrent misses for the same key wait on one in-flight
  fetch rather than stampeding the feed. Without this, N simultaneous users
  asking a question on a cold cache means N identical upstream calls.
* **Stale-on-error** — if a refresh fails but we still hold an expired value,
  serve the stale value rather than failing the request. A chatbot answering
  from data 90 seconds old beats a chatbot returning an error.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    value: T
    stored_at: float

    def age(self, now: float) -> float:
        return now - self.stored_at


class TTLCache:
    """Async cache keyed by string, with per-key single-flight locking."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry[Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def _lock_for(self, key: str) -> asyncio.Lock:
        # Guard the lock registry itself so two coroutines racing on a cold key
        # can't each create a *different* lock and both proceed to fetch.
        async with self._guard:
            return self._locks.setdefault(key, asyncio.Lock())

    def peek(self, key: str) -> tuple[Any | None, float | None]:
        """Return (value, age_seconds) without fetching. Both None if absent."""
        entry = self._entries.get(key)
        if entry is None:
            return None, None
        return entry.value, entry.age(time.monotonic())

    async def get_or_fetch(
        self,
        key: str,
        ttl: float,
        fetch: Callable[[], Awaitable[T]],
    ) -> T:
        """Return a cached value newer than `ttl`, else fetch and store it.

        Raises whatever `fetch` raises, but only if there is no stale value to
        fall back on.
        """
        now = time.monotonic()
        entry = self._entries.get(key)
        if entry is not None and entry.age(now) < ttl:
            return entry.value

        lock = await self._lock_for(key)
        async with lock:
            # Re-check: while we waited for the lock, the coroutine that held it
            # may have already refreshed this key. This is what collapses a
            # stampede into a single upstream call.
            now = time.monotonic()
            entry = self._entries.get(key)
            if entry is not None and entry.age(now) < ttl:
                return entry.value

            try:
                value = await fetch()
            except Exception as exc:
                if entry is not None:
                    logger.warning(
                        "refresh failed for %s (%s: %s); serving stale value %.1fs old",
                        key,
                        type(exc).__name__,
                        exc,
                        entry.age(time.monotonic()),
                    )
                    return entry.value
                raise

            self._entries[key] = _Entry(value=value, stored_at=time.monotonic())
            return value

    def clear(self) -> None:
        self._entries.clear()


# Process-wide cache. Single-process uvicorn is the deployment target; for a
# multi-worker or multi-instance deploy this would move to Redis behind the
# same get_or_fetch interface.
market_cache = TTLCache()
