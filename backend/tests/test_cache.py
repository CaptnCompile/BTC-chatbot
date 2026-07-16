"""Cache behaviour tests.

The cache is the component the requirements care about most concretely ("so you
don't hit rate limits, add lag, or run up API costs"), so its three behaviours
are each pinned down here: TTL reuse, single-flight collapsing, and
stale-on-error.
"""

from __future__ import annotations

import asyncio

import pytest

from app.market.cache import TTLCache


@pytest.mark.asyncio
class TestTTLCache:
    async def test_fetches_on_miss(self):
        cache = TTLCache()
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            return "value"

        assert await cache.get_or_fetch("k", 60, fetch) == "value"
        assert calls == 1

    async def test_reuses_fresh_value_without_refetching(self):
        cache = TTLCache()
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            return calls

        first = await cache.get_or_fetch("k", 60, fetch)
        second = await cache.get_or_fetch("k", 60, fetch)
        assert first == second == 1
        assert calls == 1, "a fresh entry must not trigger a second upstream call"

    async def test_refetches_after_ttl_expires(self):
        cache = TTLCache()
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            return calls

        assert await cache.get_or_fetch("k", 0.05, fetch) == 1
        await asyncio.sleep(0.08)
        assert await cache.get_or_fetch("k", 0.05, fetch) == 2
        assert calls == 2

    async def test_keys_are_independent(self):
        cache = TTLCache()

        async def fetch_a():
            return "a"

        async def fetch_b():
            return "b"

        assert await cache.get_or_fetch("a", 60, fetch_a) == "a"
        assert await cache.get_or_fetch("b", 60, fetch_b) == "b"

    async def test_single_flight_collapses_concurrent_misses(self):
        """20 simultaneous cold reads must produce exactly one upstream call.

        This is the stampede guard: without it, a burst of chat messages on a
        cold cache would fan out into one feed request each.
        """
        cache = TTLCache()
        calls = 0

        async def slow_fetch():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)
            return "value"

        results = await asyncio.gather(
            *(cache.get_or_fetch("k", 60, slow_fetch) for _ in range(20))
        )

        assert results == ["value"] * 20
        assert calls == 1, f"expected 1 upstream call, got {calls}"

    async def test_serves_stale_value_when_refresh_fails(self):
        cache = TTLCache()
        state = {"fail": False}

        async def fetch():
            if state["fail"]:
                raise RuntimeError("feed is down")
            return "good"

        assert await cache.get_or_fetch("k", 0.05, fetch) == "good"

        await asyncio.sleep(0.08)  # let it go stale
        state["fail"] = True
        # Refresh fails, but a stale value beats an error for a chatbot.
        assert await cache.get_or_fetch("k", 0.05, fetch) == "good"

    async def test_raises_when_fetch_fails_with_no_stale_value(self):
        cache = TTLCache()

        async def fetch():
            raise RuntimeError("feed is down")

        with pytest.raises(RuntimeError, match="feed is down"):
            await cache.get_or_fetch("k", 60, fetch)

    async def test_peek_reports_age_without_fetching(self):
        cache = TTLCache()

        async def fetch():
            return "value"

        assert cache.peek("k") == (None, None)
        await cache.get_or_fetch("k", 60, fetch)
        value, age = cache.peek("k")
        assert value == "value"
        assert 0 <= age < 1.0
