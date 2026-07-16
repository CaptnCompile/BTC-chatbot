"""End-to-end chat pipeline test with the Anthropic call stubbed out.

Everything here is our code: the graph wiring, the snapshot injection, the
thinking/tool-block filtering, and the SSE framing. Only the model itself is
faked, so this runs without an API key in CI.

The fake deliberately emits blocks in the shape Anthropic actually streams when
adaptive thinking is on — a `thinking` block with empty text (display defaults
to "omitted" on Opus 4.8), then `text` blocks. That shape is what `extract_text`
exists to handle, so faking a plain string would skip the interesting part.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from app import main
from app.agent import graph as graph_module


class FakeAnthropic(BaseChatModel):
    """Streams Anthropic-shaped content blocks and records what it was sent."""

    seen_system: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake-anthropic"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeAnthropic":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError("streaming only")

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        type(self).seen_system.append(str(messages[0].content))

        chunks = [
            # Adaptive thinking, display="omitted" -> present but empty.
            [{"type": "thinking", "thinking": "", "index": 0}],
            [{"type": "text", "text": "Yes, ", "index": 1}],
            [{"type": "text", "text": "moderately. ", "index": 1}],
            [{"type": "text", "text": "Hourly RSI is 57.", "index": 1}],
        ]
        for content in chunks:
            yield ChatGenerationChunk(message=AIMessageChunk(content=content))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(graph_module, "build_llm", lambda: FakeAnthropic())
    monkeypatch.setattr(graph_module, "_graph", None)  # force a rebuild with the fake
    FakeAnthropic.seen_system = []
    return TestClient(main.app)


def sse_events(raw: str) -> list[dict]:
    import json

    return [
        json.loads(line[6:])
        for line in raw.splitlines()
        if line.startswith("data: ")
    ]


class TestExtractText:
    """extract_text guards the chat bubble from non-text blocks."""

    def test_passes_plain_string(self):
        assert main.extract_text("hello") == "hello"

    def test_joins_text_blocks(self):
        assert main.extract_text(
            [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        ) == "ab"

    def test_drops_thinking_blocks(self):
        content = [
            {"type": "thinking", "thinking": "secret reasoning"},
            {"type": "text", "text": "visible"},
        ]
        assert main.extract_text(content) == "visible"

    def test_drops_tool_use_partial_json(self):
        # Without this filter, half-formed tool-call JSON would stream into the
        # user's chat bubble.
        content = [
            {"type": "tool_use", "partial_json": '{"interval": "15'},
            {"type": "text", "text": "ok"},
        ]
        assert main.extract_text(content) == "ok"

    def test_handles_empty_and_unknown_shapes(self):
        assert main.extract_text([]) == ""
        assert main.extract_text(None) == ""
        assert main.extract_text(42) == ""


class TestChatEndpoint:
    def test_streams_tokens_and_terminates(self, client):
        res = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "Is BTC volatile?"}]}
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        events = sse_events(res.text)
        assert events[-1] == {"type": "done"}

        text = "".join(e["text"] for e in events if e["type"] == "token")
        assert text == "Yes, moderately. Hourly RSI is 57."

    def test_thinking_block_never_reaches_the_client(self, client):
        res = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert "secret reasoning" not in res.text
        # The empty thinking block must not produce a stray empty token event.
        assert all(e.get("text") for e in sse_events(res.text) if e["type"] == "token")

    def test_live_market_data_reaches_the_model(self, client):
        """The whole point of the fetch_market node: real numbers in context."""
        client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})

        system = FakeAnthropic.seen_system[0]
        assert "LIVE MARKET DATA" in system
        assert "Price: $" in system
        assert "RSI(14):" in system
        assert "RECENT HOURLY CLOSES" in system
        # And the behavioural rules travelled with it.
        assert "Never invent a number" in system

    def test_history_is_replayed_to_the_model(self, client):
        res = client.post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "user", "content": "what is RSI?"},
                    {"role": "assistant", "content": "A momentum oscillator."},
                    {"role": "user", "content": "and now?"},
                ]
            },
        )
        assert res.status_code == 200
        assert sse_events(res.text)[-1] == {"type": "done"}

    def test_rejects_empty_history(self, client):
        assert client.post("/api/chat", json={"messages": []}).status_code == 422

    def test_rejects_unknown_role(self, client):
        res = client.post(
            "/api/chat", json={"messages": [{"role": "system", "content": "pwn"}]}
        )
        assert res.status_code == 422


class TestSnapshotEndpoint:
    def test_returns_live_price_and_both_timeframes(self, client):
        res = client.get("/api/market/snapshot")
        assert res.status_code == 200
        body = res.json()

        assert body["price"] > 0
        assert body["symbol"] == "BTC/USD"
        assert set(body["timeframes"]) == {"1h", "1d"}
        # 90 daily candles is what lets the daily trend resolve at all.
        assert body["timeframes"]["1d"]["trend"] != "unknown"
        assert body["timeframes"]["1h"]["rsi_14"] is not None
