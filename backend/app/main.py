"""FastAPI service: market data endpoints + streaming chat."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from .agent.graph import get_graph
from .config import get_settings
from .market.service import get_snapshot
from .schemas import (
    ChatRequest,
    HealthResponse,
    SnapshotResponse,
    TimeframeSummary,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="BTC Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- helpers ---------------------------------------------------------------


def extract_text(content: Any) -> str:
    """Pull only user-facing text out of a streamed chunk.

    Necessary because adaptive thinking is on: a chunk's content is a list of
    blocks that may be `thinking` (empty text, since display defaults to
    "omitted") or `tool_use` (partial JSON) as well as `text`. Concatenating
    the list blindly would leak tool-call JSON fragments into the chat bubble.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text", ""))
        return "".join(out)
    return ""


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# --- routes ----------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    s = get_settings()
    feed_ok, detail = True, None
    try:
        await get_snapshot()
    except Exception as exc:
        feed_ok, detail = False, str(exc)

    return HealthResponse(
        status="ok" if feed_ok else "degraded",
        feed_ok=feed_ok,
        llm_configured=bool(s.anthropic_api_key),
        model=s.anthropic_model,
        detail=detail,
    )


@app.get("/api/market/snapshot", response_model=SnapshotResponse)
async def market_snapshot() -> SnapshotResponse:
    """Current price + indicator summary. Powers the ticker strip in the UI."""
    snap = await get_snapshot()
    return SnapshotResponse(
        symbol=snap.symbol,
        price=snap.quote.price,
        change_24h_pct=snap.quote.change_24h_pct,
        high_24h=snap.quote.high_24h,
        low_24h=snap.quote.low_24h,
        volume_24h=snap.quote.volume_24h,
        source=snap.quote.source,
        generated_at=snap.generated_at.isoformat(),
        timeframes={
            key: TimeframeSummary(
                interval=a.interval,
                trend=a.trend,
                trend_rationale=a.trend_rationale,
                rsi_14=a.rsi_14,
                atr_pct=a.atr_pct,
                realized_vol_annual_pct=a.realized_vol_annual_pct,
                sma_20=a.sma_20,
                sma_50=a.sma_50,
                range_high=a.range_high,
                range_low=a.range_low,
                changes=a.changes,
            )
            for key, a in snap.timeframes.items()
        },
    )


async def chat_stream(request: ChatRequest) -> AsyncIterator[str]:
    """Run the graph and stream assistant tokens as Server-Sent Events."""
    history = [
        HumanMessage(content=m.content)
        if m.role == "user"
        else AIMessage(content=m.content)
        for m in request.messages
    ]

    try:
        graph = get_graph()
    except Exception as exc:
        logger.error("graph unavailable: %s", exc)
        yield sse({"type": "error", "message": str(exc)})
        return

    try:
        # stream_mode="messages" yields (chunk, metadata) per token.
        async for chunk, metadata in graph.astream(
            {"messages": history, "market_context": ""},
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node")

            if node == "agent":
                # Surface tool calls so the UI can show what it's doing rather
                # than sitting silent during a fetch.
                for call in getattr(chunk, "tool_calls", None) or []:
                    if call.get("name"):
                        yield sse({"type": "tool", "name": call["name"]})

                text = extract_text(chunk.content)
                if text:
                    yield sse({"type": "token", "text": text})

        yield sse({"type": "done"})

    except asyncio.CancelledError:
        # Client navigated away or aborted the fetch; not an error.
        logger.info("chat stream cancelled by client")
        raise
    except Exception as exc:
        logger.exception("chat stream failed")
        yield sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        chat_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this, nginx buffers the whole response and the stream
            # arrives as one lump — defeating the point of streaming.
            "X-Accel-Buffering": "no",
        },
    )
