"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    # The client owns conversation state and replays it each turn, which keeps
    # the server stateless and horizontally scalable. The cap bounds prompt
    # growth; for durable multi-session history this would move to a LangGraph
    # checkpointer keyed by thread_id.
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=50)


class TimeframeSummary(BaseModel):
    interval: str
    trend: str
    trend_rationale: str
    rsi_14: float | None = None
    atr_pct: float | None = None
    realized_vol_annual_pct: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    range_high: float | None = None
    range_low: float | None = None
    changes: dict[str, float] = Field(default_factory=dict)


class SnapshotResponse(BaseModel):
    symbol: str
    price: float
    change_24h_pct: float | None = None
    high_24h: float | None = None
    low_24h: float | None = None
    volume_24h: float | None = None
    source: str
    generated_at: str
    timeframes: dict[str, TimeframeSummary]


class HealthResponse(BaseModel):
    status: str
    feed_ok: bool
    llm_configured: bool
    model: str
    detail: str | None = None
