"""LangGraph agent for the BTC assistant.

    START -> fetch_market -> agent -> (tools -> agent)* -> END

`fetch_market` runs unconditionally before the model, so the snapshot is always
in context. That is the design decision that makes answers grounded: if fetching
were a tool the model could choose to skip, "is BTC volatile today?" would
sometimes be answered from training priors instead of today's tape. The model
only needs a tool for what the snapshot *doesn't* carry — finer intraday
timeframes.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from ..config import get_settings
from ..market.indicators import analyze
from ..market.service import get_candles, get_snapshot, render_snapshot
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    market_context: str


@tool
async def get_candles_for_timeframe(interval: str, limit: int = 48) -> str:
    """Fetch and analyse candles for a timeframe not in the market data block.

    The block already covers hourly and daily. Use this only for finer intraday
    resolution than that.

    Args:
        interval: Either "5m" or "15m".
        limit: How many candles to analyse, between 20 and 200.
    """
    if interval not in ("5m", "15m"):
        return (
            f"Unsupported interval {interval!r}. This tool serves '5m' and '15m'; "
            "hourly and daily are already in the market data block."
        )
    limit = max(20, min(int(limit), 200))
    try:
        candles = await get_candles(interval, limit)
        a = analyze(candles, interval)
    except Exception as exc:
        logger.warning("tool candle fetch failed: %s", exc)
        return f"Could not fetch {interval} candles: {exc}"

    def f(v: float | None, spec: str = ",.2f") -> str:
        return f"{v:{spec}}" if v is not None else "n/a"

    changes = ", ".join(f"{k} {v:+.2f}%" for k, v in a.changes.items()) or "n/a"
    return (
        f"{interval} analysis ({a.candles_analyzed} candles):\n"
        f"  Last close: ${f(a.last_close)}\n"
        f"  Trend: {a.trend} ({a.trend_rationale})\n"
        f"  Changes: {changes}\n"
        f"  RSI(14): {f(a.rsi_14, '.1f')}\n"
        f"  ATR(14): ${f(a.atr_14)} ({f(a.atr_pct, '.2f')}% of price)\n"
        f"  Realised volatility (annualised): {f(a.realized_vol_annual_pct, '.1f')}%\n"
        f"  Range: ${f(a.range_low)} - ${f(a.range_high)}"
    )


TOOLS = [get_candles_for_timeframe]


def build_llm() -> ChatAnthropic:
    """Construct the chat model.

    Note what is deliberately absent: `temperature`. Opus 4.8 removed the
    sampling parameters and rejects them with a 400. ChatAnthropic defaults
    `temperature` to None and omits it from the payload, but setting it here
    would pass it straight through and break every request.
    """
    s = get_settings()
    if not s.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    return ChatAnthropic(
        model=s.anthropic_model,
        api_key=s.anthropic_api_key,
        max_tokens=s.llm_max_tokens,
        # Adaptive thinking is the only supported on-mode for 4.8-family models.
        # Left on deliberately: with thinking disabled, Opus 4.8 tends to spill
        # its reasoning into the visible reply, which is worse in a chat panel
        # than the small latency cost. `effort` is what keeps it snappy.
        thinking={"type": "adaptive"},
        output_config={"effort": s.llm_effort},
    )


async def fetch_market_node(state: AgentState) -> dict[str, Any]:
    """Pull the snapshot and render it into the turn's context."""
    try:
        snapshot = await get_snapshot()
        return {"market_context": render_snapshot(snapshot)}
    except Exception as exc:
        # Degrade honestly rather than answering as if we had data. The prompt
        # forbids inventing numbers, so tell the model plainly that it has none.
        logger.error("market snapshot unavailable: %s", exc)
        return {
            "market_context": (
                "=== LIVE MARKET DATA UNAVAILABLE ===\n"
                f"The market data feed could not be reached ({exc}).\n"
                "You have NO current price or indicator data for this turn. Tell the "
                "user the feed is down and that you cannot assess the market right "
                "now. Do not estimate or recall any prices. You may still answer "
                "general trading-education questions that need no live data."
            )
        }


def agent_node(llm_with_tools: Any):
    async def _agent(state: AgentState) -> dict[str, Any]:
        system = SystemMessage(
            content=f"{SYSTEM_PROMPT}\n\n{state['market_context']}"
        )
        response = await llm_with_tools.ainvoke([system, *state["messages"]])
        return {"messages": [response]}

    return _agent


def build_graph():
    """Compile the agent graph."""
    llm_with_tools = build_llm().bind_tools(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("fetch_market", fetch_market_node)
    graph.add_node("agent", agent_node(llm_with_tools))
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "fetch_market")
    graph.add_edge("fetch_market", "agent")
    # tools_condition routes to "tools" when the model emitted tool calls,
    # else to END.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


_graph = None


def get_graph():
    """Compiled graph, built lazily so import doesn't require an API key."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
