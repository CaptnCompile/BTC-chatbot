"""Technical indicators computed over recent candles.

These exist so the model reasons over *history* rather than the latest tick. We
compute the numbers deterministically here and hand the model a structured
summary; the model's job is interpretation and explanation, not arithmetic.
That split keeps answers grounded — an LLM asked to eyeball 168 candles and
estimate volatility will confabulate a plausible number.

Pure stdlib, no numpy: the series are a few hundred points at most.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from .feeds import Candle

# Candles per year, used to annualise realised volatility.
PERIODS_PER_YEAR: dict[str, float] = {
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "1h": 365 * 24,
    "1d": 365,
}


def sma(values: list[float], period: int) -> float | None:
    if len(values) < period or period <= 0:
        return None
    return sum(values[-period:]) / period


def rsi(closes: list[float], period: int = 14) -> float | None:
    """Relative Strength Index using Wilder's smoothing."""
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    # Seed with a simple average of the first `period` deltas, then smooth.
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def true_ranges(candles: list[Candle]) -> list[float]:
    out: list[float] = []
    for i in range(1, len(candles)):
        c, prev = candles[i], candles[i - 1]
        out.append(
            max(
                c.high - c.low,
                abs(c.high - prev.close),
                abs(c.low - prev.close),
            )
        )
    return out


def atr(candles: list[Candle], period: int = 14) -> float | None:
    """Average True Range using Wilder's smoothing."""
    trs = true_ranges(candles)
    if len(trs) < period:
        return None
    value = sum(trs[:period]) / period
    for tr in trs[period:]:
        value = (value * (period - 1) + tr) / period
    return value


def log_returns(closes: list[float]) -> list[float]:
    return [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0 and closes[i] > 0
    ]


def realized_volatility(closes: list[float], interval: str) -> float | None:
    """Annualised realised volatility, in percent."""
    rets = log_returns(closes)
    if len(rets) < 2:
        return None
    per_year = PERIODS_PER_YEAR.get(interval)
    if per_year is None:
        return None
    return statistics.stdev(rets) * math.sqrt(per_year) * 100.0


def bollinger_width_pct(closes: list[float], period: int = 20, k: float = 2.0) -> float | None:
    """Bollinger band width as a percent of the middle band.

    A useful regime read: narrow bands mean compression/consolidation, wide
    bands mean an active, volatile market.
    """
    if len(closes) < period:
        return None
    window = closes[-period:]
    mid = sum(window) / period
    if mid == 0:
        return None
    sd = statistics.pstdev(window)
    return (2 * k * sd) / mid * 100.0


def pct_change(closes: list[float], periods_back: int) -> float | None:
    if len(closes) <= periods_back or periods_back <= 0:
        return None
    old = closes[-1 - periods_back]
    if old == 0:
        return None
    return (closes[-1] - old) / old * 100.0


@dataclass
class TimeframeAnalysis:
    """Everything we derive from one candle series."""

    interval: str
    candles_analyzed: int
    last_close: float
    sma_20: float | None = None
    sma_50: float | None = None
    rsi_14: float | None = None
    atr_14: float | None = None
    atr_pct: float | None = None
    realized_vol_annual_pct: float | None = None
    bollinger_width_pct: float | None = None
    range_high: float | None = None
    range_low: float | None = None
    pct_from_range_high: float | None = None
    pct_from_range_low: float | None = None
    changes: dict[str, float] = field(default_factory=dict)
    trend: str = "unknown"
    trend_rationale: str = ""


def classify_trend(
    last: float,
    sma_20: float | None,
    sma_50: float | None,
    slope_pct: float | None,
) -> tuple[str, str]:
    """Label the trend from moving-average structure plus slope.

    Deliberately coarse and explainable — the model gets the label *and* the
    reason, so it can hedge appropriately rather than treating a label as
    gospel.
    """
    if sma_20 is None or sma_50 is None:
        return "unknown", "not enough history for a 50-period average"

    above_20 = last > sma_20
    above_50 = last > sma_50
    stacked_up = sma_20 > sma_50
    bits = [
        f"price {'above' if above_20 else 'below'} SMA20",
        f"{'above' if above_50 else 'below'} SMA50",
        f"SMA20 {'above' if stacked_up else 'below'} SMA50",
    ]
    if slope_pct is not None:
        bits.append(f"SMA20 slope {slope_pct:+.2f}% over last 10 periods")
    rationale = ", ".join(bits)

    if above_20 and above_50 and stacked_up:
        label = "uptrend"
    elif not above_20 and not above_50 and not stacked_up:
        label = "downtrend"
    elif stacked_up:
        label = "choppy / pulling back within an uptrend"
    else:
        label = "choppy / bouncing within a downtrend"

    # A flat slope overrides the structural read: stacked averages with no
    # movement is a range, not a trend.
    if slope_pct is not None and abs(slope_pct) < 0.25 and label in ("uptrend", "downtrend"):
        label = "range-bound / flat"
        rationale += " — slope is flat, so treating as range-bound"

    return label, rationale


def analyze(candles: list[Candle], interval: str) -> TimeframeAnalysis:
    """Compute the full indicator set for one timeframe."""
    if not candles:
        raise ValueError("cannot analyze an empty candle series")

    closes = [c.close for c in candles]
    last = closes[-1]

    s20 = sma(closes, 20)
    s50 = sma(closes, 50)
    a14 = atr(candles, 14)

    # SMA20 slope: compare the current SMA20 against the SMA20 as of 10
    # candles ago, as a percentage. Captures direction *and* pace.
    slope_pct = None
    if len(closes) >= 30:
        prev_s20 = sma(closes[:-10], 20)
        if prev_s20:
            slope_pct = (s20 - prev_s20) / prev_s20 * 100.0 if s20 else None

    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    range_high, range_low = max(highs), min(lows)

    trend, rationale = classify_trend(last, s20, s50, slope_pct)

    # Window sizes chosen per interval so the labels mean what they say.
    change_windows = {
        "1h": {"1h": 1, "6h": 6, "24h": 24, "7d": 168},
        "1d": {"1d": 1, "7d": 7, "30d": 30},
        "15m": {"15m": 1, "1h": 4, "4h": 16, "24h": 96},
        "5m": {"5m": 1, "1h": 12, "4h": 48},
    }.get(interval, {})

    changes = {}
    for label, back in change_windows.items():
        v = pct_change(closes, back)
        if v is not None:
            changes[label] = round(v, 2)

    return TimeframeAnalysis(
        interval=interval,
        candles_analyzed=len(candles),
        last_close=last,
        sma_20=s20,
        sma_50=s50,
        rsi_14=rsi(closes, 14),
        atr_14=a14,
        atr_pct=(a14 / last * 100.0) if (a14 and last) else None,
        realized_vol_annual_pct=realized_volatility(closes, interval),
        bollinger_width_pct=bollinger_width_pct(closes),
        range_high=range_high,
        range_low=range_low,
        pct_from_range_high=((last - range_high) / range_high * 100.0) if range_high else None,
        pct_from_range_low=((last - range_low) / range_low * 100.0) if range_low else None,
        changes=changes,
        trend=trend,
        trend_rationale=rationale,
    )
