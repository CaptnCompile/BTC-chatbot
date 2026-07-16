"""Indicator correctness tests.

RSI and ATR use Wilder's smoothing, which is easy to implement subtly wrong
(seeding with the wrong window, or using a plain average throughout). The RSI
case below checks against the reference series from Wilder's "New Concepts in
Technical Trading Systems", whose first RSI(14) value is a published 70.53.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.market.feeds import Candle
from app.market.indicators import (
    analyze,
    atr,
    bollinger_width_pct,
    classify_trend,
    pct_change,
    realized_volatility,
    rsi,
    sma,
)

# Wilder's canonical RSI dataset, at the full 4dp precision the published
# values are derived from. Rounding these to 2dp shifts the first RSI to 70.46
# and no longer matches the reference.
WILDER_CLOSES = [
    44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245,
    45.8433, 46.0826, 45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0028,
    46.0328, 46.4116, 46.2222, 45.6439, 46.2122, 46.2521, 45.7137, 46.4515,
    45.7835, 45.3548, 44.0288, 44.1783, 44.2181, 44.5672, 43.4205, 42.6628,
    43.1314,
]


def _candles(rows: list[tuple[float, float, float, float]]) -> list[Candle]:
    """rows are (open, high, low, close)."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(ts=t0 + timedelta(hours=i), open=o, high=h, low=lo, close=c, volume=1.0)
        for i, (o, h, lo, c) in enumerate(rows)
    ]


class TestSMA:
    def test_simple_average(self):
        assert sma([1, 2, 3, 4, 5], 5) == 3.0

    def test_uses_only_trailing_window(self):
        assert sma([100, 100, 1, 2, 3], 3) == 2.0

    def test_none_when_insufficient_history(self):
        assert sma([1, 2], 5) is None


class TestRSI:
    def test_matches_wilder_reference_value(self):
        # First RSI(14) value in Wilder's published series is 70.53.
        assert rsi(WILDER_CLOSES[:15], period=14) == pytest.approx(70.53, abs=0.01)

    def test_matches_wilder_reference_after_smoothing_step(self):
        # Second published value is 66.32. This one only comes out right if the
        # smoothing recurrence is Wilder's; a plain rolling mean drifts here.
        assert rsi(WILDER_CLOSES[:16], period=14) == pytest.approx(66.32, abs=0.01)

    def test_all_gains_saturates_at_100(self):
        assert rsi([float(i) for i in range(1, 30)], 14) == pytest.approx(100.0)

    def test_all_losses_saturates_at_0(self):
        assert rsi([float(i) for i in range(30, 1, -1)], 14) == pytest.approx(0.0)

    def test_flat_series_is_neutral(self):
        assert rsi([50.0] * 30, 14) == 50.0

    def test_none_when_insufficient_history(self):
        assert rsi([1.0, 2.0, 3.0], 14) is None


class TestATR:
    def test_true_range_uses_prior_close_gap(self):
        # Candle 2 gaps up: its low (20) sits far above candle 1's close (10),
        # so TR must measure from the prior close (20 -> 30 = 20), not just the
        # candle's own 10-point high-low span.
        candles = _candles([(10, 10, 10, 10)] + [(20, 30, 20, 25)] * 14)
        value = atr(candles, period=14)
        assert value is not None
        assert value > 10.0

    def test_constant_range_gives_that_range(self):
        candles = _candles([(10, 12, 8, 10)] * 20)  # TR is a flat 4.0
        assert atr(candles, period=14) == pytest.approx(4.0)

    def test_none_when_insufficient_history(self):
        assert atr(_candles([(1, 2, 0.5, 1)] * 5), period=14) is None


class TestRealizedVolatility:
    def test_flat_series_has_zero_volatility(self):
        assert realized_volatility([100.0] * 50, "1h") == pytest.approx(0.0)

    def test_constant_growth_has_zero_volatility(self):
        # Identical log returns every period => zero dispersion => zero vol.
        closes = [100 * (1.01**i) for i in range(50)]
        assert realized_volatility(closes, "1h") == pytest.approx(0.0, abs=1e-9)

    def test_choppier_series_is_more_volatile(self):
        calm = [100 + (i % 2) * 0.1 for i in range(50)]
        wild = [100 + (i % 2) * 10.0 for i in range(50)]
        assert realized_volatility(wild, "1h") > realized_volatility(calm, "1h")

    def test_annualisation_scales_with_interval(self):
        # Same returns sampled hourly annualise higher than daily: more
        # periods per year => sqrt(8760) vs sqrt(365).
        closes = [100 + (i % 2) for i in range(50)]
        assert realized_volatility(closes, "1h") > realized_volatility(closes, "1d")


class TestBollingerWidth:
    def test_flat_series_has_zero_width(self):
        assert bollinger_width_pct([100.0] * 25) == pytest.approx(0.0)

    def test_wider_dispersion_widens_bands(self):
        tight = [100 + (i % 2) * 0.5 for i in range(25)]
        loose = [100 + (i % 2) * 20.0 for i in range(25)]
        assert bollinger_width_pct(loose) > bollinger_width_pct(tight)


class TestPctChange:
    def test_computes_change_over_window(self):
        assert pct_change([100.0, 110.0], 1) == pytest.approx(10.0)

    def test_none_when_window_exceeds_history(self):
        assert pct_change([100.0, 110.0], 50) is None


class TestClassifyTrend:
    def test_uptrend_when_stacked_and_rising(self):
        label, why = classify_trend(last=110, sma_20=105, sma_50=100, slope_pct=2.0)
        assert label == "uptrend"
        assert "above SMA20" in why

    def test_downtrend_when_stacked_down_and_falling(self):
        label, _ = classify_trend(last=90, sma_20=95, sma_50=100, slope_pct=-2.0)
        assert label == "downtrend"

    def test_flat_slope_overrides_structure(self):
        # Averages stacked bullishly but going nowhere is a range, not a trend.
        label, why = classify_trend(last=110, sma_20=105, sma_50=100, slope_pct=0.05)
        assert label == "range-bound / flat"
        assert "flat" in why

    def test_unknown_without_enough_history(self):
        label, _ = classify_trend(last=110, sma_20=105, sma_50=None, slope_pct=None)
        assert label == "unknown"


class TestAnalyze:
    def test_produces_full_summary_for_rising_series(self):
        rows = [(100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(80)]
        result = analyze(_candles(rows), "1h")

        assert result.interval == "1h"
        assert result.candles_analyzed == 80
        assert result.trend == "uptrend"
        assert result.rsi_14 > 90  # relentless one-way series
        assert result.sma_20 > result.sma_50  # rising series stacks bullishly
        assert result.range_high == pytest.approx(180.0)
        assert result.range_low == pytest.approx(99.0)
        # At the highs, so the gap to the range high is ~0 and to the low is large.
        assert result.pct_from_range_high == pytest.approx(0.0, abs=1.0)
        assert result.pct_from_range_low > 50
        assert "24h" in result.changes

    def test_rejects_empty_series(self):
        with pytest.raises(ValueError, match="empty candle series"):
            analyze([], "1h")
