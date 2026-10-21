"""allie/stdlib/indicators.py against plain-Python reference
implementations of the textbook definitions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, PriceSeriesValue, SeriesValue
from allie.stdlib.indicators import ATR, RSI, count, rolling_max, rolling_min

# 33 daily closes: long enough for a 14-period RSI to warm up and then
# move through both halves of its range.
_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89,
    46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25,
    45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57, 43.42, 42.66, 43.13,
]  # fmt: skip


def _series(values: list[float]) -> SeriesValue:
    return SeriesValue(np.array(values, dtype=np.float64))


def _reference_rsi(closes: list[float], period: int) -> list[float]:
    """Wilder's definition, one bar at a time."""
    changes = [b - a for a, b in zip(closes, closes[1:], strict=False)]
    gain = sum(max(x, 0.0) for x in changes[:period]) / period
    loss = sum(max(-x, 0.0) for x in changes[:period]) / period
    out = [100 - 100 / (1 + gain / loss)]
    for x in changes[period:]:
        gain = (gain * (period - 1) + max(x, 0.0)) / period
        loss = (loss * (period - 1) + max(-x, 0.0)) / period
        out.append(100 - 100 / (1 + gain / loss))
    return out


def test_rsi_matches_wilders_definition() -> None:
    rsi = RSI(_series(_CLOSES), 14).data
    assert np.isnan(rsi[:14]).all()
    np.testing.assert_allclose(rsi[14:], _reference_rsi(_CLOSES, 14), rtol=0, atol=1e-10)
    assert rsi[14] > 70 > 50 > rsi[-1] > 30  # overbought at the start, weak at the end


def test_rsi_without_losses_is_100_and_without_movement_is_50() -> None:
    np.testing.assert_array_equal(
        RSI(_series([1.0, 2.0, 3.0, 4.0]), 2).data, [np.nan, np.nan, 100.0, 100.0]
    )
    np.testing.assert_array_equal(
        RSI(_series([3.0, 3.0, 3.0, 3.0]), 2).data, [np.nan, np.nan, 50.0, 50.0]
    )


def test_rsi_of_a_warming_up_indicator_starts_after_its_warm_up() -> None:
    # RSI(SMA(c, 3), 2): the SMA's two NaNs delay RSI, they don't erase it.
    prices = [1.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0]
    smoothed = [np.nan, np.nan, *prices[2:]]
    rsi = RSI(_series(smoothed), 2).data
    assert np.isnan(rsi[:4]).all()
    np.testing.assert_allclose(rsi[4:], _reference_rsi(prices[2:], 2))


def test_rsi_skips_a_missing_bar_instead_of_losing_everything_after_it() -> None:
    # The change across the gap is from the last bar with a value.
    with_gap = RSI(_series([1.0, 2.0, 3.0, np.nan, 2.5, 4.0, 3.0]), 2).data
    without = RSI(_series([1.0, 2.0, 3.0, 2.5, 4.0, 3.0]), 2).data
    assert np.isnan(with_gap[3])
    np.testing.assert_allclose(np.delete(with_gap, 3), without, equal_nan=True)
    assert not np.isnan(with_gap[-1])


def test_atr_skips_a_bar_missing_part_of_its_prices() -> None:
    complete = _prices([10.0, 11.0, 12.0, 13.0], [9.0, 10.0, 11.0, 12.0], [9.5, 10.5, 11.5, 12.5])
    gappy = _prices(
        [10.0, 11.0, np.nan, 12.0, 13.0],
        [9.0, 10.0, 10.5, 11.0, 12.0],
        [9.5, 10.5, 11.0, 11.5, 12.5],
    )
    result = ATR(gappy, 2).data
    assert np.isnan(result[2])
    np.testing.assert_allclose(np.delete(result, 2), ATR(complete, 2).data, equal_nan=True)


def test_rsi_shorter_than_its_period_is_all_warm_up() -> None:
    assert np.isnan(RSI(_series([1.0, 2.0]), 14).data).all()
    assert len(RSI(_series([]), 14)) == 0


def _prices(high: list[float], low: list[float], close: list[float]) -> PriceSeriesValue:
    return PriceSeriesValue(
        open=_series(close),
        high=_series(high),
        low=_series(low),
        close=_series(close),
        volume=_series([0.0] * len(close)),
    )


def test_atr_counts_gaps_as_range_and_uses_wilder_smoothing() -> None:
    high = [10.0, 11.0, 15.0, 12.0, 13.0]
    low = [9.0, 10.0, 14.0, 11.0, 12.5]
    close = [9.5, 10.5, 14.5, 11.5, 13.0]
    # True ranges from bar 1: max(h-l, |h-prev c|, |l-prev c|).
    true_ranges = [
        max(h - lo, abs(h - pc), abs(lo - pc))
        for h, lo, pc in zip(high[1:], low[1:], close[:-1], strict=True)
    ]
    assert true_ranges == [1.5, 4.5, 3.5, 1.5]  # bar 2 gapped up 4.5 from 10.5
    first = (true_ranges[0] + true_ranges[1]) / 2
    expected = [np.nan, np.nan, first]
    for tr in true_ranges[2:]:
        expected.append((expected[-1] * 1 + tr) / 2)
    np.testing.assert_allclose(ATR(_prices(high, low, close), 2).data, expected)


def test_rolling_max_and_min_include_the_current_bar() -> None:
    values = [1.0, 3.0, 2.0, 5.0, 4.0, 0.5]
    for fn, reduce in ((rolling_max, max), (rolling_min, min)):
        result = fn(_series(values), 3).data
        expected = [np.nan, np.nan] + [reduce(values[i - 2 : i + 1]) for i in range(2, 6)]
        np.testing.assert_array_equal(result, expected)


def test_a_missing_value_only_poisons_the_windows_it_is_in() -> None:
    result = rolling_max(_series([1.0, np.nan, 2.0, 3.0, 4.0]), 2).data
    assert [math.isnan(x) for x in result] == [True, True, True, False, False]


def test_count_is_the_number_of_true_bars() -> None:
    assert count(BoolSeriesValue(np.array([True, False, True, True]))) == 3
    assert count(BoolSeriesValue(np.array([], dtype=bool))) == 0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: RSI(_series(_CLOSES), 0), "RSI: period must be at least 1, got 0"),
        (lambda: ATR(_prices([1.0], [1.0], [1.0]), -2), "ATR: period must be at least 1, got -2"),
        (lambda: rolling_max(_series(_CLOSES), 0), "rolling_max: window must be at least 1"),
        (lambda: rolling_min(_series(_CLOSES), -1), "rolling_min: window must be at least 1"),
    ],
)
def test_bad_periods_are_errors(call: object, message: str) -> None:
    with pytest.raises(AllieValueError, match=message):
        call()  # type: ignore[operator]
