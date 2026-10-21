"""Technical indicators that can't be composed from operators.

Most indicators are one line of Allie once series support operators --
MACD is `EMA(c, 12) - EMA(c, 26)`, a Bollinger band is
`SMA(c, 20) + 2.0 * rolling_std(c, 20)` -- so they are written in
Allie, not added here (examples/strategies.qs). What is here needs
something operators can't express: a recurrence (RSI's and ATR's
Wilder smoothing), a window reduction other than the mean/std that
series_ops already has (rolling_max/rolling_min), or a reduction of a
Series<Bool> (count).

Conventions shared with series_ops: an indicator's warm-up bars are
NaN, so a comparison against them is false and a strategy can't signal
off a value that doesn't exist yet. Smoothing follows Wilder (and
TA-Lib): the first value is the plain mean of the first `period`
changes, at index `period`, then `avg = (avg * (period - 1) + x) /
period`.
"""

from __future__ import annotations

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, PriceSeriesValue, SeriesValue
from allie.stdlib.missing import over_present, rolling
from allie.stdlib.registry import register
from allie.typechecker.types import BOOL, FLOAT, INT, PRICE_SERIES, FunctionType, SeriesType

_SERIES_FLOAT = SeriesType(FLOAT)


def _wilder(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing of `values[1:]` (index 0 has no change behind
    it), aligned with `values`: NaN up to index `period`, then the
    running average. A recurrence -- each value depends on the previous
    one -- so it is a loop, like EMA (see series_ops.EMA on why the
    closed form isn't safe)."""
    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) <= period:
        return result
    average = float(np.mean(values[1 : period + 1]))
    result[period] = average
    for i in range(period + 1, len(values)):
        average = (average * (period - 1) + values[i]) / period
        result[i] = average
    return result


def _check_window(name: str, window: int) -> None:
    if window < 1:
        raise AllieValueError(f"{name}: window must be at least 1, got {window}")


def _check_period(name: str, period: int) -> None:
    if period < 1:
        raise AllieValueError(f"{name}: period must be at least 1, got {period}")


def RSI(series: SeriesValue, period: int) -> SeriesValue:  # noqa: N802 - the name traders use
    """Relative Strength Index, 0-100, with Wilder smoothing:
    `100 - 100 / (1 + average gain / average loss)`.

    NaN for the first `period` bars. A window with no losses is 100 and
    one with no movement at all is 50, rather than a division by zero.
    Missing bars are skipped (allie/stdlib/missing.py): the change across
    a gap is from the last bar with a value, and `RSI(SMA(c, 5), 14)`
    starts once the SMA has values instead of never.
    """
    _check_period("RSI", period)

    def over(values: np.ndarray) -> np.ndarray:
        changes = np.diff(values, prepend=np.nan)  # index 0 is never read: _wilder starts at 1
        gains = _wilder(np.maximum(changes, 0.0), period)
        losses = _wilder(np.maximum(-changes, 0.0), period)
        with np.errstate(divide="ignore", invalid="ignore"):
            rsi = 100.0 - 100.0 / (1.0 + gains / losses)
        rsi = np.where((losses == 0) & (gains > 0), 100.0, rsi)
        return np.where((losses == 0) & (gains == 0), 50.0, rsi)

    return SeriesValue(over_present(series.data, over))


def ATR(prices: PriceSeriesValue, period: int) -> SeriesValue:  # noqa: N802 - the name traders use
    """Average True Range with Wilder smoothing. A bar's true range is
    the largest of high - low, |high - previous close| and |low -
    previous close|, so an overnight gap counts as range. NaN for the
    first `period` bars, like RSI. A bar missing its high, low or close
    is skipped; the next bar's gap is measured from the last full one."""
    _check_period("ATR", period)
    high, low, close = prices.high.data, prices.low.data, prices.close.data
    complete = ~(np.isnan(high) | np.isnan(low) | np.isnan(close))
    h, lo, c = high[complete], low[complete], close[complete]
    previous_close = np.concatenate(([np.nan], c[:-1]))
    true_range = np.maximum(
        h - lo, np.maximum(np.abs(h - previous_close), np.abs(lo - previous_close))
    )
    result = np.full(len(close), np.nan, dtype=np.float64)
    result[complete] = _wilder(true_range, period)
    return SeriesValue(result)


def rolling_max(series: SeriesValue, window: int) -> SeriesValue:
    """Highest value of the last `window` bars, this one included --
    a Donchian channel's upper edge. NaN for the first `window - 1`
    bars, and wherever the window contains a NaN."""
    _check_window("rolling_max", window)
    return SeriesValue(rolling(series.data, window, np.max))


def rolling_min(series: SeriesValue, window: int) -> SeriesValue:
    """Lowest value of the last `window` bars; see rolling_max."""
    _check_window("rolling_min", window)
    return SeriesValue(rolling(series.data, window, np.min))


def count(signal: BoolSeriesValue) -> int:
    """How many bars are true: `count(fast crosses_above slow)` is the
    number of crossovers."""
    return int(np.count_nonzero(signal.data))


register("RSI", FunctionType(params=(_SERIES_FLOAT, INT), ret=_SERIES_FLOAT), RSI)
register("ATR", FunctionType(params=(PRICE_SERIES, INT), ret=_SERIES_FLOAT), ATR)
register("rolling_max", FunctionType(params=(_SERIES_FLOAT, INT), ret=_SERIES_FLOAT), rolling_max)
register("rolling_min", FunctionType(params=(_SERIES_FLOAT, INT), ret=_SERIES_FLOAT), rolling_min)
register("count", FunctionType(params=(SeriesType(BOOL),), ret=INT), count)
