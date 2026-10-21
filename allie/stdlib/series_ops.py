"""Time-series functions: moving averages, rolling statistics, shifts and
differences -- each one series in, one series out, one value per bar.

All of them are *trailing*: the value at bar i is computed from bar i and
earlier bars only, so a `when` condition may use them freely
(allie/semantic/lookahead.py). `shift` with a negative period is the one
way to break that, and the look-ahead analysis rejects it in a condition.
Missing bars follow allie/stdlib/missing.py: a window holding one is
missing, and only that window.

`crosses_above`/`crosses_below` used to be builtins here too. They are
operators now (`fast crosses_above slow`, `rsi crosses_below 30.0`),
implemented with the other series operators in allie/runtime/
elementwise.py. Indicators that need more than operators (RSI, ATR,
rolling_max/min) are in allie/stdlib/indicators.py.

`rolling` as a generic higher-order function (window, fn) isn't built:
Allie has no first-class function values usable as arguments from
source. The `rolling_*` functions are the concrete instantiations.
"""

from __future__ import annotations

import functools

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import SeriesValue
from allie.stdlib.missing import over_present, rolling, windows
from allie.stdlib.registry import register
from allie.typechecker.types import FLOAT, INT, FunctionType, SeriesType

_SERIES_FLOAT = SeriesType(FLOAT)


def _check_window(fn_name: str, window: int, at_least: int = 1) -> None:
    if window < at_least:
        if at_least == 1:
            raise AllieValueError(f"{fn_name}: window must be positive, got {window}")
        raise AllieValueError(f"{fn_name}: window must be at least {at_least}, got {window}")


def SMA(series: SeriesValue, window: int) -> SeriesValue:  # noqa: N802 - matches spec's exact name
    """Simple moving average. The first `window - 1` bars are NaN,
    matching pandas' `.rolling(window).mean()`.

    Computed per window rather than from a running sum: a running sum
    carries a single missing bar into every later window, which made one
    NaN erase the whole rest of the series."""
    _check_window("SMA", window)
    return SeriesValue(rolling(series.data, window, np.mean))


def EMA(series: SeriesValue, span: int) -> SeriesValue:  # noqa: N802 - matches spec's exact name
    """Exponential moving average, smoothing factor alpha = 2/(span+1),
    seeded with the first value. Missing bars are skipped -- the
    average runs over the bars that have a value -- so `EMA(SMA(c, 20),
    9)` starts where the SMA does instead of being NaN throughout.

    **This loop is deliberate -- do not "vectorize" it.** EMA is a
    first-order recurrence (`y[i] = a*x[i] + (1-a)*y[i-1]`), and the
    closed form that lets numpy do it in one pass,
    `y[n] = (1-a)**n * cumsum(a*x[i] / (1-a)**i)`, divides by
    `(1-a)**i`. That underflows to zero for large `i`: measured against
    a 10,000-element series with span=20, results go non-finite from
    index 7043 onward. It looks ~9x faster right up until it silently
    returns garbage on real-length data. The loop is O(n), ~3 ms per
    10,000 elements -- not a bottleneck worth trading correctness for.
    """
    if span <= 0:
        raise AllieValueError(f"EMA: span must be positive, got {span}")
    alpha = 2.0 / (span + 1)

    def recurrence(values: np.ndarray) -> np.ndarray:
        result = np.empty_like(values)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
        return result

    return SeriesValue(over_present(series.data, recurrence))


def rolling_mean(series: SeriesValue, window: int) -> SeriesValue:
    return SMA(series, window)


def rolling_sum(series: SeriesValue, window: int) -> SeriesValue:
    _check_window("rolling_sum", window)
    return SeriesValue(rolling(series.data, window, np.sum))


def rolling_std(series: SeriesValue, window: int) -> SeriesValue:
    """Rolling sample standard deviation (`ddof=1`), NaN-padding the
    first `window - 1` bars like `SMA`.

    Vectorized with `sliding_window_view`, which builds a strided *view*
    of every window and lets numpy do the reduction in C -- ~190x faster
    than the per-window Python loop it replaced, and bit-identical to it.

    The obvious O(n) alternative -- cumulative sums of `x` and `x**2`,
    then `var = (sum2 - sum1**2/w) / (w-1)` -- is another ~4x faster
    still, and was rejected. On price-like data (mean ~160, std ~2) it
    loses about 1.7e-08 absolute accuracy to catastrophic cancellation,
    because it subtracts two large nearly-equal numbers to recover a
    small one. That is the exact input shape this function exists for.
    """
    _check_window("rolling_std", window, at_least=2)
    return SeriesValue(rolling(series.data, window, functools.partial(np.std, ddof=1)))


def rolling_zscore(series: SeriesValue, window: int) -> SeriesValue:
    """How many rolling standard deviations each bar sits from its
    rolling mean: `(x - SMA(x, w)) / rolling_std(x, w)`. The trailing,
    point-in-time version of `zscore` -- the one a strategy can use.
    A window with no variation has no z-score (NaN)."""
    _check_window("rolling_zscore", window, at_least=2)
    data = series.data
    mean = rolling(data, window, np.mean)
    std = rolling(data, window, functools.partial(np.std, ddof=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return SeriesValue(np.where(std > 0, (data - mean) / std, np.nan))


def rolling_corr(a: SeriesValue, b: SeriesValue, window: int) -> SeriesValue:
    """Pearson correlation of the last `window` bars of `a` and `b`.
    NaN before the first full window, for a window holding a missing
    bar, and for one where either series doesn't vary."""
    _check_window("rolling_corr", window, at_least=2)
    if len(a) != len(b):
        raise AllieValueError(
            f"rolling_corr: series must be the same length, got {len(a)} and {len(b)}"
        )
    result = np.full(len(a), np.nan, dtype=np.float64)
    if window <= len(a):
        x, y = windows(a.data, window), windows(b.data, window)
        dx = x - x.mean(axis=-1, keepdims=True)
        dy = y - y.mean(axis=-1, keepdims=True)
        spread = np.sqrt((dx**2).sum(axis=-1) * (dy**2).sum(axis=-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = np.where(spread > 0, (dx * dy).sum(axis=-1) / spread, np.nan)
        result[window - 1 :] = correlations
    return SeriesValue(result)


def cumsum(series: SeriesValue) -> SeriesValue:
    """Running total. A missing bar adds nothing and is itself missing."""
    return SeriesValue(over_present(series.data, np.cumsum))


def shift(series: SeriesValue, periods: int) -> SeriesValue:
    """Shift by `periods` (positive shifts forward/later), NaN-filling
    the vacated positions -- matches pandas' `.shift()` convention."""
    data = series.data
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if periods >= 0:
        if periods < n:
            result[periods:] = data[: n - periods]
    else:
        k = -periods
        if k < n:
            result[: n - k] = data[k:]
    return SeriesValue(result)


def diff(series: SeriesValue) -> SeriesValue:
    """First-difference: result[i] = data[i] - data[i-1], result[0] = NaN."""
    data = series.data
    result = np.full(len(data), np.nan, dtype=np.float64)
    if len(data) > 1:
        result[1:] = data[1:] - data[:-1]
    return SeriesValue(result)


def pct_change(series: SeriesValue) -> SeriesValue:
    """Percent change: result[i] = (data[i]-data[i-1])/data[i-1],
    result[0] = NaN. A change from 0 is IEEE: `inf`, or NaN from 0 to 0."""
    data = series.data
    result = np.full(len(data), np.nan, dtype=np.float64)
    if len(data) > 1:
        result[1:] = (data[1:] - data[:-1]) / data[:-1]
    return SeriesValue(result)


def append(series: SeriesValue, value: float) -> SeriesValue:
    """`series` with `value` added at the end, as a new series -- series
    are immutable. How a loop collects one number per pass:

        let sharpes: Series[Float] = [];
        for ... { sharpes = append(sharpes, r.sharpe); }
    """
    return SeriesValue(np.append(series.data, float(value)))


_SERIES_INT = (_SERIES_FLOAT, INT)
register("SMA", FunctionType(params=_SERIES_INT, ret=_SERIES_FLOAT), SMA)
register("EMA", FunctionType(params=_SERIES_INT, ret=_SERIES_FLOAT), EMA)
register("rolling_mean", FunctionType(params=_SERIES_INT, ret=_SERIES_FLOAT), rolling_mean)
register("rolling_sum", FunctionType(params=_SERIES_INT, ret=_SERIES_FLOAT), rolling_sum)
register("rolling_std", FunctionType(params=_SERIES_INT, ret=_SERIES_FLOAT), rolling_std)
register("rolling_zscore", FunctionType(params=_SERIES_INT, ret=_SERIES_FLOAT), rolling_zscore)
register(
    "rolling_corr",
    FunctionType(params=(_SERIES_FLOAT, _SERIES_FLOAT, INT), ret=_SERIES_FLOAT),
    rolling_corr,
)
register("cumsum", FunctionType(params=(_SERIES_FLOAT,), ret=_SERIES_FLOAT), cumsum)
register("shift", FunctionType(params=_SERIES_INT, ret=_SERIES_FLOAT), shift)
register("diff", FunctionType(params=(_SERIES_FLOAT,), ret=_SERIES_FLOAT), diff)
register("append", FunctionType(params=(_SERIES_FLOAT, FLOAT), ret=_SERIES_FLOAT), append)
register("pct_change", FunctionType(params=(_SERIES_FLOAT,), ret=_SERIES_FLOAT), pct_change)
