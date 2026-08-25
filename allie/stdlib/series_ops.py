"""Time-series / indicator stdlib functions (Phase 4.3, "quant/
timeseries/" and "quant/indicators/" in the merge notes -- kept at this
path since that's what was explicitly given as the heading).

`crosses_above`/`crosses_below` landed here in Phase 7, as flagged back
in Phase 4.3 -- they exist only for the strategy/when DSL, so they were
deferred until the backtest engine that consumes them existed. They
return a `BoolSeriesValue` (`Series<Bool>`), not a `SeriesValue`: see
that class's docstring for why a boolean series isn't just float64
1.0/0.0.

`rolling` as a generic higher-order function (window, fn) isn't built:
Allie has no first-class function *values* usable as arguments from
source yet (BuiltinFunction/Closure are runtime-only wrapper types, not
something a `.ql` program can pass around as data). rolling_mean and
rolling_std are the concrete instantiations the spec allows as a
fallback.
"""

from __future__ import annotations

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, SeriesValue
from allie.stdlib.registry import register
from allie.typechecker.types import BOOL, FLOAT, INT, FunctionType, SeriesType

_SERIES_FLOAT = SeriesType(FLOAT)
_SERIES_BOOL = SeriesType(BOOL)


def SMA(series: SeriesValue, window: int) -> SeriesValue:  # noqa: N802 - matches spec's exact name
    """Simple moving average. The first `window - 1` points are NaN,
    matching pandas' `.rolling(window).mean()` convention."""
    if window <= 0:
        raise AllieValueError(f"SMA: window must be positive, got {window}")
    data = series.data
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if window <= n:
        cumsum = np.cumsum(np.insert(data, 0, 0.0))
        result[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    return SeriesValue(result)


def EMA(series: SeriesValue, span: int) -> SeriesValue:  # noqa: N802 - matches spec's exact name
    """Exponential moving average, standard smoothing factor
    alpha = 2/(span+1), seeded with the series' first value.

    **This loop is deliberate -- do not "vectorize" it.** EMA is a
    first-order recurrence (`y[i] = a*x[i] + (1-a)*y[i-1]`), and the
    closed form that lets numpy do it in one pass,
    `y[n] = (1-a)**n * cumsum(a*x[i] / (1-a)**i)`, divides by
    `(1-a)**i`. That underflows to zero for large `i`: measured against
    a 10,000-element series with span=20, results go non-finite from
    index 7043 onward. It looks ~9x faster right up until it silently
    returns garbage on real-length data.

    Unlike `rolling_std` (which had a genuine O(n*window) problem worth
    fixing), this loop is O(n) and costs ~3 ms per 10,000 elements --
    not a bottleneck worth trading correctness for.
    """
    if span <= 0:
        raise AllieValueError(f"EMA: span must be positive, got {span}")
    if len(series) == 0:
        raise AllieValueError("EMA: series must not be empty")
    data = series.data
    alpha = 2.0 / (span + 1)
    result = np.empty_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return SeriesValue(result)


def rolling_mean(series: SeriesValue, window: int) -> SeriesValue:
    return SMA(series, window)


def rolling_std(series: SeriesValue, window: int) -> SeriesValue:
    """Rolling sample standard deviation (`ddof=1`), NaN-padding the
    first `window - 1` bars like `SMA`.

    Vectorized with `sliding_window_view`, which builds a strided *view*
    of every window and lets numpy do the reduction in C. ~190x faster
    than the per-window Python loop this replaced, and **bit-identical**
    to it -- same `np.std` call, just applied to all rows at once.

    The obvious O(n) alternative -- cumulative sums of `x` and `x**2`,
    then `var = (sum2 - sum1**2/w) / (w-1)` -- is another ~4x faster
    still, and was rejected. On price-like data (mean ~160, std ~2) it
    loses about 1.7e-08 absolute accuracy to catastrophic cancellation,
    because it subtracts two large nearly-equal numbers to recover a
    small one. That is the exact input shape this function exists to
    handle, so trading correctness for a speedup that no longer matters
    (the loop was the bottleneck, not the reduction) is a bad deal.
    """
    if window < 2:
        raise AllieValueError(f"rolling_std: window must be at least 2, got {window}")
    data = series.data
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if window <= n:
        windows = np.lib.stride_tricks.sliding_window_view(data, window)
        result[window - 1 :] = np.std(windows, axis=-1, ddof=1)
    return SeriesValue(result)


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
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        result[1:] = data[1:] - data[:-1]
    return SeriesValue(result)


def pct_change(series: SeriesValue) -> SeriesValue:
    """Percent change: result[i] = (data[i]-data[i-1])/data[i-1], result[0] = NaN."""
    data = series.data
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        result[1:] = (data[1:] - data[:-1]) / data[:-1]
    return SeriesValue(result)


def _crossing(a: SeriesValue, b: SeriesValue, fn_name: str, above: bool) -> BoolSeriesValue:
    if len(a) != len(b):
        raise AllieValueError(
            f"{fn_name}: series must be the same length, got {len(a)} and {len(b)}"
        )
    prev_a = shift(a, 1).data
    prev_b = shift(b, 1).data
    if above:
        now = a.data > b.data
        before = prev_a <= prev_b
    else:
        now = a.data < b.data
        before = prev_a >= prev_b
    # shift() NaN-fills index 0, and every NaN comparison is False, so
    # the first bar can never be a crossing -- correct, since there is
    # no prior bar to have crossed from.
    return BoolSeriesValue(now & before)


def crosses_above(a: SeriesValue, b: SeriesValue) -> BoolSeriesValue:
    """True exactly at each bar where `a` moves from at-or-below `b` to
    strictly above it: `(a > b) & (shift(a,1) <= shift(b,1))`."""
    return _crossing(a, b, "crosses_above", above=True)


def crosses_below(a: SeriesValue, b: SeriesValue) -> BoolSeriesValue:
    """Symmetric to `crosses_above`."""
    return _crossing(a, b, "crosses_below", above=False)


register("SMA", FunctionType(params=(_SERIES_FLOAT, INT), ret=_SERIES_FLOAT), SMA)
register("EMA", FunctionType(params=(_SERIES_FLOAT, INT), ret=_SERIES_FLOAT), EMA)
register(
    "rolling_mean", FunctionType(params=(_SERIES_FLOAT, INT), ret=_SERIES_FLOAT), rolling_mean
)
register("rolling_std", FunctionType(params=(_SERIES_FLOAT, INT), ret=_SERIES_FLOAT), rolling_std)
register("shift", FunctionType(params=(_SERIES_FLOAT, INT), ret=_SERIES_FLOAT), shift)
register("diff", FunctionType(params=(_SERIES_FLOAT,), ret=_SERIES_FLOAT), diff)
register("pct_change", FunctionType(params=(_SERIES_FLOAT,), ret=_SERIES_FLOAT), pct_change)
register(
    "crosses_above",
    FunctionType(params=(_SERIES_FLOAT, _SERIES_FLOAT), ret=_SERIES_BOOL),
    crosses_above,
)
register(
    "crosses_below",
    FunctionType(params=(_SERIES_FLOAT, _SERIES_FLOAT), ret=_SERIES_BOOL),
    crosses_below,
)
