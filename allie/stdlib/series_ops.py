"""Time-series / indicator stdlib functions (Phase 4.3, "quant/
timeseries/" and "quant/indicators/" in the merge notes -- kept at this
path since that's what was explicitly given as the heading).

`crosses_above`/`crosses_below` deliberately do NOT live here yet even
though they're named in this same file in the spec -- they only exist
for the strategy/when DSL and are explicitly Phase 7's responsibility to
implement and test. Not building them now; don't be surprised when they
land in this exact file later.

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
from allie.runtime.values import SeriesValue
from allie.stdlib.registry import register
from allie.typechecker.types import FLOAT, INT, FunctionType, SeriesType

_SERIES_FLOAT = SeriesType(FLOAT)


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
    alpha = 2/(span+1), seeded with the series' first value."""
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
    if window < 2:
        raise AllieValueError(f"rolling_std: window must be at least 2, got {window}")
    data = series.data
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        result[i] = np.std(data[i - window + 1 : i + 1], ddof=1)
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


register("SMA", FunctionType(params=(_SERIES_FLOAT, INT), ret=_SERIES_FLOAT), SMA)
register("EMA", FunctionType(params=(_SERIES_FLOAT, INT), ret=_SERIES_FLOAT), EMA)
register(
    "rolling_mean", FunctionType(params=(_SERIES_FLOAT, INT), ret=_SERIES_FLOAT), rolling_mean
)
register("rolling_std", FunctionType(params=(_SERIES_FLOAT, INT), ret=_SERIES_FLOAT), rolling_std)
register("shift", FunctionType(params=(_SERIES_FLOAT, INT), ret=_SERIES_FLOAT), shift)
register("diff", FunctionType(params=(_SERIES_FLOAT,), ret=_SERIES_FLOAT), diff)
register("pct_change", FunctionType(params=(_SERIES_FLOAT,), ret=_SERIES_FLOAT), pct_change)
