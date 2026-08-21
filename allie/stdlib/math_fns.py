"""Statistics stdlib functions (Phase 4.3, "quant/statistics/" in the
merge notes -- kept at this path since that's what was explicitly given
as the heading). Pure numpy, no pandas at runtime (see
allie/runtime/values's module docstring); pandas is used only in this
module's tests, to cross-check these against a known-good reference.

Every function raises AllieValueError on an empty series, rather than
silently returning NaN -- an empty series is almost always a bug (a
misconfigured data load, an off-by-one window), and NaN would propagate
silently through everything downstream instead of surfacing it.
"""

from __future__ import annotations

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import SeriesValue
from allie.stdlib.registry import register
from allie.typechecker.types import FLOAT, FunctionType, SeriesType

_SERIES_FLOAT = SeriesType(FLOAT)


def _require_nonempty(series: SeriesValue, fn_name: str) -> None:
    if len(series) == 0:
        raise AllieValueError(f"{fn_name}: series must not be empty")


def _require_at_least_two(series: SeriesValue, fn_name: str) -> None:
    _require_nonempty(series, fn_name)
    if len(series) < 2:
        raise AllieValueError(f"{fn_name}: series must have at least 2 elements (ddof=1)")


def mean(series: SeriesValue) -> float:
    _require_nonempty(series, "mean")
    return float(np.mean(series.data))


def median(series: SeriesValue) -> float:
    _require_nonempty(series, "median")
    return float(np.median(series.data))


def std(series: SeriesValue) -> float:
    """Sample standard deviation (ddof=1 -- matches pandas' default and
    the usual convention for an observed sample rather than a full
    population)."""
    _require_at_least_two(series, "std")
    return float(np.std(series.data, ddof=1))


def variance(series: SeriesValue) -> float:
    _require_at_least_two(series, "variance")
    return float(np.var(series.data, ddof=1))


def covariance(a: SeriesValue, b: SeriesValue) -> float:
    _require_at_least_two(a, "covariance")
    _require_at_least_two(b, "covariance")
    if len(a) != len(b):
        raise AllieValueError(
            f"covariance: series must be the same length, got {len(a)} and {len(b)}"
        )
    return float(np.cov(a.data, b.data, ddof=1)[0, 1])


def correlation(a: SeriesValue, b: SeriesValue) -> float:
    _require_at_least_two(a, "correlation")
    _require_at_least_two(b, "correlation")
    if len(a) != len(b):
        raise AllieValueError(
            f"correlation: series must be the same length, got {len(a)} and {len(b)}"
        )
    return float(np.corrcoef(a.data, b.data)[0, 1])


register("mean", FunctionType(params=(_SERIES_FLOAT,), ret=FLOAT), mean)
register("median", FunctionType(params=(_SERIES_FLOAT,), ret=FLOAT), median)
register("std", FunctionType(params=(_SERIES_FLOAT,), ret=FLOAT), std)
register("variance", FunctionType(params=(_SERIES_FLOAT,), ret=FLOAT), variance)
register("covariance", FunctionType(params=(_SERIES_FLOAT, _SERIES_FLOAT), ret=FLOAT), covariance)
register(
    "correlation", FunctionType(params=(_SERIES_FLOAT, _SERIES_FLOAT), ret=FLOAT), correlation
)
