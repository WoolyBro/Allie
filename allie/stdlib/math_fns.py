"""Statistics over a whole series: `mean`, `median`, `std`, `variance`,
`sum`, `min`, `max`, `covariance`, `correlation`. Pure numpy, no pandas
at runtime (see allie/runtime/values's module docstring); pandas is used
only in the tests, as a reference to check these against.

Missing bars (NaN) are skipped, and a series with no values left raises
rather than returning NaN -- see allie/stdlib/missing.py. These read the
whole series, every bar included, so a `when` condition can't use them
(allie/semantic/lookahead.py); the trailing versions are `SMA`,
`rolling_std`, `rolling_max` and the like.
"""

from __future__ import annotations

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import SeriesValue
from allie.stdlib.missing import present, present_pairs
from allie.stdlib.registry import register
from allie.typechecker.types import FLOAT, FunctionType, SeriesType

_SERIES_FLOAT = SeriesType(FLOAT)


def mean(series: SeriesValue) -> float:
    return float(np.mean(present(series, "mean")))


def median(series: SeriesValue) -> float:
    return float(np.median(present(series, "median")))


def std(series: SeriesValue) -> float:
    """Sample standard deviation (ddof=1 -- pandas' default, and the
    usual convention for an observed sample rather than a population)."""
    return float(np.std(present(series, "std", at_least=2), ddof=1))


def variance(series: SeriesValue) -> float:
    return float(np.var(present(series, "variance", at_least=2), ddof=1))


def sum_(series: SeriesValue) -> float:
    return float(np.sum(present(series, "sum")))


def min_(series: SeriesValue) -> float:
    return float(np.min(present(series, "min")))


def max_(series: SeriesValue) -> float:
    return float(np.max(present(series, "max")))


def covariance(a: SeriesValue, b: SeriesValue) -> float:
    """Sample covariance over the bars where both have a value."""
    x, y = present_pairs(a, b, "covariance")
    return float(np.cov(x, y, ddof=1)[0, 1])


def correlation(a: SeriesValue, b: SeriesValue) -> float:
    """Pearson correlation over the bars where both have a value. A
    series that doesn't vary has no correlation with anything, which is
    an error rather than a NaN."""
    x, y = present_pairs(a, b, "correlation")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise AllieValueError("correlation: undefined when a series doesn't vary")
    return float(np.corrcoef(x, y)[0, 1])


register("mean", FunctionType(params=(_SERIES_FLOAT,), ret=FLOAT), mean)
register("median", FunctionType(params=(_SERIES_FLOAT,), ret=FLOAT), median)
register("std", FunctionType(params=(_SERIES_FLOAT,), ret=FLOAT), std)
register("variance", FunctionType(params=(_SERIES_FLOAT,), ret=FLOAT), variance)
register("sum", FunctionType(params=(_SERIES_FLOAT,), ret=FLOAT), sum_)
register("min", FunctionType(params=(_SERIES_FLOAT,), ret=FLOAT), min_)
register("max", FunctionType(params=(_SERIES_FLOAT,), ret=FLOAT), max_)
register("covariance", FunctionType(params=(_SERIES_FLOAT, _SERIES_FLOAT), ret=FLOAT), covariance)
register("correlation", FunctionType(params=(_SERIES_FLOAT, _SERIES_FLOAT), ret=FLOAT), correlation)
