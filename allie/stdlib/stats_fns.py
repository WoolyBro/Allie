"""Sampling and distribution functions: `normal`, `zscore`, `percentile`.

`normal` takes an explicit `seed`, honoring docs/spec.md's determinism
decision: "All stochastic stdlib functions ... take an explicit seed; no
hidden global RNG state." A seedless `normal` would make reruns
irreproducible, which matters for a backtest's credibility as much as
for debugging.

`zscore` and `percentile` read the whole series -- `zscore(c)[i]` uses
the mean and spread of every bar, later ones included -- so a `when`
condition can't use them (allie/semantic/lookahead.py); the trailing
version is `rolling_zscore`. Missing bars are skipped
(allie/stdlib/missing.py).
"""

from __future__ import annotations

import math

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import SeriesValue
from allie.stdlib.missing import present
from allie.stdlib.registry import register
from allie.typechecker.types import FLOAT, INT, FunctionType, SeriesType

_SERIES_FLOAT = SeriesType(FLOAT)


def normal(mean: float, std: float, n: int, seed: int) -> SeriesValue:
    """`n` draws from a normal distribution, reproducible from `seed`."""
    if n <= 0:
        raise AllieValueError(f"normal: n must be positive, got {n}")
    if not (math.isfinite(mean) and math.isfinite(std)) or std < 0:
        raise AllieValueError(
            f"normal: needs a finite mean and a std of at least 0, got {mean} and {std}"
        )
    if seed < 0:
        raise AllieValueError(f"normal: the seed must be 0 or more, got {seed}")
    rng = np.random.default_rng(seed)
    return SeriesValue(rng.normal(loc=mean, scale=std, size=n))


def zscore(series: SeriesValue) -> SeriesValue:
    """Each bar's distance from the series' mean, in standard deviations
    (ddof=1). A missing bar stays missing."""
    values = present(series, "zscore", at_least=2)
    mean_, std_ = np.mean(values), np.std(values, ddof=1)
    if std_ == 0:
        raise AllieValueError("zscore: series has zero standard deviation")
    return SeriesValue((series.data - mean_) / std_)


def percentile(series: SeriesValue, p: float) -> float:
    """`p` on the 0-100 scale, matching numpy's convention (`p=50` is
    the median), with linear interpolation between values."""
    if not (0.0 <= p <= 100.0):
        raise AllieValueError(f"percentile: p must be in [0, 100], got {p}")
    return float(np.percentile(present(series, "percentile"), p))


register("normal", FunctionType(params=(FLOAT, FLOAT, INT, INT), ret=_SERIES_FLOAT), normal)
register("zscore", FunctionType(params=(_SERIES_FLOAT,), ret=_SERIES_FLOAT), zscore)
register("percentile", FunctionType(params=(_SERIES_FLOAT, FLOAT), ret=FLOAT), percentile)
