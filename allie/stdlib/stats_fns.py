"""Statistical sampling stdlib functions -- Phase 4.3.

`normal`'s signature adds an explicit `seed: Int` parameter beyond what
was literally given (`normal(Float, Float, Int) -> Series<Float>`), to
honor the already-locked determinism decision in docs/spec.md: "All
stochastic stdlib functions ... take an explicit seed; no hidden global
RNG state." A no-seed `normal` would be exactly the hidden-global-RNG
case that decision rules out -- reruns wouldn't be reproducible, which
matters for the paper's credibility as much as for debugging. Treating
the missing parameter as an omission, not a new decision -- say so if a
seedless version was actually intended.
"""

from __future__ import annotations

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import SeriesValue
from allie.stdlib.registry import register
from allie.typechecker.types import FLOAT, INT, FunctionType, SeriesType

_SERIES_FLOAT = SeriesType(FLOAT)


def normal(mean: float, std: float, n: int, seed: int) -> SeriesValue:
    if n <= 0:
        raise AllieValueError(f"normal: n must be positive, got {n}")
    rng = np.random.default_rng(seed)
    return SeriesValue(rng.normal(loc=mean, scale=std, size=n))


def zscore(series: SeriesValue) -> SeriesValue:
    if len(series) < 2:
        raise AllieValueError("zscore: series must have at least 2 elements")
    mean_ = np.mean(series.data)
    std_ = np.std(series.data, ddof=1)
    if std_ == 0:
        raise AllieValueError("zscore: series has zero standard deviation")
    return SeriesValue((series.data - mean_) / std_)


def percentile(series: SeriesValue, p: float) -> float:
    """`p` on the 0-100 scale, matching numpy's convention (`p=50` is
    the median)."""
    if len(series) == 0:
        raise AllieValueError("percentile: series must not be empty")
    if not (0.0 <= p <= 100.0):
        raise AllieValueError(f"percentile: p must be in [0, 100], got {p}")
    return float(np.percentile(series.data, p))


register(
    "normal", FunctionType(params=(FLOAT, FLOAT, INT, INT), ret=_SERIES_FLOAT), normal
)
register("zscore", FunctionType(params=(_SERIES_FLOAT,), ret=_SERIES_FLOAT), zscore)
register("percentile", FunctionType(params=(_SERIES_FLOAT, FLOAT), ret=FLOAT), percentile)
