"""The statistics the benchmark report rests on, kept apart from the
harness so they can be tested on known inputs (tests/unit/test_benchmarks.py).

- `summarize`: median, interquartile range and minimum of a sample.
- `ratio`: how many times slower one sample is than another, as a ratio
  of medians with a bootstrap 95% confidence interval. A comparison
  whose interval contains 1 is reported as *within noise* -- not as a
  0.4% improvement.
- `linear_fit`: least squares, for separating a backend's fixed cost
  (compilation, setup) from its cost per loop iteration.

Every resampling is seeded, so a report regenerated from the same
samples says the same thing.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

_BOOTSTRAP_ROUNDS = 2000
_SEED = 20261020


@dataclass(frozen=True)
class Summary:
    median: float
    q1: float
    q3: float
    minimum: float
    n: int

    @property
    def spread(self) -> float:
        """The interquartile range relative to the median."""
        return (self.q3 - self.q1) / self.median if self.median else 0.0


def summarize(samples: list[float]) -> Summary:
    if not samples:
        raise ValueError("summarize: no samples")
    ordered = sorted(samples)
    if len(ordered) == 1:
        q1 = q3 = ordered[0]
    else:
        q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return Summary(statistics.median(ordered), q1, q3, ordered[0], len(ordered))


@dataclass(frozen=True)
class Ratio:
    """`numerator / denominator` of the medians, with a 95% interval."""

    point: float
    low: float
    high: float

    @property
    def significant(self) -> bool:
        return self.low > 1.0 or self.high < 1.0

    def times(self) -> str:
        """`12.3x [11.8, 12.9]`."""
        return f"{_fmt(self.point)}x [{_fmt(self.low)}, {_fmt(self.high)}]"


def ratio(numerator: list[float], denominator: list[float], seed: int = _SEED) -> Ratio:
    """Bootstrap: resample both samples with replacement and take the
    ratio of their medians, many times; the 2.5th and 97.5th percentiles
    of those ratios bound the interval."""
    if not numerator or not denominator:
        raise ValueError("ratio: both samples need at least one value")
    point = statistics.median(numerator) / statistics.median(denominator)
    rng = random.Random(seed)
    estimates = sorted(
        statistics.median(rng.choices(numerator, k=len(numerator)))
        / statistics.median(rng.choices(denominator, k=len(denominator)))
        for _ in range(_BOOTSTRAP_ROUNDS)
    )
    low = estimates[int(0.025 * _BOOTSTRAP_ROUNDS)]
    high = estimates[int(0.975 * _BOOTSTRAP_ROUNDS) - 1]
    return Ratio(point, low, high)


@dataclass(frozen=True)
class Fit:
    slope: float
    intercept: float
    r_squared: float


def linear_fit(xs: list[float], ys: list[float]) -> Fit:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("linear_fit: needs at least two (x, y) pairs")
    slope, intercept = statistics.linear_regression(xs, ys)
    mean_y = statistics.fmean(ys)
    total = sum((y - mean_y) ** 2 for y in ys)
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    return Fit(slope, intercept, 1.0 - residual / total if total else 1.0)


def geometric_mean(values: list[float]) -> float:
    return statistics.geometric_mean(values)


def _fmt(x: float) -> str:
    """Three significant figures, never scientific notation."""
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"
