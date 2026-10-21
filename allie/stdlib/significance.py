"""Statistical rigor: is a result real, or the best of many tries?

`docs/spec.md` calls this out as a *structural* requirement, not a nicety:

> The project tests 5 indicators (MA, MACD, RSI, Bollinger, EMA).
> Reporting the best Sharpe ratio across 5 strategies without correction
> (Bonferroni, or a permutation/Monte Carlo null) is p-hacking by
> construction.

Three questions, three groups of tools:

**Is a backtest's Sharpe ratio evidence of anything?** `probabilistic_sharpe`
is the probability that the true Sharpe exceeds a benchmark, given how
many bars there are and how skewed and fat-tailed the returns are.
`deflated_sharpe` asks the same of the *best* of several strategies,
against the Sharpe the best of that many zero-skill strategies would
reach by luck -- the correction for a sweep (Bailey & López de Prado,
"The Deflated Sharpe Ratio", 2014).

**Is a signal or a strategy's timing related to returns beyond chance?**
`permutation_pvalue` and `backtest_pvalue` compare what was observed with
every circular shift of one series against the other. A shift keeps
each series exactly as it is -- its trends, its persistence, how long
positions are held -- and breaks only their alignment. Shuffling bars
independently instead destroys that persistence, which makes chance
look too unlikely: on two independent but persistent series (AR(1),
phi = 0.9) a shuffle test calls the association significant at 5% in
48% of trials. The shift test stays at its nominal rate
(tests/unit/test_rigor_calibration.py).

**How many of several results survive having tried them all?**
`bonferroni`, `holm` (family-wise error) and `benjamini_hochberg` (false
discovery rate).

Everything here is exact: the shift tests enumerate every shift (by FFT,
in O(n log n)) rather than sampling some, so a p-value needs no seed and
is the same on every run and every backend.

## NaN handling

Indicator series arrive NaN-padded (`SMA` pads its first `window - 1`
bars). Rather than propagating NaN into a p-value -- which would be
silently useless -- the functions here drop bars where a value is
missing, and raise if too few remain.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from allie.quant.backtest_engine import BacktestResult, annualized_sharpe, effective_std
from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, SeriesValue
from allie.stdlib.registry import register
from allie.typechecker.types import BACKTEST_RESULT, BOOL, FLOAT, INT, FunctionType, SeriesType

_SERIES_FLOAT = SeriesType(FLOAT)
_SERIES_BOOL = SeriesType(BOOL)
_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649015329


# ---- correcting for many tests ----


def bonferroni(p_value: float, num_tests: int) -> float:
    """Bonferroni-adjusted p-value: `min(1, p * n)`.

    The blunt correction: it controls the family-wise error rate (the
    probability of *any* false positive) with no assumptions about how
    the tests relate. Conservative when the strategies are correlated --
    which sweeping adjacent windows certainly is -- so a result that
    survives Bonferroni has genuinely survived something. `holm` controls
    the same error rate and never rejects less.

        let p: Float = 0.01;
        print(bonferroni(p, 20));   // 0.2 -- no longer significant
    """
    if not 0.0 <= p_value <= 1.0:
        raise AllieValueError(f"bonferroni: p_value must be in [0, 1], got {p_value}")
    if num_tests < 1:
        raise AllieValueError(f"bonferroni: num_tests must be at least 1, got {num_tests}")
    return min(1.0, p_value * num_tests)


def _p_values(p_values: SeriesValue, alpha: float, fn_name: str) -> np.ndarray:
    if not 0.0 < alpha < 1.0:
        raise AllieValueError(f"{fn_name}: alpha must be in (0, 1), got {alpha}")
    values = p_values.data
    if len(values) == 0:
        raise AllieValueError(f"{fn_name}: p_values must not be empty")
    if not np.all(np.isfinite(values)):
        raise AllieValueError(f"{fn_name}: p_values must all be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise AllieValueError(f"{fn_name}: every p_value must be in [0, 1]")
    return values


def holm(p_values: SeriesValue, alpha: float) -> BoolSeriesValue:
    """Holm's step-down procedure. Returns a `Series<Bool>` aligned with
    the input: True where that hypothesis is rejected (significant).

    Controls the family-wise error rate, as Bonferroni does, under the
    same no-assumptions terms -- and rejects everything Bonferroni would,
    often more. Procedure: sort ascending, compare the k-th smallest
    (from 1) with `alpha / (m - k + 1)`, and reject until the first one
    that fails; it and everything after it stand.
    """
    values = _p_values(p_values, alpha, "holm")
    m = len(values)
    order = np.argsort(values, kind="stable")
    thresholds = alpha / (m - np.arange(m))
    failing = np.nonzero(values[order] > thresholds)[0]
    stop = int(failing[0]) if len(failing) else m
    rejected = np.zeros(m, dtype=bool)
    rejected[order[:stop]] = True
    return BoolSeriesValue(rejected)


def benjamini_hochberg(p_values: SeriesValue, alpha: float) -> BoolSeriesValue:
    """Benjamini-Hochberg FDR control. Returns a `Series<Bool>` aligned
    with the input: True where that hypothesis is rejected (significant).

    Controls the *expected proportion of false discoveries* among the
    rejections, rather than the probability of any false discovery. More
    powerful than Bonferroni or Holm when a decent fraction of the tested
    strategies are genuinely predictive.

    Procedure: sort ascending, find the largest `k` with
    `p_(k) <= (k/m) * alpha`, reject every hypothesis with a p-value at
    or below that threshold. Rejecting *all* of the smallest k -- rather
    than only those individually under the line -- is the step that is
    easy to get wrong; it is what makes the procedure monotone.
    """
    values = _p_values(p_values, alpha, "benjamini_hochberg")
    m = len(values)
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    under = np.nonzero(ranked <= thresholds)[0]

    rejected = np.zeros(m, dtype=bool)
    if len(under) > 0:
        cutoff = int(under[-1])
        rejected[order[: cutoff + 1]] = True
    return BoolSeriesValue(rejected)


# ---- is a Sharpe ratio evidence? ----


def sharpe(returns: SeriesValue, periods_per_year: int) -> float:
    """Annualized Sharpe ratio, risk-free rate 0 -- the same convention
    and formula the backtest engine reports, so a Sharpe computed here
    is comparable to one from a `backtest` block.

    Non-finite entries are dropped first (indicator warmup padding),
    and a zero-variance return series gives 0.0 rather than dividing by
    zero.
    """
    _check_periods(periods_per_year, "sharpe")
    values = returns.data[np.isfinite(returns.data)]
    if len(values) < 2:
        raise AllieValueError(f"sharpe: need at least 2 finite observations, got {len(values)}")
    # Delegates to the backtest engine's definition rather than
    # reimplementing it, so the two cannot drift apart.
    return annualized_sharpe(values, periods_per_year)


def _check_periods(periods_per_year: int, fn_name: str) -> None:
    if periods_per_year < 1:
        raise AllieValueError(
            f"{fn_name}: periods_per_year must be at least 1, got {periods_per_year}"
        )


def _sharpe_evidence(returns: SeriesValue, fn_name: str) -> tuple[float, float, float, int]:
    """(Sharpe per bar, skewness, kurtosis -- 3 for a normal distribution
    -- and the number of bars) of the finite returns."""
    values = returns.data[np.isfinite(returns.data)]
    if len(values) < 3:
        raise AllieValueError(f"{fn_name}: need at least 3 finite returns, got {len(values)}")
    if effective_std(values) == 0.0:
        raise AllieValueError(
            f"{fn_name}: the returns are constant, so their Sharpe ratio is undefined"
        )
    per_bar = annualized_sharpe(values, 1)  # the engine's estimator, not annualized
    deviations = values - values.mean()
    m2 = float(np.mean(deviations**2))
    skew = float(np.mean(deviations**3)) / m2**1.5
    kurtosis = float(np.mean(deviations**4)) / m2**2
    return per_bar, skew, kurtosis, len(values)


def _probability_above(returns: SeriesValue, benchmark_per_bar: float, fn_name: str) -> float:
    sr, skew, kurtosis, n = _sharpe_evidence(returns, fn_name)
    # The standard error of an estimated Sharpe ratio grows with skew
    # against it and with fat tails (Mertens 2002); at most 0 only for
    # degenerate samples, since kurtosis >= 1 + skew**2 always holds.
    spread = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if spread <= 0.0:
        raise AllieValueError(f"{fn_name}: the returns are too degenerate to judge")
    return _NORMAL.cdf((sr - benchmark_per_bar) * math.sqrt(n - 1) / math.sqrt(spread))


def probabilistic_sharpe(returns: SeriesValue, benchmark: float, periods_per_year: int) -> float:
    """The probability that the true Sharpe ratio of these returns
    exceeds `benchmark` (annualized, like `sharpe`): Bailey & López de
    Prado's Probabilistic Sharpe Ratio.

    A Sharpe ratio is an estimate, and a noisy one: its standard error
    shrinks with the number of bars and grows with negative skew and fat
    tails, which daily returns have. `probabilistic_sharpe(r.returns,
    0.0, 252) > 0.95` is the claim "this strategy makes money" at 95%
    confidence, made properly. For one strategy tested on its own; for
    the best of several, `deflated_sharpe`.
    """
    _check_periods(periods_per_year, "probabilistic_sharpe")
    return _probability_above(
        returns, benchmark / math.sqrt(periods_per_year), "probabilistic_sharpe"
    )


def deflated_sharpe(
    returns: SeriesValue, trial_sharpes: SeriesValue, periods_per_year: int
) -> float:
    """The probability that the true Sharpe of the chosen strategy is
    above zero, allowing for how many strategies were tried to find it:
    Bailey & López de Prado's Deflated Sharpe Ratio.

    `returns` are the chosen strategy's; `trial_sharpes` holds the
    (annualized) Sharpe of every strategy tried, the chosen one
    included. The best of N strategies with no skill at all still has a
    positive Sharpe, and the expected size of that luck grows with N and
    with how widely the trials' Sharpes spread. That expected maximum
    replaces 0 as the benchmark for `probabilistic_sharpe`. Similar trials
    (adjacent windows) spread less, so they cost less than N unrelated
    ones.

    It is deliberately conservative: in simulation, the best of 20
    zero-skill strategies passes 0.95 about 0.1% of the time, where a
    plain Sharpe test passes it 64% of the time.
    """
    _check_periods(periods_per_year, "deflated_sharpe")
    trials = trial_sharpes.data[np.isfinite(trial_sharpes.data)] / math.sqrt(periods_per_year)
    if len(trials) == 0:
        raise AllieValueError(
            "deflated_sharpe: trial_sharpes must hold the Sharpe of every strategy tried, "
            "the chosen one included"
        )
    expected_best = 0.0
    if len(trials) > 1:
        spread = float(np.std(trials, ddof=1))
        n = len(trials)
        expected_best = spread * (
            (1.0 - _EULER_GAMMA) * _NORMAL.inv_cdf(1.0 - 1.0 / n)
            + _EULER_GAMMA * _NORMAL.inv_cdf(1.0 - 1.0 / (n * math.e))
        )
    return _probability_above(returns, expected_best, "deflated_sharpe")


# ---- is an association, or a strategy's timing, beyond chance? ----


def _circular_products(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """`sum(x[t] * y[(t + k) % n])` for every shift k from 0 to n - 1 --
    k = 0 is the alignment observed. By FFT, so all n of them cost
    O(n log n) together."""
    n = len(x)
    products: np.ndarray = np.fft.irfft(np.conj(np.fft.rfft(x)) * np.fft.rfft(y), n)
    return products


def _shift_pvalue(statistics: np.ndarray) -> float:
    """The share of all n alignments -- the observed one, at index 0,
    included -- that score at least as high as the observed one. Ties
    count against it: an FFT's last-digit noise must not make a strategy
    look better than its own shifted copies."""
    observed = statistics[0]
    tolerance = 1e-9 * float(np.max(np.abs(statistics)))
    hits = int(np.count_nonzero(statistics[1:] >= observed - tolerance))
    return (1.0 + hits) / len(statistics)


def permutation_pvalue(signal: SeriesValue, returns: SeriesValue) -> float:
    """Two-sided test for whether `signal` is associated with `returns`
    beyond chance, by the absolute correlation between them.

    The null is every circular shift of `signal` against `returns`: each
    keeps both series exactly as they are, persistence and all, and
    breaks only their alignment. Exact -- all n shifts, no sampling, no
    seed -- and its smallest possible p-value is 1/n, which is the
    honest limit of what n bars can show when neither series is
    independent from one bar to the next.

    Test stationary quantities -- returns, spreads, z-scores -- rather
    than price levels: a series that trends doesn't look like its own
    shifted copies, whatever it is compared with. Bars where either
    value is missing are dropped first.
    """
    x, y = _finite_pairs(signal, returns, "permutation_pvalue")
    if effective_std(x) == 0.0 or effective_std(y) == 0.0:
        raise AllieValueError(
            "permutation_pvalue: correlation is undefined when either series is constant"
        )
    # Every shift has the same mean and variance, so the products of the
    # centred series rank the shifts exactly as their correlations do.
    return _shift_pvalue(np.abs(_circular_products(x - x.mean(), y - y.mean())))


def backtest_pvalue(result: BacktestResult) -> float:
    """The probability that the backtest's timing is luck: how often the
    same positions, shifted in time, would have earned at least as much
    from the market's moves as they did where the strategy put them.

    Every shift keeps the strategy's exposure and how long it holds its
    positions, and the market's returns as they were; only the timing
    changes. Measured on close-to-close returns before costs, so it
    isolates the one thing the strategy chose. Buy-and-hold, or a
    strategy that never trades, has no timing to test and scores 1.
    """
    positions = np.asarray(result.positions, dtype=np.float64)
    closes = np.asarray(result.closes, dtype=np.float64)
    if len(positions) != len(closes):
        raise AllieValueError("backtest_pvalue: the result has no price history to test against")
    with np.errstate(divide="ignore", invalid="ignore"):
        moves = closes[1:] / closes[:-1] - 1.0
    held = positions[:-1]  # the position held into each close-to-close move
    present = np.isfinite(moves) & np.isfinite(held)
    if int(present.sum()) < 3:
        raise AllieValueError("backtest_pvalue: need at least 3 bars with prices to test")
    return _shift_pvalue(_circular_products(held[present], moves[present]))


def _finite_pairs(a: SeriesValue, b: SeriesValue, fn_name: str) -> tuple[np.ndarray, np.ndarray]:
    if len(a) != len(b):
        raise AllieValueError(
            f"{fn_name}: series must be the same length, got {len(a)} and {len(b)}"
        )
    mask = np.isfinite(a.data) & np.isfinite(b.data)
    if int(mask.sum()) < 3:
        raise AllieValueError(
            f"{fn_name}: need at least 3 overlapping finite observations, got {int(mask.sum())}"
        )
    return a.data[mask], b.data[mask]


register("bonferroni", FunctionType(params=(FLOAT, INT), ret=FLOAT), bonferroni)
register("holm", FunctionType(params=(_SERIES_FLOAT, FLOAT), ret=_SERIES_BOOL), holm)
register(
    "benjamini_hochberg",
    FunctionType(params=(_SERIES_FLOAT, FLOAT), ret=_SERIES_BOOL),
    benjamini_hochberg,
)
register("sharpe", FunctionType(params=(_SERIES_FLOAT, INT), ret=FLOAT), sharpe)
register(
    "probabilistic_sharpe",
    FunctionType(params=(_SERIES_FLOAT, FLOAT, INT), ret=FLOAT),
    probabilistic_sharpe,
)
register(
    "deflated_sharpe",
    FunctionType(params=(_SERIES_FLOAT, _SERIES_FLOAT, INT), ret=FLOAT),
    deflated_sharpe,
)
register(
    "permutation_pvalue",
    FunctionType(params=(_SERIES_FLOAT, _SERIES_FLOAT), ret=FLOAT),
    permutation_pvalue,
)
register("backtest_pvalue", FunctionType(params=(BACKTEST_RESULT,), ret=FLOAT), backtest_pvalue)
