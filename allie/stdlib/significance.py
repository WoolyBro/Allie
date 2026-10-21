"""Multiple-comparisons correction and permutation testing.

`docs/spec.md` calls this out as a *structural* requirement, not a nicety:

> The project tests 5 indicators (MA, MACD, RSI, Bollinger, EMA).
> Reporting the best Sharpe ratio across 5 strategies without correction
> (Bonferroni, or a permutation/Monte Carlo null) is p-hacking by
> construction.

A window sweep is a multiple-comparisons problem whether or not anyone
names it as one. Try 20 moving-average windows on the same price series
and the best Sharpe will look impressive *even if every one of them is
noise* -- that is what "best of 20" means. Mainstream backtesting tools
report the winner and leave the correction to the user, who mostly
doesn't do it. These functions make it a one-liner.

All stochastic functions take an explicit seed (docs/spec.md
"Determinism") -- there is no hidden global RNG, so a reported p-value
is reproducible.

## NaN handling

Indicator series arrive NaN-padded (`SMA` pads its first `window - 1`
bars). Rather than propagating NaN into a p-value -- which would be
silently useless -- the paired functions here drop index pairs where
either side is non-finite, and raise if too little overlap remains. That
is stated in each docstring rather than left for the user to discover.
"""

from __future__ import annotations

import numpy as np

from allie.quant.backtest_engine import annualized_sharpe, effective_std
from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, SeriesValue
from allie.stdlib.registry import register
from allie.typechecker.types import BOOL, FLOAT, INT, FunctionType, SeriesType

_SERIES_FLOAT = SeriesType(FLOAT)
_SERIES_BOOL = SeriesType(BOOL)

TRADING_DAYS_PER_YEAR = 252


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


def bonferroni(p_value: float, num_tests: int) -> float:
    """Bonferroni-adjusted p-value: `min(1, p * n)`.

    The blunt correction, and the right default: it controls the
    family-wise error rate (the probability of *any* false positive)
    with no assumptions about how the tests relate. Conservative when
    the strategies are correlated -- which sweeping adjacent windows
    certainly is -- so a result that survives Bonferroni has genuinely
    survived something. `benjamini_hochberg` is the less conservative
    alternative when many tests are expected to be real.

        let p: Float = 0.01;
        print(bonferroni(p, 20));   // 0.2 -- no longer significant
    """
    if not 0.0 <= p_value <= 1.0:
        raise AllieValueError(f"bonferroni: p_value must be in [0, 1], got {p_value}")
    if num_tests < 1:
        raise AllieValueError(f"bonferroni: num_tests must be at least 1, got {num_tests}")
    return min(1.0, p_value * num_tests)


def benjamini_hochberg(p_values: SeriesValue, alpha: float) -> BoolSeriesValue:
    """Benjamini-Hochberg FDR control. Returns a `Series<Bool>` aligned
    with the input: True where that hypothesis is rejected (significant).

    Controls the *expected proportion of false discoveries* among the
    rejections, rather than the probability of any false discovery. More
    powerful than Bonferroni when a decent fraction of the tested
    strategies are genuinely predictive.

    Procedure: sort ascending, find the largest `k` with
    `p_(k) <= (k/m) * alpha`, reject every hypothesis with a p-value at
    or below that threshold. Rejecting *all* of the smallest k -- rather
    than only those individually under the line -- is the step that is
    easy to get wrong; it is what makes the procedure monotone.
    """
    if not 0.0 < alpha < 1.0:
        raise AllieValueError(f"benjamini_hochberg: alpha must be in (0, 1), got {alpha}")
    values = p_values.data
    if len(values) == 0:
        raise AllieValueError("benjamini_hochberg: p_values must not be empty")
    if not np.all(np.isfinite(values)):
        raise AllieValueError("benjamini_hochberg: p_values must all be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise AllieValueError("benjamini_hochberg: every p_value must be in [0, 1]")

    m = len(values)
    order = np.argsort(values)
    ranked = values[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    under = np.nonzero(ranked <= thresholds)[0]

    rejected = np.zeros(m, dtype=bool)
    if len(under) > 0:
        cutoff = int(under[-1])
        rejected[order[: cutoff + 1]] = True
    return BoolSeriesValue(rejected)


def sharpe(returns: SeriesValue, periods_per_year: int) -> float:
    """Annualized Sharpe ratio, risk-free rate 0 -- the same convention
    and formula the backtest engine reports, so a Sharpe computed here
    is comparable to one from a `backtest` block.

    Non-finite entries are dropped first (indicator warmup padding),
    and a zero-variance return series gives 0.0 rather than dividing by
    zero.
    """
    if periods_per_year < 1:
        raise AllieValueError(
            f"sharpe: periods_per_year must be at least 1, got {periods_per_year}"
        )
    values = returns.data[np.isfinite(returns.data)]
    if len(values) < 2:
        raise AllieValueError(f"sharpe: need at least 2 finite observations, got {len(values)}")
    # Delegates to the backtest engine's definition rather than
    # reimplementing it, so the two cannot drift apart.
    return annualized_sharpe(values, periods_per_year)


def permutation_pvalue(signal: SeriesValue, returns: SeriesValue, trials: int, seed: int) -> float:
    """Two-sided permutation test for whether `signal` is associated
    with `returns` beyond chance.

    Statistic is `|corr(signal, returns)|`. Under the null, `signal`
    carries no timing information, so any reordering of it is equally
    likely -- the test shuffles `signal` `trials` times and counts how
    often chance alone matches the observed association.

        p = (1 + #{shuffles with |corr| >= observed}) / (1 + trials)

    The `1 +` on both sides is deliberate, not an off-by-one: it makes
    the estimate unbiased and, more practically, stops the function ever
    returning exactly 0.0, which would overstate certainty that no
    finite number of shuffles can support. With `trials = 999` the
    smallest reportable p-value is 0.001.

    **Why shuffle the signal rather than the returns.** Permuting a
    return series does not change its Sharpe ratio at all -- mean and
    standard deviation are both order-invariant, so every shuffle scores
    identically and the test is vacuous. The information being tested is
    the *alignment* between signal and outcome, so that is what has to
    be broken.

    Non-finite pairs are dropped before testing (see the module
    docstring).
    """
    if trials < 1:
        raise AllieValueError(f"permutation_pvalue: trials must be at least 1, got {trials}")

    signal_values, return_values = _finite_pairs(signal, returns, "permutation_pvalue")
    # Same scale-relative zero test as the Sharpe path -- a series
    # that is constant up to float noise has an undefined correlation
    # just as surely as an exactly-constant one.
    if effective_std(signal_values) == 0.0 or effective_std(return_values) == 0.0:
        raise AllieValueError(
            "permutation_pvalue: correlation is undefined when either series is constant"
        )

    observed = abs(float(np.corrcoef(signal_values, return_values)[0, 1]))

    rng = np.random.default_rng(seed)
    shuffled = signal_values.copy()
    at_least_as_extreme = 0
    for _ in range(trials):
        rng.shuffle(shuffled)
        statistic = abs(float(np.corrcoef(shuffled, return_values)[0, 1]))
        if statistic >= observed:
            at_least_as_extreme += 1

    return (1.0 + at_least_as_extreme) / (1.0 + trials)


register("bonferroni", FunctionType(params=(FLOAT, INT), ret=FLOAT), bonferroni)
register(
    "benjamini_hochberg",
    FunctionType(params=(_SERIES_FLOAT, FLOAT), ret=_SERIES_BOOL),
    benjamini_hochberg,
)
register("sharpe", FunctionType(params=(_SERIES_FLOAT, INT), ret=FLOAT), sharpe)
register(
    "permutation_pvalue",
    FunctionType(params=(_SERIES_FLOAT, _SERIES_FLOAT, INT, INT), ret=FLOAT),
    permutation_pvalue,
)
