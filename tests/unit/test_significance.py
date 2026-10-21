"""Tests for allie/stdlib/significance.py -- multiple-comparisons
correction and permutation testing.

Assertions are against known-correct values (textbook BH worked example,
hand-computed Bonferroni) rather than against whatever the code happens
to return, so a plausible-but-wrong rewrite fails here.
"""

from __future__ import annotations

import numpy as np
import pytest

import allie.stdlib as stdlib
from allie.runtime.errors import AllieValueError
from allie.runtime.values import SeriesValue
from allie.stdlib.significance import (
    benjamini_hochberg,
    bonferroni,
    permutation_pvalue,
    sharpe,
)

# ---- bonferroni ----


@pytest.mark.parametrize(
    ("p", "n", "expected"),
    [(0.01, 20, 0.20), (0.05, 1, 0.05), (0.30, 20, 1.0), (0.0, 100, 0.0)],
)
def test_bonferroni_known_values(p: float, n: int, expected: float) -> None:
    assert bonferroni(p, n) == pytest.approx(expected)


def test_bonferroni_clamps_at_one() -> None:
    """An adjusted p-value above 1 is meaningless as a probability."""
    assert bonferroni(0.9, 50) == 1.0


def test_bonferroni_rejects_invalid_input() -> None:
    with pytest.raises(AllieValueError):
        bonferroni(1.5, 10)
    with pytest.raises(AllieValueError):
        bonferroni(0.05, 0)


def test_bonferroni_makes_a_sweep_winner_insignificant() -> None:
    """The whole point: p=0.03 looks significant until you admit you
    tried 20 windows to find it."""
    assert bonferroni(0.03, 1) < 0.05
    assert bonferroni(0.03, 20) > 0.05


# ---- benjamini_hochberg ----


def test_benjamini_hochberg_textbook_example() -> None:
    """Classic worked example: p = .005 .011 .02 .04 .13, alpha = .05,
    m = 5. Thresholds are .01 .02 .03 .04 .05; the largest k under the
    line is k=4, so the first four are rejected."""
    result = benjamini_hochberg(SeriesValue([0.005, 0.011, 0.02, 0.04, 0.13]), 0.05)
    np.testing.assert_array_equal(result.data, [True, True, True, True, False])


def test_benjamini_hochberg_is_monotone_step_up() -> None:
    """The step-up property, and the easiest part to get wrong: with
    p = [.001, .049], alpha = .05, m = 2, the thresholds are .025 and
    .05. The second value clears its own threshold, so *both* are
    rejected -- even though .049 > .025."""
    result = benjamini_hochberg(SeriesValue([0.001, 0.049]), 0.05)
    np.testing.assert_array_equal(result.data, [True, True])


def test_benjamini_hochberg_preserves_input_order() -> None:
    """Results align with the input, not with the sorted order."""
    result = benjamini_hochberg(SeriesValue([0.9, 0.001, 0.8]), 0.05)
    np.testing.assert_array_equal(result.data, [False, True, False])


def test_benjamini_hochberg_rejects_nothing_when_all_are_large() -> None:
    result = benjamini_hochberg(SeriesValue([0.4, 0.6, 0.9]), 0.05)
    assert not result.data.any()


def test_benjamini_hochberg_is_less_conservative_than_bonferroni() -> None:
    """The reason to offer both."""
    p_values = [0.001, 0.008, 0.02, 0.03]
    bh = benjamini_hochberg(SeriesValue(p_values), 0.05).data
    bonf = [bonferroni(p, len(p_values)) <= 0.05 for p in p_values]
    assert int(bh.sum()) >= sum(bonf)


def test_benjamini_hochberg_rejects_invalid_input() -> None:
    with pytest.raises(AllieValueError):
        benjamini_hochberg(SeriesValue([0.1]), 0.0)
    with pytest.raises(AllieValueError):
        benjamini_hochberg(SeriesValue([]), 0.05)
    with pytest.raises(AllieValueError):
        benjamini_hochberg(SeriesValue([0.5, 1.5]), 0.05)
    with pytest.raises(AllieValueError):
        benjamini_hochberg(SeriesValue([0.5, np.nan]), 0.05)


# ---- sharpe ----


def test_sharpe_matches_the_backtest_engines_convention() -> None:
    """Must agree with allie.quant.backtest_engine so a Sharpe computed
    in source is comparable to one from a `backtest` block."""
    from allie.quant.backtest_engine import compute_metrics

    equity = [100.0, 102.0, 101.0, 105.0, 103.0, 107.0]
    returns = np.diff(equity) / np.asarray(equity[:-1])
    engine = compute_metrics(equity, num_trades=0).sharpe
    assert sharpe(SeriesValue(returns), 252) == pytest.approx(engine)


def test_sharpe_zero_variance_returns_zero_not_infinity() -> None:
    assert sharpe(SeriesValue([0.01] * 20), 252) == 0.0


def test_sharpe_drops_nan_warmup() -> None:
    values = [0.01, -0.02, 0.03, 0.005]
    clean = sharpe(SeriesValue(values), 252)
    padded = sharpe(SeriesValue([np.nan, np.nan, *values]), 252)
    assert padded == pytest.approx(clean)


def test_sharpe_rejects_invalid_input() -> None:
    with pytest.raises(AllieValueError):
        sharpe(SeriesValue([0.01]), 252)  # needs >= 2 observations
    with pytest.raises(AllieValueError):
        sharpe(SeriesValue([0.01, 0.02]), 0)


# ---- permutation_pvalue ----


def _rng_series(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n)


def test_permutation_pvalue_finds_unrelated_series_insignificant() -> None:
    a, b = _rng_series(400, 1), _rng_series(400, 2)
    assert permutation_pvalue(SeriesValue(a), SeriesValue(b), 199, 7) > 0.05


def test_permutation_pvalue_finds_a_real_association_significant() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=400)
    b = a * 0.5 + rng.normal(0, 0.5, size=400)
    assert permutation_pvalue(SeriesValue(a), SeriesValue(b), 199, 7) < 0.01


def test_permutation_pvalue_never_returns_exactly_zero() -> None:
    """The `1 +` on both sides of the ratio. Reporting p = 0 would claim
    certainty no finite number of shuffles can support."""
    rng = np.random.default_rng(4)
    a = rng.normal(size=200)
    b = a * 10.0  # perfectly correlated
    p = permutation_pvalue(SeriesValue(a), SeriesValue(b), 99, 7)
    assert p > 0.0
    assert p == pytest.approx(1.0 / 100.0)  # the floor: 1/(1+trials)


def test_permutation_pvalue_is_reproducible_from_its_seed() -> None:
    """docs/spec.md "Determinism": no hidden global RNG."""
    a, b = _rng_series(200, 5), _rng_series(200, 6)
    first = permutation_pvalue(SeriesValue(a), SeriesValue(b), 99, 42)
    second = permutation_pvalue(SeriesValue(a), SeriesValue(b), 99, 42)
    assert first == second


def test_permutation_pvalue_differs_across_seeds() -> None:
    """Reproducibility must come from the seed, not from the function
    ignoring randomness altogether."""
    a, b = _rng_series(200, 5), _rng_series(200, 6)
    results = {permutation_pvalue(SeriesValue(a), SeriesValue(b), 99, s) for s in range(8)}
    assert len(results) > 1


def test_permutation_pvalue_drops_nan_pairs() -> None:
    a, b = _rng_series(100, 8), _rng_series(100, 9)
    padded_a = np.concatenate([[np.nan] * 5, a])
    padded_b = np.concatenate([[np.nan] * 5, b])
    assert permutation_pvalue(SeriesValue(padded_a), SeriesValue(padded_b), 99, 7) == pytest.approx(
        permutation_pvalue(SeriesValue(a), SeriesValue(b), 99, 7)
    )


def test_permutation_pvalue_rejects_invalid_input() -> None:
    a = _rng_series(50, 10)
    with pytest.raises(AllieValueError):
        permutation_pvalue(SeriesValue(a), SeriesValue(a[:10]), 99, 7)  # length mismatch
    with pytest.raises(AllieValueError):
        permutation_pvalue(SeriesValue(a), SeriesValue(a), 0, 7)  # no trials
    with pytest.raises(AllieValueError):
        # Correlation is undefined against a constant series.
        permutation_pvalue(SeriesValue(a), SeriesValue(np.ones(50)), 99, 7)


def test_permutation_pvalue_needs_enough_overlap() -> None:
    with pytest.raises(AllieValueError):
        permutation_pvalue(SeriesValue([1.0, np.nan, np.nan]), SeriesValue([1.0, 2.0, 3.0]), 99, 7)


# ---- registration ----


def test_all_four_are_registered_builtins() -> None:
    signatures = stdlib.builtin_function_types()
    assert str(signatures["bonferroni"]) == "(Float, Int) -> Float"
    assert str(signatures["benjamini_hochberg"]) == "(Series<Float>, Float) -> Series<Bool>"
    assert str(signatures["sharpe"]) == "(Series<Float>, Int) -> Float"
    assert (
        str(signatures["permutation_pvalue"]) == "(Series<Float>, Series<Float>, Int, Int) -> Float"
    )
