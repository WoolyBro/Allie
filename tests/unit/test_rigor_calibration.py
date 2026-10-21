"""Do the rigor tools hold their error rates? Monte Carlo, seeded.

A significance test is only as good as its null. These run each test
many times on data where the null is *true* -- no association, no skill,
no real Sharpe -- and check that it calls the result significant at 5%
about 5% of the time, not more. Where a naive method fails that, the
naive method is run too, so the failure the tool exists to prevent is
on the record:

- shuffling bars independently, on two unrelated but persistent series;
- testing the best of 20 strategies as if it were the only one.

And each is run where something real is there, to show it can find it.
The bounds allow for the sampling error of the number of trials used.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from allie.quant.backtest_engine import BacktestResult, compute_metrics
from allie.runtime.values import SeriesValue
from allie.stdlib.significance import (
    backtest_pvalue,
    deflated_sharpe,
    permutation_pvalue,
    probabilistic_sharpe,
)


def _ar1(rng: np.random.Generator, phi: float, n: int, count: int) -> np.ndarray:
    """`count` independent AR(1) series of length n, past their burn-in."""
    shocks = rng.normal(size=(count, n + 200))
    series = np.empty_like(shocks)
    series[:, 0] = shocks[:, 0]
    for t in range(1, n + 200):
        series[:, t] = phi * series[:, t - 1] + shocks[:, t]
    return series[:, 200:]


def _shuffle_pvalue(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    """The test permutation_pvalue used to be: shuffle the bars
    independently, 199 times."""
    observed = abs(np.corrcoef(x, y)[0, 1])
    shuffled = x[np.argsort(rng.random((199, len(x))), axis=1)]
    shuffled = shuffled - shuffled.mean(axis=1, keepdims=True)
    centred = y - y.mean()
    scores = np.abs(shuffled @ centred) / (
        np.sqrt((shuffled**2).sum(axis=1)) * np.sqrt((centred**2).sum())
    )
    return (1 + int(np.count_nonzero(scores >= observed))) / 200


@pytest.mark.parametrize(("phi", "shuffle_at_least"), [(0.0, 0.0), (0.9, 0.3)])
def test_the_shift_test_holds_its_level_where_shuffling_does_not(
    phi: float, shuffle_at_least: float
) -> None:
    rng = np.random.default_rng(100)
    signals, returns = _ar1(rng, phi, 300, 300), _ar1(rng, phi, 300, 300)
    shift = np.mean(
        [
            permutation_pvalue(SeriesValue(s), SeriesValue(r)) <= 0.05
            for s, r in zip(signals, returns, strict=True)
        ]
    )
    shuffle = np.mean(
        [_shuffle_pvalue(s, r, rng) <= 0.05 for s, r in zip(signals, returns, strict=True)]
    )
    assert 0.01 <= shift <= 0.09
    # Independent bars: both are fine. Persistent ones: shuffling calls
    # chance significant several times as often as it should.
    assert shuffle >= shuffle_at_least


def test_the_shift_test_finds_a_real_association() -> None:
    """Persistent series carry less information than their length
    suggests: 300 bars at phi = 0.9 are worth about 31 independent ones.
    A correlation of 0.6 is still found; one of 0.3 honestly is not,
    usually -- which a shuffle test would hide by overstating n."""
    rng = np.random.default_rng(101)
    signals = _ar1(rng, 0.9, 300, 100)
    returns = 0.8 * signals + _ar1(rng, 0.9, 300, 100)
    found = np.mean(
        [
            permutation_pvalue(SeriesValue(s), SeriesValue(r)) <= 0.05
            for s, r in zip(signals, returns, strict=True)
        ]
    )
    assert found >= 0.8


@pytest.mark.parametrize("tails", ["normal", "fat"])
def test_probabilistic_sharpe_holds_its_level(tails: str) -> None:
    rng = np.random.default_rng(102)
    if tails == "normal":
        draws = rng.normal(0.0, 0.01, (2000, 250))
    else:
        draws = rng.standard_t(5, (2000, 250)) * 0.01
    passed = np.mean([probabilistic_sharpe(SeriesValue(r), 0.0, 252) > 0.95 for r in draws])
    assert 0.03 <= passed <= 0.07


def _annual_sharpes(returns: np.ndarray) -> np.ndarray:
    sharpes: np.ndarray = returns.mean(axis=1) / returns.std(axis=1, ddof=1) * math.sqrt(252)
    return sharpes


def test_the_best_of_twenty_needs_deflating() -> None:
    """Twenty strategies with no skill; report the best. A plain Sharpe
    test is fooled most of the time -- the selection bias a sweep
    creates. The deflated Sharpe is not."""
    rng = np.random.default_rng(103)
    plain = deflated = 0
    runs = 500
    for _ in range(runs):
        returns = rng.normal(0.0, 0.01, (20, 250))
        sharpes = _annual_sharpes(returns)
        best = SeriesValue(returns[int(np.argmax(sharpes))])
        plain += probabilistic_sharpe(best, 0.0, 252) > 0.95
        deflated += deflated_sharpe(best, SeriesValue(sharpes), 252) > 0.95
    assert plain / runs >= 0.5
    assert deflated / runs <= 0.05


def test_the_deflated_sharpe_still_finds_real_skill() -> None:
    """One strategy with a true annual Sharpe of 3.5 among 19 without,
    over three years: found, and found as the best, most of the time."""
    rng = np.random.default_rng(104)
    found = 0
    runs = 300
    for _ in range(runs):
        returns = rng.normal(0.0, 0.01, (20, 750))
        returns[0] += 3.5 / math.sqrt(252) * 0.01
        sharpes = _annual_sharpes(returns)
        best = int(np.argmax(sharpes))
        found += (
            best == 0 and deflated_sharpe(SeriesValue(returns[0]), SeriesValue(sharpes), 252) > 0.95
        )
    assert found / runs >= 0.8


def _result(positions: np.ndarray, moves: np.ndarray) -> BacktestResult:
    closes = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + moves)))
    return BacktestResult(
        strategy_name="S",
        symbol="X",
        metrics=compute_metrics([1.0, 1.0]),
        trades=[],
        equity_curve=[1.0] * len(closes),
        positions=list(positions),
        closes=list(closes),
    )


def _trend_rule(moves: np.ndarray) -> np.ndarray:
    """Long after the last ten bars rose, flat otherwise -- a position
    for each close, the first one before any move."""
    trailing = np.convolve(moves, np.ones(10))[: len(moves)]
    return np.concatenate(([0.0], (trailing > 0).astype(float)))


def test_backtest_pvalue_holds_its_level_for_a_trend_rule_on_a_random_walk() -> None:
    """No skill is possible on a random walk, but a trend rule holds its
    positions for many bars -- exactly the persistence a shuffle would
    destroy. The shift test keeps it."""
    rng = np.random.default_rng(105)
    passed = 0
    runs = 600
    for _ in range(runs):
        moves = rng.normal(0.0, 0.01, 500)
        passed += backtest_pvalue(_result(_trend_rule(moves), moves)) <= 0.05
    assert 0.01 <= passed / runs <= 0.08


def test_backtest_pvalue_finds_timing_that_works() -> None:
    """A market with weak momentum (each move 15% of the last); being long
    after an up bar exploits it."""
    rng = np.random.default_rng(106)
    found = 0
    runs = 200
    for _ in range(runs):
        shocks = rng.normal(0.0, 0.01, 520)
        moves = np.empty(520)
        moves[0] = shocks[0]
        for t in range(1, 520):
            moves[t] = 0.15 * moves[t - 1] + shocks[t]
        moves = moves[20:]
        after_up = np.concatenate(([0.0], (moves > 0).astype(float)))
        found += backtest_pvalue(_result(after_up, moves)) <= 0.05
    assert found / runs >= 0.6


def test_the_result_helper_matches_the_engine_layout() -> None:
    """One position per close, as the engine records them."""
    result = _result(np.zeros(5), np.zeros(4))
    assert len(result.positions) == len(result.closes) == 5
    assert backtest_pvalue(replace(result, positions=[1.0] * 5)) == 1.0
