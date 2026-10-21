"""Tests for allie/stdlib/significance.py -- the statistical rigor tools.

Assertions are against known-correct values -- textbook worked examples,
the published formulas written out independently, the shift tests'
definitions computed shift by shift -- rather than against whatever the
code happens to return, so a plausible-but-wrong rewrite fails here.
Whether the tests hold their error rates is tests/unit/test_rigor_calibration.py.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

import allie.stdlib as stdlib
from allie.cli import run_pipeline
from allie.quant.backtest_engine import (
    BacktestConfig,
    BacktestResult,
    WhenSignal,
    current_engine,
    run_backtest,
)
from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, PriceSeriesValue, SeriesValue
from allie.stdlib.significance import (
    backtest_pvalue,
    benjamini_hochberg,
    bonferroni,
    deflated_sharpe,
    holm,
    permutation_pvalue,
    probabilistic_sharpe,
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


# ---- holm ----


def test_holm_textbook_example() -> None:
    # Sorted: 0.005 <= 0.05/4, 0.01 <= 0.05/3, then 0.03 > 0.05/2 stops it.
    result = holm(SeriesValue([0.01, 0.04, 0.03, 0.005]), 0.05)
    assert list(result.data) == [True, False, False, True]


def test_holm_sits_between_bonferroni_and_benjamini_hochberg() -> None:
    """Never rejects less than Bonferroni (same error rate, step-down),
    never more than BH (which controls a weaker one)."""
    rng = np.random.default_rng(11)
    for _ in range(200):
        p = np.concatenate([rng.uniform(0, 0.02, 3), rng.uniform(0, 1, 7)])
        by_holm = holm(SeriesValue(p), 0.05).data
        by_bonferroni = p * len(p) <= 0.05
        by_bh = benjamini_hochberg(SeriesValue(p), 0.05).data
        assert np.all(by_holm >= by_bonferroni)
        assert np.all(by_holm <= by_bh)


def test_holm_rejects_invalid_input() -> None:
    with pytest.raises(AllieValueError):
        holm(SeriesValue([0.1]), 1.0)
    with pytest.raises(AllieValueError):
        holm(SeriesValue([]), 0.05)
    with pytest.raises(AllieValueError):
        holm(SeriesValue([0.5, -0.1]), 0.05)


# ---- probabilistic_sharpe and deflated_sharpe ----

_RETURNS = [0.012, -0.004, 0.021, 0.0, 0.015, -0.011, 0.006, 0.009, -0.002, 0.013]


def _psr_by_hand(returns: list[float], benchmark_per_bar: float) -> float:
    """Bailey & López de Prado (2012), eq. 9, written out independently."""
    x = np.asarray(returns)
    sr = x.mean() / x.std(ddof=1)
    z = x - x.mean()
    m2 = (z**2).mean()
    skew, kurtosis = (z**3).mean() / m2**1.5, (z**4).mean() / m2**2
    t = (sr - benchmark_per_bar) * math.sqrt(len(x) - 1)
    t /= math.sqrt(1 - skew * sr + (kurtosis - 1) / 4 * sr**2)
    return 0.5 * (1 + math.erf(t / math.sqrt(2)))


def test_probabilistic_sharpe_hand_computed() -> None:
    expected = _psr_by_hand(_RETURNS, 1.0 / math.sqrt(252))
    assert probabilistic_sharpe(SeriesValue(_RETURNS), 1.0, 252) == pytest.approx(expected)


def test_a_benchmark_equal_to_the_estimate_is_a_coin_flip() -> None:
    estimate = sharpe(SeriesValue(_RETURNS), 252)
    assert probabilistic_sharpe(SeriesValue(_RETURNS), estimate, 252) == pytest.approx(0.5)


def test_more_bars_of_the_same_returns_are_more_convincing() -> None:
    once = probabilistic_sharpe(SeriesValue(_RETURNS), 0.0, 252)
    four_times = probabilistic_sharpe(SeriesValue(_RETURNS * 4), 0.0, 252)
    assert 0.5 < once < four_times < 1.0


def test_negative_skew_makes_a_positive_sharpe_less_convincing() -> None:
    rng = np.random.default_rng(12)
    right_skewed = 0.001 + rng.exponential(0.01, 250) - 0.01
    left_skewed = 2 * right_skewed.mean() - right_skewed  # same mean and spread, skew negated
    assert sharpe(SeriesValue(right_skewed), 252) == pytest.approx(
        sharpe(SeriesValue(left_skewed), 252)
    )
    assert probabilistic_sharpe(SeriesValue(left_skewed), 0.0, 252) < probabilistic_sharpe(
        SeriesValue(right_skewed), 0.0, 252
    )


def test_one_trial_deflates_nothing() -> None:
    r = SeriesValue(_RETURNS)
    alone = SeriesValue([sharpe(r, 252)])
    assert deflated_sharpe(r, alone, 252) == pytest.approx(probabilistic_sharpe(r, 0.0, 252))


def test_the_benchmark_is_the_expected_best_of_the_trials() -> None:
    """Bailey & López de Prado (2014), eq. 2, written out independently."""
    r = SeriesValue(_RETURNS)
    trials = [0.4, -0.2, 1.1, 0.7, sharpe(r, 252)]
    per_bar = np.asarray(trials) / math.sqrt(252)
    n, gamma, normal = len(trials), 0.5772156649015329, NormalDist()
    expected_best = per_bar.std(ddof=1) * (
        (1 - gamma) * normal.inv_cdf(1 - 1 / n) + gamma * normal.inv_cdf(1 - 1 / (n * math.e))
    )
    assert deflated_sharpe(r, SeriesValue(trials), 252) == pytest.approx(
        probabilistic_sharpe(r, expected_best * math.sqrt(252), 252)
    )


def _trials(n: int, spread: float) -> SeriesValue:
    """n trial Sharpes with exactly this standard deviation."""
    x = np.linspace(-1.0, 1.0, n)
    return SeriesValue(x / x.std(ddof=1) * spread)


def test_more_trials_deflate_more() -> None:
    r = SeriesValue(_RETURNS)
    assert deflated_sharpe(r, _trials(100, 1.0), 252) < deflated_sharpe(r, _trials(5, 1.0), 252)


def test_trials_that_disagree_more_deflate_more() -> None:
    """Adjacent windows give similar Sharpes, so ten of them cost less
    than ten unrelated strategies would."""
    r = SeriesValue(_RETURNS)
    assert deflated_sharpe(r, _trials(10, 2.0), 252) < deflated_sharpe(r, _trials(10, 0.2), 252)


def test_sharpe_inference_rejects_invalid_input() -> None:
    with pytest.raises(AllieValueError, match="constant"):
        probabilistic_sharpe(SeriesValue([0.01] * 10), 0.0, 252)
    with pytest.raises(AllieValueError, match="at least 3"):
        probabilistic_sharpe(SeriesValue([0.01, 0.02]), 0.0, 252)
    with pytest.raises(AllieValueError, match="periods_per_year"):
        probabilistic_sharpe(SeriesValue(_RETURNS), 0.0, 0)
    with pytest.raises(AllieValueError, match="every strategy tried"):
        deflated_sharpe(SeriesValue(_RETURNS), SeriesValue([np.nan]), 252)


# ---- permutation_pvalue: every circular shift ----


def _shift_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a - a.mean(), b - b.mean()
    return np.abs([np.dot(a, np.roll(b, -k)) for k in range(len(a))])


def test_permutation_pvalue_is_the_share_of_shifts_at_least_as_extreme() -> None:
    """Checked against the definition, shift by shift with np.roll."""
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=60), rng.normal(size=60)
    scores = _shift_scores(a, b)
    expected = np.count_nonzero(scores >= scores[0]) / 60
    assert permutation_pvalue(SeriesValue(a), SeriesValue(b)) == pytest.approx(expected)


def test_permutation_pvalue_finds_unrelated_series_insignificant() -> None:
    rng = np.random.default_rng(2)
    assert (
        permutation_pvalue(SeriesValue(rng.normal(size=400)), SeriesValue(rng.normal(size=400)))
        > 0.05
    )


def test_permutation_pvalue_floors_at_one_over_n() -> None:
    """A real association beats every shift; with n bars, 1/n is as
    small as an honest p-value gets -- never 0."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=400)
    b = a * 0.5 + rng.normal(0, 0.5, size=400)
    assert permutation_pvalue(SeriesValue(a), SeriesValue(b)) == pytest.approx(1 / 400)


def test_permutation_pvalue_drops_nan_pairs() -> None:
    rng = np.random.default_rng(8)
    a, b = rng.normal(size=100), rng.normal(size=100)
    padded = np.concatenate([[np.nan] * 5, a]), np.concatenate([[np.nan] * 5, b])
    assert permutation_pvalue(SeriesValue(padded[0]), SeriesValue(padded[1])) == pytest.approx(
        permutation_pvalue(SeriesValue(a), SeriesValue(b))
    )


def test_permutation_pvalue_rejects_invalid_input() -> None:
    a = np.random.default_rng(10).normal(size=50)
    with pytest.raises(AllieValueError):
        permutation_pvalue(SeriesValue(a), SeriesValue(a[:10]))  # length mismatch
    with pytest.raises(AllieValueError):
        # Correlation is undefined against a constant series.
        permutation_pvalue(SeriesValue(a), SeriesValue(np.ones(50)))
    with pytest.raises(AllieValueError):
        permutation_pvalue(SeriesValue([1.0, np.nan, np.nan]), SeriesValue([1.0, 2.0, 3.0]))


# ---- backtest_pvalue: the strategy's timing against its own shifts ----

_AT_CLOSE = BacktestConfig(fill="close", slippage=0.0, cash=1000.0)


def _backtest(closes: np.ndarray, long_on: np.ndarray) -> BacktestResult:
    """Long on the bars in `long_on`, flat otherwise, filling at the
    signal bar's close so the position is exactly the signal."""
    n = len(closes)
    prices = PriceSeriesValue(
        open=SeriesValue(closes),
        high=SeriesValue(closes),
        low=SeriesValue(closes),
        close=SeriesValue(closes),
        volume=SeriesValue(np.ones(n)),
    )

    def order(fraction: float) -> Callable[[], None]:
        def run() -> None:
            engine = current_engine()
            assert engine is not None
            engine.target(fraction)

        return run

    clauses = [
        WhenSignal(BoolSeriesValue(long_on), order(1.0)),
        WhenSignal(BoolSeriesValue(~long_on), order(0.0)),
    ]
    return run_backtest("S", "X", prices, clauses, 0, n, config=_AT_CLOSE)


def _random_walk(n: int, seed: int) -> np.ndarray:
    return 100.0 * np.cumprod(1.0 + np.random.default_rng(seed).normal(0, 0.01, n))


def test_backtest_pvalue_is_the_share_of_shifted_timings_that_did_as_well() -> None:
    closes = _random_walk(80, 20)
    result = _backtest(closes, np.random.default_rng(21).random(80) > 0.5)
    held = np.asarray(result.positions)[:-1]
    moves = closes[1:] / closes[:-1] - 1.0
    earned = np.array([np.dot(np.roll(held, k), moves) for k in range(len(held))])
    expected = np.count_nonzero(earned >= earned[0] - 1e-12) / len(held)
    assert backtest_pvalue(result) == pytest.approx(expected)


def test_knowing_tomorrow_beats_every_shift() -> None:
    closes = _random_walk(100, 22)
    tomorrow_up = np.append(closes[1:] > closes[:-1], False)
    assert backtest_pvalue(_backtest(closes, tomorrow_up)) == pytest.approx(1 / 99)


def test_buy_and_hold_and_never_trading_have_no_timing_to_test() -> None:
    closes = _random_walk(60, 23)
    assert backtest_pvalue(_backtest(closes, np.ones(60, dtype=bool))) == 1.0
    assert backtest_pvalue(_backtest(closes, np.zeros(60, dtype=bool))) == 1.0


def test_backtest_pvalue_needs_the_price_history() -> None:
    result = _backtest(_random_walk(30, 24), np.ones(30, dtype=bool))
    with pytest.raises(AllieValueError, match="price history"):
        backtest_pvalue(replace(result, closes=[]))


# ---- in the language, on every backend ----


def test_a_sweep_corrected_end_to_end_on_every_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    program = tmp_path / "sweep.qs"
    program.write_text(
        'let AAPL = load("tests/fixtures/AAPL_sample.csv");\n'
        "let c: Series[Float] = AAPL.close;\n"
        "let sharpes: Series[Float] = [];\n"
        "let pvalues: Series[Float] = [];\n"
        "for let w: Int = 3; w <= 12; w = w + 3 {\n"
        "    let fast: Series[Float] = SMA(c, w);\n"
        "    strategy S { when c crosses_above fast: buy(); when c crosses_below fast: sell(); }\n"
        "    let r = backtest S on AAPL from 0 to 60;\n"
        "    sharpes = append(sharpes, r.sharpe);\n"
        "    pvalues = append(pvalues, backtest_pvalue(r));\n"
        "}\n"
        "print(holm(pvalues, 0.05));\n"
        "let slow: Series[Float] = SMA(c, 6);\n"
        "strategy Best { when c crosses_above slow: buy(); when c crosses_below slow: sell(); }\n"
        "let best = backtest Best on AAPL from 0 to 60;\n"
        "print(deflated_sharpe(best.returns, sharpes, 252));\n"
        "print(permutation_pvalue(shift(c / SMA(c, 5) - 1.0, 1), pct_change(c)));\n"
    )
    assert run_pipeline(program, verify=True) == 0
    captured = capsys.readouterr()
    assert "[verified: tree, ir, llvm agree]" in captured.err
    assert captured.out.splitlines()[0] == "Series[False, False, False, False]"


# ---- registration ----


def test_the_rigor_tools_are_registered_builtins() -> None:
    signatures = {name: str(t) for name, t in stdlib.builtin_function_types().items()}
    assert signatures["bonferroni"] == "(Float, Int) -> Float"
    assert signatures["holm"] == "(Series<Float>, Float) -> Series<Bool>"
    assert signatures["benjamini_hochberg"] == "(Series<Float>, Float) -> Series<Bool>"
    assert signatures["sharpe"] == "(Series<Float>, Int) -> Float"
    assert signatures["probabilistic_sharpe"] == "(Series<Float>, Float, Int) -> Float"
    assert signatures["deflated_sharpe"] == "(Series<Float>, Series<Float>, Int) -> Float"
    assert signatures["permutation_pvalue"] == "(Series<Float>, Series<Float>) -> Float"
    assert signatures["backtest_pvalue"] == "(BacktestResult) -> Float"
