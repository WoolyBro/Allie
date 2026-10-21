"""The strategy DSL end to end: series operators, `when` conditions,
crossings, `backtest` as an expression, and the indicators strategies are
built from.

Every program runs six ways -- the tree-walking interpreter, the IR
interpreter and the LLVM JIT, each optimized and not -- and all six must
print exactly the same thing, errors included. Values are also checked
against an independent numpy computation, so "every backend agrees" can't
hide "every backend is wrong".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import allie.stdlib  # noqa: F401 - registers the builtins
from allie.cli import BACKENDS, run_pipeline
from allie.runtime.elementwise import crossing
from allie.stdlib.indicators import RSI
from allie.stdlib.io import load
from allie.stdlib.series_ops import EMA, SMA, rolling_std

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "AAPL_sample.csv"
_PRELUDE = f'let AAPL = load("{_FIXTURE.as_posix()}");\nlet c: Series[Float] = AAPL.close;\n'
_PRICES = load(str(_FIXTURE))
_CLOSE = _PRICES.close.data


def _run_everywhere(
    tmp_path: Path, source: str, capsys: pytest.CaptureFixture[str]
) -> dict[str, tuple[int, str, str]]:
    path = tmp_path / "p.qs"
    path.write_text(_PRELUDE + source, encoding="utf-8")
    results = {}
    for backend in BACKENDS:
        for optimize in (True, False):
            code = run_pipeline(path, backend=backend, optimize=optimize)
            captured = capsys.readouterr()
            results[f"{backend}{'' if optimize else '-O0'}"] = (code, captured.out, captured.err)
    return results


def _output(tmp_path: Path, source: str, capsys: pytest.CaptureFixture[str]) -> list[str]:
    """What the program prints, after checking all six runs printed it
    and none failed."""
    results = _run_everywhere(tmp_path, source, capsys)
    reference = results["tree"]
    disagreeing = {name: r for name, r in results.items() if r != reference}
    assert not disagreeing, f"backends disagree:\ntree: {reference}\n{disagreeing}"
    code, out, err = reference
    assert (code, err) == (0, ""), err
    return out.splitlines()


def _error(tmp_path: Path, source: str, capsys: pytest.CaptureFixture[str]) -> str:
    """The error every run ends with -- the same one everywhere."""
    results = _run_everywhere(tmp_path, source, capsys)
    assert len(set(results.values())) == 1, results
    code, out, err = results["tree"]
    assert code == 1 and out == "", (code, out)
    return err


# ---- series operators ----


def test_arithmetic_on_series_is_bar_by_bar_with_numbers_broadcast(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = _output(
        tmp_path,
        "let spread: Series[Float] = (c - AAPL.open) / AAPL.open * 100.0;\n"
        "print(spread[7]);\n"
        "let typical: Series[Float] = (AAPL.high + AAPL.low + c) / 3;\n"  # Int divisor widens
        "print(typical[30]);\n"
        "print((-c)[0]);\n"
        "print(len(c - 1));\n",
        capsys,
    )
    spread = (_CLOSE - _PRICES.open.data) / _PRICES.open.data * 100.0
    typical = (_PRICES.high.data + _PRICES.low.data + _CLOSE) / 3
    assert [float(x) for x in lines[:3]] == [spread[7], typical[30], -_CLOSE[0]]
    assert lines[3] == str(len(_CLOSE))


def test_comparisons_and_logic_give_one_bool_per_bar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = _output(
        tmp_path,
        "let s: Series[Float] = [1.0, 5.0, 3.0, 8.0];\n"
        "print(s > 2.5);\n"
        "print(s > 2.5 and s < 6.0);\n"
        "print(!(s > 2.5) or s >= 8.0);\n"
        "print([true, false, true, true]);\n"
        "let enabled: Bool = false;\n"
        "print(enabled and s > 0.0);\n",
        capsys,
    )
    assert lines == [
        "Series[False, True, True, True]",
        "Series[False, True, True, False]",
        "Series[True, False, False, True]",
        "Series[True, False, True, True]",
        "Series[False, False, False, False]",
    ]


def test_warm_up_bars_never_satisfy_a_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # SMA's first 19 bars are NaN; NaN compares false either way round.
    lines = _output(
        tmp_path,
        "let slow: Series[Float] = SMA(c, 20);\nprint(count(c > slow or c <= slow));\n",
        capsys,
    )
    assert lines == [str(len(_CLOSE) - 19)]


def test_division_by_zero_in_a_series_is_ieee_not_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = _output(tmp_path, "print([1.0, -1.0, 0.0] / 0.0);\n", capsys)
    assert lines == ["Series[inf, -inf, nan]"]


def test_series_of_different_lengths_are_an_error_never_aligned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `unused` is never read; DCE must keep the comparison anyway, or the
    # optimized runs would succeed where the others fail.
    err = _error(
        tmp_path,
        "let a: Series[Float] = [1.0, 2.0];\n"
        "let unused: Series[Bool] = a < [1.0, 2.0, 3.0];\n"
        "print(1);\n",
        capsys,
    )
    assert "'<' needs series of the same length, got 2 and 3" in err


# ---- indicators, written in Allie ----


def test_indicator_recipes_match_numpy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lines = _output(
        tmp_path,
        "let macd: Series[Float] = EMA(c, 12) - EMA(c, 26);\n"
        "let signal: Series[Float] = EMA(macd, 9);\n"
        "let upper: Series[Float] = SMA(c, 20) + 2.0 * rolling_std(c, 20);\n"
        "let rsi: Series[Float] = RSI(c, 14);\n"
        "print(macd[40]);\n"
        "print(signal[40]);\n"
        "print(upper[40]);\n"
        "print(rsi[40]);\n"
        "print(count(macd crosses_above signal));\n"
        "print(count(rsi crosses_above 50.0));\n"
        "print(rolling_max(c, 10)[25] >= c[25]);\n"
        "print(ATR(AAPL, 14)[30] > 0.0);\n",
        capsys,
    )
    macd = EMA(_PRICES.close, 12).data - EMA(_PRICES.close, 26).data
    signal = EMA(type(_PRICES.close)(macd), 9).data
    upper = SMA(_PRICES.close, 20).data + 2.0 * rolling_std(_PRICES.close, 20).data
    rsi = RSI(_PRICES.close, 14).data
    assert [float(x) for x in lines[:4]] == [macd[40], signal[40], upper[40], rsi[40]]
    assert int(lines[4]) == int(np.count_nonzero(crossing("crosses_above", macd, signal).data))
    level = np.full(len(rsi), 50.0)
    assert int(lines[5]) == int(np.count_nonzero(crossing("crosses_above", rsi, level).data))
    assert lines[6:] == ["True", "True"]


# ---- strategies ----

_CROSSOVER = """
let fast: Series[Float] = SMA(c, 5);
let slow: Series[Float] = SMA(c, 20);
let rsi: Series[Float] = RSI(c, 14);
strategy Filtered {
    when fast crosses_above slow and rsi < 70.0: buy();
    when fast crosses_below slow or rsi > 80.0: sell();
}
"""


def test_a_compound_condition_is_the_same_as_its_precomputed_signal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = _output(
        tmp_path,
        _CROSSOVER + "let entry: Series[Bool] = fast crosses_above slow and rsi < 70.0;\n"
        "let exit: Series[Bool] = fast crosses_below slow or rsi > 80.0;\n"
        "strategy Precomputed { when entry: buy(); when exit: sell(); }\n"
        "let a = backtest Filtered on AAPL from 0 to 60;\n"
        "let b = backtest Precomputed on AAPL from 0 to 60;\n"
        "print(a.final_equity == b.final_equity);\n"
        "print(a.trades == b.trades);\n"
        "print(a.trades);\n",
        capsys,
    )
    assert lines[:2] == ["True", "True"]
    assert int(lines[2]) > 0  # the comparison isn't vacuous


def test_a_number_is_a_level_to_cross(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lines = _output(
        tmp_path,
        "let rsi: Series[Float] = RSI(c, 5);\n"
        "strategy MeanReversion {\n"
        "    when rsi crosses_below 40.0: buy();\n"
        "    when rsi crosses_above 60.0: sell();\n"
        "}\n"
        "let r = backtest MeanReversion on AAPL from 0 to 60;\n"
        "print(r.trades);\n"
        "print(count(rsi crosses_below 40.0));\n",
        capsys,
    )
    rsi = RSI(_PRICES.close, 5).data
    expected = np.count_nonzero(crossing("crosses_below", rsi, np.full(len(rsi), 40.0)).data)
    assert int(lines[1]) == expected > 0


def test_the_backtest_statement_prints_what_the_expression_returns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = _output(
        tmp_path,
        _CROSSOVER + "backtest Filtered on AAPL from 0 to 60;\n"
        "let r = backtest Filtered on AAPL from 0 to 60;\n"
        "print(r);\n",
        capsys,
    )
    half = len(lines) // 2
    assert lines[:half] == lines[half:]
    assert "BACKTEST RESULTS" in lines[1]


def test_backtest_result_fields(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lines = _output(
        tmp_path,
        _CROSSOVER + "let r: BacktestResult = backtest Filtered on AAPL from 5 to 55;\n"
        "print(len(r.equity));\n"
        "print(r.equity[0] == r.initial_equity);\n"
        "print(r.equity[len(r.equity) - 1] == r.final_equity);\n"
        "print(sharpe(r.returns, 252) == r.sharpe);\n"
        "print(r.max_drawdown >= 0.0 and r.volatility >= 0.0);\n"
        "let trades: Int = r.trades;\n"
        "let growth: Float = r.cagr;\n"
        "print(trades >= 0);\n",
        capsys,
    )
    assert lines == ["50", "True", "True", "True", "True", "True"]


def test_a_train_test_split_is_two_backtests_over_computed_bounds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = _output(
        tmp_path,
        _CROSSOVER + "let split: Int = 40;\n"
        "let train = backtest Filtered on AAPL from 0 to split;\n"
        "let test = backtest Filtered on AAPL from split to len(c);\n"
        "print(len(train.equity) + len(test.equity) == len(c));\n"
        "print(len(test.equity));\n",
        capsys,
    )
    assert lines == ["True", str(len(_CLOSE) - 40)]


def test_a_parameter_sweep_declares_its_strategy_in_a_loop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each iteration's `strategy` captures that iteration's `fast`, so
    the sweep's result for a window equals a program written for just
    that window."""
    sweep = _output(
        tmp_path,
        "let slow: Series[Float] = SMA(c, 20);\n"
        "let tested: Int = 0;\n"
        "for let w: Int = 3; w <= 9; w = w + 3 {\n"
        "    let fast: Series[Float] = SMA(c, w);\n"
        "    strategy Sweep {\n"
        "        when fast crosses_above slow: buy();\n"
        "        when fast crosses_below slow: sell();\n"
        "    }\n"
        "    let r = backtest Sweep on AAPL from 0 to len(c);\n"
        "    tested = tested + 1;\n"
        "    print(r.sharpe);\n"
        "}\n"
        "print(bonferroni(0.01, tested));\n",
        capsys,
    )
    for window, sharpe_line in zip((3, 6, 9), sweep[:3], strict=True):
        single = _output(
            tmp_path,
            f"let fast: Series[Float] = SMA(c, {window});\n"
            "let slow: Series[Float] = SMA(c, 20);\n"
            "strategy One {\n"
            "    when fast crosses_above slow: buy();\n"
            "    when fast crosses_below slow: sell();\n"
            "}\n"
            "print((backtest One on AAPL from 0 to len(c)).sharpe);\n",
            capsys,
        )
        assert single == [sharpe_line], window
    assert len(set(sweep[:3])) == 3  # the windows really differ
    assert sweep[3] == "0.03"


def test_a_condition_is_captured_when_the_strategy_is_declared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    declare = (
        "strategy S { when fast crosses_above slow: buy(); when fast crosses_below slow: sell(); }"
    )
    source = "\n".join(
        [
            "let fast: Series[Float] = SMA(c, 3);",
            "let slow: Series[Float] = SMA(c, 20);",
            declare,
            "let before = backtest S on AAPL from 0 to 60;",
            "fast = SMA(c, 9);",
            "let after = backtest S on AAPL from 0 to 60;",  # still SMA(c, 3)
            declare,
            "let redeclared = backtest S on AAPL from 0 to 60;",  # now SMA(c, 9)
            "print(before.final_equity == after.final_equity);",
            "print(before.final_equity == redeclared.final_equity);\n",
        ]
    )
    lines = _output(tmp_path, source, capsys)
    assert lines == ["True", "False"]


def test_a_condition_shorter_than_the_prices_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    err = _error(
        tmp_path,
        "strategy S { when [true, false]: buy(); }\nbacktest S on AAPL from 0 to 2;\n",
        capsys,
    )
    assert "the condition of 'when' clause 1 of 'S' has 2 bars, but AAPL has 60" in err


# ---- compile-time errors ----


@pytest.mark.parametrize(
    ("source", "message", "hint"),
    [
        (
            "strategy S { when c: buy(); }",
            "a 'when' condition must be a Series<Bool> (one value per bar), got Series<Float>",
            "compare it with something",
        ),
        (
            "strategy S { when c[0] > 1.0: buy(); }",
            "a 'when' condition must be a Series<Bool> (one value per bar), got Bool",
            "this is one true/false value",
        ),
        (
            "strategy S { when 1.0 crosses_above 2.0: buy(); }",
            "'crosses_above' compares series over time, got Float and Float",
            "at least one side must be a series",
        ),
        (
            "let same = c == AAPL.open;",
            "'==' can't compare whole series",
            "per-bar condition",
        ),
        (
            "let x = (c > 1.0) + 1.0;",
            "'+' isn't defined for Series<Bool> and Float",
            "series support",
        ),
        (
            "strategy Crossover { when c > 1.0: buy(); }\n"
            "let r = backtest Crosover on AAPL from 0 to 10;",
            "backtest refers to undeclared strategy 'Crosover'",
            "did you mean 'Crossover'?",
        ),
        (
            "strategy S { when c > 1.0: buy(); }\nlet r = backtest S on AAPL from 0 to 1.5 * 2;",
            "a bar index must be an Int, got Float",
            "bars are counted",
        ),
        (
            "strategy S { when c > 1.0: buy(); }\nlet r = backtest S on AAPL from 0 to 10;\n"
            "print(r.sharp);",
            "no field 'sharp'",
            "sharpe",
        ),
    ],
)
def test_mistakes_are_caught_before_anything_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str, message: str, hint: str | None
) -> None:
    path = tmp_path / "p.qs"
    path.write_text(_PRELUDE + source + "\n", encoding="utf-8")
    assert run_pipeline(path, backend="tree") == 1
    err = capsys.readouterr().err
    assert message in err
    if hint is not None:
        assert "help:" in err and hint in err


def test_look_ahead_is_rejected_inside_a_compound_condition(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "p.qs"
    path.write_text(
        _PRELUDE + "let rsi: Series[Float] = RSI(c, 14);\n"
        "strategy Cheat { when rsi < 30.0 and shift(c, -1) > c: buy(); }\n"
        "backtest Cheat on AAPL from 0 to 60;\n",
        encoding="utf-8",
    )
    for backend in BACKENDS:
        assert run_pipeline(path, backend=backend) == 1
        captured = capsys.readouterr()
        assert captured.out == ""  # nothing ran
        assert "look-ahead bias" in captured.err
        assert "'shift(...)' is derived from a backwards time shift" in captured.err
