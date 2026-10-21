"""The backtest engine from source: `with` options, the order builtins,
stops, and the result fields -- each program run on every backend,
optimized and not, which must agree exactly (see test_strategy_dsl.py).
The engine's arithmetic is checked by hand in test_backtest_engine.py;
here the question is whether source reaches it intact."""

from __future__ import annotations

from pathlib import Path

import pytest

from allie.cli import run_pipeline
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from tests.unit.test_strategy_dsl import _PRELUDE, _error, _output

_CROSSOVER = """
let fast: Series[Float] = SMA(c, 5);
let slow: Series[Float] = SMA(c, 20);
strategy Cross {
    when fast crosses_above slow: buy();
    when fast crosses_below slow: sell();
}
"""


def _floats(lines: list[str]) -> list[float]:
    return [float(line) for line in lines]


# ---- options ----


def test_options_reach_the_engine_on_every_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    default, free, close, costly = _floats(
        _output(
            tmp_path,
            _CROSSOVER + "print((backtest Cross on AAPL from 0 to 60).final_equity);\n"
            "print((backtest Cross on AAPL from 0 to 60 with slippage: 0.0).final_equity);\n"
            'print((backtest Cross on AAPL from 0 to 60 with fill: "close", slippage: 0.0)'
            ".final_equity);\n"
            "print((backtest Cross on AAPL from 0 to 60 with commission: 0.01).final_equity);\n",
            capsys,
        )
    )
    assert default < free  # costs are on by default
    assert costly < default
    # v1's frictionless same-bar-close model, kept as an option.
    assert close == pytest.approx(9930.307025805236)
    # The fixture has no overnight gaps -- every open is the previous
    # close -- so on it the two fill points trade at the same prices.
    # test_backtest_engine.py tells them apart on gapped prices.
    assert close == free


def test_option_values_are_expressions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lines = _output(
        tmp_path,
        _CROSSOVER + "let bps: Float = 0.0001;\n"
        "let a = backtest Cross on AAPL from 0 to 60 with slippage: 10 * bps, cash: 50000;\n"
        "let b = backtest Cross on AAPL from 0 to 60 with slippage: 0.001, cash: 50000.0;\n"
        "print(a.final_equity == b.final_equity);\n"
        "print(a.initial_equity);\n",
        capsys,
    )
    assert lines == ["True", "50000.0"]  # the Int literal widened to a Float


def test_a_computed_option_is_checked_when_it_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    err = _error(
        tmp_path,
        _CROSSOVER + "let bps: Float = 0.5;\n"
        "backtest Cross on AAPL from 0 to 60 with slippage: bps * 4.0;\n",
        capsys,
    )
    assert "backtest: 'slippage' is a fraction of the price from 0 up to 1" in err


# ---- orders ----


def test_short_cover_and_target_agree_everywhere(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = _output(
        tmp_path,
        _CROSSOVER + "strategy LongShort {\n"
        "    when fast crosses_above slow: buy();\n"
        "    when fast crosses_below slow: short();\n"
        "}\n"
        "strategy HalfIn {\n"
        "    when fast crosses_above slow: target(0.5);\n"
        "    when fast crosses_below slow: target(0.0);\n"
        "}\n"
        "let ls = backtest LongShort on AAPL from 0 to 60 with slippage: 0.0;\n"
        "let half = backtest HalfIn on AAPL from 0 to 60 with slippage: 0.0;\n"
        "let full = backtest Cross on AAPL from 0 to 60 with slippage: 0.0;\n"
        "print(ls.exposure > full.exposure);\n"  # a reversal is never flat
        "print(half.final_equity - half.initial_equity);\n"
        "print(full.final_equity - full.initial_equity);\n",
        capsys,
    )
    assert lines[0] == "True"
    half_pnl, full_pnl = _floats(lines[1:])
    assert half_pnl == pytest.approx(full_pnl / 2, rel=0.02)  # half the bet, about half the P&L


def test_stops_change_the_exits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lines = _output(
        tmp_path,
        "let rsi: Series[Float] = RSI(c, 5);\n"
        "strategy Dip { when rsi < 45.0: buy(); }\n"  # re-enters whenever it's flat and dipped
        "let plain = backtest Dip on AAPL from 0 to 60;\n"
        "let stopped = backtest Dip on AAPL from 0 to 60 with stop_loss: 0.01, take_profit: 0.02;\n"
        "print(plain.trades);\n"
        "print(stopped.trades);\n"
        "print(len(stopped.trade_returns) == stopped.trades);\n",
        capsys,
    )
    plain, stopped = int(lines[0]), int(lines[1])
    assert plain == 0  # held from the first dip to the end
    assert stopped > 1
    assert lines[2] == "True"


# ---- result fields ----


def test_result_fields_are_consistent_with_each_other(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = _output(
        tmp_path,
        _CROSSOVER + "let r = backtest Cross on AAPL from 0 to 60 with commission: 0.001;\n"
        "print(r.total_return == r.final_equity / r.initial_equity - 1.0);\n"
        "print(len(r.position) == len(r.equity) and len(r.drawdown) == len(r.equity));\n"
        "print(rolling_max(r.drawdown, len(r.drawdown))[59] == r.max_drawdown);\n"
        "print(r.costs > 0.0);\n"
        "print(r.exposure > 0.0 and r.exposure < 1.0);\n"
        "print(r.buy_hold_return == c[59] / c[0] - 1.0);\n"
        "print(len(r.trade_returns) == r.trades);\n"
        "print(r.win_rate >= 0.0 and r.win_rate <= 1.0);\n"
        "print(r.sortino);\n"
        "print(r.profit_factor >= 0.0);\n",
        capsys,
    )
    assert lines[:8] == ["True"] * 8
    float(lines[8])  # a number, possibly inf
    assert lines[9] == "True"


def test_trade_statistics_are_nan_without_trades(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = _output(
        tmp_path,
        "strategy Never { when c < 0.0: buy(); }\n"
        "let r = backtest Never on AAPL from 0 to 60;\n"
        "print(r.trades);\n"
        "print(r.win_rate);\n"
        "print(len(r.trade_returns));\n",
        capsys,
    )
    assert lines == ["0", "nan", "0"]


# ---- compile-time errors ----


@pytest.mark.parametrize(
    ("options", "message", "hint"),
    [
        ("slipage: 0.001", "unknown backtest option 'slipage'", "did you mean 'slippage'?"),
        ("leverage: 2.0", "unknown backtest option 'leverage'", "the options are cash"),
        ("fill: 1.0", "backtest option 'fill' must be String, got Float", "in quotes"),
        ("cash: true", "backtest option 'cash' must be Float, got Bool", None),
        ("slippage: 5.0", "'slippage' is a fraction of the price from 0 up to 1", None),
        ("commission: -0.1", "'commission' is a fraction of the value traded", None),
        ("stop_loss: 1.5", "'stop_loss' is a fraction of the entry price between 0 and 1", None),
        ('fill: "midday"', '\'fill\' is "open" (the next bar\'s open) or "close"', None),
        ("slippage: 0.001, slippage: 0.002", "'slippage' is set twice", "set it once"),
    ],
)
def test_option_mistakes_are_caught_before_anything_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    options: str,
    message: str,
    hint: str | None,
) -> None:
    path = tmp_path / "p.qs"
    path.write_text(
        _PRELUDE + _CROSSOVER + f"backtest Cross on AAPL from 0 to 60 with {options};\n",
        encoding="utf-8",
    )
    assert run_pipeline(path, backend="tree") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
    if hint is not None:
        assert hint in captured.err


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ("", "expected an option name after 'with', found ';'"),
        ("slippage 0.001", "expected ':' after the option name 'slippage', found '0.001'"),
        ("slippage: 0.001,", "expected an option name after ',', found ';'"),
        ("fill: close", 'the fill point is a string, not a name: fill: "close"'),
    ],
)
def test_malformed_with_clauses_are_syntax_errors(options: str, message: str) -> None:
    parser = Parser(Lexer(f"backtest S on A from 0 to 10 with {options};").scan_tokens())
    parser.parse()
    assert [d.message for d in parser.diagnostics.diagnostics] == [message]


def test_options_parse_in_order_on_both_forms() -> None:
    source = (
        'backtest S on A from 0 to 10 with fill: "close", stop_loss: 0.05;\n'
        "let r = backtest S on A from 0 to 10 with cash: 1.0 * 2;\n"
    )
    statement, let = Parser(Lexer(source).scan_tokens()).parse().statements
    assert [o.name for o in statement.options] == ["fill", "stop_loss"]  # type: ignore[attr-defined]
    assert [o.name for o in let.initializer.options] == ["cash"]  # type: ignore[attr-defined]
