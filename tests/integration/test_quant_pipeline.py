"""End-to-end integration tests: real .ql-shaped source through the full
lex -> parse -> resolve -> type-check -> run pipeline, exercising the
stdlib against the synthetic fixture in tests/fixtures/.

The first test is the smallest quant program there is (`load` ->
`SMA` -> `print`), with the semicolons docs/grammar.md's
`letStmt`/`exprStmt` require; that's a syntax difference from matching
this project's real, already-committed grammar, not a semantic one.

The second combines this with a `for` loop (computing SMA at several
window sizes and printing each), the same cross-family concern a later
`blended.qs`-style test would check, in miniature, now -- catching
anything the control flow and the quant types don't compose
cleanly on yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import allie.stdlib as stdlib
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.runtime.environment import Environment
from allie.runtime.interpreter import run_program
from allie.runtime.values import SeriesValue
from allie.semantic.resolver import Resolver
from allie.stdlib.series_ops import SMA
from allie.typechecker.checker import TypeChecker

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "AAPL_sample.csv"


def _run_full_pipeline(source: str) -> Environment:
    tokens = Lexer(source).scan_tokens()
    parser = Parser(tokens)
    program = parser.parse()
    assert not parser.diagnostics.has_errors(), parser.diagnostics.diagnostics

    resolver = Resolver()
    resolver.resolve(program)
    assert not resolver.diagnostics.has_errors(), resolver.diagnostics.diagnostics

    checker = TypeChecker()
    for name, sig in stdlib.builtin_function_types().items():
        checker.env.define(name, sig)
    checker.check(program)

    env = stdlib.create_global_environment()
    run_program(program, env)
    return env


def test_load_sma_print_reference_example(capsys) -> None:  # type: ignore[no-untyped-def]
    source = f"""
    let prices = load("{_FIXTURE.as_posix()}");
    let sma20 = SMA(prices.close, 20);
    print(sma20);
    """
    env = _run_full_pipeline(source)

    prices = env.get("prices")
    sma20 = env.get("sma20")
    assert isinstance(sma20, SeriesValue)
    assert len(sma20) == 60

    # First window-1 = 19 points are NaN, the 20th onward are real.
    assert all(np.isnan(x) for x in sma20.data[:19])
    assert not np.isnan(sma20.data[19])

    # Cross-check against computing SMA directly on the same loaded data.
    expected = SMA(prices.close, 20).data
    np.testing.assert_allclose(sma20.data, expected, equal_nan=True)

    captured = capsys.readouterr()
    assert "Series[" in captured.out


def test_for_loop_over_multiple_sma_windows(capsys) -> None:  # type: ignore[no-untyped-def]
    source = f"""
    let prices = load("{_FIXTURE.as_posix()}");
    for let w = 5; w <= 20; w = w + 5 {{
        let s = SMA(prices.close, w);
        print(s);
    }}
    """
    _run_full_pipeline(source)

    captured = capsys.readouterr()
    printed_series = [line for line in captured.out.splitlines() if line.startswith("Series[")]
    # windows 5, 10, 15, 20 -> 4 prints.
    assert len(printed_series) == 4
