"""Cross-validation: run the same source through both the tree-walking
interpreter (allie.runtime.interpreter) and the lowering -> IR
interpreter path (allie.ir), and assert identical results. This is what
proves the `for` desugar preserves loop semantics (`continue` running
the increment, `break` skipping it), not just that lowering produces
*some* IR -- so it reuses the exact for/break/continue programs from
tests/unit/test_interpreter.py, plus the quant integration programs.

Two of test_interpreter.py's programs are NOT reused here: the
closure-capture tests (a function returning a function; two functions
sharing one mutable variable). They run untyped programs on the tree
walker alone. The language has no closures -- the type checker rejects
them -- and the IR has no first-class functions (allie.ir.lowering's
module docstring), so those programs have no way to run through this
path at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import allie.stdlib as stdlib
from allie.ir.interpreter import run_ir_program
from allie.ir.lowering import lower_program
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.runtime.environment import Environment
from allie.runtime.errors import AllieValueError
from allie.runtime.interpreter import run_program
from allie.runtime.values import SeriesValue

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "AAPL_sample.csv"


def _run_tree_walker(source: str) -> Environment:
    tokens = Lexer(source).scan_tokens()
    program = Parser(tokens).parse()
    env = stdlib.create_global_environment()
    run_program(program, env)
    return env


def _run_ir(source: str) -> Environment:
    tokens = Lexer(source).scan_tokens()
    program = Parser(tokens).parse()
    main_program, functions, _register_types, _function_info = lower_program(program)
    env = stdlib.create_global_environment()
    run_ir_program(main_program, env, functions, stdlib.BUILTINS)
    return env


def _assert_cross_validates(source: str, var_name: str) -> None:
    tree_env = _run_tree_walker(source)
    ir_env = _run_ir(source)
    tree_value = tree_env.get(var_name)
    ir_value = ir_env.get(var_name)
    if isinstance(tree_value, SeriesValue):
        assert isinstance(ir_value, SeriesValue)
        np.testing.assert_allclose(tree_value.data, ir_value.data, equal_nan=True)
    else:
        assert tree_value == ir_value


@pytest.mark.parametrize(
    ("source", "var_name"),
    [
        ("let x = 1 + 2 * 3;", "x"),
        ('let x = "hello, " + "allie";', "x"),
        ("let x = 1 + 2.5;", "x"),
        (
            """
            let i = 0;
            let count = 0;
            while i < 5 {
                count = count + 1;
                i = i + 1;
            }
            """,
            "count",
        ),
        (
            """
            let total = 0;
            for let i = 0; i < 5; i = i + 1 {
                total = total + 1;
            }
            """,
            "total",
        ),
        (
            """
            let last = -1;
            for let i = 0; i < 10; i = i + 1 {
                last = i;
                if i == 3 {
                    break;
                }
            }
            """,
            "last",
        ),
        (
            """
            let i = 0;
            while i < 100 {
                if i == 4 {
                    break;
                }
                i = i + 1;
            }
            """,
            "i",
        ),
        (
            """
            let sum = 0;
            for let i = 0; i < 5; i = i + 1 {
                if i == 2 {
                    continue;
                }
                sum = sum + i;
            }
            """,
            "sum",
        ),
        ("let x = false and (1 / 0 > 0);", "x"),
        ("let x = true or (1 / 0 > 0);", "x"),
        (
            """
            func factorial(n) {
                if n <= 1 {
                    return 1;
                } else {
                    return n * factorial(n - 1);
                }
            }
            let result = factorial(5);
            """,
            "result",
        ),
        (
            """
            func fib(n) {
                if n < 2 {
                    return n;
                } else {
                    return fib(n - 1) + fib(n - 2);
                }
            }
            let result = fib(10);
            """,
            "result",
        ),
    ],
    ids=[
        "arithmetic_precedence",
        "string_concatenation",
        "int_float_promotion",
        "while_counts_to_n",
        "for_counts_to_n",
        "for_break_at_known_iteration",
        "while_break_at_known_iteration",
        "for_continue_skips_known_iteration",
        "and_short_circuits",
        "or_short_circuits",
        "recursive_factorial",
        "recursive_fibonacci",
    ],
)
def test_tree_walker_and_ir_produce_identical_results(source: str, var_name: str) -> None:
    _assert_cross_validates(source, var_name)


def test_division_by_zero_raises_in_both_interpreters() -> None:
    source = "let x = 1 / 0;"
    with pytest.raises(AllieValueError):
        _run_tree_walker(source)
    with pytest.raises(AllieValueError):
        _run_ir(source)


def test_quant_load_sma_mean_cross_validates() -> None:
    source = f"""
    let prices = load("{_FIXTURE.as_posix()}");
    let sma20 = SMA(prices.close, 20);
    let m = mean(prices.close);
    """
    _assert_cross_validates(source, "sma20")
    _assert_cross_validates(source, "m")


def test_quant_for_loop_over_windows_cross_validates() -> None:
    # Same shape as tests/integration/test_quant_pipeline.py's second
    # test -- control flow (for) composed with quant types (SMA),
    # cross-validated end to end.
    source = f"""
    let prices = load("{_FIXTURE.as_posix()}");
    let last_sma = [0.0];
    for let w = 5; w <= 20; w = w + 5 {{
        last_sma = SMA(prices.close, w);
    }}
    """
    _assert_cross_validates(source, "last_sma")
