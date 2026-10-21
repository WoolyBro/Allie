"""Phase 7.5 end-to-end tests for the backtest DSL.

Three things this covers that nothing earlier did:

1. The full `MovingAverage` example running on the real fixture through
   the whole pipeline, asserting the results table renders and metrics
   land in sane *bounds* (not exact values -- it's realistic data).
2. **Three-backend agreement on the quant DSL specifically.** Phase 6.5
   established the "three independent execution paths agreeing"
   standard, but only ever tested it on scalar/loop programs. This is
   where it gets applied to `strategy`/`when`/`backtest`.
3. Re-running the Phase 2.5 semantic rules (`break`/`continue` inside a
   `when` action, `buy`/`sell` outside a strategy) against the *fully
   wired* checker and interpreter, confirming enforcement survived the
   stubs being replaced by real execution paths.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import allie.stdlib as stdlib
from allie.codegen.jit import compile_and_run
from allie.codegen.llvm_lowering import LLVMLowering
from allie.ir.interpreter import run_ir_program
from allie.ir.lowering import lower_program
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.quant.backtest_engine import reset_strategy_registry
from allie.runtime.errors import AllieValueError
from allie.runtime.interpreter import run_program
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker
from allie.typechecker.errors import QSTypeError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _REPO_ROOT / "examples" / "moving_average.qs"
_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "AAPL_sample.csv"
_FIXTURE_4_BARS = _REPO_ROOT / "tests" / "fixtures" / "AAPL_4bars.csv"


def _example_source() -> str:
    # The example hardcodes a repo-relative fixture path; rewrite it to
    # an absolute one so the test doesn't depend on the working dir.
    return _EXAMPLE.read_text(encoding="utf-8").replace(
        '"tests/fixtures/AAPL_sample.csv"', f'"{_FIXTURE.as_posix()}"'
    )


def _parse_and_check(source: str):  # noqa: ANN202
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
    return program


def _run_tree(source: str) -> str:
    reset_strategy_registry()
    program = _parse_and_check(source)
    env = stdlib.create_global_environment()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        run_program(program, env)
    return buffer.getvalue()


def _run_ir(source: str) -> str:
    reset_strategy_registry()
    program = _parse_and_check(source)
    main_program, functions, _rt, _fi = lower_program(program)
    env = stdlib.create_global_environment()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        run_ir_program(main_program, env, functions, stdlib.BUILTINS)
    return buffer.getvalue()


def _run_llvm(source: str) -> str:
    reset_strategy_registry()
    program = _parse_and_check(source)
    main_program, functions, register_types, function_info = lower_program(program)
    lowering = LLVMLowering(functions, function_info)
    lowering.declare_user_functions()
    for name in function_info:
        lowering.lower_function(name)
    main_func = lowering.lower_main(main_program, register_types)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        compile_and_run(lowering.module, main_func.name)
    return buffer.getvalue()


def _metric(output: str, label: str) -> float:
    match = re.search(rf"^{re.escape(label)}\s+(-?[\d,.]+)%?$", output, re.MULTILINE)
    assert match is not None, f"{label!r} not found in:\n{output}"
    return float(match.group(1).replace(",", ""))


# ---- the MovingAverage example, end to end ----


def test_moving_average_example_renders_and_metrics_are_sane() -> None:
    output = _run_tree(_example_source())

    assert "BACKTEST RESULTS" in output
    assert "MovingAverage" in output
    assert "AAPL" in output

    # Realistic data -- assert bounds, not exact values.
    assert -1.0 < _metric(output, "CAGR") / 100.0 < 1.0
    assert 0.0 <= _metric(output, "Volatility") / 100.0 < 2.0
    assert -10.0 < _metric(output, "Sharpe") < 10.0
    assert 0.0 <= _metric(output, "Max Drawdown") / 100.0 <= 1.0
    assert _metric(output, "Trades") >= 0
    assert _metric(output, "Bars") == 60  # the fixture's length, clamped from `to 100`
    assert _metric(output, "Initial Equity") == 10000


# ---- three-backend agreement on the quant DSL ----

_DSL_PROGRAMS: dict[str, str] = {
    "moving_average_example": _example_source(),
    "single_clause_only_buys": f"""
        let AAPL = load("{_FIXTURE.as_posix()}");
        let fast: Series[Float] = SMA(AAPL.close, 3);
        let slow: Series[Float] = SMA(AAPL.close, 10);
        strategy BuyOnly {{
            when fast crosses_above slow: buy();
        }}
        backtest BuyOnly on AAPL from 0 to 60;
    """,
    "narrow_bar_window": f"""
        let AAPL = load("{_FIXTURE.as_posix()}");
        let fast: Series[Float] = SMA(AAPL.close, 3);
        let slow: Series[Float] = SMA(AAPL.close, 10);
        strategy Narrow {{
            when fast crosses_above slow: buy();
            when fast crosses_below slow: sell();
        }}
        backtest Narrow on AAPL from 10 to 40;
    """,
    # A condition has one value per price bar, so four hand-written bars
    # need a four-bar price file.
    "array_literal_signals": """
        let up: Series[Float] = [1.0, 4.0, 1.0, 4.0];
        let flat: Series[Float] = [3.0, 3.0, 3.0, 3.0];
        let px = load("REPLACED");
        strategy Toy {
            when up crosses_above flat: buy();
            when up crosses_below flat: sell();
        }
        backtest Toy on px from 0 to 4;
    """.replace("REPLACED", _FIXTURE_4_BARS.as_posix()),
}


@pytest.mark.parametrize("name", sorted(_DSL_PROGRAMS), ids=sorted(_DSL_PROGRAMS))
def test_all_three_backends_agree_on_backtest_metrics(name: str) -> None:
    source = _DSL_PROGRAMS[name]
    tree_out = _run_tree(source)
    ir_out = _run_ir(source)
    llvm_out = _run_llvm(source)

    assert tree_out == ir_out, f"tree-walker vs IR interpreter disagree on {name}"
    assert tree_out == llvm_out, f"tree-walker vs LLVM/JIT disagree on {name}"
    assert "BACKTEST RESULTS" in tree_out


# ---- Phase 2.5 semantic rules, re-checked against the wired pipeline ----


@pytest.mark.parametrize("order", ["buy()", "sell()", "short()", "cover()", "target(0.5)"])
def test_an_order_outside_a_strategy_is_a_compile_error(order: str) -> None:
    name = order.split("(")[0]
    message = f"'{name}()' places an order, which only a strategy's 'when' action can do"
    with pytest.raises(QSTypeError, match=re.escape(message)):
        _run_tree(f"if true {{ {order}; }}")


def test_an_order_from_a_function_called_outside_a_backtest_raises_at_runtime() -> None:
    # A function may place orders -- an action can call it -- so this one
    # is only knowable when it runs.
    with pytest.raises(AllieValueError, match="only valid inside a strategy"):
        _run_tree("func enter() -> Void { buy(); }\nenter();")


def test_a_user_function_named_like_an_order_is_just_a_function() -> None:
    assert _run_tree('func target(x: Float) -> Void { print(x); }\ntarget(2.5);') == "2.5\n"


def _resolver_errors(source: str) -> list[str]:
    tokens = Lexer(source).scan_tokens()
    program = Parser(tokens).parse()
    resolver = Resolver()
    resolver.resolve(program)
    return [d.message for d in resolver.diagnostics.diagnostics]


def test_break_inside_a_when_action_is_rejected() -> None:
    errors = _resolver_errors(
        """
        strategy S {
            when a crosses_above b: break;
        }
        """
    )
    assert any("break" in e for e in errors)


def test_continue_inside_a_when_action_is_rejected() -> None:
    errors = _resolver_errors(
        """
        strategy S {
            when a crosses_above b: continue;
        }
        """
    )
    assert any("continue" in e for e in errors)


def test_break_in_a_when_action_cannot_target_an_enclosing_loop() -> None:
    # The `inside_when` flag existed since Phase 2.5 but was never read,
    # so this case silently passed until Phase 7. A `when` action is a
    # per-bar callback -- it cannot jump out to a loop enclosing the
    # strategy declaration.
    errors = _resolver_errors(
        """
        for let i: Int = 0; i < 5; i = i + 1 {
            strategy S {
                when a crosses_above b: break;
            }
        }
        """
    )
    assert any("when" in e and "break" in e for e in errors)


def test_break_in_a_loop_written_inside_a_when_action_is_allowed() -> None:
    # The negative case: `inside_when` must not subsume `loop_depth`.
    errors = _resolver_errors(
        """
        strategy S {
            when a crosses_above b: { while true { break; } }
        }
        """
    )
    assert not any("break" in e for e in errors)


def test_break_inside_a_function_cannot_target_an_enclosing_loop() -> None:
    # Same class of bug as the `when` case, found alongside it: a
    # function body is its own execution context.
    errors = _resolver_errors("while true { func f() -> Void { break; } }")
    assert any("break" in e for e in errors)


# ---- backtest / strategy type rules (Phase 7.2) ----


def test_backtest_on_undeclared_strategy_is_rejected() -> None:
    errors = _resolver_errors(
        f'let AAPL = load("{_FIXTURE.as_posix()}"); backtest Ghost on AAPL from 0 to 5;'
    )
    assert any("Ghost" in e for e in errors)


def test_backtest_on_a_non_strategy_is_a_type_error() -> None:
    with pytest.raises(QSTypeError, match="not a strategy"):
        _run_tree(
            f'let AAPL = load("{_FIXTURE.as_posix()}");\n'
            "let NotAStrat: Int = 5;\n"
            "backtest NotAStrat on AAPL from 0 to 5;"
        )


def test_backtest_symbol_must_be_a_price_series() -> None:
    with pytest.raises(QSTypeError, match="must be a PriceSeries"):
        _run_tree(
            "let fast: Series[Float] = [1.0, 4.0];\n"
            "let slow: Series[Float] = [3.0, 3.0];\n"
            "let notprices: Int = 5;\n"
            "strategy S { when fast crosses_above slow: buy(); }\n"
            "backtest S on notprices from 0 to 2;"
        )


def test_a_when_condition_must_be_a_series_of_bools() -> None:
    with pytest.raises(QSTypeError, match="must be a Series<Bool>"):
        _run_tree(
            "let slow: Series[Float] = [3.0, 3.0];\n"
            "strategy S { when slow: buy(); }"
        )
