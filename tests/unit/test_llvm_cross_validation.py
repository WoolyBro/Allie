"""Three-way cross-validation -- Phase 6.5's required suite: run the same
source through the tree-walking interpreter, the IR interpreter, and the
LLVM/MCJIT backend, and assert all three agree. Three independent
execution paths agreeing is a much stronger correctness claim than any
one of them passing its own tests.

Comparison is by *printed output*, not by inspecting variables: an LLVM
function's locals are gone once it returns, so `print(...)` is the only
observable all three backends share. Each program below therefore ends
by printing whatever it wants checked.

Scope, carried forward and re-stated rather than silently narrowed:

- **Closure-capture programs are excluded**, as in Phase 5.3 -- the IR
  path has no first-class function values (allie.ir.lowering's
  docstring), so neither the IR interpreter nor the LLVM backend can run
  them.
- **Every program here is fully type-annotated**, unlike Phase 5.3's
  suite. The LLVM backend needs the Phase 6 `register_types` retrofit,
  which is populated from the type checker's annotations -- so unlike
  the two interpreters, it can't run a program that never type-checked.
- **`for`/`break`/`continue` programs are included**, per Phase 6.5's
  explicit ask, with exact iteration counts asserted (test_iteration_*
  below) -- the desugar-then-lower path is new machinery neither
  interpreter exercises, and off-by-one is the classic failure here.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import allie.stdlib as stdlib
from allie.codegen.jit import compile_and_run
from allie.codegen.llvm_lowering import LLVMLowering
from allie.ir.interpreter import run_ir_program
from allie.ir.lowering import lower_program
from allie.lexer.lexer import Lexer
from allie.optimizer import constant_fold, cse, dce, run_passes
from allie.parser.parser import Parser
from allie.runtime.interpreter import run_program
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "AAPL_sample.csv"


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


def _run_tree_walker(source: str) -> str:
    program = _parse_and_check(source)
    env = stdlib.create_global_environment()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        run_program(program, env)
    return buffer.getvalue()


def _run_ir(source: str) -> str:
    program = _parse_and_check(source)
    main_program, functions, _register_types, _function_info = lower_program(program)
    env = stdlib.create_global_environment()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        run_ir_program(main_program, env, functions, stdlib.BUILTINS)
    return buffer.getvalue()


def _run_llvm(source: str, *, optimize: bool = False) -> str:
    program = _parse_and_check(source)
    main_program, functions, register_types, function_info = lower_program(program)
    if optimize:
        passes = [constant_fold.run, dce.run, cse.run]
        main_program = run_passes(main_program, passes, enabled=True)
        functions = {
            name: (run_passes(prog, passes, enabled=True), params)
            for name, (prog, params) in functions.items()
        }
    lowering = LLVMLowering(functions, function_info)
    lowering.declare_user_functions()
    for name in function_info:
        lowering.lower_function(name)
    main_func = lowering.lower_main(main_program, register_types)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        compile_and_run(lowering.module, main_func.name)
    return buffer.getvalue()


_PROGRAMS: dict[str, str] = {
    "arithmetic_precedence": "let x: Int = 1 + 2 * 3; print(x);",
    "int_float_promotion": "let x: Float = 1 + 2.5; print(x);",
    "modulo": "let x: Int = 17 % 5; print(x);",
    "unary_negate": "let x: Int = 0 - 7; let y: Int = -x; print(y);",
    "comparison_chain": "let x: Bool = (3 > 2) and (1 < 2); print(x);",
    "string_concatenation": 'let x: String = "hello, " + "allie"; print(x);',
    "and_short_circuits": "let x: Bool = false and (1 / 0 > 0); print(x);",
    "or_short_circuits": "let x: Bool = true or (1 / 0 > 0); print(x);",
    "if_else_both_branches": """
        let x: Int = 0;
        if 3 > 2 { x = 1; } else { x = 2; }
        let y: Int = 0;
        if 2 > 3 { y = 1; } else { y = 2; }
        print(x);
        print(y);
    """,
    "while_counts_to_n": """
        let i: Int = 0;
        let count: Int = 0;
        while i < 5 {
            count = count + 1;
            i = i + 1;
        }
        print(count);
    """,
    "while_break": """
        let i: Int = 0;
        while i < 100 {
            if i == 4 { break; }
            i = i + 1;
        }
        print(i);
    """,
    "for_counts_to_n": """
        let total: Int = 0;
        for let i: Int = 0; i < 5; i = i + 1 {
            total = total + 1;
        }
        print(total);
    """,
    "for_break": """
        let last: Int = -1;
        for let i: Int = 0; i < 10; i = i + 1 {
            last = i;
            if i == 3 { break; }
        }
        print(last);
    """,
    "for_continue": """
        let sum: Int = 0;
        for let i: Int = 0; i < 5; i = i + 1 {
            if i == 2 { continue; }
            sum = sum + i;
        }
        print(sum);
    """,
    "for_break_and_continue_together": """
        let sum: Int = 0;
        for let i: Int = 0; i < 10; i = i + 1 {
            if i == 8 { break; }
            if i == 3 { continue; }
            sum = sum + i;
        }
        print(sum);
    """,
    "nested_for_loops": """
        let total: Int = 0;
        for let i: Int = 0; i < 3; i = i + 1 {
            for let j: Int = 0; j < 3; j = j + 1 {
                total = total + 1;
            }
        }
        print(total);
    """,
    "recursive_factorial": """
        func factorial(n: Int) -> Int {
            if n <= 1 { return 1; } else { return n * factorial(n - 1); }
        }
        print(factorial(10));
    """,
    "recursive_fibonacci": """
        func fib(n: Int) -> Int {
            if n < 2 { return n; } else { return fib(n - 1) + fib(n - 2); }
        }
        print(fib(15));
    """,
    "function_with_loop": """
        func sum_to_n(n: Int) -> Int {
            let total: Int = 0;
            for let i: Int = 0; i < n; i = i + 1 {
                total = total + i;
            }
            return total;
        }
        print(sum_to_n(10));
    """,
    "array_literal_and_mean": """
        let xs: Series[Float] = [1.0, 2.0, 3.0, 4.0, 5.0];
        print(mean(xs));
    """,
    "quant_load_sma_mean": f"""
        let prices = load("{_FIXTURE.as_posix()}");
        let sma20 = SMA(prices.close, 20);
        print(sma20);
        print(mean(prices.close));
    """,
    "quant_for_loop_over_windows": f"""
        let prices = load("{_FIXTURE.as_posix()}");
        for let w: Int = 5; w <= 20; w = w + 5 {{
            print(mean(SMA(prices.close, w)));
        }}
    """,
}


@pytest.mark.parametrize("name", sorted(_PROGRAMS), ids=sorted(_PROGRAMS))
def test_all_three_backends_agree(name: str) -> None:
    source = _PROGRAMS[name]
    tree_out = _run_tree_walker(source)
    ir_out = _run_ir(source)
    llvm_out = _run_llvm(source)

    assert tree_out == ir_out, f"tree-walker vs IR interpreter disagree on {name}"
    assert tree_out == llvm_out, f"tree-walker vs LLVM backend disagree on {name}"


# ---- exact iteration counts through the JIT path specifically ----
#
# Phase 6.5 asks for these separately from the agreement suite above:
# off-by-one is the classic loop-lowering bug, and asserting an exact
# expected number (rather than just "all three match") is what catches a
# case where every backend is consistently wrong.


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("let n: Int = 0; for let i: Int = 0; i < 5; i = i + 1 { n = n + 1; } print(n);", "5"),
        ("let n: Int = 0; for let i: Int = 0; i < 1; i = i + 1 { n = n + 1; } print(n);", "1"),
        ("let n: Int = 0; for let i: Int = 0; i < 0; i = i + 1 { n = n + 1; } print(n);", "0"),
        ("let n: Int = 0; for let i: Int = 1; i <= 5; i = i + 1 { n = n + 1; } print(n);", "5"),
        (
            "let n: Int = 0; for let i: Int = 0; i < 10; i = i + 1 "
            "{ if i == 3 { break; } n = n + 1; } print(n);",
            "3",
        ),
        (
            "let n: Int = 0; for let i: Int = 0; i < 5; i = i + 1 "
            "{ if i == 2 { continue; } n = n + 1; } print(n);",
            "4",
        ),
        ("let n: Int = 0; let i: Int = 0; while i < 7 { n = n + 1; i = i + 1; } print(n);", "7"),
        (
            "let n: Int = 0; let i: Int = 0; while i < 100 "
            "{ if n == 6 { break; } n = n + 1; i = i + 1; } print(n);",
            "6",
        ),
    ],
    ids=[
        "for_lt_5",
        "for_lt_1",
        "for_lt_0_never_runs",
        "for_1_through_5_inclusive",
        "for_break_at_3",
        "for_continue_skips_one",
        "while_lt_7",
        "while_break_at_6",
    ],
)
def test_exact_iteration_count_through_jit(source: str, expected: str) -> None:
    assert _run_llvm(source).strip() == expected


@pytest.mark.parametrize("name", sorted(_PROGRAMS), ids=sorted(_PROGRAMS))
def test_optimized_llvm_matches_unoptimized(name: str) -> None:
    """The optimizer (Phase 5.4) runs over the same IR the LLVM backend
    then lowers, and its passes rewrite register references (CSE) and
    delete instructions (DCE) -- so it could in principle invalidate the
    `register_types` map LLVM lowering depends on. Running every program
    through the JIT both ways and comparing is what actually verifies
    it doesn't."""
    source = _PROGRAMS[name]
    assert _run_llvm(source, optimize=False) == _run_llvm(source, optimize=True)
