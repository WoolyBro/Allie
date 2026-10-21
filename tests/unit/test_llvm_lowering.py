"""Tests for allie/codegen/{llvm_lowering,jit}.py -- Phase 6. Verifies
the LLVM backend on its own (execution correctness, native-vs-bridged
dispatch, the documented non-scalar-user-function limitation). See
test_llvm_cross_validation.py for the required three-way agreement
suite against the tree-walking and IR interpreters.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import allie.stdlib as stdlib
from allie.codegen.jit import compile_and_run
from allie.codegen.llvm_lowering import LLVMLowering
from allie.ir.lowering import lower_program
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.runtime.errors import AllieValueError
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker


def _run_llvm(source: str) -> str:
    """Type-checks and JIT-compiles `source`, returning everything it
    printed. Programs are expected to `print(...)` whatever they want
    verified -- there's no way to read an LLVM function's local
    variables back out after it returns, so output is the interface."""
    tokens = Lexer(source).scan_tokens()
    program = Parser(tokens).parse()
    Resolver().resolve(program)

    checker = TypeChecker()
    for name, sig in stdlib.builtin_function_types().items():
        checker.env.define(name, sig)
    checker.check(program)

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


def test_native_arithmetic() -> None:
    assert _run_llvm("let x: Int = 1 + 2 * 3; print(x);").strip() == "7"


def test_int_float_promotion() -> None:
    assert _run_llvm("let x: Float = 1 + 2.5; print(x);").strip() == "3.5"


def test_division_by_zero_raises() -> None:
    with pytest.raises(AllieValueError):
        _run_llvm("let x: Float = 1 / 0; print(x);")


def test_division_by_runtime_zero_raises_without_crashing() -> None:
    # Non-constant divisor, so LLVM can't fold it away: this is the case
    # that would execute a real `sdiv i64 %a, 0` -- undefined behavior
    # that traps and kills the process -- if the emitted guard fell
    # through to the division instead of returning early.
    with pytest.raises(AllieValueError):
        _run_llvm("let x: Int = 10; let y: Int = 0; let z: Float = x / y; print(z);")


def test_division_by_zero_inside_non_void_function_raises() -> None:
    # Exercises the guard's dummy-return path (a non-void function has
    # to return *something* when it bails out early).
    with pytest.raises(AllieValueError):
        _run_llvm(
            """
            func divide(a: Int, b: Int) -> Float {
                return a / b;
            }
            print(divide(10, 0));
            """
        )


def test_modulo_by_zero_raises() -> None:
    with pytest.raises(AllieValueError):
        _run_llvm("let a: Int = 7; let b: Int = 0; let c: Int = a % b; print(c);")


def test_comparison_and_if_else() -> None:
    out = _run_llvm(
        """
        let x: Int = 0;
        if 3 > 2 {
            x = 1;
        } else {
            x = 2;
        }
        print(x);
        """
    )
    assert out.strip() == "1"


def test_while_loop() -> None:
    out = _run_llvm(
        """
        let i: Int = 0;
        let count: Int = 0;
        while i < 5 {
            count = count + 1;
            i = i + 1;
        }
        print(count);
        """
    )
    assert out.strip() == "5"


def test_for_loop_with_break_and_continue() -> None:
    out = _run_llvm(
        """
        let sum: Int = 0;
        for let i: Int = 0; i < 10; i = i + 1 {
            if i == 8 { break; }
            if i == 3 { continue; }
            sum = sum + i;
        }
        print(sum);
        """
    )
    # i = 0,1,2,4,5,6,7 (3 skipped, stops before 8): sum = 25.
    assert out.strip() == "25"


def test_short_circuit_and() -> None:
    # Would raise (division by zero) if the right side were evaluated --
    # proves the native comparison short-circuits, same as the other
    # two interpreters.
    out = _run_llvm("let x: Bool = false and (1 / 0 > 0); print(x);")
    assert out.strip() == "False"


def test_recursive_factorial() -> None:
    out = _run_llvm(
        """
        func factorial(n: Int) -> Int {
            if n <= 1 {
                return 1;
            } else {
                return n * factorial(n - 1);
            }
        }
        print(factorial(10));
        """
    )
    assert out.strip() == "3628800"


def test_recursive_fibonacci() -> None:
    out = _run_llvm(
        """
        func fib(n: Int) -> Int {
            if n < 2 {
                return n;
            } else {
                return fib(n - 1) + fib(n - 2);
            }
        }
        print(fib(15));
        """
    )
    assert out.strip() == "610"


def test_string_concatenation_via_bridge() -> None:
    out = _run_llvm('let x: String = "hello, " + "allie"; print(x);')
    assert out.strip() == "hello, allie"


def test_quant_sma_via_bridge() -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "AAPL_sample.csv"
    out = _run_llvm(
        f"""
        let prices = load("{fixture.as_posix()}");
        let sma20 = SMA(prices.close, 20);
        print(sma20);
        """
    )
    assert "Series[" in out
    assert "..." in out  # 60 elements, more than the compact-print threshold


def test_mean_bridge_result_used_in_native_arithmetic() -> None:
    # mean() returns a boxed Float handle; using it in further native
    # arithmetic (+ 1) requires unboxing back to a native double first.
    fixture_source = """
    let prices = [1.0, 2.0, 3.0, 4.0, 5.0];
    let m: Float = mean(prices);
    let result: Float = m + 1.0;
    print(result);
    """
    out = _run_llvm(fixture_source)
    assert out.strip() == "4.0"


def test_user_functions_can_take_and_return_series() -> None:
    # Series cross a native call as boxed handles, like any other
    # non-scalar value.
    out = _run_llvm(
        """
        func total(prices: Series[Float]) -> Float {
            return mean(prices);
        }
        func smooth(prices: Series[Float], n: Int) -> Series[Float] {
            return SMA(prices, n);
        }
        print(total([1.0, 2.0]));
        print(smooth([1.0, 2.0, 3.0], 2));
        """
    )
    assert out.split("\n")[:2] == ["1.5", "Series[nan, 1.5, 2.5]"]
