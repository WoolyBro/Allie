"""Tests for allie/typechecker/ (Phase 3). One test per rule, both a
passing and a failing case where the spec asked for it."""

from __future__ import annotations

import pytest

from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.typechecker.checker import TypeChecker
from allie.typechecker.errors import QSTypeError


def _check(source: str) -> None:
    tokens = Lexer(source).scan_tokens()
    program = Parser(tokens).parse()
    TypeChecker().check(program)


def _check_expr_type(source: str) -> str:
    """Type-checks `let __t = <source>;` and returns the inferred type's
    str() for the initializer expression."""
    tokens = Lexer(f"let __t = {source};").scan_tokens()
    program = Parser(tokens).parse()
    TypeChecker().check(program)
    return str(program.statements[0].initializer.inferred_type)  # type: ignore[attr-defined]


# ---- let / assignability ----


def test_let_int_from_int_literal_passes() -> None:
    _check("let x: Int = 10;")


def test_let_int_from_string_fails_with_exact_message() -> None:
    with pytest.raises(QSTypeError) as exc_info:
        _check('let x: Int = "hello";')
    assert str(exc_info.value) == "TypeError: cannot assign String to Int. Line 1, Column 14"


def test_let_float_from_int_widens() -> None:
    _check("let x: Float = 10;")


def test_let_int_from_float_narrowing_rejected() -> None:
    with pytest.raises(QSTypeError):
        _check("let x: Int = 3.14;")


# ---- binary operators ----


def test_binary_arithmetic_type_mismatch_fails() -> None:
    with pytest.raises(QSTypeError):
        _check_expr_type('1 + "a"')


def test_binary_arithmetic_int_float_passes_with_float_result() -> None:
    assert _check_expr_type("1 + 2.0") == "Float"


def test_comparison_result_is_bool() -> None:
    assert _check_expr_type("1 < 2") == "Bool"


def test_using_comparison_result_where_int_expected_fails() -> None:
    with pytest.raises(QSTypeError):
        _check("let x: Int = 1 < 2;")


def test_equality_between_same_types_passes() -> None:
    assert _check_expr_type('"a" == "b"') == "Bool"


def test_equality_between_different_types_fails() -> None:
    with pytest.raises(QSTypeError):
        _check_expr_type('1 == "a"')


def test_and_or_require_bool_operands() -> None:
    assert _check_expr_type("true and false") == "Bool"
    with pytest.raises(QSTypeError):
        _check_expr_type("1 and 2")


# ---- unary operators ----


def test_unary_minus_requires_numeric() -> None:
    assert _check_expr_type("-1") == "Int"
    with pytest.raises(QSTypeError):
        _check_expr_type('-"a"')


def test_unary_bang_requires_bool() -> None:
    assert _check_expr_type("!true") == "Bool"
    with pytest.raises(QSTypeError):
        _check_expr_type("!1")


# ---- function calls ----


def test_function_call_arity_mismatch_fails() -> None:
    with pytest.raises(QSTypeError):
        _check("func add(a: Int, b: Int) -> Int { return a + b; } let x = add(1);")


def test_function_call_arg_type_mismatch_fails() -> None:
    with pytest.raises(QSTypeError):
        _check(
            'func add(a: Int, b: Int) -> Int { return a + b; } let x = add(1, "two");'
        )


def test_function_call_correct_types_passes() -> None:
    _check("func add(a: Int, b: Int) -> Int { return a + b; } let x = add(1, 2);")


def test_recursive_function_type_checks() -> None:
    _check(
        """
        func factorial(n: Int) -> Int {
            if n <= 1 {
                return 1;
            } else {
                return n * factorial(n - 1);
            }
        }
        let result: Int = factorial(5);
        """
    )


# ---- condition must be Bool ----


def test_if_condition_not_bool_fails() -> None:
    with pytest.raises(QSTypeError):
        _check("if 1 { }")


def test_while_condition_not_bool_fails() -> None:
    with pytest.raises(QSTypeError):
        _check("while 1 { }")


def test_for_condition_not_bool_fails() -> None:
    # Tested separately from if/while, not assumed to work by analogy.
    with pytest.raises(QSTypeError):
        _check("for let i = 0; i; i = i + 1 { }")


def test_if_while_for_condition_bool_all_pass() -> None:
    _check("if true { }")
    _check("while false { }")
    _check("for let i = 0; i < 1; i = i + 1 { }")


# ---- return-on-every-path ----


def test_missing_return_on_non_void_function_fails() -> None:
    with pytest.raises(QSTypeError):
        _check(
            """
            func maybe(n: Int) -> Int {
                if n > 0 {
                    return n;
                }
            }
            """
        )


def test_return_on_every_path_passes() -> None:
    _check(
        """
        func abs(n: Int) -> Int {
            if n < 0 {
                return -n;
            } else {
                return n;
            }
        }
        """
    )


def test_void_function_does_not_need_a_return() -> None:
    _check('func log(msg: String) -> Void { let x = msg; }')


# ---- regression net: every valid Phase-2 evaluator program still type-checks ----


def test_phase2_for_loop_program_type_checks_cleanly() -> None:
    _check(
        """
        func run() -> Int {
            let total: Int = 0;
            for let i = 0; i < 5; i = i + 1 {
                total = total + i;
            }
            return total;
        }
        """
    )


def test_phase2_break_program_type_checks_cleanly() -> None:
    _check(
        """
        func run() -> Int {
            let last: Int = -1;
            for let i = 0; i < 10; i = i + 1 {
                last = i;
                if i == 3 {
                    break;
                }
            }
            return last;
        }
        """
    )


def test_phase2_continue_program_type_checks_cleanly() -> None:
    _check(
        """
        func run() -> Int {
            let sum: Int = 0;
            for let i = 0; i < 5; i = i + 1 {
                if i == 2 {
                    continue;
                }
                sum = sum + i;
            }
            return sum;
        }
        """
    )


def test_phase2_recursive_fibonacci_type_checks_cleanly() -> None:
    _check(
        """
        func fib(n: Int) -> Int {
            if n < 2 {
                return n;
            } else {
                return fib(n - 1) + fib(n - 2);
            }
        }
        """
    )


def test_phase2_closure_style_nested_function_type_checks_cleanly() -> None:
    _check(
        """
        func makeAdder(x: Int) -> Int {
            func adder(n: Int) -> Int {
                return x + n;
            }
            return adder(5);
        }
        """
    )
