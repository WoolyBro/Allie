"""Tests for allie/runtime/interpreter/ against real source text through
the full lex -> parse -> eval pipeline. Supersedes test_phase1_eval.py's
hand-built tuples: the interpreter no longer accepts tuples (see
allie/runtime/interpreter/__init__.py's module docstring for the AST
adaptation this reflects) -- but every semantic property that file
proved (short-circuit and/or, closures by reference, recursion, for/
break/continue, division-by-zero raising a custom error) is re-proved
here, just driven by source text instead of hand-built trees.
"""

from __future__ import annotations

import pytest

from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.runtime.environment import Environment
from allie.runtime.errors import AllieValueError
from allie.runtime.interpreter import run_program


def _run(source: str) -> Environment:
    tokens = Lexer(source).scan_tokens()
    program = Parser(tokens).parse()
    env = Environment()
    run_program(program, env)
    return env


def test_arithmetic_precedence() -> None:
    env = _run("let x = 1 + 2 * 3;")
    assert env.get("x") == 7


def test_string_concatenation() -> None:
    env = _run('let x = "hello, " + "allie";')
    assert env.get("x") == "hello, allie"


def test_int_float_promotion() -> None:
    env = _run("let x = 1 + 2.5;")
    result = env.get("x")
    assert result == 3.5
    assert isinstance(result, float)


def test_division_by_zero_raises_custom_error() -> None:
    with pytest.raises(AllieValueError):
        _run("let x = 1 / 0;")


def test_if_true_branch() -> None:
    env = _run(
        """
        let x = 0;
        if true {
            x = 1;
        } else {
            x = 2;
        }
        """
    )
    assert env.get("x") == 1


def test_if_false_branch() -> None:
    env = _run(
        """
        let x = 0;
        if false {
            x = 1;
        } else {
            x = 2;
        }
        """
    )
    assert env.get("x") == 2


def test_while_counts_to_n() -> None:
    env = _run(
        """
        let i = 0;
        let count = 0;
        while i < 5 {
            count = count + 1;
            i = i + 1;
        }
        """
    )
    assert env.get("count") == 5


def test_for_counts_to_n() -> None:
    env = _run(
        """
        let total = 0;
        for let i = 0; i < 5; i = i + 1 {
            total = total + 1;
        }
        """
    )
    assert env.get("total") == 5


def test_for_break_at_known_iteration() -> None:
    env = _run(
        """
        let last = -1;
        for let i = 0; i < 10; i = i + 1 {
            last = i;
            if i == 3 {
                break;
            }
        }
        """
    )
    assert env.get("last") == 3


def test_while_break_at_known_iteration() -> None:
    env = _run(
        """
        let i = 0;
        while i < 100 {
            if i == 4 {
                break;
            }
            i = i + 1;
        }
        """
    )
    assert env.get("i") == 4


def test_for_continue_skips_known_iteration() -> None:
    env = _run(
        """
        let sum = 0;
        for let i = 0; i < 5; i = i + 1 {
            if i == 2 {
                continue;
            }
            sum = sum + i;
        }
        """
    )
    # i = 0,1,2,3,4; the sum-update is skipped exactly when i == 2.
    assert env.get("sum") == 0 + 1 + 3 + 4


def test_and_short_circuits() -> None:
    # Would raise AllieValueError if the right side were eagerly
    # evaluated -- proves it never is.
    env = _run("let x = false and (1 / 0 > 0);")
    assert env.get("x") is False


def test_or_short_circuits() -> None:
    env = _run("let x = true or (1 / 0 > 0);")
    assert env.get("x") is True


def test_recursive_factorial() -> None:
    env = _run(
        """
        func factorial(n) {
            if n <= 1 {
                return 1;
            } else {
                return n * factorial(n - 1);
            }
        }
        let result = factorial(5);
        """
    )
    assert env.get("result") == 120


def test_recursive_fibonacci() -> None:
    env = _run(
        """
        func fib(n) {
            if n < 2 {
                return n;
            } else {
                return fib(n - 1) + fib(n - 2);
            }
        }
        let result = fib(10);
        """
    )
    assert env.get("result") == 55


def test_returned_closure_captures_outer_variable() -> None:
    env = _run(
        """
        func make_adder() {
            let x = 100;
            func adder(n) {
                return x + n;
            }
            return adder;
        }
        let adder_ref = make_adder();
        let result = adder_ref(5);
        """
    )
    assert env.get("result") == 105


def test_closures_capture_environment_by_reference_not_by_value() -> None:
    # get/set share one enclosing scope's `x` binding. Mutating it
    # through set() must be visible through get() -- proves closures
    # capture their defining Environment by reference, not a snapshot
    # of its values at definition time.
    env = _run(
        """
        let x = 10;
        func get() {
            return x;
        }
        func set(new_x) {
            x = new_x;
            return x;
        }
        let before = get();
        set(99);
        let after = get();
        """
    )
    assert env.get("before") == 10
    assert env.get("after") == 99
