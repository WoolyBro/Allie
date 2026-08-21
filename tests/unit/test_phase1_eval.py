"""Phase 1 evaluator tests. Every tree below is hand-built -- no lexer or
parser exists yet (docs/architecture.md "Phase 1 -- hand-built evaluator").
"""

from __future__ import annotations

import pytest

from runtime.environment import Environment
from runtime.errors import QuantLangValueError
from runtime.interpreter import eval_expr, exec_stmt
from runtime.values import Closure


def test_arithmetic_precedence() -> None:
    # 1 + 2 * 3
    tree = ("binop", "+", ("num", 1), ("binop", "*", ("num", 2), ("num", 3)))
    assert eval_expr(tree, Environment()) == 7


def test_string_concatenation() -> None:
    tree = ("binop", "+", ("str", "hello, "), ("str", "quantlang"))
    assert eval_expr(tree, Environment()) == "hello, quantlang"


def test_int_float_promotion() -> None:
    tree = ("binop", "+", ("num", 1), ("num", 2.5))
    result = eval_expr(tree, Environment())
    assert result == 3.5
    assert isinstance(result, float)


def test_division_by_zero_raises_custom_error() -> None:
    tree = ("binop", "/", ("num", 1), ("num", 0))
    with pytest.raises(QuantLangValueError):
        eval_expr(tree, Environment())


def test_if_true_branch() -> None:
    env = Environment()
    env.define("x", 0)
    then_block = ("block", [("expr_stmt", ("assign", "x", ("num", 1)))])
    else_block = ("block", [("expr_stmt", ("assign", "x", ("num", 2)))])

    exec_stmt(("if", ("bool", True), then_block, else_block), env)

    assert env.get("x") == 1


def test_if_false_branch() -> None:
    env = Environment()
    env.define("x", 0)
    then_block = ("block", [("expr_stmt", ("assign", "x", ("num", 1)))])
    else_block = ("block", [("expr_stmt", ("assign", "x", ("num", 2)))])

    exec_stmt(("if", ("bool", False), then_block, else_block), env)

    assert env.get("x") == 2


def test_while_counts_to_n() -> None:
    env = Environment()
    env.define("i", 0)
    env.define("count", 0)
    body = (
        "block",
        [
            ("expr_stmt", ("assign", "count", ("binop", "+", ("var", "count"), ("num", 1)))),
            ("expr_stmt", ("assign", "i", ("binop", "+", ("var", "i"), ("num", 1)))),
        ],
    )

    exec_stmt(("while", ("binop", "<", ("var", "i"), ("num", 5)), body), env)

    assert env.get("count") == 5


def _counting_for_loop(total_var: str, limit: int) -> tuple:
    init_let = ("let", "i", ("num", 0))
    cond = ("binop", "<", ("var", "i"), ("num", limit))
    increment = ("assign", "i", ("binop", "+", ("var", "i"), ("num", 1)))
    body = (
        "block",
        [("expr_stmt", ("assign", total_var, ("binop", "+", ("var", total_var), ("num", 1))))],
    )
    return ("for", init_let, cond, increment, body)


def test_for_counts_to_n() -> None:
    env = Environment()
    env.define("total", 0)

    exec_stmt(_counting_for_loop("total", 5), env)

    assert env.get("total") == 5


def test_for_break_at_known_iteration() -> None:
    env = Environment()
    env.define("last", -1)
    init_let = ("let", "i", ("num", 0))
    cond = ("binop", "<", ("var", "i"), ("num", 10))
    increment = ("assign", "i", ("binop", "+", ("var", "i"), ("num", 1)))
    body = (
        "block",
        [
            ("expr_stmt", ("assign", "last", ("var", "i"))),
            ("if", ("binop", "==", ("var", "i"), ("num", 3)), ("block", [("break",)]), None),
        ],
    )

    exec_stmt(("for", init_let, cond, increment, body), env)

    assert env.get("last") == 3


def test_while_break_at_known_iteration() -> None:
    env = Environment()
    env.define("i", 0)
    body = (
        "block",
        [
            ("if", ("binop", "==", ("var", "i"), ("num", 4)), ("block", [("break",)]), None),
            ("expr_stmt", ("assign", "i", ("binop", "+", ("var", "i"), ("num", 1)))),
        ],
    )

    exec_stmt(("while", ("binop", "<", ("var", "i"), ("num", 100)), body), env)

    assert env.get("i") == 4


def test_for_continue_skips_known_iteration() -> None:
    env = Environment()
    env.define("sum", 0)
    init_let = ("let", "i", ("num", 0))
    cond = ("binop", "<", ("var", "i"), ("num", 5))
    increment = ("assign", "i", ("binop", "+", ("var", "i"), ("num", 1)))
    body = (
        "block",
        [
            ("if", ("binop", "==", ("var", "i"), ("num", 2)), ("block", [("continue",)]), None),
            ("expr_stmt", ("assign", "sum", ("binop", "+", ("var", "sum"), ("var", "i")))),
        ],
    )

    exec_stmt(("for", init_let, cond, increment, body), env)

    # i = 0,1,2,3,4; the sum-update is skipped exactly when i == 2.
    assert env.get("sum") == 0 + 1 + 3 + 4


def test_and_short_circuits() -> None:
    env = Environment()
    false_node = ("bool", False)
    # Would raise QuantLangValueError if evaluated -- proves it never is.
    side_effect_node = ("binop", "/", ("num", 1), ("num", 0))

    assert eval_expr(("and", false_node, side_effect_node), env) is False


def test_or_short_circuits() -> None:
    env = Environment()
    true_node = ("bool", True)
    side_effect_node = ("binop", "/", ("num", 1), ("num", 0))

    assert eval_expr(("or", true_node, side_effect_node), env) is True


def test_recursive_factorial() -> None:
    env = Environment()
    # func factorial(n) { if (n <= 1) { return 1; } else { return n * factorial(n - 1); } }
    body = (
        "block",
        [
            (
                "if",
                ("binop", "<=", ("var", "n"), ("num", 1)),
                ("block", [("return", ("num", 1))]),
                (
                    "block",
                    [
                        (
                            "return",
                            (
                                "binop",
                                "*",
                                ("var", "n"),
                                (
                                    "call",
                                    ("var", "factorial"),
                                    [("binop", "-", ("var", "n"), ("num", 1))],
                                ),
                            ),
                        )
                    ],
                ),
            )
        ],
    )

    exec_stmt(("func", "factorial", ["n"], body), env)
    result = eval_expr(("call", ("var", "factorial"), [("num", 5)]), env)

    assert result == 120


def test_recursive_fibonacci() -> None:
    env = Environment()
    # func fib(n) { if (n < 2) { return n; } else { return fib(n-1) + fib(n-2); } }
    body = (
        "block",
        [
            (
                "if",
                ("binop", "<", ("var", "n"), ("num", 2)),
                ("block", [("return", ("var", "n"))]),
                (
                    "block",
                    [
                        (
                            "return",
                            (
                                "binop",
                                "+",
                                (
                                    "call",
                                    ("var", "fib"),
                                    [("binop", "-", ("var", "n"), ("num", 1))],
                                ),
                                (
                                    "call",
                                    ("var", "fib"),
                                    [("binop", "-", ("var", "n"), ("num", 2))],
                                ),
                            ),
                        )
                    ],
                ),
            )
        ],
    )

    exec_stmt(("func", "fib", ["n"], body), env)
    result = eval_expr(("call", ("var", "fib"), [("num", 10)]), env)

    assert result == 55


def test_returned_closure_captures_outer_variable() -> None:
    env = Environment()
    # func make_adder() { let x = 100; func adder(n) { return x + n; } return adder; }
    inner_body = ("block", [("return", ("binop", "+", ("var", "x"), ("var", "n")))])
    outer_body = (
        "block",
        [
            ("let", "x", ("num", 100)),
            ("func", "adder", ["n"], inner_body),
            ("return", ("var", "adder")),
        ],
    )

    exec_stmt(("func", "make_adder", [], outer_body), env)
    adder = eval_expr(("call", ("var", "make_adder"), []), env)
    assert isinstance(adder, Closure)

    env.define("adder_ref", adder)
    assert eval_expr(("call", ("var", "adder_ref"), [("num", 5)]), env) == 105


def test_closures_capture_environment_by_reference_not_by_value() -> None:
    # Two closures sharing one defining Environment -- mutating `x` through
    # one must be visible through the other, proving capture is by
    # reference to the Environment, not a snapshot of its values.
    outer_env = Environment()
    shared_scope = Environment(parent=outer_env)
    shared_scope.define("x", 10)

    get_closure = Closure(
        params=[], body=("block", [("return", ("var", "x"))]), defining_env=shared_scope
    )
    set_closure = Closure(
        params=["new_x"],
        body=("block", [("expr_stmt", ("assign", "x", ("var", "new_x")))]),
        defining_env=shared_scope,
    )
    outer_env.define("get", get_closure)
    outer_env.define("set", set_closure)

    assert eval_expr(("call", ("var", "get"), []), outer_env) == 10

    eval_expr(("call", ("var", "set"), [("num", 99)]), outer_env)

    assert eval_expr(("call", ("var", "get"), []), outer_env) == 99
