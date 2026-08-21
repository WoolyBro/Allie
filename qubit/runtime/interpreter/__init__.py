"""Phase 1: hand-built tree-walking evaluator.

Nodes are plain tuples: ('num', 1), ('binop', '+', left, right), etc. --
see docs/architecture.md "Phase 1 -- hand-built evaluator" and
tests/unit/test_phase1_eval.py, which construct every tree by hand. This
module is deliberately parser-free: Phase 2's real parser targets this same
node shape, so this evaluator should not need to change when it lands.
"""

from __future__ import annotations

from typing import Any

from qubit.runtime.environment import Environment
from qubit.runtime.errors import QubitValueError
from qubit.runtime.values import Closure, Value, promote_numeric

Node = tuple[Any, ...]


class BreakSignal(Exception):
    """Internal control-flow signal for `break`. Never surfaces to the user."""


class ContinueSignal(Exception):
    """Internal control-flow signal for `continue`. Never surfaces to the user."""


class ReturnSignal(Exception):
    """Internal control-flow signal for `return`. Never surfaces to the user."""

    def __init__(self, value: Value) -> None:
        super().__init__()
        self.value = value


def truthy(value: Value) -> bool:
    return bool(value)


_COMPARISON_OPS: dict[str, Any] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}


def eval_expr(node: Node, env: Environment) -> Value:
    tag = node[0]

    if tag in ("num", "str", "bool"):
        return node[1]

    if tag == "var":
        return env.get(node[1])

    if tag == "assign":
        _, name, expr = node
        value = eval_expr(expr, env)
        env.assign(name, value)
        return value

    if tag == "unary":
        _, op, operand_node = node
        operand = eval_expr(operand_node, env)
        if op == "-":
            if isinstance(operand, bool) or not isinstance(operand, (int, float)):
                raise QubitValueError(f"unary '-' requires a number, got {operand!r}")
            return -operand
        if op == "!":
            return not truthy(operand)
        raise QubitValueError(f"unknown unary operator: {op!r}")

    if tag == "and":
        _, left_node, right_node = node
        left = eval_expr(left_node, env)
        if not truthy(left):
            return left
        return eval_expr(right_node, env)

    if tag == "or":
        _, left_node, right_node = node
        left = eval_expr(left_node, env)
        if truthy(left):
            return left
        return eval_expr(right_node, env)

    if tag == "binop":
        return _eval_binop(node, env)

    if tag == "call":
        return _eval_call(node, env)

    raise QubitValueError(f"unknown expression node: {tag!r}")


def _eval_binop(node: Node, env: Environment) -> Value:
    _, op, left_node, right_node = node
    left = eval_expr(left_node, env)
    right = eval_expr(right_node, env)

    if op in _COMPARISON_OPS:
        return bool(_COMPARISON_OPS[op](left, right))

    if op == "+":
        if isinstance(left, str) or isinstance(right, str):
            return f"{left}{right}"
        left, right = promote_numeric(left, right)
        return left + right

    if op in ("-", "*"):
        left, right = promote_numeric(left, right)
        return (left - right) if op == "-" else (left * right)

    if op == "/":
        left, right = promote_numeric(left, right)
        if right == 0:
            raise QubitValueError("division by zero")
        return left / right

    if op == "%":
        left, right = promote_numeric(left, right)
        if right == 0:
            raise QubitValueError("modulo by zero")
        return left % right

    raise QubitValueError(f"unknown binary operator: {op!r}")


def _eval_call(node: Node, env: Environment) -> Value:
    _, callee_node, arg_nodes = node
    callee = eval_expr(callee_node, env)
    if not isinstance(callee, Closure):
        raise QubitValueError(f"cannot call non-function value: {callee!r}")

    args = [eval_expr(arg, env) for arg in arg_nodes]
    if len(args) != len(callee.params):
        raise QubitValueError(
            f"{callee!r} expects {len(callee.params)} argument(s), got {len(args)}"
        )

    call_env = Environment(parent=callee.defining_env)
    for name, value in zip(callee.params, args, strict=True):
        call_env.define(name, value)

    try:
        exec_stmt(callee.body, call_env)
    except ReturnSignal as ret:
        return ret.value
    return None


def exec_stmt(node: Node, env: Environment) -> None:
    tag = node[0]

    if tag == "let":
        _, name, expr = node
        env.define(name, eval_expr(expr, env))
        return

    if tag == "expr_stmt":
        eval_expr(node[1], env)
        return

    if tag == "if":
        _, cond, then_block, else_block = node
        if truthy(eval_expr(cond, env)):
            exec_stmt(then_block, env)
        elif else_block is not None:
            exec_stmt(else_block, env)
        return

    if tag == "while":
        _, cond, body = node
        while truthy(eval_expr(cond, env)):
            try:
                exec_stmt(body, env)
            except BreakSignal:
                break
            except ContinueSignal:
                continue
        return

    if tag == "for":
        _exec_for(node, env)
        return

    if tag == "block":
        _, statements = node
        block_env = Environment(parent=env)
        for stmt in statements:
            exec_stmt(stmt, block_env)
        return

    if tag == "break":
        raise BreakSignal()

    if tag == "continue":
        raise ContinueSignal()

    if tag == "return":
        _, expr = node
        value = eval_expr(expr, env) if expr is not None else None
        raise ReturnSignal(value)

    if tag == "func":
        _, name, params, body = node
        env.define(name, Closure(params=params, body=body, defining_env=env))
        return

    raise QubitValueError(f"unknown statement node: {tag!r}")


def _exec_for(node: Node, env: Environment) -> None:
    _, init_let, cond, increment, body = node
    loop_env = Environment(parent=env)
    exec_stmt(init_let, loop_env)
    while truthy(eval_expr(cond, loop_env)):
        try:
            exec_stmt(body, loop_env)
        except BreakSignal:
            break
        except ContinueSignal:
            pass
        eval_expr(increment, loop_env)
