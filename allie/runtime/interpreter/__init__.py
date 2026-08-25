"""Tree-walking evaluator -- Phase 1 validated these semantics on
hand-built tuples; Phase 4 needs real execution (ArrayLiteral producing
a SeriesValue, FieldAccess reading a PriceSeriesValue, calling stdlib
builtins), which requires walking the actual AST allie.parser.parser
produces (allie.ast.nodes), not tuples. This module has been rewritten
to dispatch on node *type* (isinstance checks against the dataclasses in
allie.ast.nodes) instead of a tuple's first-element tag string.

Every evaluation rule below is unchanged from Phase 1 -- short-circuit
and/or, break/continue/return via internal signal exceptions, closures
capturing their defining Environment by reference, division/modulo by
zero raising AllieValueError -- only the node shape changed. See
tests/unit/test_interpreter.py, which re-proves each of these properties
by running real .ql source text through the full lex -> parse -> eval
pipeline (superseding tests/unit/test_phase1_eval.py's hand-built
tuples, which can no longer run against this module's new contract).

Each `allie run` invocation should construct a fresh Environment/
Interpreter so runs are isolated (docs/spec.md "Concurrency model").
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from allie.ast.nodes import (
    ArrayLiteral,
    Assignment,
    BacktestStmt,
    Binary,
    Block,
    BoolLiteral,
    BreakStmt,
    Call,
    ContinueStmt,
    Expr,
    ExprStmt,
    FieldAccess,
    ForStmt,
    FuncDecl,
    Identifier,
    IfStmt,
    ImportStmt,
    LetStmt,
    Logical,
    NumberLiteral,
    Program,
    ReturnStmt,
    Stmt,
    StrategyDecl,
    StringLiteral,
    Unary,
    WhenClause,
    WhileStmt,
)
from allie.quant.backtest_engine import WhenSignal, run_backtest
from allie.runtime.environment import Environment
from allie.runtime.errors import AllieValueError
from allie.runtime.values import (
    BoolSeriesValue,
    BuiltinFunction,
    Closure,
    PriceSeriesValue,
    SeriesValue,
    StrategyValue,
    Value,
    promote_numeric,
)
from allie.stdlib.series_ops import crosses_above, crosses_below

_COMPARISON_OPS: dict[str, Any] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}

_PRICE_SERIES_FIELDS = ("open", "high", "low", "close", "volume", "dates")


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


def run_program(program: Program, env: Environment) -> None:
    for stmt in program.statements:
        exec_stmt(stmt, env)


def eval_expr(node: Expr, env: Environment) -> Value:
    if isinstance(node, (NumberLiteral, StringLiteral, BoolLiteral)):
        return node.value

    if isinstance(node, Identifier):
        return env.get(node.name)

    if isinstance(node, Assignment):
        value = eval_expr(node.value, env)
        env.assign(node.name, value)
        return value

    if isinstance(node, Unary):
        return _eval_unary(node, env)

    if isinstance(node, Logical):
        left = eval_expr(node.left, env)
        if node.operator == "and":
            if not truthy(left):
                return left
            return eval_expr(node.right, env)
        if node.operator == "or":
            if truthy(left):
                return left
            return eval_expr(node.right, env)
        raise AllieValueError(f"unknown logical operator: {node.operator!r}")

    if isinstance(node, Binary):
        return _eval_binary(node, env)

    if isinstance(node, Call):
        return _eval_call(node, env)

    if isinstance(node, FieldAccess):
        return _eval_field_access(node, env)

    if isinstance(node, ArrayLiteral):
        elements = [eval_expr(e, env) for e in node.elements]
        return SeriesValue(np.array(elements, dtype=np.float64))

    raise AllieValueError(f"unknown expression node: {type(node).__name__}")


def _eval_unary(node: Unary, env: Environment) -> Value:
    operand = eval_expr(node.operand, env)
    if node.operator == "-":
        if isinstance(operand, bool) or not isinstance(operand, (int, float)):
            raise AllieValueError(f"unary '-' requires a number, got {operand!r}")
        return -operand
    if node.operator == "!":
        return not truthy(operand)
    raise AllieValueError(f"unknown unary operator: {node.operator!r}")


def _eval_binary(node: Binary, env: Environment) -> Value:
    left = eval_expr(node.left, env)
    right = eval_expr(node.right, env)
    op = node.operator

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
            raise AllieValueError("division by zero")
        return left / right

    if op == "%":
        left, right = promote_numeric(left, right)
        if right == 0:
            raise AllieValueError("modulo by zero")
        return left % right

    raise AllieValueError(f"unknown binary operator: {op!r}")


def _eval_call(node: Call, env: Environment) -> Value:
    callee = eval_expr(node.callee, env)
    args = [eval_expr(arg, env) for arg in node.arguments]

    if isinstance(callee, Closure):
        if len(args) != len(callee.params):
            raise AllieValueError(
                f"expected {len(callee.params)} argument(s), got {len(args)}"
            )
        call_env = Environment(parent=callee.defining_env)
        for name, value in zip(callee.params, args, strict=True):
            call_env.define(name, value)
        try:
            exec_stmt(callee.body, call_env)
        except ReturnSignal as ret:
            return ret.value
        return None

    if isinstance(callee, BuiltinFunction):
        return callee.fn(*args)

    raise AllieValueError(f"cannot call non-function value: {callee!r}")


def _eval_field_access(node: FieldAccess, env: Environment) -> Value:
    obj = eval_expr(node.obj, env)
    if isinstance(obj, PriceSeriesValue):
        if node.name in _PRICE_SERIES_FIELDS:
            return getattr(obj, node.name)
        raise AllieValueError(f"PriceSeriesValue has no field '{node.name}'")
    raise AllieValueError(f"cannot access field '{node.name}' on {type(obj).__name__}")


def exec_stmt(node: Stmt, env: Environment) -> None:
    if isinstance(node, LetStmt):
        env.define(node.name, eval_expr(node.initializer, env))
        return

    if isinstance(node, ExprStmt):
        eval_expr(node.expression, env)
        return

    if isinstance(node, IfStmt):
        if truthy(eval_expr(node.condition, env)):
            exec_stmt(node.then_branch, env)
        elif node.else_branch is not None:
            exec_stmt(node.else_branch, env)
        return

    if isinstance(node, WhileStmt):
        while truthy(eval_expr(node.condition, env)):
            try:
                exec_stmt(node.body, env)
            except BreakSignal:
                break
            except ContinueSignal:
                continue
        return

    if isinstance(node, ForStmt):
        _exec_for(node, env)
        return

    if isinstance(node, Block):
        block_env = Environment(parent=env)
        for stmt in node.statements:
            exec_stmt(stmt, block_env)
        return

    if isinstance(node, BreakStmt):
        raise BreakSignal()

    if isinstance(node, ContinueStmt):
        raise ContinueSignal()

    if isinstance(node, ReturnStmt):
        value = eval_expr(node.value, env) if node.value is not None else None
        raise ReturnSignal(value)

    if isinstance(node, FuncDecl):
        params = [param.name for param in node.params]
        env.define(node.name, Closure(params=params, body=node.body, defining_env=env))
        return

    if isinstance(node, ImportStmt):
        # No module system exists yet -- inert, same treatment as the
        # type checker's resolver/checker give it.
        return

    if isinstance(node, StrategyDecl):
        # Declaration binds a value; the clause operands aren't evaluated
        # until a `backtest` statement actually runs (see StrategyValue).
        env.define(
            node.name,
            StrategyValue(name=node.name, clauses=node.when_clauses, defining_env=env),
        )
        return

    if isinstance(node, BacktestStmt):
        _exec_backtest(node, env)
        return

    raise AllieValueError(f"unknown statement node: {type(node).__name__}")


def _exec_backtest(node: BacktestStmt, env: Environment) -> None:
    """`backtest S on SYM from N to M;` -- resolves both names, computes
    each `when` clause's signal series, and hands everything to the one
    shared engine entry point (allie.quant.backtest_engine.run_backtest)
    that every other backend also calls."""
    strategy = env.get(node.strategy_name)
    if not isinstance(strategy, StrategyValue):
        raise AllieValueError(f"'{node.strategy_name}' is not a strategy")

    # `on SYM` names an ordinary variable holding a PriceSeriesValue --
    # e.g. `let AAPL = load("...")`. The grammar makes it an IDENTIFIER,
    # so this is the natural reading and needs no new syntax.
    prices = env.get(node.symbol)
    if not isinstance(prices, PriceSeriesValue):
        raise AllieValueError(
            f"backtest: '{node.symbol}' must be a PriceSeries (from load(...)), "
            f"got {type(prices).__name__}"
        )

    clauses = [
        WhenSignal(
            signal=_eval_when_signal(clause, strategy.defining_env),
            action=_make_when_action(clause, strategy.defining_env),
        )
        for clause in strategy.clauses
    ]

    result = run_backtest(
        strategy_name=node.strategy_name,
        symbol=node.symbol,
        prices=prices,
        clauses=clauses,
        start=int(node.start),
        end=int(node.end),
    )
    print(result.render())


def _eval_when_signal(clause: WhenClause, env: Environment) -> BoolSeriesValue:
    left = eval_expr(clause.left, env)
    right = eval_expr(clause.right, env)
    if not isinstance(left, SeriesValue) or not isinstance(right, SeriesValue):
        raise AllieValueError("'when' clause operands must be Series<Float>")
    crossing = crosses_above if clause.operator == "crosses_above" else crosses_below
    return crossing(left, right)


def _make_when_action(clause: WhenClause, env: Environment) -> Callable[[], None]:
    def action() -> None:
        exec_stmt(clause.body, Environment(parent=env))

    return action


def _exec_for(node: ForStmt, env: Environment) -> None:
    loop_env = Environment(parent=env)
    exec_stmt(node.initializer, loop_env)
    while truthy(eval_expr(node.condition, loop_env)):
        try:
            exec_stmt(node.body, loop_env)
        except BreakSignal:
            break
        except ContinueSignal:
            pass
        eval_expr(node.increment, loop_env)
