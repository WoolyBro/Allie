"""IR interpreter -- Phase 5.3. A second, independent execution path for
the same programs the tree-walking interpreter (allie.runtime.
interpreter) already covers -- see tests/unit/test_ir_cross_validation.py,
which runs the same source through both and asserts identical results.
Operator/field-access logic is deliberately re-implemented here rather
than imported from the tree-walker: sharing those helpers would let a
bug common to both paths hide from cross-validation, defeating the
point of having two independent implementations to compare.

Variable scoping is deliberately flat: one Environment per execution
(module-level, or one per function call -- functions don't capture a
lexical parent, per allie.ir.lowering's documented "no true closures"
simplification), with no child scope per if/while/for block. Safe for
every program this phase's cross-validation actually covers (none of
them shadow a variable in a nested block); `StoreVar` always
defines-or-overwrites in that one flat scope rather than distinguishing
"new binding" from "reassignment" the way the tree-walking interpreter's
`Environment.define`/`assign` do.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from allie.ir.instructions import (
    BinOp,
    Call,
    IRProgram,
    Jump,
    JumpIfFalse,
    Label,
    LoadConst,
    LoadSeries,
    LoadVar,
    MemberGet,
    Move,
    Return,
    StoreVar,
    UnaryOp,
)
from allie.quant.backtest_engine import (
    add_strategy_clause,
    declare_strategy,
    register_when_body,
    run_registered_backtest,
)
from allie.runtime.environment import Environment
from allie.runtime.errors import AllieValueError
from allie.runtime.values import PriceSeriesValue, Value, promote_numeric
from allie.runtime.values import SeriesValue as _SeriesValue

_PRICE_SERIES_FIELDS = ("open", "high", "low", "close", "volume", "dates")

_COMPARISON_OPS: dict[str, Any] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}

Functions = dict[str, tuple[IRProgram, list[str]]]
Builtins = dict[str, tuple[Any, Any]]


def _truthy(value: Value) -> bool:
    return bool(value)


def _apply_binop(op: str, left: Value, right: Value) -> Value:
    # promote_numeric's result is assigned to fresh names, not `left`/
    # `right` themselves: reassigning an explicitly-annotated *parameter*
    # keeps mypy holding it to the declared Value union (so `a + b` still
    # gets checked against every non-numeric member of that union and
    # fails); a plain local narrows to the call's Any result instead. The
    # tree-walking interpreter's equivalent code only avoids this because
    # its `left`/`right` are inferred locals, not parameters.
    if op in _COMPARISON_OPS:
        return bool(_COMPARISON_OPS[op](left, right))
    if op == "+":
        if isinstance(left, str) or isinstance(right, str):
            return f"{left}{right}"
        num_left, num_right = promote_numeric(left, right)
        return num_left + num_right
    if op in ("-", "*"):
        num_left, num_right = promote_numeric(left, right)
        return (num_left - num_right) if op == "-" else (num_left * num_right)
    if op == "/":
        num_left, num_right = promote_numeric(left, right)
        if num_right == 0:
            raise AllieValueError("division by zero")
        return num_left / num_right
    if op == "%":
        num_left, num_right = promote_numeric(left, right)
        if num_right == 0:
            raise AllieValueError("modulo by zero")
        return num_left % num_right
    raise AllieValueError(f"unknown binary operator: {op!r}")


def _apply_unary(op: str, operand: Value) -> Value:
    if op == "-":
        if isinstance(operand, bool) or not isinstance(operand, (int, float)):
            raise AllieValueError(f"unary '-' requires a number, got {operand!r}")
        return -operand
    if op == "!":
        return not _truthy(operand)
    raise AllieValueError(f"unknown unary operator: {op!r}")


def run_ir_program(
    program: IRProgram,
    env: Environment,
    functions: Functions,
    builtins: Builtins | None = None,
) -> Value:
    """Executes `program`'s flat instruction list against `env` and
    `functions`/`builtins` for Call dispatch. Returns the value of the
    first Return instruction reached, or None if execution falls off
    the end of the list."""
    builtins = builtins if builtins is not None else {}
    _register_when_bodies(env, functions, builtins)
    instructions = program.instructions
    label_index = {
        instr.name: i for i, instr in enumerate(instructions) if isinstance(instr, Label)
    }
    registers: dict[str, Value] = {}

    pc = 0
    while pc < len(instructions):
        instr = instructions[pc]

        if isinstance(instr, Label):
            pc += 1
        elif isinstance(instr, LoadConst):
            registers[instr.dest] = instr.value
            pc += 1
        elif isinstance(instr, LoadVar):
            registers[instr.dest] = env.get(instr.name)
            pc += 1
        elif isinstance(instr, StoreVar):
            env.define(instr.name, registers[instr.src])
            pc += 1
        elif isinstance(instr, Move):
            registers[instr.dest] = registers[instr.src]
            pc += 1
        elif isinstance(instr, BinOp):
            registers[instr.dest] = _apply_binop(
                instr.op, registers[instr.left], registers[instr.right]
            )
            pc += 1
        elif isinstance(instr, UnaryOp):
            registers[instr.dest] = _apply_unary(instr.op, registers[instr.operand])
            pc += 1
        elif isinstance(instr, MemberGet):
            obj = registers[instr.obj]
            if isinstance(obj, PriceSeriesValue) and instr.field in _PRICE_SERIES_FIELDS:
                registers[instr.dest] = getattr(obj, instr.field)
            else:
                raise AllieValueError(
                    f"cannot access field '{instr.field}' on {type(obj).__name__}"
                )
            pc += 1
        elif isinstance(instr, Call):
            result = _exec_call(instr, registers, functions, builtins)
            if instr.dest is not None:
                registers[instr.dest] = result
            pc += 1
        elif isinstance(instr, Jump):
            pc = label_index[instr.label]
        elif isinstance(instr, JumpIfFalse):
            pc = label_index[instr.label] if not _truthy(registers[instr.cond]) else pc + 1
        elif isinstance(instr, Return):
            return registers[instr.value] if instr.value is not None else None
        elif isinstance(instr, LoadSeries):
            raise AllieValueError(
                "LoadSeries is not emitted by the lowering pass and has no interpreter support"
            )
        else:
            raise AllieValueError(f"unknown IR instruction: {type(instr).__name__}")

    return None


def _register_when_bodies(env: Environment, functions: Functions, builtins: Builtins) -> None:
    """Publishes every lowered `when` body as a zero-argument callable
    the backtest engine can invoke per-bar. The LLVM backend does the
    same thing with JIT'd function addresses (see
    allie.codegen.jit.compile_and_run) -- both fill the one dispatch
    table in allie.quant.backtest_engine, so neither needs an engine of
    its own."""
    for name, (body_program, _params) in functions.items():
        if not name.startswith("__when_body_"):
            continue

        def action(prog: IRProgram = body_program) -> None:
            run_ir_program(prog, Environment(parent=env), functions, builtins)

        register_when_body(name, action)


def _exec_call(
    instr: Call, registers: dict[str, Value], functions: Functions, builtins: Builtins
) -> Value:
    args = [registers[a] for a in instr.args]

    if instr.fn_name == "__make_series":
        return _SeriesValue(np.array(args, dtype=np.float64))
    if instr.fn_name == "__strategy_declare":
        declare_strategy(args[0])  # type: ignore[arg-type]
        return None
    if instr.fn_name == "__strategy_clause":
        add_strategy_clause(args[0], args[1], args[2])  # type: ignore[arg-type]
        return None
    if instr.fn_name == "__backtest_run":
        run_registered_backtest(args[0], args[1], args[2], args[3])  # type: ignore[arg-type]
        return None
    if instr.fn_name in functions:
        fn_program, fn_params = functions[instr.fn_name]
        if len(args) != len(fn_params):
            raise AllieValueError(f"expected {len(fn_params)} argument(s), got {len(args)}")
        call_env = Environment()
        for name, value in zip(fn_params, args, strict=True):
            call_env.define(name, value)
        return run_ir_program(fn_program, call_env, functions, builtins)
    if instr.fn_name in builtins:
        _, fn = builtins[instr.fn_name]
        return fn(*args)
    raise AllieValueError(f"cannot call undefined function '{instr.fn_name}'")
