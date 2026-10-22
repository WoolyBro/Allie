"""IR interpreter. A second, independent execution path for
the same programs the tree-walking interpreter (allie.runtime.
interpreter) already covers -- see tests/unit/test_ir_cross_validation.py,
which runs the same source through both and asserts identical results.
Operator/field-access logic is deliberately re-implemented here rather
than imported from the tree-walker: sharing those helpers would let a
bug common to both paths hide from cross-validation, defeating the
point of having two independent implementations to compare.

Variable scoping is flat: one Environment per execution -- the main
program's, or a fresh one per function call -- with no child scope per
block (lowering alpha-renames shadowed variables, so one flat scope is
unambiguous). Functions don't capture a lexical parent; the one thing a
function body can see beyond its own locals is the main program's
variables, which lowering writes as `@name` (allie.ir.instructions.
GLOBAL_PREFIX). Those are read from, and assigned in, the main
program's environment, which every call passes down as `globals_env`.

## Execution: compiled once, then run

A program is translated once into a list of small Python closures, one
per instruction, each doing its instruction's work and returning the
index of the next one. Labels are resolved to indices at translation,
operators are chosen then rather than looked up per execution, and
Int/Float arithmetic has a fast path that skips the generic operator
code. Running is then a three-line loop -- several times faster than
re-inspecting every instruction with an `isinstance` chain each time it
runs (see benchmarks/results.md).

A runtime error escaping an instruction is given that instruction's
source span (allie.ir.instructions.IRInstr.span) if it has none yet, so
the innermost failing expression is the one reported -- the same line
and column the tree-walking interpreter and the JIT report.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from allie.errors import SourceSpan
from allie.ir.instructions import (
    GLOBAL_PREFIX,
    TO_FLOAT,
    BinOp,
    Call,
    IRInstr,
    IRProgram,
    Jump,
    JumpIfFalse,
    Label,
    LoadConst,
    LoadVar,
    MemberGet,
    Move,
    Return,
    StoreVar,
    UnaryOp,
    writes,
)
from allie.quant.backtest_engine import (
    BacktestResult,
    add_strategy_clause,
    declare_strategy,
    register_when_body,
    run_registered_backtest,
    set_backtest_option,
)
from allie.runtime import elementwise
from allie.runtime.environment import Environment
from allie.runtime.errors import AllieNameError, AllieRuntimeError, AllieValueError
from allie.runtime.values import PriceSeriesValue, Value, promote_numeric

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


_INT_MIN = -(2**63)
_INT_MAX = 2**63 - 1


def _truthy(value: Value) -> bool:
    return bool(value)


def _in_int_range(value: Value) -> Value:
    """Int is 64-bit on every backend; past that range is an overflow."""
    if isinstance(value, int) and not isinstance(value, bool):
        if value < _INT_MIN or value > _INT_MAX:
            raise AllieValueError("integer overflow: the result doesn't fit in a 64-bit Int")
    return value


def _apply_binop(op: str, left: Value, right: Value) -> Value:
    # promote_numeric's result is assigned to fresh names, not `left`/
    # `right` themselves: reassigning an explicitly-annotated *parameter*
    # keeps mypy holding it to the declared Value union (so `a + b` still
    # gets checked against every non-numeric member of that union and
    # fails); a plain local narrows to the call's Any result instead. The
    # tree-walking interpreter's equivalent code only avoids this because
    # its `left`/`right` are inferred locals, not parameters.
    if elementwise.is_series(left) or elementwise.is_series(right):
        return elementwise.apply_binary(op, left, right)
    if op in _COMPARISON_OPS:
        return bool(_COMPARISON_OPS[op](left, right))
    if op == "+":
        if isinstance(left, str) or isinstance(right, str):
            return f"{left}{right}"
        num_left, num_right = promote_numeric(left, right)
        return _in_int_range(num_left + num_right)
    if op in ("-", "*"):
        num_left, num_right = promote_numeric(left, right)
        return _in_int_range((num_left - num_right) if op == "-" else (num_left * num_right))
    if op == "/":
        # Always true division: `/` is Float-typed even for two Ints.
        num_left, num_right = promote_numeric(left, right)
        if num_right == 0:
            raise AllieValueError("division by zero")
        return float(num_left) / float(num_right)
    if op == "%":
        num_left, num_right = promote_numeric(left, right)
        if num_right == 0:
            raise AllieValueError("modulo by zero")
        return num_left % num_right
    raise AllieValueError(f"unknown binary operator: {op!r}")


def _apply_unary(op: str, operand: Value) -> Value:
    if elementwise.is_series(operand):
        return elementwise.apply_unary(op, operand)  # type: ignore[arg-type]
    if op == "-":
        if isinstance(operand, bool) or not isinstance(operand, (int, float)):
            raise AllieValueError(f"unary '-' requires a number, got {operand!r}")
        return _in_int_range(-operand)
    if op == "!":
        return not _truthy(operand)
    if op == "to_float":
        return float(operand)  # type: ignore[arg-type]
    raise AllieValueError(f"unknown unary operator: {op!r}")


def run_ir_program(
    program: IRProgram,
    env: Environment,
    functions: Functions,
    builtins: Builtins | None = None,
    globals_env: Environment | None = None,
) -> Value:
    """Executes `program` against `env`, with `functions`/`builtins` for
    Call dispatch. Returns the value of the first Return instruction
    reached, or None if execution falls off the end. `globals_env` is the
    main program's environment, where `@name` variables live; for the
    main program itself it is `env`."""
    builtins = builtins if builtins is not None else {}
    globals_env = globals_env if globals_env is not None else env
    if globals_env is env:
        _register_when_bodies(env, functions, builtins)
    return _execute(_compiled(program, functions, builtins), env, globals_env)


# ---- running compiled code ----

# An op: does one instruction's work on (registers, env, globals_env) and
# returns the index of the next op to run. Registers are list slots,
# numbered at translation.
Registers = list[Value]
Op = Callable[[Registers, Environment, Environment], int]


class _Code:
    def __init__(
        self, ops: list[Op], spans: list[SourceSpan | None], template: Registers, result: int
    ) -> None:
        self.ops = ops
        self.spans = spans
        # The registers a run starts with: None, except the constants
        # (see _translate), already in place.
        self.template = template
        self.result = result  # the slot a Return writes


def _execute(code: _Code, env: Environment, globals_env: Environment) -> Value:
    ops = code.ops
    end = len(ops)
    registers = code.template[:]
    pc = 0
    try:
        while pc < end:
            pc = ops[pc](registers, env, globals_env)
    except AllieRuntimeError as error:
        if error.span is None:
            error.span = code.spans[pc]
        raise
    return registers[code.result]


def _compiled(program: IRProgram, functions: Functions, builtins: Builtins) -> _Code:
    """The program's ops, translated on first use and kept on the program
    -- keyed by the function and builtin tables they were bound against."""
    cached = getattr(program, "_ir_code", None)
    if cached is not None and cached[0] is functions and cached[1] is builtins:
        return cast(_Code, cached[2])
    code = _translate(program, functions, builtins)
    program._ir_code = (functions, builtins, code)  # type: ignore[attr-defined]
    return code


def _translate(program: IRProgram, functions: Functions, builtins: Builtins) -> _Code:
    """Ops for every instruction but two kinds, which need none: a Label
    (its jumps go straight to the op after it) and a LoadConst into a
    register nothing else ever writes -- the constant is put in the
    register before the program starts, so a loop doesn't reload it on
    every iteration."""
    slots: dict[str, int] = {}

    def slot(register: str) -> int:
        return slots.setdefault(register, len(slots))

    writers: dict[str, int] = {}
    for instr in program.instructions:
        written = writes(instr)
        if written is not None:
            writers[written] = writers.get(written, 0) + 1
    constants = {
        instr.dest: instr.value
        for instr in program.instructions
        if isinstance(instr, LoadConst) and writers[instr.dest] == 1
    }

    def needs_op(instr: IRInstr) -> bool:
        return not isinstance(instr, Label) and not (
            isinstance(instr, LoadConst) and instr.dest in constants
        )

    kept = [instr for instr in program.instructions if needs_op(instr)]
    targets: dict[str, int] = {}
    index = 0
    for instr in program.instructions:
        if isinstance(instr, Label):
            targets[instr.name] = index
        elif needs_op(instr):
            index += 1
    end = len(kept)
    result = slot("\0return")  # no IR register is named this
    ops = [
        _op(instr, position + 1, targets, functions, builtins, end, slot, result)
        for position, instr in enumerate(kept)
    ]
    for register in constants:
        slot(register)
    template: Registers = [None] * len(slots)
    for register, value in constants.items():
        template[slots[register]] = value
    return _Code(ops, [instr.span for instr in kept], template, result)


def _op(
    instr: IRInstr,
    after: int,
    targets: dict[str, int],
    functions: Functions,
    builtins: Builtins,
    end: int,
    slot: Callable[[str], int],
    result: int,
) -> Op:
    if isinstance(instr, LoadConst):
        return _load_const(slot(instr.dest), instr.value, after)
    if isinstance(instr, LoadVar):
        return _load_var(slot(instr.dest), instr.name, after)
    if isinstance(instr, StoreVar):
        return _store_var(instr.name, slot(instr.src), after)
    if isinstance(instr, Move):
        return _move(slot(instr.dest), slot(instr.src), after)
    if isinstance(instr, BinOp):
        return _binop(slot(instr.dest), instr.op, slot(instr.left), slot(instr.right), after)
    if isinstance(instr, UnaryOp):
        return _unary(slot(instr.dest), instr.op, slot(instr.operand), after)
    if isinstance(instr, MemberGet):
        return _member_get(slot(instr.dest), slot(instr.obj), instr.field, after)
    if isinstance(instr, Call):
        dest = slot(instr.dest) if instr.dest is not None else None
        return _call(dest, instr.fn_name, [slot(a) for a in instr.args], after, functions, builtins)
    if isinstance(instr, Jump):
        target = targets[instr.label]
        return lambda registers, env, globals_env: target
    if isinstance(instr, JumpIfFalse):
        return _jump_if_false(slot(instr.cond), targets[instr.label], after)
    if isinstance(instr, Return):
        value = slot(instr.value) if instr.value is not None else None
        return _return(value, result, end)
    raise AllieValueError(f"unknown IR instruction: {type(instr).__name__}")


def _load_const(dest: int, value: Value, after: int) -> Op:
    def op(registers: Registers, env: Environment, globals_env: Environment) -> int:
        registers[dest] = value
        return after

    return op


def _load_var(dest: int, name: str, after: int) -> Op:
    if name.startswith(GLOBAL_PREFIX):
        global_name = name[1:]

        def load_global(registers: Registers, env: Environment, globals_env: Environment) -> int:
            registers[dest] = globals_env.get(global_name)
            return after

        return load_global

    def load(registers: Registers, env: Environment, globals_env: Environment) -> int:
        try:
            registers[dest] = env.vars[name]
        except KeyError:
            registers[dest] = env.get(name)  # an outer scope, or the error
        return after

    return load


def _store_var(name: str, src: int, after: int) -> Op:
    if name.startswith(GLOBAL_PREFIX):
        global_name = name[1:]

        def store_global(registers: Registers, env: Environment, globals_env: Environment) -> int:
            globals_env.assign(global_name, registers[src])
            return after

        return store_global

    def store(registers: Registers, env: Environment, globals_env: Environment) -> int:
        env.vars[name] = registers[src]
        return after

    return store


def _move(dest: int, src: int, after: int) -> Op:
    def op(registers: Registers, env: Environment, globals_env: Environment) -> int:
        registers[dest] = registers[src]
        return after

    return op


def _overflow() -> AllieValueError:
    return AllieValueError("integer overflow: the result doesn't fit in a 64-bit Int")


def _binop(dest: int, op_name: str, left: int, right: int, after: int) -> Op:
    """A fast path for two Ints or two Floats -- what scalar code is made
    of -- and `_apply_binop` for everything else (mixed numbers, strings,
    series, the errors)."""
    if op_name in _ARITHMETIC:
        arithmetic = _ARITHMETIC[op_name]

        def arith(registers: Registers, env: Environment, globals_env: Environment) -> int:
            a, b = registers[left], registers[right]
            kind = type(a)
            if kind is type(b):
                if kind is int:
                    value = arithmetic(a, b)
                    if _INT_MIN <= value <= _INT_MAX:
                        registers[dest] = value
                        return after
                    raise _overflow()
                if kind is float:
                    registers[dest] = arithmetic(a, b)
                    return after
            registers[dest] = _apply_binop(op_name, a, b)
            return after

        return arith

    comparison = _FAST_COMPARISONS.get(op_name)
    if comparison is not None:

        def compare(registers: Registers, env: Environment, globals_env: Environment) -> int:
            a, b = registers[left], registers[right]
            kind = type(a)
            if (kind is int or kind is float) and type(b) is kind:
                registers[dest] = comparison(a, b)
                return after
            registers[dest] = _apply_binop(op_name, a, b)
            return after

        return compare

    def generic(registers: Registers, env: Environment, globals_env: Environment) -> int:
        registers[dest] = _apply_binop(op_name, registers[left], registers[right])
        return after

    return generic


_ARITHMETIC: dict[str, Callable[[Any, Any], Any]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}
_FAST_COMPARISONS: dict[str, Callable[[Any, Any], bool]] = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _unary(dest: int, op_name: str, operand: int, after: int) -> Op:
    if op_name == "!":

        def negate_bool(registers: Registers, env: Environment, globals_env: Environment) -> int:
            value = registers[operand]
            registers[dest] = (not value) if type(value) is bool else _apply_unary("!", value)
            return after

        return negate_bool
    if op_name == TO_FLOAT:

        def widen(registers: Registers, env: Environment, globals_env: Environment) -> int:
            registers[dest] = float(registers[operand])  # type: ignore[arg-type]
            return after

        return widen

    def op(registers: Registers, env: Environment, globals_env: Environment) -> int:
        registers[dest] = _apply_unary(op_name, registers[operand])
        return after

    return op


def _member_get(dest: int, obj_reg: int, field: str, after: int) -> Op:
    def op(registers: Registers, env: Environment, globals_env: Environment) -> int:
        obj = registers[obj_reg]
        if isinstance(obj, PriceSeriesValue) and field in _PRICE_SERIES_FIELDS:
            registers[dest] = getattr(obj, field)
        elif isinstance(obj, BacktestResult):
            registers[dest] = cast(Value, obj.field(field))
        else:
            raise AllieValueError(f"cannot access field '{field}' on {type(obj).__name__}")
        return after

    return op


def _jump_if_false(cond: int, target: int, after: int) -> Op:
    def op(registers: Registers, env: Environment, globals_env: Environment) -> int:
        return after if registers[cond] else target

    return op


def _return(value: int | None, result: int, end: int) -> Op:
    def op(registers: Registers, env: Environment, globals_env: Environment) -> int:
        registers[result] = registers[value] if value is not None else None
        return end

    return op


def _call(
    dest: int | None,
    name: str,
    arg_slots: list[int],
    after: int,
    functions: Functions,
    builtins: Builtins,
) -> Op:
    target = _call_target(name, functions, builtins)

    def op(registers: Registers, env: Environment, globals_env: Environment) -> int:
        result = target([registers[a] for a in arg_slots], globals_env)
        if dest is not None:
            registers[dest] = result
        return after

    return op


def _call_target(
    name: str, functions: Functions, builtins: Builtins
) -> Callable[[list[Value], Environment], Value]:
    if name == "__make_series":
        return lambda args, globals_env: elementwise.series_literal(list(args))
    if name == "__strategy_declare":
        return lambda args, globals_env: declare_strategy(args[0])  # type: ignore[arg-type]
    if name == "__strategy_clause":
        return lambda args, globals_env: add_strategy_clause(args[0], args[1])  # type: ignore[arg-type]
    if name == "__backtest_option":
        return lambda args, globals_env: set_backtest_option(args[0], args[1])  # type: ignore[arg-type]
    if name == "__backtest_run":
        return lambda args, globals_env: cast(
            Value,
            run_registered_backtest(args[0], args[1], args[2], args[3]),  # type: ignore[arg-type]
        )
    if name in functions:
        program, params = functions[name]

        def call_function(args: list[Value], globals_env: Environment) -> Value:
            if len(args) != len(params):
                raise AllieValueError(f"expected {len(params)} argument(s), got {len(args)}")
            call_env = Environment()
            call_env.vars.update(zip(params, args, strict=True))
            return _execute(_compiled(program, functions, builtins), call_env, globals_env)

        return call_function
    if name in builtins:
        _, fn = builtins[name]
        return lambda args, globals_env: fn(*args)

    def undefined(args: list[Value], globals_env: Environment) -> Value:
        raise AllieNameError(f"cannot call undefined function '{name}'")

    return undefined


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
            _execute(_compiled(prog, functions, builtins), Environment(), env)

        register_when_body(name, action)
