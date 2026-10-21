"""Constant propagation and folding. `run(program) -> IRProgram`, pure
function, no shared state with the other passes.

Finds every register and variable whose value is the same constant on
every path that reaches it, replaces the instructions that compute them
with a `LoadConst`, and turns a branch on a known condition into a plain
jump (or into nothing). The never-taken side of a folded branch, and the
stores and loads nothing needs any more, are left for DCE, which runs
next; the original operand constants a fold consumed are dropped here.

## How

A forward dataflow problem over the control-flow graph
(allie/ir/cfg.py). The state maps each register and each variable to a
known constant, to NAC ("not a constant": it may hold different values
on different paths), or to nothing yet (no path has assigned it).
Joining two paths keeps a constant only if both carry the same one --
compared by type and `repr`, so `1` and `1.0`, or `0.0` and `-0.0`, are
different constants. Loops iterate to a fixpoint: a variable the loop
body changes becomes NAC at the loop header and stays NAC.

This pass used to fold only a `BinOp` whose two operands were literals
in the same straight-line stretch, so `let a = 2; let b = a * 3;`
survived: `a` went through a variable. Now it folds to 6, and
`if a > 1 { ... }` folds the branch away.

## What is never folded

The folded value must be exactly what running the code would produce,
on every backend:

- A division or modulo by zero, or an Int result outside 64 bits, is a
  runtime error, so it's left for the runtime to raise.
- A `Call`'s result is never assumed. A call to anything other than a
  known builtin may assign module-level variables, so every variable is
  NAC after one.
- In a function body, a variable the body never assigns may be a
  module-level one with an unknown value, so variables start as NAC
  there rather than "unassigned".
"""

from __future__ import annotations

from typing import Any

from allie.ir.cfg import BasicBlock, build_cfg, forward
from allie.ir.instructions import (
    TO_FLOAT,
    BinOp,
    Call,
    IRInstr,
    IRProgram,
    Jump,
    JumpIfFalse,
    LoadConst,
    LoadVar,
    MemberGet,
    Move,
    StoreVar,
    UnaryOp,
    reads,
)
from allie.optimizer.pure_functions import VARIABLE_SAFE_CALLS

_ARITHMETIC: dict[str, Any] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: float(a) / float(b),
    "%": lambda a, b: a % b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}

_INT_MIN = -(2**63)
_INT_MAX = 2**63 - 1


class _Known:
    """A known constant. Equal to another only if type and repr match, so
    `1`, `1.0` and `True`, or `0.0` and `-0.0`, stay distinct."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def _key(self) -> tuple[type, str]:
        return type(self.value), repr(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Known) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())


class _Marker:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


_NAC = _Marker("NAC")  # may differ between paths
_UNSET = _Marker("UNSET")  # no path has assigned it yet -- the join identity

Fact = _Known | _Marker


class _State:
    __slots__ = ("regs", "vars", "default_var")

    def __init__(self, regs: dict[str, Fact], variables: dict[str, Fact], default_var: Fact):
        self.regs = regs
        self.vars = variables
        self.default_var = default_var

    def copy(self) -> _State:
        return _State(dict(self.regs), dict(self.vars), self.default_var)

    def var(self, name: str) -> Fact:
        fact = self.vars.get(name, self.default_var)
        return _NAC if fact is _UNSET else fact

    def reg(self, name: str) -> Fact:
        return self.regs.get(name, _NAC)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _State)
            and self.regs == other.regs
            and self.vars == other.vars
            and self.default_var is other.default_var
        )


def _join_fact(a: Fact, b: Fact) -> Fact:
    if a is _UNSET:
        return b
    if b is _UNSET:
        return a
    return a if a == b else _NAC


def _join(a: _State, b: _State) -> _State:
    regs = {k: _join_fact(a.regs.get(k, _UNSET), b.regs.get(k, _UNSET)) for k in a.regs | b.regs}
    variables = {
        k: _join_fact(a.vars.get(k, a.default_var), b.vars.get(k, b.default_var))
        for k in a.vars | b.vars
    }
    default = _NAC if _NAC in (a.default_var, b.default_var) else _UNSET
    return _State(regs, variables, default)


def _overflows(value: Any) -> bool:
    """An Int result outside i64 is a runtime overflow error on every
    backend, so it must not be folded into a constant that holds it."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and not (_INT_MIN <= value <= _INT_MAX)
    )


def _fold_binop(op: str, left: Any, right: Any) -> _Known | None:
    if op not in _ARITHMETIC:
        return None
    try:
        value = _ARITHMETIC[op](left, right)
    except (ZeroDivisionError, TypeError, OverflowError):
        return None  # leave it to raise at runtime
    return None if _overflows(value) else _Known(value)


def _fold_unary(op: str, operand: Any) -> _Known | None:
    if op == "!" and isinstance(operand, bool):
        return _Known(not operand)
    if isinstance(operand, bool) or not isinstance(operand, (int, float)):
        return None
    if op == "-":
        value = -operand
        return None if _overflows(value) else _Known(value)
    if op == TO_FLOAT:
        return _Known(float(operand))
    return None


def _value(instr: IRInstr, state: _State) -> Fact:
    """The fact `instr` writes to its destination, given `state`."""
    if isinstance(instr, LoadConst):
        return _Known(instr.value)
    if isinstance(instr, LoadVar):
        return state.var(instr.name)
    if isinstance(instr, Move):
        return state.reg(instr.src)
    if isinstance(instr, BinOp):
        left, right = state.reg(instr.left), state.reg(instr.right)
        if isinstance(left, _Known) and isinstance(right, _Known):
            return _fold_binop(instr.op, left.value, right.value) or _NAC
        return _NAC
    if isinstance(instr, UnaryOp):
        operand = state.reg(instr.operand)
        if isinstance(operand, _Known):
            return _fold_unary(instr.op, operand.value) or _NAC
        return _NAC
    return _NAC  # Call, MemberGet: never assumed


def _step(instr: IRInstr, state: _State) -> None:
    """Applies `instr` to `state` in place."""
    if isinstance(instr, StoreVar):
        state.vars[instr.name] = state.reg(instr.src)
        return
    if isinstance(instr, Call):
        if instr.dest is not None:
            state.regs[instr.dest] = _NAC
        if instr.fn_name not in VARIABLE_SAFE_CALLS:
            state.vars = {}
            state.default_var = _NAC
        return
    if isinstance(instr, (LoadConst, LoadVar, Move, BinOp, UnaryOp, MemberGet)):
        state.regs[instr.dest] = _value(instr, state)


def _transfer(block: BasicBlock, state: _State) -> _State:
    after = state.copy()
    for instr in block.instructions:
        _step(instr, after)
    return after


def run(program: IRProgram) -> IRProgram:
    cfg = build_cfg(program)
    entry = _State({}, {}, _NAC if program.is_function else _UNSET)
    in_states = forward(cfg, entry, _transfer, _join)

    rewritten: list[IRInstr] = []
    for block in cfg.blocks:
        if block.index not in in_states:
            rewritten.extend(block.instructions)  # unreachable: DCE removes it
            continue
        state = in_states[block.index].copy()
        for instr in block.instructions:
            rewritten.extend(_rewrite(instr, state))
            _step(instr, state)

    # Drop operand constants a fold consumed. A fold's own result stays
    # even if unread here -- whether anything needs it is DCE's call.
    original_consts = {i.dest for i in program.instructions if isinstance(i, LoadConst)}
    read = {register for instr in rewritten for register in reads(instr)}
    result = [
        instr
        for instr in rewritten
        if not (
            isinstance(instr, LoadConst)
            and instr.dest in original_consts
            and instr.dest not in read
        )
    ]
    return IRProgram(instructions=result, is_function=program.is_function)


def _rewrite(instr: IRInstr, state: _State) -> list[IRInstr]:
    """`instr` with what `state` proves about it applied: a computed
    constant becomes a LoadConst, a branch on a known condition becomes
    a Jump or disappears. Moves are left alone -- the register they
    write is shared by two Moves, and must stay Move-written only."""
    if isinstance(instr, (LoadVar, BinOp, UnaryOp)):
        fact = _value(instr, state)
        if isinstance(fact, _Known):
            return [LoadConst(dest=instr.dest, value=fact.value)]
    if isinstance(instr, JumpIfFalse):
        cond = state.reg(instr.cond)
        if isinstance(cond, _Known) and isinstance(cond.value, bool):
            return [] if cond.value else [Jump(label=instr.label)]
    return [instr]
