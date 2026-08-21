"""Constant folding -- Phase 5.4. `run(program) -> IRProgram`, pure
function, no shared state with the other passes.

Two-pass: (1) walk forward tracking which registers are known constants
(written by LoadConst, or by a BinOp both of whose operands are already
known constants -- folded in place as we go, so a chain of constant
operations collapses in one pass, not just one level); (2) drop any
LoadConst whose register ends up unread in the resulting program --
"remove the now-dead original LoadConsts if unused elsewhere" from the
spec, generalized slightly: any constant register left unread after
folding is dead, not just the specific ones a fold consumed.
"""

from __future__ import annotations

from typing import Any

from allie.ir.instructions import (
    BinOp,
    Call,
    IRInstr,
    IRProgram,
    JumpIfFalse,
    LoadConst,
    MemberGet,
    Move,
    Return,
    StoreVar,
    UnaryOp,
)

_ARITHMETIC: dict[str, Any] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "%": lambda a, b: a % b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}


def run(program: IRProgram) -> IRProgram:
    const_values: dict[str, Any] = {}
    original_loadconst_dests: set[str] = set()
    folded: list[IRInstr] = []

    for instr in program.instructions:
        if isinstance(instr, LoadConst):
            const_values[instr.dest] = instr.value
            original_loadconst_dests.add(instr.dest)
            folded.append(instr)
            continue

        if (
            isinstance(instr, BinOp)
            and instr.left in const_values
            and instr.right in const_values
            and instr.op in _ARITHMETIC
        ):
            try:
                value = _ARITHMETIC[instr.op](const_values[instr.left], const_values[instr.right])
            except ZeroDivisionError:
                # A folded 1/0 would raise at compile time instead of
                # runtime -- leave it as a real BinOp so the existing
                # AllieValueError("division by zero") path still fires
                # when the program actually runs.
                folded.append(instr)
                continue
            # This LoadConst represents the fold's *result*, not one of
            # the original operand constants -- it must survive
            # regardless of whether anything reads it here (that's a
            # DCE concern, not this pass's), so it's deliberately NOT
            # added to original_loadconst_dests below.
            new_instr: IRInstr = LoadConst(dest=instr.dest, value=value)
            const_values[instr.dest] = value
            folded.append(new_instr)
            continue

        folded.append(instr)

    read_registers = _collect_read_registers(folded)
    result = [
        instr
        for instr in folded
        if not (
            isinstance(instr, LoadConst)
            and instr.dest in original_loadconst_dests
            and instr.dest not in read_registers
        )
    ]
    return IRProgram(instructions=result)


def _collect_read_registers(instructions: list[IRInstr]) -> set[str]:
    read: set[str] = set()
    for instr in instructions:
        if isinstance(instr, BinOp):
            read.add(instr.left)
            read.add(instr.right)
        elif isinstance(instr, UnaryOp):
            read.add(instr.operand)
        elif isinstance(instr, Call):
            read.update(instr.args)
        elif isinstance(instr, MemberGet):
            read.add(instr.obj)
        elif isinstance(instr, Move):
            read.add(instr.src)
        elif isinstance(instr, StoreVar):
            read.add(instr.src)
        elif isinstance(instr, JumpIfFalse):
            read.add(instr.cond)
        elif isinstance(instr, Return) and instr.value is not None:
            read.add(instr.value)
    return read
