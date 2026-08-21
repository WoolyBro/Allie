"""IR instruction set -- Phase 5.1.

Flat list, not a basic-block/CFG structure: `IRProgram.instructions` is a
single `list[IRInstr]` for v1. This is a deliberate simplification, not
an oversight -- it constrains how sophisticated DCE/CSE can be (see
allie/optimizer/dce.py's module docstring for the loop-carried-liveness
consequence this creates), and it's exactly what lets Phase 6's LLVM
lowering skip almost all control-flow-specific code: `for`/`while`/`if`
are all just `Label`/`Jump`/`JumpIfFalse` here, and LLVM basic blocks
fall out of a straightforward label-to-block translation later.

Confirmed per Phase 5.1's own ask, not assumed: no new instruction is
needed for `for`/`break`/`continue`, or for `strategy`/`when`/`backtest`
-- see allie/ir/lowering.py, which desugars all of them onto exactly the
ops below (`ForStmt` reuses `WhileStmt`'s Label/Jump/JumpIfFalse shape;
`break`/`continue` are just `Jump` to the right label; `strategy`/`when`/
`backtest` lower to `Call`s into the not-yet-built Phase 7 backtest
engine, invoked rather than inlined).

One addition beyond the literal list, because lowering genuinely has no
other way to do it: `Move(dest, src)`. Short-circuit `and`/`or` needs to
converge two different possible source registers (the left operand, if
it short-circuits; the right operand, if not) into one destination
register representing the expression's result -- and this instruction
set has no `JumpIfTrue` and no copy/phi instruction to do that
convergence with. `Move` is the minimal fix; see `_lower_logical` in
lowering.py for exactly how it's used.

`dest`/operand fields reference virtual registers ("r0", "r1", ...),
never variable names directly -- `RegisterAllocator.new_reg()` just
increments a counter; real register allocation is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class IRInstr:
    """Marker base for every IR instruction."""


@dataclass
class LoadConst(IRInstr):
    dest: str
    value: Any


@dataclass
class LoadVar(IRInstr):
    dest: str
    name: str


@dataclass
class StoreVar(IRInstr):
    name: str
    src: str


@dataclass
class LoadSeries(IRInstr):
    """Defined per the Phase 5.1 instruction list, but not currently
    emitted by the lowering pass: array-literal lowering builds a
    SeriesValue at runtime through a `Call` to an internal
    `__make_series` builtin instead (see lowering.py's `_lower_expr` for
    ArrayLiteral), which was the explicit Phase 5.2 decision. Kept here,
    unused, rather than silently dropped from the instruction set."""

    dest: str
    source_expr_desc: str


@dataclass
class BinOp(IRInstr):
    dest: str
    op: str
    left: str
    right: str


@dataclass
class UnaryOp(IRInstr):
    dest: str
    op: str
    operand: str


@dataclass
class Call(IRInstr):
    dest: str | None
    fn_name: str
    args: list[str]


@dataclass
class MemberGet(IRInstr):
    dest: str
    obj: str
    field: str


@dataclass
class Move(IRInstr):
    """Not in the original Phase 5.1 list -- see this module's docstring
    for why short-circuit Logical lowering needs it."""

    dest: str
    src: str


@dataclass
class Label(IRInstr):
    name: str


@dataclass
class Jump(IRInstr):
    label: str


@dataclass
class JumpIfFalse(IRInstr):
    cond: str
    label: str


@dataclass
class Return(IRInstr):
    value: str | None


@dataclass
class IRProgram:
    instructions: list[IRInstr] = field(default_factory=list)


class RegisterAllocator:
    """`new_reg()` just increments a counter -- real register allocation
    (reuse, spilling, ...) is out of scope for this IR."""

    def __init__(self) -> None:
        self._next_reg = 0
        self._next_label = 0

    def new_reg(self) -> str:
        reg = f"r{self._next_reg}"
        self._next_reg += 1
        return reg

    def new_label(self, hint: str) -> str:
        label = f"{hint}_{self._next_label}"
        self._next_label += 1
        return label
