"""IR instruction set.

Stored as a flat list: `IRProgram.instructions` is a single
`list[IRInstr]`, which is what lowering emits and what both the IR
interpreter and the LLVM backend consume -- `for`/`while`/`if` are all
just `Label`/`Jump`/`JumpIfFalse` here, and LLVM basic blocks fall out of
a straightforward label-to-block translation. Optimizer passes that need
control flow build a graph on demand (allie/ir/cfg.py) instead of
reasoning about text order, and allie/ir/verify.py checks the invariants
every program must keep.

No instruction is needed for `for`/`break`/`continue` or for
`strategy`/`when`/`backtest`: allie/ir/lowering.py desugars all of them
onto the ops below (`ForStmt` reuses `WhileStmt`'s Label/Jump/JumpIfFalse
shape; `break`/`continue` are `Jump`s to the right label; the quant
constructs lower to `Call`s into the backtest engine).

`Move(dest, src)` exists for short-circuit `and`/`or`, which has to
converge two possible source registers (the left operand if it
short-circuits, the right one if not) into one result register; it is
the only instruction allowed to write a register more than once. Every
other register is written exactly once, which the verifier enforces.

`UnaryOp` carries `-`, `!`, and `to_float` -- an implicit Int -> Float
widening the type checker made explicit (allie.ast.nodes.Widen).

`dest`/operand fields reference virtual registers ("r0", "r1", ...),
never variable names directly -- `RegisterAllocator.new_reg()` just
increments a counter; real register allocation is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

from allie.errors import SourceSpan

# UnaryOp operator for an implicit Int -> Float widening (a Widen node).
TO_FLOAT = "to_float"

# Prefix on a LoadVar/StoreVar name inside a function or `when` body that
# refers to a module-level variable: `@rate` is the main program's `rate`.
# `@` can't start a source identifier, so it never collides with a local.
GLOBAL_PREFIX = "@"


class IRInstr:
    """Marker base for every IR instruction.

    `span` is the source position of the expression an instruction came
    from, set by lowering on the instructions that can fail at run time
    (LoadVar, BinOp, UnaryOp, MemberGet, Call), so a runtime error names
    the same line and column on every backend. It is a plain attribute
    rather than a dataclass field: it is not part of what an instruction
    *does*, so it stays out of equality, `repr` and the golden snapshots.
    A pass that rebuilds one of these carries it over with `located`."""

    span: SourceSpan | None = None


_Instr = TypeVar("_Instr", bound=IRInstr)


def located(new: _Instr, old: IRInstr) -> _Instr:
    """`new`, with `old`'s source span."""
    new.span = old.span
    return new


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
    # True for a function (or `when`) body. Liveness needs it: a
    # function may assign a module-level variable someone reads after
    # it returns, so at a function's exit every variable counts as live;
    # at the end of the main program, none does.
    is_function: bool = False


def reads(instr: IRInstr) -> list[str]:
    """The registers `instr` reads."""
    if isinstance(instr, BinOp):
        return [instr.left, instr.right]
    if isinstance(instr, UnaryOp):
        return [instr.operand]
    if isinstance(instr, (StoreVar, Move)):
        return [instr.src]
    if isinstance(instr, Call):
        return list(instr.args)
    if isinstance(instr, MemberGet):
        return [instr.obj]
    if isinstance(instr, JumpIfFalse):
        return [instr.cond]
    if isinstance(instr, Return):
        return [instr.value] if instr.value is not None else []
    return []


def writes(instr: IRInstr) -> str | None:
    """The register `instr` writes, if any."""
    if isinstance(instr, (LoadConst, LoadVar, BinOp, UnaryOp, MemberGet, Move)):
        return instr.dest
    if isinstance(instr, Call):
        return instr.dest
    return None


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
