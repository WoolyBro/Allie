"""Dead-code elimination. `run(program) -> IRProgram`, pure function, no
shared state with the other passes.

Removes, until nothing more can go:

- **unreachable blocks** -- code after a `return`, the never-taken side
  of a branch constant folding resolved;
- **dead values** -- an instruction whose result register nothing reads
  later on any path;
- **dead stores** -- a `StoreVar` whose value is overwritten, or never
  read, before any path reads the variable again;
- **jumps to the very next instruction**, and labels nothing jumps to,
  so the straight-line code left behind is one block again.

## Liveness, not "read anywhere"

Whether a value is dead is a backward dataflow problem over the
control-flow graph (allie/ir/cfg.py): live after an instruction means
some path from there reads it before overwriting it. An earlier version
had no control-flow graph, and its forward-only "is this read *later in
the list*?" check deleted a `for` loop's counter increment -- the read
is at the loop's condition check, *earlier* in the list, reached again
through the back-edge -- making the loop infinite. It was fixed by
asking "is this read *anywhere*?", which is safe but can't see that the
first store in `x = 1; x = 2; print(x);` is dead. Liveness gets both
right.

Two conservative assumptions keep it sound:

- At a function's exit every variable counts as live, since a function
  may assign a module-level variable the caller reads afterwards; at
  the end of the main program none does (`IRProgram.is_function`).
- A call to anything other than a known builtin -- a user function, or
  `__backtest_run` invoking `when` bodies -- may read any variable.

## What is never removed, even if unused

- A `Call`: it may print, trade, or raise.
- An instruction that can raise (`_can_trap`): a division or modulo by
  a divisor not known to be nonzero, Int `+`/`-`/`*`/unary `-`, which
  raise on 64-bit overflow, and the operators that raise on two series
  of different lengths. Deleting one would turn "fails" into
  "succeeds silently" on the optimized backends only -- the tree-walking
  interpreter never runs this pass. `allie run --verify` found exactly
  that divergence once, with `let x: Int = 1 / 0;` and `x` never read.
"""

from __future__ import annotations

from allie.ir.cfg import BasicBlock, backward, build_cfg
from allie.ir.instructions import (
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
    StoreVar,
    UnaryOp,
    reads,
    writes,
)
from allie.optimizer.pure_functions import VARIABLE_SAFE_CALLS

_DIVIDING_OPS = frozenset({"/", "%"})
# Int `+`, `-`, `*` raise on 64-bit overflow. The IR is untyped here, so
# a Float one is kept too -- a missed optimization, never a wrong result.
_OVERFLOWING_OPS = frozenset({"+", "-", "*"})
# On two series these raise when the lengths differ. The IR can't tell a
# series comparison from a scalar one, so every one is kept.
_SERIES_OPS = frozenset({"<", ">", "<=", ">=", "and", "or", "crosses_above", "crosses_below"})

_REMOVABLE_IF_UNUSED = (LoadConst, LoadVar, BinOp, UnaryOp, MemberGet, Move)


class _Live:
    """Registers and variables live at a program point.

    Variables are a finite set (`all_vars` False: exactly `vars` are
    live) or a co-finite one (`all_vars` True: every variable *except*
    `vars` is live). The co-finite form is what a call that may read any
    variable produces, and it still lets a store just before such a call
    be recognized as dead when an even later store overwrites it first.
    """

    __slots__ = ("regs", "vars", "all_vars")

    def __init__(self, regs: frozenset[str], variables: frozenset[str], all_vars: bool) -> None:
        self.regs = regs
        self.vars = variables
        self.all_vars = all_vars

    def has_var(self, name: str) -> bool:
        return (name not in self.vars) if self.all_vars else (name in self.vars)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _Live)
            and self.regs == other.regs
            and self.vars == other.vars
            and self.all_vars == other.all_vars
        )


_NOTHING_LIVE = _Live(frozenset(), frozenset(), False)


def _join(a: _Live, b: _Live) -> _Live:
    regs = a.regs | b.regs
    if not a.all_vars and not b.all_vars:
        return _Live(regs, a.vars | b.vars, False)
    if a.all_vars and b.all_vars:
        return _Live(regs, a.vars & b.vars, True)  # live unless dead on both
    finite, cofinite = (b, a) if a.all_vars else (a, b)
    return _Live(regs, cofinite.vars - finite.vars, True)


def _before(instr: IRInstr, after: _Live) -> _Live:
    """What is live just before `instr`, given what is live after it."""
    regs = set(after.regs)
    variables = set(after.vars)
    all_vars = after.all_vars
    dest = writes(instr)
    if dest is not None:
        regs.discard(dest)
    if isinstance(instr, StoreVar):  # kills the variable
        (variables.add if all_vars else variables.discard)(instr.name)
    elif isinstance(instr, LoadVar):  # reads it
        (variables.discard if all_vars else variables.add)(instr.name)
    elif isinstance(instr, Call) and instr.fn_name not in VARIABLE_SAFE_CALLS:
        all_vars, variables = True, set()  # may read every variable
    regs.update(reads(instr))
    return _Live(frozenset(regs), frozenset(variables), all_vars)


def _transfer(block: BasicBlock, live_out: _Live) -> _Live:
    live = live_out
    for instr in reversed(block.instructions):
        live = _before(instr, live)
    return live


def _can_trap(instr: IRInstr, nonzero_registers: set[str]) -> bool:
    """Whether removing `instr` could delete a runtime error -- and with
    it, an observable difference from the unoptimized backends."""
    if isinstance(instr, UnaryOp):
        return instr.op == "-"
    if not isinstance(instr, BinOp):
        return False
    if instr.op in _OVERFLOWING_OPS or instr.op in _SERIES_OPS:
        return True
    if instr.op in _DIVIDING_OPS:
        # A divisor that is a literal nonzero constant cannot trap, so
        # `x / 2` in dead code is still removable.
        return instr.right not in nonzero_registers
    return False


def _dead(instr: IRInstr, live_after: _Live, nonzero_registers: set[str]) -> bool:
    if isinstance(instr, StoreVar):
        return not live_after.has_var(instr.name)
    if isinstance(instr, _REMOVABLE_IF_UNUSED):
        return instr.dest not in live_after.regs and not _can_trap(instr, nonzero_registers)
    return False  # Call, Label, Jump, JumpIfFalse, Return


def _sweep(program: IRProgram) -> tuple[list[IRInstr], bool]:
    """One round: drop unreachable blocks and dead instructions."""
    cfg = build_cfg(program)
    reachable = cfg.reachable()
    # At a function's exit every variable is live (co-finite, nothing excepted).
    exit_live = _Live(frozenset(), frozenset(), program.is_function)
    live_out = backward(cfg, exit_live, _NOTHING_LIVE, _transfer, _join)
    nonzero = {
        i.dest
        for i in program.instructions
        if isinstance(i, LoadConst) and not isinstance(i.value, str) and i.value != 0
    }

    kept: list[IRInstr] = []
    changed = False
    for block in cfg.blocks:
        if block.index not in reachable:
            changed = changed or bool(block.instructions)
            continue
        live = live_out[block.index]
        survivors: list[IRInstr] = []
        for instr in reversed(block.instructions):
            if _dead(instr, live, nonzero):
                changed = True
                continue
            survivors.append(instr)
            live = _before(instr, live)
        kept.extend(reversed(survivors))
    return kept, changed


def _tidy_control_flow(instructions: list[IRInstr]) -> list[IRInstr]:
    """Drops a `Jump` to the label right after it, then every label no
    jump targets -- both no-ops once dead code is gone."""
    without_jumps = [
        instr
        for i, instr in enumerate(instructions)
        if not (
            isinstance(instr, Jump)
            and i + 1 < len(instructions)
            and isinstance(instructions[i + 1], Label)
            and instructions[i + 1].name == instr.label  # type: ignore[attr-defined]
        )
    ]
    targets = {i.label for i in without_jumps if isinstance(i, (Jump, JumpIfFalse))}
    return [i for i in without_jumps if not (isinstance(i, Label) and i.name not in targets)]


def run(program: IRProgram) -> IRProgram:
    current = program
    while True:
        instructions, changed = _sweep(current)
        current = IRProgram(instructions=instructions, is_function=program.is_function)
        if not changed:
            break
    return IRProgram(
        instructions=_tidy_control_flow(current.instructions), is_function=program.is_function
    )
