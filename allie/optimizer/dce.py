"""Dead-code elimination -- Phase 5.4. `run(program) -> IRProgram`, pure
function, no shared state with the other passes.

**The loop-carried-liveness trap, and how this avoids it.** A naive
single-forward-pass DCE asks "is this register/variable read *later* in
program order" -- and a `for`-loop's increment `StoreVar` (from the
allie.ir.lowering desugar) fails that check: nothing *after* it in the
flat instruction list reads the loop variable, since the read happens at
the *top* of the loop, reached only via the back-edge `Jump`. A naive
scan would call the increment dead and delete it, breaking the loop
(the counter would never advance).

The fix used here is simpler than tracking loop ranges explicitly:
liveness is computed as "is this register/variable read *anywhere in the
whole program*", not "read strictly after this instruction in list
order". Position doesn't matter, so the loop's back-edge is a non-issue
by construction -- the condition check's `LoadVar i` counts as a read of
`i` regardless of where in the list it sits relative to the increment's
`StoreVar i`. Trade-off, stated plainly: this is *more conservative*
than a proper forward-flow analysis -- it won't catch `let x = 1; x = 2;
print(x);`'s first assignment as dead (that store IS read somewhere, just
not by the read that will actually observe it), where a position-aware
analysis would. Given this IR is a flat list with no CFG (see
allie/ir/instructions.py's module docstring), a whole-program check is
the tractable option that's still correct for what's actually asked:
removing a genuinely-never-read variable, and never removing a loop's
carried variable.

`Call` instructions are never removed regardless of their destination's
liveness -- they may be side-effecting (`print`, and eventually `buy`/
`sell`), and this pass has no way to know which without consulting
allie.optimizer.pure_functions, which is cse.py's job, not this one's:
DCE's job is simpler and safer -- assume every Call might matter.

**Fixpoint, not a single pass.** Removing one dead instruction can make
another dead: a `LoadConst` that only ever fed a since-removed `StoreVar`
still looks "read" during the very pass that deletes that `StoreVar`
(its operand scan still sees the reference), so a single pass leaves it
behind. `run()` repeats the pass until nothing more gets removed.
"""

from __future__ import annotations

from allie.ir.instructions import (
    BinOp,
    Call,
    IRInstr,
    IRProgram,
    JumpIfFalse,
    LoadConst,
    LoadVar,
    MemberGet,
    Move,
    Return,
    StoreVar,
    UnaryOp,
)

_TRAPPING_OPS = frozenset({"/", "%"})


def _can_trap(instr: IRInstr, instructions: list[IRInstr]) -> bool:
    """Whether removing `instr` could delete an observable side effect.

    Raising is a side effect. `let x: Int = 1 / 0;` where `x` is never
    read still has to raise "division by zero" -- deleting the division
    because its result is unused changes the program's behaviour from
    "fails" to "succeeds silently", which is strictly worse than the
    missed optimization.

    Found by `allie run --verify`: with the optimizer on, the tree-walker
    raised while the IR and LLVM backends exited 0, because only they run
    this pass. That is exactly the class of divergence three-backend
    cross-validation exists to catch.

    A divisor that is a literal non-zero constant cannot trap, so those
    stay removable -- otherwise every `x / 2` in dead code would be
    pinned for no reason.
    """
    if not isinstance(instr, BinOp) or instr.op not in _TRAPPING_OPS:
        return False
    for candidate in instructions:
        if isinstance(candidate, LoadConst) and candidate.dest == instr.right:
            return bool(candidate.value == 0)
    return True  # divisor not a known constant -- must assume it can be zero


def run(program: IRProgram) -> IRProgram:
    # Fixpoint, not a single pass: removing a dead StoreVar can make the
    # LoadConst that only ever fed *it* dead too (nothing else reads that
    # register once the StoreVar referencing it is gone) -- a single
    # pass would leave that LoadConst behind because it still looked
    # "read" by the very instruction this pass was about to delete.
    instructions = list(program.instructions)
    while True:
        read_vars, read_regs = _collect_reads(instructions)
        kept: list[IRInstr] = []
        changed = False
        for instr in instructions:
            if isinstance(instr, StoreVar):
                if instr.name not in read_vars:
                    changed = True
                    continue  # dead: this variable is never read anywhere
            elif isinstance(instr, (LoadConst, LoadVar, BinOp, UnaryOp, MemberGet, Move)):
                if instr.dest not in read_regs and not _can_trap(instr, instructions):
                    changed = True
                    continue  # dead: this register is never read anywhere
            # Call, Label, Jump, JumpIfFalse, Return always survive.
            kept.append(instr)
        instructions = kept
        if not changed:
            return IRProgram(instructions=instructions)


def _collect_reads(instructions: list[IRInstr]) -> tuple[set[str], set[str]]:
    read_vars: set[str] = set()
    read_regs: set[str] = set()

    for instr in instructions:
        if isinstance(instr, LoadVar):
            read_vars.add(instr.name)
        elif isinstance(instr, StoreVar):
            read_regs.add(instr.src)
        elif isinstance(instr, BinOp):
            read_regs.add(instr.left)
            read_regs.add(instr.right)
        elif isinstance(instr, UnaryOp):
            read_regs.add(instr.operand)
        elif isinstance(instr, Call):
            read_regs.update(instr.args)
        elif isinstance(instr, MemberGet):
            read_regs.add(instr.obj)
        elif isinstance(instr, Move):
            read_regs.add(instr.src)
        elif isinstance(instr, JumpIfFalse):
            read_regs.add(instr.cond)
        elif isinstance(instr, Return) and instr.value is not None:
            read_regs.add(instr.value)

    return read_vars, read_regs
