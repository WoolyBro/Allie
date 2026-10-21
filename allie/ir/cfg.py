"""Control-flow graph over the flat IR, and a small dataflow solver.

The IR is *stored* as one flat instruction list -- lowering produces it
that way and both backends consume it that way. But any analysis that
reasons about values flowing between instructions (liveness, constants,
definite assignment) needs to know which instruction can run after
which, and in a flat list that is not "the next one in text order": a
loop's back-edge runs an earlier instruction next, and a branch skips
ahead. Both optimizer bugs in this project's history came from exactly
that confusion (docs/how-it-works.md section 3.4). This module makes
control flow explicit instead of leaving each pass to approximate it.

`build_cfg` splits a program into basic blocks -- maximal straight-line
runs, each entered only at its first instruction -- with successor and
predecessor edges; `flatten` turns a graph back into a program. Block
order is preserved, so `flatten(build_cfg(p))` is `p` exactly.

`forward` and `backward` solve a dataflow problem to its fixpoint with
a worklist. States are opaque to the solver; the caller supplies the
transfer function for a block and the join for two states. A block that
hasn't been reached yet has state None, which the solver treats as the
identity for join -- so "may" analyses (union) and "must" analyses
(intersection, constants) both work without a separate top element.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from allie.ir.instructions import IRInstr, IRProgram, Jump, JumpIfFalse, Label, Return

S = TypeVar("S")


class CFGError(Exception):
    """The program's control flow is malformed (e.g. a jump to a label
    that doesn't exist) -- always a compiler bug, never a user error."""


@dataclass
class BasicBlock:
    index: int
    instructions: list[IRInstr]
    successors: list[int] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)

    @property
    def label(self) -> str | None:
        first = self.instructions[0] if self.instructions else None
        return first.name if isinstance(first, Label) else None


@dataclass
class CFG:
    blocks: list[BasicBlock]

    def reachable(self) -> set[int]:
        """Indices of every block reachable from the entry block."""
        if not self.blocks:
            return set()
        seen = {0}
        stack = [0]
        while stack:
            for successor in self.blocks[stack.pop()].successors:
                if successor not in seen:
                    seen.add(successor)
                    stack.append(successor)
        return seen


def build_cfg(program: IRProgram) -> CFG:
    instructions = program.instructions
    leaders = {0}
    for i, instr in enumerate(instructions):
        if isinstance(instr, Label):
            leaders.add(i)
        if isinstance(instr, (Jump, JumpIfFalse, Return)):
            leaders.add(i + 1)
    starts = sorted(i for i in leaders if i < len(instructions)) or [0]
    bounds = list(zip(starts, [*starts[1:], len(instructions)], strict=True))
    blocks = [BasicBlock(index, list(instructions[a:b])) for index, (a, b) in enumerate(bounds)]

    by_label: dict[str, int] = {}
    for block in blocks:
        if block.label is not None:
            if block.label in by_label:
                raise CFGError(f"label '{block.label}' is defined twice")
            by_label[block.label] = block.index

    def target(label: str) -> int:
        if label not in by_label:
            raise CFGError(f"jump to undefined label '{label}'")
        return by_label[label]

    for block in blocks:
        last = block.instructions[-1] if block.instructions else None
        falls_through = block.index + 1 < len(blocks)
        if isinstance(last, Jump):
            block.successors = [target(last.label)]
        elif isinstance(last, JumpIfFalse):
            targets = [block.index + 1] if falls_through else []
            if target(last.label) not in targets:
                targets.append(target(last.label))
            block.successors = targets
        elif isinstance(last, Return):
            block.successors = []
        else:
            block.successors = [block.index + 1] if falls_through else []
    for block in blocks:
        for successor in block.successors:
            blocks[successor].predecessors.append(block.index)
    return CFG(blocks)


def flatten(cfg: CFG, keep: set[int] | None = None, *, is_function: bool = False) -> IRProgram:
    """The blocks' instructions in block order -- only those in `keep`,
    if given."""
    return IRProgram(
        instructions=[
            instr
            for block in cfg.blocks
            if keep is None or block.index in keep
            for instr in block.instructions
        ],
        is_function=is_function,
    )


def without_unreachable_code(program: IRProgram) -> IRProgram:
    """`program` minus every block no path from the entry reaches."""
    cfg = build_cfg(program)
    return flatten(cfg, cfg.reachable(), is_function=program.is_function)


def forward(
    cfg: CFG,
    entry_state: S,
    transfer: Callable[[BasicBlock, S], S],
    join: Callable[[S, S], S],
) -> dict[int, S]:
    """Solves a forward problem; returns each reachable block's IN state."""
    if not cfg.blocks:
        return {}
    in_states: dict[int, S] = {0: entry_state}
    out_states: dict[int, S] = {}
    worklist = deque([0])
    while worklist:
        index = worklist.popleft()
        out = transfer(cfg.blocks[index], in_states[index])
        if index in out_states and out_states[index] == out:
            continue
        out_states[index] = out
        for successor in cfg.blocks[index].successors:
            merged = out if successor not in in_states else join(in_states[successor], out)
            if successor not in in_states or merged != in_states[successor]:
                in_states[successor] = merged
                worklist.append(successor)
    return in_states


def backward(
    cfg: CFG,
    exit_state: S,
    bottom: S,
    transfer: Callable[[BasicBlock, S], S],
    join: Callable[[S, S], S],
) -> dict[int, S]:
    """Solves a backward problem; returns each block's OUT state (the
    state just after its last instruction). Blocks with no successors
    start from `exit_state`; a successor not solved yet contributes
    `bottom` (the empty state), so states only ever grow."""
    out_states: dict[int, S] = {}
    in_states: dict[int, S] = {}
    worklist = deque(reversed(range(len(cfg.blocks))))
    queued = set(worklist)
    while worklist:
        index = worklist.popleft()
        queued.discard(index)
        block = cfg.blocks[index]
        out = exit_state if not block.successors else bottom
        for successor in block.successors:
            out = join(out, in_states.get(successor, bottom))
        out_states[index] = out
        new_in = transfer(block, out)
        if index in in_states and in_states[index] == new_in:
            continue
        in_states[index] = new_in
        for predecessor in block.predecessors:
            if predecessor not in queued:
                queued.add(predecessor)
                worklist.append(predecessor)
    return out_states
