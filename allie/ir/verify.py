"""IR verifier: the structural invariants every IR program must keep.

Lowering produces IR and every optimizer pass rewrites it. A pass that
breaks an invariant rarely crashes; it produces IR that one backend runs
slightly wrong, and the symptom shows up far from the cause. The
verifier turns that into an immediate, named failure:
`run_passes(..., verify=True)` checks the program before the first pass
and after each one, and says which pass broke it. The test suite runs
every program in the repository, and every program the optimizer fuzz
test generates, through it; `allie run --debug-ir` does too.

Invariants:

- labels are unique, and every jump targets one that exists;
- every register is written by exactly one instruction -- except `Move`
  destinations, which short-circuit `and`/`or` writes from two branches
  by design -- and never by both a `Move` and something else;
- on every path from the entry, a register is written before it is read
  (definite assignment, a forward "must" dataflow problem over the CFG);
- a call to a user function passes as many arguments as it takes, when
  the function table is supplied.
"""

from __future__ import annotations

from collections.abc import Mapping

from allie.ir.cfg import BasicBlock, CFGError, build_cfg, forward
from allie.ir.instructions import Call, IRProgram, Move, reads, writes


class IRVerificationError(Exception):
    """The IR breaks an invariant -- a compiler bug, never a user error."""


def verify(
    program: IRProgram,
    *,
    functions: Mapping[str, tuple[IRProgram, list[str]]] | None = None,
    where: str = "",
) -> None:
    problems: list[str] = []

    move_written: set[str] = set()
    other_written: set[str] = set()
    for instr in program.instructions:
        dest = writes(instr)
        if dest is None:
            continue
        if isinstance(instr, Move):
            move_written.add(dest)
        elif dest in other_written:
            problems.append(f"register {dest} is written twice")
        else:
            other_written.add(dest)
    for register in sorted(move_written & other_written):
        problems.append(f"register {register} is written by both a Move and another instruction")

    try:
        cfg = build_cfg(program)
    except CFGError as error:
        problems.append(str(error))
    else:
        def defined_after(block: BasicBlock, defined: frozenset[str]) -> frozenset[str]:
            return defined | {w for w in map(writes, block.instructions) if w is not None}

        def on_every_path(a: frozenset[str], b: frozenset[str]) -> frozenset[str]:
            return a & b

        nothing: frozenset[str] = frozenset()
        in_states = forward(cfg, nothing, defined_after, on_every_path)
        for index, defined in in_states.items():
            current = set(defined)
            for instr in cfg.blocks[index].instructions:
                for register in reads(instr):
                    if register not in current:
                        problems.append(f"{instr} reads {register} before it is written")
                dest = writes(instr)
                if dest is not None:
                    current.add(dest)

    if functions is not None:
        for instr in program.instructions:
            if isinstance(instr, Call) and instr.fn_name in functions:
                expected = len(functions[instr.fn_name][1])
                if len(instr.args) != expected:
                    problems.append(
                        f"{instr} passes {len(instr.args)} argument(s); "
                        f"'{instr.fn_name}' takes {expected}"
                    )

    if problems:
        prefix = f"{where}: " if where else ""
        shown = "; ".join(problems[:5])
        more = f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""
        raise IRVerificationError(f"{prefix}invalid IR: {shown}{more}")
