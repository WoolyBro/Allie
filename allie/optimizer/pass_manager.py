"""Pass manager -- Phase 5.4. A single on/off switch, used by the
Phase 8 benchmark harness to produce "optimized vs unoptimized" numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from allie.ir.instructions import IRProgram

Pass = Callable[[IRProgram], IRProgram]


# Sequence, not list: `list` is invariant, so a caller's
# `list[<concrete pass function>]` wouldn't satisfy a `list[Pass]`
# parameter even though every element is a valid Pass.
def run_passes(program: IRProgram, passes: Sequence[Pass], enabled: bool) -> IRProgram:
    """Runs `passes` in order over `program` if `enabled`, else returns
    `program` unchanged. Each pass is a pure `IRProgram -> IRProgram`
    function with no shared state, so the order they're listed in is the
    order they run in -- callers decide that, this just threads the
    result of each into the next."""
    if not enabled:
        return program
    for pass_fn in passes:
        program = pass_fn(program)
    return program
