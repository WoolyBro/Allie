"""Pass manager: runs the optimizer passes over every IR program, with
an on/off switch (the benchmark harness compares the two), optional IR
verification after each pass, and an optional per-pass report.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from allie.ir.instructions import IRProgram
from allie.ir.verify import verify as verify_ir

Pass = Callable[[IRProgram], IRProgram]
Functions = Mapping[str, tuple[IRProgram, list[str]]]


def pass_name(pass_fn: Pass) -> str:
    """`constant_fold` for `allie.optimizer.constant_fold.run`."""
    return pass_fn.__module__.rsplit(".", 1)[-1]


# Sequence, not list: `list` is invariant, so a caller's
# `list[<concrete pass function>]` wouldn't satisfy a `list[Pass]`
# parameter even though every element is a valid Pass.
def run_passes(
    program: IRProgram,
    passes: Sequence[Pass],
    enabled: bool,
    *,
    verify: bool = False,
    functions: Functions | None = None,
    name: str = "main",
    report: list[str] | None = None,
) -> IRProgram:
    """Runs `passes` in order over `program` if `enabled`, else returns
    `program` unchanged. Each pass is a pure `IRProgram -> IRProgram`
    function with no shared state, so the order they're listed in is the
    order they run in.

    With `verify`, the IR is checked (allie/ir/verify.py) before the
    first pass and after each one, so a pass that breaks an invariant is
    named in the error rather than discovered later as a wrong answer.
    With `report`, one line per pass is appended: how many instructions
    it removed from `name`.
    """
    if not enabled:
        return program
    if verify:
        verify_ir(program, functions=functions, where=f"{name}, as lowered")
    for pass_fn in passes:
        before = len(program.instructions)
        program = pass_fn(program)
        if verify:
            verify_ir(program, functions=functions, where=f"{name}, after {pass_name(pass_fn)}")
        if report is not None:
            report.append(
                f"{name}: {pass_name(pass_fn)} {before} -> {len(program.instructions)} instructions"
            )
    return program


def optimize(
    main: IRProgram,
    functions: dict[str, tuple[IRProgram, list[str]]],
    passes: Sequence[Pass],
    enabled: bool,
    *,
    verify: bool = False,
    report: list[str] | None = None,
) -> tuple[IRProgram, dict[str, tuple[IRProgram, list[str]]]]:
    """`run_passes` over the main program and every function body -- the
    one place callers (the CLI, `--verify`, the benchmarks) get the
    optimized program from, so they can't drift apart."""
    optimized_main = run_passes(
        main, passes, enabled, verify=verify, functions=functions, report=report
    )
    optimized_functions = {
        fn_name: (
            run_passes(
                body,
                passes,
                enabled,
                verify=verify,
                functions=functions,
                name=fn_name,
                report=report,
            ),
            params,
        )
        for fn_name, (body, params) in functions.items()
    }
    return optimized_main, optimized_functions
