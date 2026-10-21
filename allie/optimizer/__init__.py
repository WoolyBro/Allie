"""IR optimization passes -- constant propagation and folding,
dead-code elimination, common subexpression elimination -- plus a pass
manager with an on/off switch, IR verification after each pass, and a
per-pass report.

Each pass (`constant_fold.run`, `dce.run`, `cse.run`) is a pure
`IRProgram -> IRProgram` function with no shared state between passes.
The first two are dataflow analyses over the control-flow graph
(allie/ir/cfg.py); CSE is local to each straight-line stretch.
`pure_functions.PURE_FUNCTIONS` is the one shared source of truth `dce.py`
and `cse.py` both import for which builtins are side-effect-free, so the
two passes can't silently disagree about it.
"""

from allie.optimizer import constant_fold, cse, dce, pure_functions
from allie.optimizer.pass_manager import optimize, run_passes
from allie.optimizer.pure_functions import KNOWN_IMPURE_FUNCTIONS, PURE_FUNCTIONS

# The pipeline, in order: propagate and fold constants (resolving
# constant branches), remove what that left dead or unreachable, then
# share repeated computations. Defined once; the CLI and the benchmark
# harness both use it.
DEFAULT_PASSES = (constant_fold.run, dce.run, cse.run)

__all__ = [
    "DEFAULT_PASSES",
    "KNOWN_IMPURE_FUNCTIONS",
    "PURE_FUNCTIONS",
    "constant_fold",
    "cse",
    "dce",
    "optimize",
    "pure_functions",
    "run_passes",
]
