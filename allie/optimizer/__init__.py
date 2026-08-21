"""Phase 5.4: IR optimization passes -- constant folding, dead-code
elimination, common subexpression elimination -- plus a pass manager
with a single on/off switch for Phase 8's benchmark harness.

Each pass (`constant_fold.run`, `dce.run`, `cse.run`) is a pure
`IRProgram -> IRProgram` function with no shared state between passes.
`pure_functions.PURE_FUNCTIONS` is the one shared source of truth `dce.py`
and `cse.py` both import for which builtins are side-effect-free, so the
two passes can't silently disagree about it.
"""

from allie.optimizer import constant_fold, cse, dce, pure_functions
from allie.optimizer.pass_manager import run_passes
from allie.optimizer.pure_functions import KNOWN_IMPURE_FUNCTIONS, PURE_FUNCTIONS

__all__ = [
    "KNOWN_IMPURE_FUNCTIONS",
    "PURE_FUNCTIONS",
    "constant_fold",
    "cse",
    "dce",
    "pure_functions",
    "run_passes",
]
