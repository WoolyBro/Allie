"""Single shared source of truth for which builtin functions are pure --
written once here, imported by both dce.py and cse.py, rather than each
file defining its own copy. The two passes must never be able to
silently disagree about what's side-effecting: dce.py never removes a
Call to an impure function (it might be observed via a side effect even
if its result is unused); cse.py never collapses two calls to one unless
the function is in this set.

Deliberately a hand-picked allowlist, not "everything in allie.stdlib.
registry.BUILTINS except a denylist": a new builtin defaults to NOT being
optimized across, which is the safe direction to be wrong in. `load` and
`print` are excluded because they're I/O; `normal` is excluded even
though it's technically deterministic given the same seed, because
silently deduplicating a call the user wrote out twice would be a
surprising thing for a "random sample" builtin to do. `buy`/`sell` don't
exist as registered builtins yet (Phase 7), but are listed here too, per
the merge notes: both are new, both side-effecting, both easy to forget
if this set were duplicated instead of shared.
"""

from __future__ import annotations

PURE_FUNCTIONS: frozenset[str] = frozenset(
    {
        "mean",
        "median",
        "std",
        "variance",
        "covariance",
        "correlation",
        "SMA",
        "EMA",
        "rolling_mean",
        "rolling_std",
        "shift",
        "diff",
        "pct_change",
        "transpose",
        "inverse",
        "determinant",
        "dot",
        "zscore",
        "percentile",
    }
)

# Never eligible for CSE/DCE-across-calls, even though some names below
# aren't registered builtins yet -- listed explicitly so nobody adds them
# to PURE_FUNCTIONS above without noticing this comment.
KNOWN_IMPURE_FUNCTIONS: frozenset[str] = frozenset({"load", "print", "normal", "buy", "sell"})
