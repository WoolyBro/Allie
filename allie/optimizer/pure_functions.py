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
        "RSI",
        "ATR",
        "rolling_max",
        "rolling_min",
        "count",
        "sum",
        "min",
        "max",
        "rolling_sum",
        "rolling_zscore",
        "rolling_corr",
        "cumsum",
        # Numbers and series alike; raising on a bad number, identically
        # on every call.
        "abs",
        "sqrt",
        "log",
        "exp",
        "floor",
        "ceil",
        "round",
        # Returns and risk.
        "log_returns",
        "cumulative_returns",
        "drawdown",
        "max_drawdown",
        "volatility",
        "sortino",
        "value_at_risk",
        "expected_shortfall",
        "beta",
        # Matrices.
        "matrix",
        "with_column",
        "identity",
        "rows",
        "cols",
        "row",
        "column",
        "covariance_matrix",
        "ols",
        # Significance testing. permutation_pvalue is pure despite
        # sampling: it takes an explicit seed, so the same arguments
        # always give the same p-value (docs/spec.md "Determinism").
        "bonferroni",
        "benjamini_hochberg",
        "sharpe",
        "permutation_pvalue",
        "len",
        # The `__index_*` functions behind `s[i]` raise on a bad index,
        # but identically on every call, so deduplicating two identical
        # calls can't change behavior. (DCE never removes a Call at all,
        # so an unused out-of-range `s[99];` still raises.)
        "__index_float",
        "__index_int",
        "__index_bool",
    }
)

# Never eligible for CSE/DCE-across-calls. The optimizer passes don't
# consult this set -- `PURE_FUNCTIONS` is an allowlist, so anything
# absent from it is already treated as impure. Its job is to make the
# classification *deliberate*: `tests/unit/test_optimizer.py` asserts
# every registered builtin appears in exactly one of these two sets, so
# adding a builtin without deciding whether it is pure fails the build
# rather than silently defaulting to "not optimized".
#
# That test was added in Phase 9 after the audit found the then-builtin
# `crosses_above`/`crosses_below` (Phase 7) in neither set.
KNOWN_IMPURE_FUNCTIONS: frozenset[str] = frozenset(
    {"load", "print", "normal", "buy", "sell", "short", "cover", "target"}
)

# Calls that can't read or assign any variable: every builtin, plus the
# array-literal constructor. Anything else -- a user function, a `when`
# body that `__backtest_run` invokes -- may touch module-level
# variables, so dataflow passes must assume it reads and writes them all.
VARIABLE_SAFE_CALLS: frozenset[str] = PURE_FUNCTIONS | KNOWN_IMPURE_FUNCTIONS | {"__make_series"}
