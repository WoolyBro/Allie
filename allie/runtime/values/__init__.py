"""Value representations -- Phase 1's scalars, extended with Phase 4's
quant runtime values (SeriesValue/MatrixValue/PriceSeriesValue) and the
BuiltinFunction wrapper that lets the interpreter call a stdlib function
through the exact same path as a user-defined Closure.

Scalars are raw Python int/float/bool/str -- no wrapper class.
Numeric promotion rule: int op float -> float.

SeriesValue/MatrixValue are backed by plain numpy float64 ndarrays, NOT
pandas.Series/DataFrame. That's a deliberate simplification versus
docs/spec.md's original "Series wraps pandas.Series, don't reimplement
DatetimeIndex" aspiration -- Phase 4 asked for a thin numpy wrapper
specifically, so pandas stays a test-only dependency for the stdlib
functions (cross-checking hand-computed values), not a runtime one, for
now. Revisit the pandas-backed version if/when real DatetimeIndex
alignment is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from allie.runtime.errors import AllieValueError

if TYPE_CHECKING:
    from allie.runtime.environment import Environment


def promote_numeric(left: Any, right: Any) -> tuple[Any, Any]:
    """int op float -> float. Leaves bool and str operands untouched."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left, right
    if isinstance(left, float) or isinstance(right, float):
        return float(left), float(right)
    return left, right


@dataclass
class Closure:
    """A function value: parameter names, an unevaluated body node, and the
    environment active where the function was defined. Captured by
    reference to that Environment, not a snapshot of its values -- see
    docs/architecture.md "Closures"."""

    params: list[str]
    body: Any
    defining_env: Environment
    # The body compiled by the tree-walking interpreter (a callable taking
    # the call's Environment), so a call never re-inspects the AST.
    code: Any = field(default=None, repr=False, compare=False)


@dataclass
class BuiltinFunction:
    """Wraps a stdlib Python callable (allie.stdlib.registry.BUILTINS) so
    the interpreter's Call handling doesn't need a separate code path for
    "user function" vs. "builtin" -- both are just callable Values."""

    name: str
    fn: Callable[..., Any]


@dataclass(eq=False)
class SeriesValue:
    """A 1-D numeric series -- Phase 4's default array-literal type and
    the return type of most stdlib functions (SMA, mean, ...).

    `eq=False`: numpy arrays don't support the plain `==` a dataclass's
    auto-generated __eq__ would produce (it returns an elementwise array,
    not a bool, so the auto __eq__ would raise on the boolean context the
    dataclass machinery needs). Compare `.data` with `numpy.testing`
    helpers or `numpy.array_equal` directly instead of `==` on the whole
    value.
    """

    data: np.ndarray
    dates: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim != 1:
            raise AllieValueError(f"SeriesValue requires a 1D array, got {self.data.ndim}D")

    def __len__(self) -> int:
        return len(self.data)


@dataclass(eq=False)
class BoolSeriesValue:
    """A 1-D boolean series -- the runtime value behind `Series<Bool>`,
    produced by `crosses_above`/`crosses_below` (Phase 7.3).

    Separate from SeriesValue rather than encoding True/False as
    1.0/0.0 float64: SeriesValue enforces float64 in `__post_init__` by
    construction, so a boolean series stored there would either break
    that invariant or silently become numeric data that `mean`/`SMA`
    would happily (and meaninglessly) accept. See SeriesValue's
    docstring for why `eq=False`.
    """

    data: np.ndarray

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=bool)
        if self.data.ndim != 1:
            raise AllieValueError(f"BoolSeriesValue requires a 1D array, got {self.data.ndim}D")

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class StrategyValue:
    """A declared `strategy`: for each `when` clause, its signal -- the
    condition, evaluated at the declaration -- and its action body, plus
    the environment the actions run in.

    Evaluating the conditions at the declaration (not at `backtest`) is
    what the IR and LLVM backends do; doing it here too means reassigning
    an indicator after declaring a strategy can't change the strategy on
    one backend only. Action bodies are typed loosely (`Any`) for the
    same circular-import reason `Closure.body` is.
    """

    name: str
    clauses: list[tuple[BoolSeriesValue, Any]]
    defining_env: Environment


@dataclass(eq=False)
class MatrixValue:
    """A 2-D numeric matrix. See SeriesValue's docstring for why
    `eq=False`."""

    data: np.ndarray

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim != 2:
            raise AllieValueError(f"MatrixValue requires a 2D array, got {self.data.ndim}D")


@dataclass(eq=False)
class PriceSeriesValue:
    """An OHLCV price history, as returned by the stdlib `load` function.
    `open`/`high`/`low`/`close`/`volume` are each a SeriesValue, matching
    the spec. `dates` is a plain numpy datetime64 array rather than a
    SeriesValue -- SeriesValue is float64-only by construction
    (`__post_init__` above enforces it), so forcing dates into one would
    either break that invariant or silently store a nonsensical numeric
    encoding of a date. A documented, deliberate deviation from a literal
    reading of "each a SeriesValue" for this one field.
    """

    open: SeriesValue
    high: SeriesValue
    low: SeriesValue
    close: SeriesValue
    volume: SeriesValue
    dates: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="datetime64[ns]"))


Value = (
    int
    | float
    | bool
    | str
    | Closure
    | BuiltinFunction
    | SeriesValue
    | BoolSeriesValue
    | MatrixValue
    | PriceSeriesValue
    | StrategyValue
    | None
)
