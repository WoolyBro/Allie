"""Phase 1 value representations (docs/architecture.md "Phase 1 -- hand-built
evaluator").

Values are raw Python int/float/bool/str for now -- no wrapper class.
Numeric promotion rule: int op float -> float, decided here so it does not
need retrofitting once real numeric literals exist. The Decimal-backed
Money type and the pandas-backed Series type (docs/spec.md "Numeric
precision", "Pandas dependency") land once the type system needs them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runtime.environment import Environment


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


Value = int | float | bool | str | Closure | None
