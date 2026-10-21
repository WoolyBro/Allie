"""Operators on whole series: arithmetic, comparisons, boolean logic and
crossings, bar by bar.

Every backend calls these for an operator with a series operand -- the
two interpreters directly, the LLVM backend through the bridge -- and a
scalar operand is broadcast to every bar, so `close - open`,
`close > 100.0` and `rsi crosses_below 30.0` all work. Unlike scalar
operators, which each interpreter implements independently so
cross-validation can catch a mistake in one, this is a single numpy
implementation shared the way stdlib functions are: the arithmetic is
numpy's, not ours.

Semantics:

- `+ - * /` produce a Series<Float>. Division by zero follows IEEE 754
  -- inf, or NaN for 0/0 -- instead of raising: one zero-volume bar
  shouldn't abort a whole computation, and NaN already means "no value at
  this bar" (an indicator's warm-up, a missing price).
- `< > <= >=` produce a Series<Bool>. Any comparison involving NaN is
  false, so warm-up bars never signal.
- `and` / `or` / `!` combine Series<Bool> bar by bar. Both sides are
  always evaluated -- there is nothing to short-circuit.
- `a crosses_above b` is true exactly at the bars where `a` moves from at
  or below `b` to strictly above it; `crosses_below` is symmetric. The
  first bar never crosses: there is no previous bar to cross from.
- Two series of different lengths are an error, never silently aligned.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, SeriesValue

_ARITHMETIC: dict[str, Any] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
}
_COMPARISON: dict[str, Any] = {
    "<": np.less,
    ">": np.greater,
    "<=": np.less_equal,
    ">=": np.greater_equal,
}
_LOGICAL: dict[str, Any] = {"and": np.logical_and, "or": np.logical_or}
CROSSINGS = ("crosses_above", "crosses_below")
OPERATORS = frozenset(_ARITHMETIC) | frozenset(_COMPARISON) | frozenset(_LOGICAL) | set(CROSSINGS)


def is_series(value: object) -> bool:
    return isinstance(value, (SeriesValue, BoolSeriesValue))


def _length(op: str, left: object, right: object) -> int:
    lengths = {len(v) for v in (left, right) if is_series(v)}  # type: ignore[arg-type]
    if len(lengths) > 1:
        a, b = (len(v) for v in (left, right))  # type: ignore[arg-type]
        raise AllieValueError(f"'{op}' needs series of the same length, got {a} and {b}")
    return lengths.pop()


def _array(value: object, length: int, dtype: type) -> np.ndarray:
    if is_series(value):
        return np.asarray(value.data, dtype=dtype)  # type: ignore[attr-defined]
    return np.full(length, value, dtype=dtype)


def apply_binary(op: str, left: object, right: object) -> SeriesValue | BoolSeriesValue:
    """`left op right` where at least one side is a series."""
    length = _length(op, left, right)
    if op in _ARITHMETIC:
        with np.errstate(all="ignore"):
            values = _ARITHMETIC[op](
                _array(left, length, np.float64), _array(right, length, np.float64)
            )
        return SeriesValue(values)
    if op in _COMPARISON:
        with np.errstate(invalid="ignore"):
            return BoolSeriesValue(
                _COMPARISON[op](_array(left, length, np.float64), _array(right, length, np.float64))
            )
    if op in _LOGICAL:
        return BoolSeriesValue(
            _LOGICAL[op](_array(left, length, bool), _array(right, length, bool))
        )
    if op in CROSSINGS:
        return crossing(op, _array(left, length, np.float64), _array(right, length, np.float64))
    raise AllieValueError(f"'{op}' is not defined on series")


def apply_unary(op: str, operand: SeriesValue | BoolSeriesValue) -> SeriesValue | BoolSeriesValue:
    if op == "-" and isinstance(operand, SeriesValue):
        return SeriesValue(-operand.data)
    if op == "!":
        return BoolSeriesValue(np.logical_not(np.asarray(operand.data, dtype=bool)))
    raise AllieValueError(f"unary '{op}' is not defined on {type(operand).__name__}")


def crossing(op: str, a: np.ndarray, b: np.ndarray) -> BoolSeriesValue:
    """`(a > b) & (a[i-1] <= b[i-1])` for crosses_above; the mirror image
    for crosses_below. Comparisons with NaN are false, so a bar whose
    previous value is missing (the first bar, a warm-up bar) can't
    cross."""
    previous_a = np.concatenate(([np.nan], a[:-1]))
    previous_b = np.concatenate(([np.nan], b[:-1]))
    with np.errstate(invalid="ignore"):
        if op == "crosses_above":
            return BoolSeriesValue((a > b) & (previous_a <= previous_b))
        return BoolSeriesValue((a < b) & (previous_a >= previous_b))


def series_literal(elements: list[object]) -> SeriesValue | BoolSeriesValue:
    """An array literal's value, on every backend. `[true, false]` is a
    Series<Bool> -- the same kind of value `close > open` produces, so
    the two print and combine identically; anything else is stored as
    float64."""
    if elements and all(isinstance(e, (bool, np.bool_)) for e in elements):
        return BoolSeriesValue(np.array(elements, dtype=bool))
    return SeriesValue(np.array(elements, dtype=np.float64))


def as_signal(value: object) -> BoolSeriesValue:
    """A `when` condition as one bool per bar."""
    if isinstance(value, BoolSeriesValue):
        return value
    if isinstance(value, SeriesValue):
        return BoolSeriesValue(value.data != 0)
    raise AllieValueError(f"a 'when' condition must be a Series<Bool>, got {type(value).__name__}")
