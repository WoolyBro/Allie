"""Series element access: the runtime half of `s[i]`, plus `len(s)`.

`s[i]` is not a function users call. IR lowering turns it into a call to
one of the `__index_*` builtins below, chosen by the series' static
element type, so the IR interpreter and the LLVM bridge run it through
the same generic builtin-call path as `SMA` or `mean`; the tree-walking
interpreter calls the same functions directly. Names starting with `__`
are internal -- `builtin_function_types()` hides them from the type
checker, so source code can't call them by name.

Indices start at 0 and must be in range. There is deliberately no
Python-style negative indexing: in a bar-by-bar language, `s[-1]`
quietly reading the *last* bar -- the future, from any earlier bar's
point of view -- is a trap rather than a convenience. Write
`s[len(s) - 1]`.

The Int variant exists because integer array literals are stored as
float64 like every other series; it converts the element back, so an
`Int`-typed `s[i]` is an Int on every backend.
"""

from __future__ import annotations

from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, SeriesValue
from allie.stdlib.registry import register
from allie.typechecker.types import ANY, BOOL, FLOAT, INT, FunctionType, SeriesType, Type


def _checked(index: int, length: int) -> int:
    if index < 0:
        raise AllieValueError(
            f"series index {index} is negative; indices start at 0 "
            "(there is no wrap-around indexing -- use len(s) - 1 for the last element)"
        )
    if index >= length:
        raise AllieValueError(
            f"series index {index} is out of range for a series of length {length}"
        )
    return index


def index_float(series: SeriesValue, index: int) -> float:
    return float(series.data[_checked(index, len(series))])


def index_int(series: SeriesValue, index: int) -> int:
    return int(series.data[_checked(index, len(series))])


def index_bool(series: SeriesValue | BoolSeriesValue, index: int) -> bool:
    # A crossing result is a BoolSeriesValue; a `[true, false]` literal
    # is a float64 SeriesValue like every other literal.
    return bool(series.data[_checked(index, len(series))])


def series_length(series: SeriesValue | BoolSeriesValue) -> int:
    return len(series)


def index_function_for(element_type: Type) -> str:
    """The `__index_*` builtin that reads one element of a series whose
    elements have `element_type` (the type checker only allows these
    three)."""
    if element_type == INT:
        return "__index_int"
    if element_type == BOOL:
        return "__index_bool"
    return "__index_float"


register("__index_float", FunctionType(params=(SeriesType(FLOAT), INT), ret=FLOAT), index_float)
register("__index_int", FunctionType(params=(SeriesType(INT), INT), ret=INT), index_int)
register("__index_bool", FunctionType(params=(SeriesType(BOOL), INT), ret=BOOL), index_bool)
# Any series: `len(close > open)` is as meaningful as `len(close)`.
register("len", FunctionType(params=(SeriesType(ANY),), ret=INT), series_length)
