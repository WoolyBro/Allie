"""Math on numbers and series: `abs`, `sqrt`, `log`, `exp` take either --
`log(c)` is the log of every bar, `sqrt(252.0)` a number -- and
`floor`, `ceil`, `round` turn a Float into an Int, so a split point can
be computed: `floor(len(c) * 0.7)`.

On a series these follow IEEE 754 bar by bar, as the series operators do
(allie/runtime/elementwise.py): `log` of 0 is `-inf` and of a negative
is NaN, a missing bar stays missing. On a single number a value outside
the function's domain is an error, as `1 / 0` is -- one bad number is a
bug, while one bad bar in a long series is data.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import SeriesValue
from allie.stdlib.registry import register
from allie.typechecker.types import (
    ELEMENTWISE,
    ELEMENTWISE_INT,
    FLOAT,
    INT,
    FunctionType,
)

Number = int | float


def _elementwise(
    name: str,
    on_array: Callable[[np.ndarray], np.ndarray],
    on_number: Callable[[float], Number],
    domain: tuple[Callable[[float], bool], str] | None = None,
) -> Callable[[SeriesValue | Number], SeriesValue | Number]:
    def apply(x: SeriesValue | Number) -> SeriesValue | Number:
        if isinstance(x, SeriesValue):
            return SeriesValue(on_array(x.data))
        if domain is not None and not domain[0](x):
            raise AllieValueError(f"{name}: needs {domain[1]}, got {x}")
        return on_number(x)

    apply.__name__ = name
    return apply


abs_ = _elementwise("abs", np.abs, abs)
sqrt = _elementwise("sqrt", np.sqrt, math.sqrt, (lambda x: x >= 0, "a number of at least 0"))
log = _elementwise("log", np.log, math.log, (lambda x: x > 0, "a positive number"))
# Overflow is `inf`, as with any Float arithmetic (`1e308 * 10.0`).
exp = _elementwise("exp", np.exp, lambda x: float(np.exp(x)))


def _to_int(name: str, rounding: Callable[[float], float]) -> Callable[[float], int]:
    def convert(x: float) -> int:
        if not math.isfinite(x):
            raise AllieValueError(f"{name}: {x} has no Int value")
        result = int(rounding(x))
        if not -(2**63) <= result < 2**63:
            raise AllieValueError(f"{name}: {x} is too large for an Int")
        return result

    convert.__name__ = name
    return convert


floor = _to_int("floor", math.floor)
ceil = _to_int("ceil", math.ceil)
# Ties go to the even neighbor, as in Python and numpy: round(2.5) is 2.
round_ = _to_int("round", round)


register("abs", FunctionType(params=(ELEMENTWISE_INT,), ret=ELEMENTWISE_INT), abs_)
register("sqrt", FunctionType(params=(ELEMENTWISE,), ret=ELEMENTWISE), sqrt)
register("log", FunctionType(params=(ELEMENTWISE,), ret=ELEMENTWISE), log)
register("exp", FunctionType(params=(ELEMENTWISE,), ret=ELEMENTWISE), exp)
register("floor", FunctionType(params=(FLOAT,), ret=INT), floor)
register("ceil", FunctionType(params=(FLOAT,), ret=INT), ceil)
register("round", FunctionType(params=(FLOAT,), ret=INT), round_)
