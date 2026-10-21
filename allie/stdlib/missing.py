"""Missing values, handled once for the whole stdlib.

A NaN in a series means *there is no value at this bar* -- an
indicator's warm-up, a gap in the data file, a bar `shift` moved past
the edge. Every builtin treats it the same way, by kind:

- **Reductions** (`mean`, `std`, `percentile`, ...) skip missing bars,
  as pandas does: `mean(SMA(c, 20))` is the mean of the bars SMA has a
  value for, not NaN. If nothing is left, they raise -- a silent NaN
  would poison everything downstream -- naming how many bars were
  missing. Paired reductions (`covariance`, `correlation`, `beta`) use
  the bars where both series have a value.
- **Windows** (`SMA`, `rolling_*`) are missing wherever the window
  holds a missing bar, and only there.
- **Recurrences** (`EMA`, `RSI`, `ATR`) run over the bars that have a
  value, in order, and leave the missing ones missing: a gap delays the
  next value instead of erasing every value after it.
- **Positional functions** (`shift`, `diff`, `pct_change`, operators)
  are missing exactly where an input they read is.

Only NaN is missing; `inf` is a value (an IEEE result, such as a
division by zero) and is not skipped. The significance tests are the
one documented exception: they drop any non-finite pair
(allie/stdlib/significance.py).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import SeriesValue


def present(series: SeriesValue, fn_name: str, at_least: int = 1) -> np.ndarray:
    """The series' values with missing bars dropped; raises if fewer than
    `at_least` remain."""
    data = series.data
    if len(data) == 0:
        raise AllieValueError(f"{fn_name}: series must not be empty")
    values = data[~np.isnan(data)]
    _require(fn_name, len(values), at_least, len(data))
    return values


def present_pairs(
    a: SeriesValue, b: SeriesValue, fn_name: str, at_least: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """The bars where both series have a value."""
    if len(a) != len(b):
        raise AllieValueError(
            f"{fn_name}: series must be the same length, got {len(a)} and {len(b)}"
        )
    if len(a) == 0:
        raise AllieValueError(f"{fn_name}: series must not be empty")
    mask = ~np.isnan(a.data) & ~np.isnan(b.data)
    _require(fn_name, int(mask.sum()), at_least, len(a), paired=True)
    return a.data[mask], b.data[mask]


def _require(fn_name: str, count: int, at_least: int, total: int, paired: bool = False) -> None:
    if count >= at_least:
        return
    where = "bars where both series have a value" if paired else "values that aren't missing"
    if count == 0:
        raise AllieValueError(f"{fn_name}: no {where} -- all {total} bars are missing (NaN)")
    raise AllieValueError(f"{fn_name}: needs at least {at_least} {where}, got {count}")


def over_present(data: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Runs `fn` over the bars that have a value, in order, and puts each
    result back at the bar it came from; missing bars stay missing."""
    mask = ~np.isnan(data)
    result = np.full(len(data), np.nan, dtype=np.float64)
    if mask.any():
        result[mask] = fn(data[mask])
    return result


def windows(data: np.ndarray, window: int) -> np.ndarray:
    """A read-only (n - window + 1, window) view of every trailing window."""
    return np.lib.stride_tricks.sliding_window_view(data, window)


def rolling(data: np.ndarray, window: int, reduce: Callable[..., Any]) -> np.ndarray:
    """`reduce(window, axis=-1)` at every bar with a full window behind it,
    NaN before that. A window holding a missing bar reduces to NaN with
    any numpy reduction that isn't `nan`-prefixed."""
    result = np.full(len(data), np.nan, dtype=np.float64)
    if window <= len(data):
        result[window - 1 :] = reduce(windows(data, window), axis=-1)
    return result
