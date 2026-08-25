"""stdlib I/O: `load` and `print` -- Phase 4.3.

`load` uses the standard-library `csv` module + numpy, not pandas, for
the same reason SeriesValue/MatrixValue are numpy-backed rather than
pandas-backed (see allie/runtime/values's module docstring): pandas
stays a test-only dependency here, not a runtime one.
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import PriceSeriesValue, SeriesValue
from allie.stdlib.registry import register
from allie.typechecker.types import ANY, PRICE_SERIES, STRING, VOID, FunctionType

_REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")
_NUMERIC_COLUMNS = tuple(col for col in _REQUIRED_COLUMNS if col != "date")


def load(path: str) -> PriceSeriesValue:
    """Loads an OHLCV CSV. Column headers are matched case-insensitively
    against date/open/high/low/close/volume; a missing column raises
    naming exactly which one, rather than a raw pandas/csv KeyError."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise AllieValueError(f"load: no such file: {path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv_module.DictReader(handle)
        if reader.fieldnames is None:
            raise AllieValueError(f"load: {path} has no header row")
        header_by_lower = {name.strip().lower(): name for name in reader.fieldnames}
        missing = [col for col in _REQUIRED_COLUMNS if col not in header_by_lower]
        if missing:
            raise AllieValueError(
                f"load: {path} is missing required column(s): {', '.join(missing)}"
            )

        dates: list[str] = []
        columns: dict[str, list[float]] = {col: [] for col in _NUMERIC_COLUMNS}
        for row in reader:
            dates.append(row[header_by_lower["date"]])
            for col in _NUMERIC_COLUMNS:
                columns[col].append(float(row[header_by_lower[col]]))

    return PriceSeriesValue(
        open=SeriesValue(np.array(columns["open"], dtype=np.float64)),
        high=SeriesValue(np.array(columns["high"], dtype=np.float64)),
        low=SeriesValue(np.array(columns["low"], dtype=np.float64)),
        close=SeriesValue(np.array(columns["close"], dtype=np.float64)),
        volume=SeriesValue(np.array(columns["volume"], dtype=np.float64)),
        dates=np.array(dates, dtype="datetime64[ns]"),
    )


_PRINT_EDGE_ELEMENTS = 3


def _format_series(series: SeriesValue) -> str:
    data = series.data
    if len(data) <= 2 * _PRINT_EDGE_ELEMENTS:
        body = ", ".join(f"{x:g}" for x in data)
    else:
        head = ", ".join(f"{x:g}" for x in data[:_PRINT_EDGE_ELEMENTS])
        tail = ", ".join(f"{x:g}" for x in data[-_PRINT_EDGE_ELEMENTS:])
        body = f"{head}, ..., {tail}"
    return f"Series[{body}]"


def print_value(value: object) -> None:
    """Formats a SeriesValue compactly (first/last few elements, `...`
    if long); anything else prints directly."""
    if isinstance(value, SeriesValue):
        print(_format_series(value))
    else:
        print(value)
    return None


register("load", FunctionType(params=(STRING,), ret=PRICE_SERIES), load)
register("print", FunctionType(params=(ANY,), ret=VOID), print_value)
