"""`load` and `print`.

`load` uses the standard-library `csv` module and numpy, not pandas, for
the same reason SeriesValue/MatrixValue are numpy-backed (see
allie/runtime/values's module docstring): pandas stays a test-only
dependency.

A price file is where a backtest's silent errors start, so `load` is
strict about what it can't interpret and explicit about what it does:

- Columns `date, open, high, low, close, volume` are found by name, in
  any case and order; other columns are ignored. A UTF-8 byte-order mark
  (as Excel writes) is fine.
- An empty cell, or `NA`, `N/A`, `NaN`, `null`, `-`, is a missing value:
  NaN, which the whole stdlib reads as "no value at this bar"
  (allie/stdlib/missing.py). Anything else that isn't a number is an
  error naming its line and column -- never a guess.
- Dates are ISO 8601, `YYYY-MM-DD` with an optional time
  (`2024-01-02 09:30`); `/` may separate the date's parts. A date in
  another form is an error: `01/02/2024` could be either of two days.
- Rows must be in time order. A file that is entirely newest-first, as
  many exports are, is reversed -- read the other way round, time would
  run backwards through every indicator and backtest. Any other
  disorder, or a repeated date, is an error naming the line.
"""

from __future__ import annotations

import csv as csv_module
import itertools
import re
from pathlib import Path

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, MatrixValue, PriceSeriesValue, SeriesValue
from allie.stdlib.registry import register
from allie.typechecker.types import ANY, PRICE_SERIES, STRING, VOID, FunctionType

_REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")
_NUMERIC_COLUMNS = tuple(col for col in _REQUIRED_COLUMNS if col != "date")
_MISSING = frozenset({"", "na", "n/a", "nan", "null", "-"})
_ZERO_TIME = np.timedelta64(0, "ns")
_DATE_PATTERN = r"\d{4}[-/]\d{2}[-/]\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
_ISO_DATE = re.compile(_DATE_PATTERN)
# A whole date column, one per line: checked in one pass, and only when
# it fails is the column checked row by row to find the bad one.
_ISO_DATES = re.compile(rf"(?:[ \t]*{_DATE_PATTERN}[ \t]*\n)*[ \t]*{_DATE_PATTERN}[ \t]*")


def load(path: str) -> PriceSeriesValue:
    """Loads an OHLCV CSV; see the module docstring for the rules."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise AllieValueError(f"load: no such file: {path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv_module.reader(handle)
        header = next(reader, None)
        if header is None:
            raise AllieValueError(f"load: {path} is empty -- it has no header row")
        position = {name.strip().lower(): i for i, name in enumerate(header)}
        missing = [col for col in _REQUIRED_COLUMNS if col not in position]
        if missing:
            raise AllieValueError(
                f"load: {path} is missing required column(s): {', '.join(missing)}"
            )

        rows: list[list[str]] = []
        lines: list[int] = []
        for row in reader:
            if "".join(row).strip():  # else a blank line
                rows.append(row)
                lines.append(reader.line_num)

    if not rows:
        raise AllieValueError(f"load: {path} has a header but no rows")
    # Rows to columns in C; a short row's missing cells read as empty.
    columns = list(itertools.zip_longest(*rows, fillvalue=""))
    dates = _dates(path, columns[position["date"]], lines)
    order = _time_order(path, dates, lines)
    series = {
        col: SeriesValue(_column(path, col, columns[position[col]], lines)[order])
        for col in _NUMERIC_COLUMNS
    }
    return PriceSeriesValue(
        open=series["open"],
        high=series["high"],
        low=series["low"],
        close=series["close"],
        volume=series["volume"],
        dates=dates[order],
    )


def _dates(path: str, cells: tuple[str, ...], lines: list[int]) -> np.ndarray:
    text = "\n".join(cells)
    if _ISO_DATES.fullmatch(text):
        cleaned = [cell.strip() for cell in cells]
        if "/" in text:
            cleaned = [cell.replace("/", "-") for cell in cleaned]
        try:
            return np.array(cleaned, dtype="datetime64[ns]")
        except ValueError:
            pass  # a well-formed but impossible date: found below
    return np.array(
        [_date(path, line, cell.strip()) for line, cell in zip(lines, cells, strict=True)],
        dtype="datetime64[ns]",
    )


def _column(path: str, name: str, cells: tuple[str, ...], lines: list[int]) -> np.ndarray:
    """A whole column at once when every cell is a number -- numpy parses
    them in C, ~5x faster than one `float()` per cell -- else cell by
    cell, which knows the missing-value spellings and names a bad cell."""
    try:
        return np.array(cells, dtype=np.float64)
    except ValueError:
        return np.array(
            [
                _number(path, line, name, cell.strip())
                for line, cell in zip(lines, cells, strict=True)
            ],
            dtype=np.float64,
        )


def _number(path: str, line: int, column: str, text: str) -> float:
    if text.lower() in _MISSING:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        raise AllieValueError(
            f"load: {path} line {line}: {column} is '{text}', which isn't a number"
        ) from None


def _date(path: str, line: int, text: str) -> np.datetime64:
    if not _ISO_DATE.fullmatch(text):
        shown = f"'{text}'" if text else "empty"
        raise AllieValueError(
            f"load: {path} line {line}: the date is {shown} -- dates must be YYYY-MM-DD, "
            "optionally with a time"
        )
    try:
        return np.datetime64(text.replace("/", "-"), "ns")
    except ValueError:
        raise AllieValueError(f"load: {path} line {line}: '{text}' isn't a real date") from None


def _time_order(path: str, dates: np.ndarray, lines: list[int]) -> np.ndarray:
    """The row order that puts the file in time order: as it is, or
    reversed for a newest-first file. Anything else is an error."""
    steps = np.diff(dates)
    if np.all(steps > _ZERO_TIME):
        return np.arange(len(dates))
    if np.all(steps < _ZERO_TIME):
        return np.arange(len(dates))[::-1]
    i = int(np.flatnonzero(steps <= _ZERO_TIME)[0])
    earlier, later = (np.datetime_as_string(d, unit="D") for d in dates[[i, i + 1]])
    if steps[i] == _ZERO_TIME:
        raise AllieValueError(
            f"load: {path} line {lines[i + 1]} repeats the date of line {lines[i]} "
            f"({later}) -- each bar needs its own date"
        )
    raise AllieValueError(
        f"load: {path} line {lines[i + 1]} ({later}) comes before line {lines[i]} "
        f"({earlier}) in time -- rows must be in time order, oldest or newest first"
    )


# ---- print ----

_PRINT_EDGE_ELEMENTS = 3


def _format_series(series: SeriesValue | BoolSeriesValue) -> str:
    data = series.data
    show = str if isinstance(series, BoolSeriesValue) else (lambda x: f"{x:g}")
    if len(data) <= 2 * _PRINT_EDGE_ELEMENTS:
        body = ", ".join(show(x) for x in data)
    else:
        head = ", ".join(show(x) for x in data[:_PRINT_EDGE_ELEMENTS])
        tail = ", ".join(show(x) for x in data[-_PRINT_EDGE_ELEMENTS:])
        body = f"{head}, ..., {tail}"
    return f"Series[{body}]"


def _format_matrix(matrix: MatrixValue) -> str:
    rows, cols = matrix.data.shape
    shown = matrix.data if rows <= 2 * _PRINT_EDGE_ELEMENTS else None
    if shown is None:
        head = [_format_row(r) for r in matrix.data[:_PRINT_EDGE_ELEMENTS]]
        tail = [_format_row(r) for r in matrix.data[-_PRINT_EDGE_ELEMENTS:]]
        body = ", ".join([*head, "...", *tail])
    else:
        body = ", ".join(_format_row(r) for r in shown)
    return f"Matrix[{rows}x{cols}: {body}]"


def _format_row(row: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:g}" for x in row) + "]"


def _format_prices(prices: PriceSeriesValue) -> str:
    n = len(prices.close)
    if len(prices.dates) == n and n:
        first, last = np.datetime_as_string(prices.dates[[0, -1]], unit="D")
        return f"PriceSeries[{n} bar{'' if n == 1 else 's'}, {first} to {last}]"
    return f"PriceSeries[{n} bar{'' if n == 1 else 's'}]"


def print_value(value: object) -> None:
    """Prints a value: series and matrices compactly (the first and last
    few elements, `...` between), a PriceSeries as its size and dates,
    anything else as itself."""
    if isinstance(value, (SeriesValue, BoolSeriesValue)):
        print(_format_series(value))
    elif isinstance(value, MatrixValue):
        print(_format_matrix(value))
    elif isinstance(value, PriceSeriesValue):
        print(_format_prices(value))
    else:
        print(value)
    return None


register("load", FunctionType(params=(STRING,), ret=PRICE_SERIES), load)
register("print", FunctionType(params=(ANY,), ret=VOID), print_value)
