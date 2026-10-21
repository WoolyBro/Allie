"""Matrices: building them from series, reading them back, and the
linear algebra a factor model or a portfolio needs.

There is no matrix literal; a matrix is built from series, which is
where the data is anyway:

    let X = with_column(matrix(market, len(market), 1), size);  // 2 columns
    let coefficients = ols(stock, X);

`matrix(values, rows, cols)` fills row by row, so
`matrix([1.0, 2.0, 3.0, 4.0], 2, 2)` is [[1, 2], [3, 4]].
"""

from __future__ import annotations

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import MatrixValue, SeriesValue
from allie.stdlib.registry import register
from allie.typechecker.types import FLOAT, INT, FunctionType, MatrixType, SeriesType

_MATRIX_FLOAT = MatrixType(FLOAT)
_SERIES_FLOAT = SeriesType(FLOAT)


# ---- building and reading ----


def matrix(values: SeriesValue, rows: int, cols: int) -> MatrixValue:
    if rows < 1 or cols < 1:
        raise AllieValueError(f"matrix: needs at least 1 row and 1 column, got {rows}x{cols}")
    if len(values) != rows * cols:
        raise AllieValueError(
            f"matrix: {rows}x{cols} needs {rows * cols} values, got {len(values)}"
        )
    return MatrixValue(values.data.reshape(rows, cols).copy())


def with_column(m: MatrixValue, column: SeriesValue) -> MatrixValue:
    """`m` with `column` appended on the right."""
    if len(column) != m.data.shape[0]:
        raise AllieValueError(
            f"with_column: the matrix has {m.data.shape[0]} rows, the column {len(column)}"
        )
    return MatrixValue(np.column_stack((m.data, column.data)))


def identity(n: int) -> MatrixValue:
    if n < 1:
        raise AllieValueError(f"identity: n must be positive, got {n}")
    return MatrixValue(np.eye(n))


def rows(m: MatrixValue) -> int:
    return int(m.data.shape[0])


def cols(m: MatrixValue) -> int:
    return int(m.data.shape[1])


def _index(fn_name: str, i: int, size: int, what: str) -> None:
    if not 0 <= i < size:
        raise AllieValueError(f"{fn_name}: {what} {i} is out of range for {size} {what}s")


def row(m: MatrixValue, i: int) -> SeriesValue:
    _index("row", i, m.data.shape[0], "row")
    return SeriesValue(m.data[i].copy())


def column(m: MatrixValue, j: int) -> SeriesValue:
    _index("column", j, m.data.shape[1], "column")
    return SeriesValue(m.data[:, j].copy())


# ---- linear algebra ----


def transpose(m: MatrixValue) -> MatrixValue:
    return MatrixValue(m.data.T.copy())


def inverse(m: MatrixValue) -> MatrixValue:
    _require_square("inverse", m)
    if np.linalg.matrix_rank(m.data) < m.data.shape[0]:
        raise AllieValueError("inverse: the matrix is singular -- it has no inverse")
    return MatrixValue(np.linalg.inv(m.data))


def determinant(m: MatrixValue) -> float:
    _require_square("determinant", m)
    return float(np.linalg.det(m.data))


def dot(a: MatrixValue, b: MatrixValue) -> MatrixValue:
    if a.data.shape[1] != b.data.shape[0]:
        raise AllieValueError(
            f"dot: a {a.data.shape[0]}x{a.data.shape[1]} matrix can't multiply a "
            f"{b.data.shape[0]}x{b.data.shape[1]} one (columns must equal rows)"
        )
    return MatrixValue(a.data @ b.data)


def _require_square(fn_name: str, m: MatrixValue) -> None:
    r, c = m.data.shape
    if r != c:
        raise AllieValueError(f"{fn_name}: the matrix must be square, got {r}x{c}")


# ---- statistics ----


def covariance_matrix(m: MatrixValue) -> MatrixValue:
    """Sample covariance (ddof=1) between the columns, each column one
    variable -- one asset's returns, say. Rows with a missing value in
    any column are left out."""
    data = m.data[~np.isnan(m.data).any(axis=1)]
    if len(data) < 2:
        raise AllieValueError(
            f"covariance_matrix: needs at least 2 rows without a missing value, got {len(data)}"
        )
    return MatrixValue(np.atleast_2d(np.cov(data, rowvar=False, ddof=1)))


def ols(y: SeriesValue, x: MatrixValue) -> SeriesValue:
    """Least-squares coefficients for `y ~ x`, one per column of `x`. No
    intercept is added: include a column of ones for one. Rows where `y`
    or any column is missing are left out."""
    if len(y) != x.data.shape[0]:
        raise AllieValueError(f"ols: y has {len(y)} values but x has {x.data.shape[0]} rows")
    keep = ~np.isnan(y.data) & ~np.isnan(x.data).any(axis=1)
    target, design = y.data[keep], x.data[keep]
    if len(target) < design.shape[1]:
        raise AllieValueError(
            f"ols: {design.shape[1]} coefficients need at least {design.shape[1]} complete rows, "
            f"got {len(target)}"
        )
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < design.shape[1]:
        raise AllieValueError(
            "ols: the columns of x are linearly dependent, so the coefficients aren't unique"
        )
    return SeriesValue(coefficients)


register("matrix", FunctionType(params=(_SERIES_FLOAT, INT, INT), ret=_MATRIX_FLOAT), matrix)
register(
    "with_column",
    FunctionType(params=(_MATRIX_FLOAT, _SERIES_FLOAT), ret=_MATRIX_FLOAT),
    with_column,
)
register("identity", FunctionType(params=(INT,), ret=_MATRIX_FLOAT), identity)
register("rows", FunctionType(params=(_MATRIX_FLOAT,), ret=INT), rows)
register("cols", FunctionType(params=(_MATRIX_FLOAT,), ret=INT), cols)
register("row", FunctionType(params=(_MATRIX_FLOAT, INT), ret=_SERIES_FLOAT), row)
register("column", FunctionType(params=(_MATRIX_FLOAT, INT), ret=_SERIES_FLOAT), column)
register("transpose", FunctionType(params=(_MATRIX_FLOAT,), ret=_MATRIX_FLOAT), transpose)
register("inverse", FunctionType(params=(_MATRIX_FLOAT,), ret=_MATRIX_FLOAT), inverse)
register("determinant", FunctionType(params=(_MATRIX_FLOAT,), ret=FLOAT), determinant)
register("dot", FunctionType(params=(_MATRIX_FLOAT, _MATRIX_FLOAT), ret=_MATRIX_FLOAT), dot)
register(
    "covariance_matrix",
    FunctionType(params=(_MATRIX_FLOAT,), ret=_MATRIX_FLOAT),
    covariance_matrix,
)
register("ols", FunctionType(params=(_SERIES_FLOAT, _MATRIX_FLOAT), ret=_SERIES_FLOAT), ols)
