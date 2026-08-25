"""Matrix stdlib functions -- Phase 4.3."""

from __future__ import annotations

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import MatrixValue
from allie.stdlib.registry import register
from allie.typechecker.types import FLOAT, FunctionType, MatrixType

_MATRIX_FLOAT = MatrixType(FLOAT)


def transpose(matrix: MatrixValue) -> MatrixValue:
    return MatrixValue(matrix.data.T.copy())


def inverse(matrix: MatrixValue) -> MatrixValue:
    try:
        return MatrixValue(np.linalg.inv(matrix.data))
    except np.linalg.LinAlgError as error:
        raise AllieValueError(f"inverse: {error}") from error


def determinant(matrix: MatrixValue) -> float:
    if matrix.data.shape[0] != matrix.data.shape[1]:
        raise AllieValueError(f"determinant: matrix must be square, got {matrix.data.shape}")
    return float(np.linalg.det(matrix.data))


def dot(a: MatrixValue, b: MatrixValue) -> MatrixValue:
    if a.data.shape[1] != b.data.shape[0]:
        raise AllieValueError(f"dot: dimension mismatch, {a.data.shape} vs {b.data.shape}")
    return MatrixValue(a.data @ b.data)


register("transpose", FunctionType(params=(_MATRIX_FLOAT,), ret=_MATRIX_FLOAT), transpose)
register("inverse", FunctionType(params=(_MATRIX_FLOAT,), ret=_MATRIX_FLOAT), inverse)
register("determinant", FunctionType(params=(_MATRIX_FLOAT,), ret=FLOAT), determinant)
register("dot", FunctionType(params=(_MATRIX_FLOAT, _MATRIX_FLOAT), ret=_MATRIX_FLOAT), dot)
