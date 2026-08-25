"""Tests for allie/stdlib/matrix_ops.py -- Phase 4.3. Known matrices with
hand-computed inverse/determinant."""

from __future__ import annotations

import numpy as np
import pytest

from allie.runtime.errors import AllieValueError
from allie.runtime.values import MatrixValue
from allie.stdlib.matrix_ops import determinant, dot, inverse, transpose


def _matrix(rows: list[list[float]]) -> MatrixValue:
    return MatrixValue(np.array(rows, dtype=np.float64))


def test_transpose_2x3() -> None:
    result = transpose(_matrix([[1, 2, 3], [4, 5, 6]])).data
    np.testing.assert_allclose(result, [[1, 4], [2, 5], [3, 6]])


def test_determinant_2x2_hand_computed() -> None:
    # det([[4, 7], [2, 6]]) = 4*6 - 7*2 = 10
    assert determinant(_matrix([[4, 7], [2, 6]])) == pytest.approx(10.0)


def test_determinant_3x3_identity() -> None:
    assert determinant(_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])) == pytest.approx(1.0)


def test_determinant_non_square_raises() -> None:
    with pytest.raises(AllieValueError):
        determinant(_matrix([[1, 2, 3], [4, 5, 6]]))


def test_inverse_2x2_hand_computed() -> None:
    # inverse([[4, 7], [2, 6]]) = (1/10) * [[6, -7], [-2, 4]]
    result = inverse(_matrix([[4, 7], [2, 6]])).data
    np.testing.assert_allclose(result, [[0.6, -0.7], [-0.2, 0.4]])


def test_inverse_3x3_identity_is_identity() -> None:
    result = inverse(_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).data
    np.testing.assert_allclose(result, np.eye(3))


def test_inverse_singular_matrix_raises_allie_error_not_numpy_error() -> None:
    with pytest.raises(AllieValueError):
        inverse(_matrix([[1, 2], [2, 4]]))  # rows are linearly dependent -> singular


def test_dot_hand_computed() -> None:
    result = dot(_matrix([[1, 2], [3, 4]]), _matrix([[5, 6], [7, 8]])).data
    np.testing.assert_allclose(result, [[19, 22], [43, 50]])


def test_dot_dimension_mismatch_raises_clear_error() -> None:
    with pytest.raises(AllieValueError):
        dot(_matrix([[1, 2, 3]]), _matrix([[1, 2]]))
