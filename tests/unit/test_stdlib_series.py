"""Tests for allie/stdlib/series_ops.py. Hand-computed
expected values, cross-checked against pandas where noted."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from allie.runtime.errors import AllieValueError
from allie.runtime.values import SeriesValue
from allie.stdlib.series_ops import EMA, SMA, diff, pct_change, rolling_mean, rolling_std, shift


def _series(*values: float) -> SeriesValue:
    return SeriesValue(list(values))


def test_sma_hand_computed() -> None:
    result = SMA(_series(1, 2, 3, 4, 5), 3).data
    # first window-1=2 points NaN, then (1+2+3)/3, (2+3+4)/3, (3+4+5)/3
    assert np.isnan(result[0]) and np.isnan(result[1])
    np.testing.assert_allclose(result[2:], [2.0, 3.0, 4.0])


def test_sma_matches_pandas_rolling_mean() -> None:
    values = [1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0]
    expected = pd.Series(values).rolling(3).mean().to_numpy()
    result = SMA(_series(*values), 3).data
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_sma_invalid_window_raises() -> None:
    with pytest.raises(AllieValueError):
        SMA(_series(1, 2, 3), 0)


def test_ema_hand_computed() -> None:
    # span=2 -> alpha=2/3. result0=1; result1=(2/3)*2+(1/3)*1=5/3;
    # result2=(2/3)*3+(1/3)*(5/3)=23/9.
    result = EMA(_series(1, 2, 3), 2).data
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(5 / 3)
    assert result[2] == pytest.approx(23 / 9)


def test_ema_matches_pandas_ewm() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    span = 3
    expected = pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()
    result = EMA(_series(*values), span).data
    np.testing.assert_allclose(result, expected)


def test_rolling_mean_matches_sma() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    np.testing.assert_allclose(
        rolling_mean(_series(*values), 2).data,
        SMA(_series(*values), 2).data,
        equal_nan=True,
    )


def test_rolling_std_hand_computed() -> None:
    result = rolling_std(_series(1, 2, 3, 4, 5), 3).data
    assert np.isnan(result[0]) and np.isnan(result[1])
    # std([1,2,3], ddof=1) = 1.0; std([2,3,4], ddof=1) = 1.0; std([3,4,5]) = 1.0
    np.testing.assert_allclose(result[2:], [1.0, 1.0, 1.0])


def test_shift_positive_and_negative() -> None:
    data = _series(1, 2, 3, 4, 5)
    forward = shift(data, 2).data
    assert np.isnan(forward[0]) and np.isnan(forward[1])
    np.testing.assert_allclose(forward[2:], [1.0, 2.0, 3.0])

    backward = shift(data, -1).data
    np.testing.assert_allclose(backward[:4], [2.0, 3.0, 4.0, 5.0])
    assert np.isnan(backward[4])


def test_diff_hand_computed() -> None:
    result = diff(_series(1, 3, 6, 10)).data
    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1:], [2.0, 3.0, 4.0])


def test_pct_change_hand_computed() -> None:
    result = pct_change(_series(100, 110, 99)).data
    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1:], [0.10, -0.10])


def test_pct_change_matches_pandas() -> None:
    values = [100.0, 105.0, 95.0, 120.0]
    expected = pd.Series(values).pct_change().to_numpy()
    result = pct_change(_series(*values)).data
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_append_returns_a_new_series_one_longer() -> None:
    from allie.stdlib.series_ops import append

    original = _series(1.0, 2.0)
    longer = append(original, 3.5)
    assert list(longer.data) == [1.0, 2.0, 3.5]
    assert list(original.data) == [1.0, 2.0]  # series are immutable
    assert list(append(SeriesValue([]), 7.0).data) == [7.0]
