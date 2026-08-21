"""Tests for allie/stdlib/math_fns.py -- Phase 4.3. Hand-computed
expected values, cross-checked against pandas where noted (pandas is a
test-only dependency here, not a runtime one)."""

from __future__ import annotations

import pandas as pd
import pytest

from allie.runtime.errors import AllieValueError
from allie.runtime.values import SeriesValue
from allie.stdlib.math_fns import correlation, covariance, mean, median, std, variance


def _series(*values: float) -> SeriesValue:
    return SeriesValue(list(values))


def test_mean() -> None:
    assert mean(_series(1, 2, 3, 4, 5)) == 3.0


def test_median() -> None:
    assert median(_series(1, 2, 3, 4, 5)) == 3.0
    assert median(_series(1, 2, 3, 4)) == 2.5


def test_std_and_variance_hand_computed() -> None:
    # 2,4,4,4,5,5,7,9: mean 5, sum of squared deviations 32.
    # sample variance (ddof=1) = 32/7, sample std = sqrt(32/7).
    data = _series(2, 4, 4, 4, 5, 5, 7, 9)
    assert variance(data) == pytest.approx(32 / 7)
    assert std(data) == pytest.approx((32 / 7) ** 0.5)


def test_std_matches_pandas_sample_std() -> None:
    values = [2, 4, 4, 4, 5, 5, 7, 9]
    assert std(_series(*values)) == pytest.approx(pd.Series(values).std())


def test_covariance_and_correlation_hand_computed() -> None:
    a = _series(1, 2, 3, 4, 5)
    b = _series(2, 4, 6, 8, 10)  # b = 2a -- perfectly correlated
    assert covariance(a, b) == pytest.approx(5.0)
    assert correlation(a, b) == pytest.approx(1.0)


def test_correlation_matches_pandas() -> None:
    a_values = [1, 3, 2, 5, 4]
    b_values = [2, 1, 4, 3, 5]
    expected = pd.Series(a_values).corr(pd.Series(b_values))
    assert correlation(_series(*a_values), _series(*b_values)) == pytest.approx(expected)


def test_empty_series_raises() -> None:
    empty = SeriesValue([])
    with pytest.raises(AllieValueError):
        mean(empty)
    with pytest.raises(AllieValueError):
        median(empty)
    with pytest.raises(AllieValueError):
        std(empty)
    with pytest.raises(AllieValueError):
        variance(empty)


def test_single_element_std_and_variance_raise() -> None:
    single = _series(1.0)
    with pytest.raises(AllieValueError):
        std(single)
    with pytest.raises(AllieValueError):
        variance(single)


def test_mismatched_length_covariance_and_correlation_raise() -> None:
    a = _series(1, 2, 3)
    b = _series(1, 2)
    with pytest.raises(AllieValueError):
        covariance(a, b)
    with pytest.raises(AllieValueError):
        correlation(a, b)
