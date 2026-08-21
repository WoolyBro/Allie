"""Tests for allie/stdlib/stats_fns.py -- Phase 4.3."""

from __future__ import annotations

import numpy as np
import pytest

from allie.runtime.errors import AllieValueError
from allie.runtime.values import SeriesValue
from allie.stdlib.stats_fns import normal, percentile, zscore


def _series(*values: float) -> SeriesValue:
    return SeriesValue(list(values))


def test_normal_statistical_properties_within_tolerance() -> None:
    # Large N -- checking statistical properties (mean/std within
    # tolerance), not exact values, per the spec.
    result = normal(mean=5.0, std=2.0, n=200_000, seed=42).data
    assert result.mean() == pytest.approx(5.0, abs=0.05)
    assert result.std(ddof=1) == pytest.approx(2.0, abs=0.05)


def test_normal_is_deterministic_given_same_seed() -> None:
    a = normal(mean=0.0, std=1.0, n=100, seed=7).data
    b = normal(mean=0.0, std=1.0, n=100, seed=7).data
    np.testing.assert_array_equal(a, b)


def test_normal_different_seeds_differ() -> None:
    a = normal(mean=0.0, std=1.0, n=100, seed=1).data
    b = normal(mean=0.0, std=1.0, n=100, seed=2).data
    assert not np.array_equal(a, b)


def test_normal_invalid_n_raises() -> None:
    with pytest.raises(AllieValueError):
        normal(mean=0.0, std=1.0, n=0, seed=1)


def test_zscore_hand_computed() -> None:
    # mean=3, sample std=sqrt(2.5)~=1.581139 for [1,2,3,4,5]
    result = zscore(_series(1, 2, 3, 4, 5)).data
    expected_std = (2.5) ** 0.5
    np.testing.assert_allclose(result, [(x - 3) / expected_std for x in (1, 2, 3, 4, 5)])


def test_zscore_zero_std_raises() -> None:
    with pytest.raises(AllieValueError):
        zscore(_series(5, 5, 5))


def test_percentile_hand_computed() -> None:
    assert percentile(_series(1, 2, 3, 4, 5), 50.0) == pytest.approx(3.0)
    assert percentile(_series(1, 2, 3, 4), 25.0) == pytest.approx(1.75)


def test_percentile_out_of_range_raises() -> None:
    with pytest.raises(AllieValueError):
        percentile(_series(1, 2, 3), 150.0)


def test_percentile_empty_series_raises() -> None:
    with pytest.raises(AllieValueError):
        percentile(SeriesValue([]), 50.0)
