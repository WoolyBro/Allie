"""Tests for crosses_above/crosses_below -- Phase 7.3."""

from __future__ import annotations

import numpy as np
import pytest

from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, SeriesValue
from allie.stdlib.series_ops import crosses_above, crosses_below


def _series(*values: float) -> SeriesValue:
    return SeriesValue(list(values))


def test_crosses_above_single_known_crossover() -> None:
    # a starts below b, crosses strictly above between index 1 and 2,
    # and stays above. The only True must be at index 2.
    a = _series(1.0, 2.0, 4.0, 5.0, 6.0)
    b = _series(3.0, 3.0, 3.0, 3.0, 3.0)
    result = crosses_above(a, b)
    assert isinstance(result, BoolSeriesValue)
    np.testing.assert_array_equal(result.data, [False, False, True, False, False])


def test_crosses_below_single_known_crossover() -> None:
    a = _series(6.0, 5.0, 2.0, 1.0, 0.0)
    b = _series(3.0, 3.0, 3.0, 3.0, 3.0)
    result = crosses_below(a, b)
    np.testing.assert_array_equal(result.data, [False, False, True, False, False])


def test_first_bar_is_never_a_crossing() -> None:
    # a is above b at index 0, but there is no prior bar to have crossed
    # from -- shift() NaN-fills index 0 and NaN comparisons are False.
    a = _series(5.0, 6.0)
    b = _series(1.0, 1.0)
    assert crosses_above(a, b).data[0] is np.False_


def test_touching_without_crossing_is_not_a_crossing() -> None:
    # a rises to exactly equal b, then falls back. `>` is strict, so
    # equality alone never fires.
    a = _series(1.0, 3.0, 1.0)
    b = _series(3.0, 3.0, 3.0)
    np.testing.assert_array_equal(crosses_above(a, b).data, [False, False, False])


def test_crossing_up_from_exactly_equal_does_fire() -> None:
    # prev_a == prev_b satisfies `shift(a,1) <= shift(b,1)`, so moving
    # from equal to strictly above is a genuine crossing.
    a = _series(3.0, 4.0)
    b = _series(3.0, 3.0)
    np.testing.assert_array_equal(crosses_above(a, b).data, [False, True])


def test_multiple_crossings_each_fire_once() -> None:
    a = _series(1.0, 4.0, 1.0, 4.0)
    b = _series(3.0, 3.0, 3.0, 3.0)
    np.testing.assert_array_equal(crosses_above(a, b).data, [False, True, False, True])
    np.testing.assert_array_equal(crosses_below(a, b).data, [False, False, True, False])


def test_crossings_are_mutually_exclusive_per_bar() -> None:
    a = _series(1.0, 4.0, 1.0, 4.0, 2.0)
    b = _series(3.0, 3.0, 3.0, 3.0, 3.0)
    up = crosses_above(a, b).data
    down = crosses_below(a, b).data
    assert not np.any(up & down)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(AllieValueError):
        crosses_above(_series(1.0, 2.0, 3.0), _series(1.0, 2.0))
    with pytest.raises(AllieValueError):
        crosses_below(_series(1.0, 2.0, 3.0), _series(1.0, 2.0))


def test_nan_padding_from_sma_does_not_produce_spurious_crossings() -> None:
    # Realistic case: SMA NaN-pads its first window-1 bars, and every
    # comparison against NaN is False -- so the warmup region must be
    # entirely free of signals rather than firing on garbage.
    from allie.stdlib.series_ops import SMA

    prices = _series(*[float(x) for x in range(1, 11)])
    fast = SMA(prices, 2)
    slow = SMA(prices, 4)
    result = crosses_above(fast, slow)
    assert not np.any(result.data[:3])  # slow's NaN warmup region


def test_registered_in_builtins() -> None:
    import allie.stdlib as stdlib

    sigs = stdlib.builtin_function_types()
    assert str(sigs["crosses_above"]) == "(Series<Float>, Series<Float>) -> Series<Bool>"
    assert str(sigs["crosses_below"]) == "(Series<Float>, Series<Float>) -> Series<Bool>"
