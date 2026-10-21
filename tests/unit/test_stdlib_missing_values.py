"""The stdlib's one missing-value rule (allie/stdlib/missing.py), checked
against pandas -- whose defaults the rule follows -- on series with gaps
in the warm-up, the middle and the end."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import allie.stdlib as stdlib
from allie.runtime.errors import AllieValueError
from allie.runtime.values import SeriesValue
from allie.stdlib.registry import _as_allie_function

_GAPPY = [np.nan, 4.0, 2.0, np.nan, 7.0, 5.0, 6.0, 1.0, np.nan, 3.0, 8.0, 9.0, 2.0]
_OTHER = [1.0, 2.0, np.nan, 4.0, 3.0, 6.0, 5.0, 2.0, 7.0, np.nan, 9.0, 8.0, 3.0]


def _call(name: str, *args: object) -> object:
    """Through the registry, as Allie code calls it."""
    _, fn = stdlib.BUILTINS[name]
    return fn(*args)


def _s(values: list[float]) -> SeriesValue:
    return SeriesValue(np.array(values, dtype=np.float64))


# ---- reductions skip missing bars ----


@pytest.mark.parametrize(
    ("name", "reference"),
    [
        ("mean", pd.Series.mean),
        ("median", pd.Series.median),
        ("std", pd.Series.std),
        ("variance", pd.Series.var),
        ("sum", pd.Series.sum),
        ("min", pd.Series.min),
        ("max", pd.Series.max),
    ],
)
def test_reductions_skip_missing_bars_like_pandas(name: str, reference: object) -> None:
    expected = reference(pd.Series(_GAPPY))  # type: ignore[operator]
    assert _call(name, _s(_GAPPY)) == pytest.approx(expected)


def test_percentile_skips_missing_bars() -> None:
    expected = pd.Series(_GAPPY).quantile(0.9)
    assert _call("percentile", _s(_GAPPY), 90.0) == pytest.approx(expected)


def test_a_reduction_of_an_indicator_uses_the_bars_it_has() -> None:
    prices = _s([float(x) for x in range(1, 31)])
    sma = _call("SMA", prices, 20)
    assert _call("mean", sma) == pytest.approx(np.mean(np.arange(1, 31)[19:] - 9.5))


@pytest.mark.parametrize("name", ["mean", "median", "sum", "min", "max", "std"])
def test_nothing_left_to_reduce_is_an_error_not_a_nan(name: str) -> None:
    with pytest.raises(AllieValueError, match=f"{name}: no values that aren't missing -- all 3"):
        _call(name, _s([np.nan, np.nan, np.nan]))


def test_too_few_values_for_a_sample_statistic() -> None:
    with pytest.raises(
        AllieValueError, match="std: needs at least 2 values that aren't missing, got 1"
    ):
        _call("std", _s([np.nan, 3.0, np.nan]))


def test_paired_statistics_use_bars_where_both_have_a_value() -> None:
    a, b = pd.Series(_GAPPY), pd.Series(_OTHER)
    assert _call("covariance", _s(_GAPPY), _s(_OTHER)) == pytest.approx(a.cov(b))
    assert _call("correlation", _s(_GAPPY), _s(_OTHER)) == pytest.approx(a.corr(b))


def test_correlation_with_a_constant_series_is_an_error() -> None:
    with pytest.raises(AllieValueError, match="correlation: undefined when a series doesn't vary"):
        _call("correlation", _s([1.0, 1.0, 1.0]), _s([1.0, 2.0, 3.0]))


def test_inf_is_a_value_not_a_missing_bar() -> None:
    assert _call("max", _s([1.0, math.inf, np.nan])) == math.inf


# ---- windows are missing only where they hold a missing bar ----


@pytest.mark.parametrize(
    ("name", "reference"),
    [
        ("SMA", lambda s, w: s.rolling(w).mean()),
        ("rolling_mean", lambda s, w: s.rolling(w).mean()),
        ("rolling_sum", lambda s, w: s.rolling(w).sum()),
        ("rolling_std", lambda s, w: s.rolling(w).std()),
        ("rolling_max", lambda s, w: s.rolling(w).max()),
        ("rolling_min", lambda s, w: s.rolling(w).min()),
    ],
)
def test_windows_match_pandas_rolling_including_gaps(name: str, reference: object) -> None:
    for window in (2, 3):
        expected = reference(pd.Series(_GAPPY), window).to_numpy()  # type: ignore[operator]
        result = _call(name, _s(_GAPPY), window).data  # type: ignore[attr-defined]
        np.testing.assert_allclose(result, expected, equal_nan=True)


def test_one_missing_bar_no_longer_erases_the_rest_of_an_sma() -> None:
    # A running sum carried the NaN into every later window.
    result = _call("SMA", _s([1.0, np.nan, 3.0, 4.0, 5.0, 6.0]), 2).data  # type: ignore[attr-defined]
    np.testing.assert_allclose(result, [np.nan, np.nan, np.nan, 3.5, 4.5, 5.5], equal_nan=True)


def test_rolling_zscore_and_corr_match_pandas() -> None:
    s, t = pd.Series(_GAPPY), pd.Series(_OTHER)
    expected_z = ((s - s.rolling(4).mean()) / s.rolling(4).std()).to_numpy()
    result_z = _call("rolling_zscore", _s(_GAPPY), 4).data  # type: ignore[attr-defined]
    np.testing.assert_allclose(result_z, expected_z, equal_nan=True)
    expected_r = s.rolling(4).corr(t).to_numpy()
    result_r = _call("rolling_corr", _s(_GAPPY), _s(_OTHER), 4).data  # type: ignore[attr-defined]
    np.testing.assert_allclose(result_r, expected_r, equal_nan=True)


def test_a_window_without_variation_has_no_zscore() -> None:
    result = _call("rolling_zscore", _s([2.0, 2.0, 2.0, 3.0]), 3).data  # type: ignore[attr-defined]
    assert np.isnan(result[2]) and not np.isnan(result[3])


# ---- recurrences run over the bars present ----


def test_ema_skips_missing_bars_like_pandas_ignore_na() -> None:
    expected = pd.Series(_GAPPY).ewm(span=3, adjust=False, ignore_na=True).mean().to_numpy()
    result = _call("EMA", _s(_GAPPY), 3).data  # type: ignore[attr-defined]
    present = ~np.isnan(np.array(_GAPPY))
    np.testing.assert_allclose(result[present], expected[present])
    assert np.isnan(result[~present]).all()  # a missing bar has no value


def test_ema_of_a_warming_up_indicator_starts_with_it() -> None:
    sma = _call("SMA", _s([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    ema = _call("EMA", sma, 2).data  # type: ignore[attr-defined]
    assert np.isnan(ema[:2]).all()
    assert ema[2] == 2.0  # seeded with the SMA's first value


def test_cumsum_matches_pandas() -> None:
    result = _call("cumsum", _s(_GAPPY)).data  # type: ignore[attr-defined]
    np.testing.assert_allclose(result, pd.Series(_GAPPY).cumsum().to_numpy(), equal_nan=True)


def test_empty_series_stay_empty() -> None:
    for name, args in (("EMA", (3,)), ("SMA", (3,)), ("cumsum", ()), ("RSI", (14,))):
        assert len(_call(name, _s([]), *args)) == 0, name  # type: ignore[arg-type]


# ---- positional functions, and IEEE ----


def test_zscore_keeps_missing_bars_missing() -> None:
    result = _call("zscore", _s(_GAPPY)).data  # type: ignore[attr-defined]
    s = pd.Series(_GAPPY)
    np.testing.assert_allclose(result, ((s - s.mean()) / s.std()).to_numpy(), equal_nan=True)


def test_pct_change_across_a_zero_is_ieee_and_silent(recwarn: pytest.WarningsRecorder) -> None:
    result = _call("pct_change", _s([1.0, 0.0, 2.0, 0.0, 0.0])).data  # type: ignore[attr-defined]
    np.testing.assert_array_equal(result, [np.nan, -1.0, np.inf, -1.0, np.nan])
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# ---- builtins fail as Allie errors ----


def test_a_library_error_inside_a_builtin_becomes_an_allie_error() -> None:
    def broken(x: float) -> float:
        raise ValueError("math domain error")

    with pytest.raises(AllieValueError, match="AllieValueError: demo: math domain error$"):
        _as_allie_function("demo", broken)(1.0)


@pytest.mark.parametrize(
    ("name", "args", "message"),
    [
        ("normal", (0.0, -1.0, 3, 1), "normal: needs a finite mean and a std of at least 0"),
        ("normal", (0.0, 1.0, 3, -5), "normal: the seed must be 0 or more"),
        ("EMA", (SeriesValue([1.0]), 0), "EMA: span must be positive"),
        ("rolling_std", (SeriesValue([1.0]), 1), "rolling_std: window must be at least 2"),
    ],
)
def test_bad_arguments_are_explained(name: str, args: tuple[object, ...], message: str) -> None:
    with pytest.raises(AllieValueError, match=message):
        _call(name, *args)
