"""Math, returns-and-risk and matrix builtins (allie/stdlib/numeric.py,
risk.py, matrix_ops.py): hand-computed values, numpy as the reference,
consistency with the backtest engine's own metrics, and the same
program run on every backend."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import allie.stdlib as stdlib
from allie.cli import run_pipeline
from allie.runtime.errors import AllieValueError
from allie.runtime.values import MatrixValue, SeriesValue
from tests.unit.test_strategy_dsl import _PRELUDE, _output


def _call(name: str, *args: object) -> object:
    _, fn = stdlib.BUILTINS[name]
    return fn(*args)


def _s(values: list[float]) -> SeriesValue:
    return SeriesValue(np.array(values, dtype=np.float64))


def _m(rows: list[list[float]]) -> MatrixValue:
    return MatrixValue(np.array(rows, dtype=np.float64))


# ---- numbers and series ----


def test_math_applies_to_numbers_and_series_alike() -> None:
    assert _call("abs", -3) == 3 and isinstance(_call("abs", -3), int)
    assert _call("sqrt", 252) == pytest.approx(math.sqrt(252))
    assert _call("exp", 0.0) == 1.0
    np.testing.assert_allclose(_call("log", _s([1.0, math.e])).data, [0.0, 1.0])  # type: ignore[attr-defined]
    np.testing.assert_array_equal(_call("abs", _s([-1.0, np.nan])).data, [1.0, np.nan])  # type: ignore[attr-defined]


def test_a_series_follows_ieee_but_a_number_outside_the_domain_is_an_error() -> None:
    np.testing.assert_array_equal(
        _call("log", _s([0.0, -1.0])).data,  # type: ignore[attr-defined]
        [-math.inf, np.nan],
    )
    with pytest.raises(AllieValueError, match="log: needs a positive number, got 0.0"):
        _call("log", 0.0)
    with pytest.raises(AllieValueError, match="sqrt: needs a number of at least 0, got -4"):
        _call("sqrt", -4)
    assert _call("exp", 1000.0) == math.inf  # overflow is inf, as Float arithmetic is


@pytest.mark.parametrize(
    ("name", "x", "expected"),
    [
        ("floor", 2.7, 2),
        ("floor", -2.1, -3),
        ("ceil", 2.1, 3),
        ("round", 2.5, 2),  # ties to even
        ("round", 3.5, 4),
    ],
)
def test_float_to_int(name: str, x: float, expected: int) -> None:
    result = _call(name, x)
    assert result == expected and isinstance(result, int)


@pytest.mark.parametrize("x", [math.nan, math.inf, 1e300])
def test_a_float_with_no_int_value_is_an_error(x: float) -> None:
    with pytest.raises(AllieValueError, match="floor: "):
        _call("floor", x)


# ---- returns and risk ----


def test_returns_and_drawdown_hand_computed() -> None:
    prices = _s([100.0, 110.0, 99.0, 108.9])
    np.testing.assert_allclose(
        _call("log_returns", prices).data,  # type: ignore[attr-defined]
        [np.nan, math.log(1.1), math.log(0.9), math.log(1.1)],
        equal_nan=True,
    )
    returns = _call("pct_change", prices)
    np.testing.assert_allclose(
        _call("cumulative_returns", returns).data,  # type: ignore[attr-defined]
        [0.0, 0.1, -0.01, 0.089],
    )
    np.testing.assert_allclose(
        _call("drawdown", prices).data,  # type: ignore[attr-defined]
        [0.0, 0.0, 0.1, 0.01],
    )
    assert _call("max_drawdown", prices) == pytest.approx(0.1)


def test_a_missing_bar_does_not_reset_the_high() -> None:
    result = _call("drawdown", _s([100.0, np.nan, 90.0])).data  # type: ignore[attr-defined]
    np.testing.assert_allclose(result, [0.0, np.nan, 0.1], equal_nan=True)


def test_tail_risk_hand_computed() -> None:
    returns = _s([np.nan] + [x / 100 for x in range(-10, 10)])  # -10% .. +9%
    values = np.arange(-10, 10) / 100
    var = -np.percentile(values, 10)
    assert _call("value_at_risk", returns, 0.90) == pytest.approx(var)
    assert _call("expected_shortfall", returns, 0.90) == pytest.approx(
        -values[values <= -var].mean()
    )
    with pytest.raises(AllieValueError, match="confidence is a fraction between 0 and 1"):
        _call("value_at_risk", returns, 95.0)


def test_beta_is_covariance_over_market_variance() -> None:
    market = _s([0.01, -0.02, 0.03, np.nan, 0.01])
    asset = _s([0.02, -0.03, 0.05, 0.04, 0.01])
    m, a = np.array([0.01, -0.02, 0.03, 0.01]), np.array([0.02, -0.03, 0.05, 0.01])
    assert _call("beta", asset, market) == pytest.approx(np.cov(a, m)[0, 1] / np.var(m, ddof=1))
    assert _call("beta", market, market) == pytest.approx(1.0)


def test_risk_statistics_agree_with_the_backtest_engine(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = _output(
        tmp_path,
        "let fast: Series[Float] = SMA(c, 5);\n"
        "let slow: Series[Float] = SMA(c, 20);\n"
        "strategy X {\n"
        "    when fast crosses_above slow: buy();\n"
        "    when fast crosses_below slow: sell();\n"
        "}\n"
        "let r = backtest X on AAPL from 0 to len(c);\n"
        "print(sortino(r.returns, 252) == r.sortino);\n"
        "print(volatility(r.returns, 252) == r.volatility);\n"
        "print(max_drawdown(r.equity) == r.max_drawdown);\n",
        capsys,
    )
    assert lines == ["True", "True", "True"]


# ---- matrices ----


def test_building_and_reading_a_matrix() -> None:
    m = _call("matrix", _s([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 2, 3)
    assert isinstance(m, MatrixValue)
    np.testing.assert_array_equal(m.data, [[1, 2, 3], [4, 5, 6]])
    wider = _call("with_column", m, _s([7.0, 8.0]))
    assert (_call("rows", wider), _call("cols", wider)) == (2, 4)
    np.testing.assert_array_equal(_call("row", wider, 1).data, [4, 5, 6, 8])  # type: ignore[attr-defined]
    np.testing.assert_array_equal(_call("column", wider, 3).data, [7, 8])  # type: ignore[attr-defined]
    np.testing.assert_array_equal(_call("identity", 2).data, np.eye(2))  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("name", "args", "message"),
    [
        ("matrix", (_s([1.0, 2.0, 3.0]), 2, 2), "matrix: 2x2 needs 4 values, got 3"),
        ("with_column", (_m([[1.0], [2.0]]), _s([1.0])), "the matrix has 2 rows, the column 1"),
        ("column", (_m([[1.0, 2.0]]), 2), "column: column 2 is out of range for 2 columns"),
        ("inverse", (_m([[1.0, 2.0], [2.0, 4.0]]),), "inverse: the matrix is singular"),
        ("determinant", (_m([[1.0, 2.0]]),), "the matrix must be square, got 1x2"),
        ("dot", (_m([[1.0, 2.0]]), _m([[1.0, 2.0]])), "a 1x2 matrix can't multiply a 1x2 one"),
    ],
)
def test_matrix_mistakes_are_explained(name: str, args: tuple[object, ...], message: str) -> None:
    with pytest.raises(AllieValueError, match=message):
        _call(name, *args)


def test_covariance_matrix_drops_incomplete_rows() -> None:
    data = np.array([[1.0, 2.0], [2.0, np.nan], [3.0, 5.0], [4.0, 4.0]])
    result = _call("covariance_matrix", MatrixValue(data)).data  # type: ignore[attr-defined]
    np.testing.assert_allclose(result, np.cov(data[[0, 2, 3]], rowvar=False))


def test_ols_recovers_known_coefficients() -> None:
    rng = np.random.default_rng(7)
    x1, x2 = rng.normal(size=50), rng.normal(size=50)
    y = 2.0 + 0.5 * x1 - 1.5 * x2
    y[10] = np.nan  # an incomplete row is left out
    design = MatrixValue(np.column_stack((np.ones(50), x1, x2)))
    coefficients = _call("ols", SeriesValue(y), design).data  # type: ignore[attr-defined]
    np.testing.assert_allclose(coefficients, [2.0, 0.5, -1.5])
    collinear = MatrixValue(np.column_stack((x1, 2 * x1)))
    with pytest.raises(AllieValueError, match="ols: the columns of x are linearly dependent"):
        _call("ols", SeriesValue(y), collinear)


# ---- from source, on every backend ----


def test_the_quant_toolkit_agrees_everywhere(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = _output(
        tmp_path,
        "let r: Series[Float] = pct_change(c);\n"
        "print(abs(-3));\n"
        "print(sqrt(252));\n"
        "print(log(c)[0] == log(c[0]));\n"
        "print(floor(len(c) * 0.7));\n"
        "print(mean(SMA(c, 20)) > 0.0);\n"  # skips the warm-up instead of being NaN
        "print(value_at_risk(r, 0.95) > 0.0);\n"
        "print(abs(beta(r, r) - 1.0) < 0.000000001);\n"
        "let X = with_column(matrix(AAPL.open, len(c), 1), AAPL.high);\n"
        "print(cols(X));\n"
        "print(len(ols(c, X)));\n"
        "print(matrix([1.0, 2.0, 3.0, 4.0], 2, 2));\n"
        "print(AAPL);\n",
        capsys,
    )
    assert lines == [
        "3",
        str(math.sqrt(252)),
        "True",
        "42",
        "True",
        "True",
        "True",
        "2",
        "2",
        "Matrix[2x2: [1, 2], [3, 4]]",
        "PriceSeries[60 bars, 2024-01-02 to 2024-03-25]",
    ]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("print(abs(true));", "argument 1 of 'abs' must be Int|Float|Series<Float>, got Bool"),
        ("let x: Int = sqrt(4);", "cannot assign Float to Int"),
        ("let s: Series[Int] = [1, 2]; let t: Float = log(s);", "cannot assign Series<Float>"),
        ("let n: Int = floor(2);", None),  # an Int argument widens to Float
    ],
)
def test_math_types(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str, message: str | None
) -> None:
    path = tmp_path / "p.qs"
    path.write_text(_PRELUDE + source + "\n", encoding="utf-8")
    code = run_pipeline(path, backend="tree")
    err = capsys.readouterr().err
    if message is None:
        assert (code, err) == (0, "")
    else:
        assert code == 1 and message in err
