"""Returns and risk: turning prices into returns and returns into the
numbers a risk report needs.

- Series out, trailing (a `when` condition may use them):
  `log_returns`, `cumulative_returns`, `drawdown`.
- Numbers out, over the whole series (a condition may not --
  allie/semantic/lookahead.py): `max_drawdown`, `volatility`, `sortino`,
  `value_at_risk`, `expected_shortfall`, `beta`.

Statistics of a return series -- `volatility`, `sortino`, the tail
measures, like `sharpe` (allie/stdlib/significance.py) -- drop non-finite
values, not only missing ones: an infinite return comes from a zero
price, a data error rather than a return, and the backtest engine drops
them from its equity returns the same way, so `sortino(r.returns, 252)`
is `r.sortino` exactly.
"""

from __future__ import annotations

import math

import numpy as np

from allie.quant.backtest_engine import annualized_sortino
from allie.runtime.errors import AllieValueError
from allie.runtime.values import SeriesValue
from allie.stdlib.missing import present, present_pairs
from allie.stdlib.registry import register
from allie.typechecker.types import FLOAT, INT, FunctionType, SeriesType

_SERIES_FLOAT = SeriesType(FLOAT)


def _finite_returns(returns: SeriesValue, fn_name: str, at_least: int) -> np.ndarray:
    data = returns.data
    if len(data) == 0:
        raise AllieValueError(f"{fn_name}: series must not be empty")
    values = data[np.isfinite(data)]
    if len(values) < at_least:
        raise AllieValueError(
            f"{fn_name}: needs at least {at_least} finite returns, got {len(values)}"
        )
    return values


def _check_periods(fn_name: str, periods_per_year: int) -> None:
    if periods_per_year < 1:
        raise AllieValueError(
            f"{fn_name}: periods_per_year must be at least 1, got {periods_per_year}"
        )


def _check_confidence(fn_name: str, confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise AllieValueError(
            f"{fn_name}: confidence is a fraction between 0 and 1 -- 0.95 is 95% -- "
            f"got {confidence}"
        )


# ---- series ----


def log_returns(prices: SeriesValue) -> SeriesValue:
    """`log(p[i] / p[i-1])`; NaN at the first bar, like `pct_change`.
    Log returns add up over time where simple returns compound."""
    data = prices.data
    result = np.full(len(data), np.nan, dtype=np.float64)
    if len(data) > 1:
        result[1:] = np.log(data[1:] / data[:-1])
    return SeriesValue(result)


def cumulative_returns(returns: SeriesValue) -> SeriesValue:
    """Compounded growth to each bar: `(1 + r1)(1 + r2)... - 1`. A
    missing return counts as no change, so the first bar of
    `pct_change(c)` doesn't erase the curve."""
    filled = np.where(np.isnan(returns.data), 0.0, returns.data)
    return SeriesValue(np.cumprod(1.0 + filled) - 1.0)


def drawdown(series: SeriesValue) -> SeriesValue:
    """How far each bar sits below the highest value so far, as a
    fraction of it: 0 at a new high, 0.2 twenty percent under. Takes a
    price or equity series (not returns). A missing bar is missing, and
    doesn't reset the high."""
    data = series.data
    peak = np.fmax.accumulate(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(peak > 0, (peak - data) / peak, np.nan)
    return SeriesValue(np.where(np.isnan(data), np.nan, result))


# ---- numbers ----


def max_drawdown(series: SeriesValue) -> float:
    """The largest `drawdown` over the series -- the worst peak-to-trough
    fall, as a positive fraction."""
    present(series, "max_drawdown")
    return float(np.max(present(drawdown(series), "max_drawdown")))


def volatility(returns: SeriesValue, periods_per_year: int) -> float:
    """Annualized standard deviation of returns (ddof=1), the backtest
    engine's convention."""
    _check_periods("volatility", periods_per_year)
    values = _finite_returns(returns, "volatility", at_least=2)
    return float(np.std(values, ddof=1)) * math.sqrt(periods_per_year)


def sortino(returns: SeriesValue, periods_per_year: int) -> float:
    """Annualized Sortino ratio, target 0 -- the backtest engine's own
    definition, so it's comparable to `r.sortino`."""
    _check_periods("sortino", periods_per_year)
    return annualized_sortino(_finite_returns(returns, "sortino", 2), periods_per_year)


def value_at_risk(returns: SeriesValue, confidence: float) -> float:
    """Historical value at risk: the loss a period's return exceeds only
    `1 - confidence` of the time, as a positive fraction --
    `value_at_risk(r, 0.95)` of 0.03 means a 3% loss or worse on one bar
    in twenty."""
    _check_confidence("value_at_risk", confidence)
    values = _finite_returns(returns, "value_at_risk", at_least=1)
    return float(-np.percentile(values, (1.0 - confidence) * 100.0))


def expected_shortfall(returns: SeriesValue, confidence: float) -> float:
    """The average loss on the bars beyond the value at risk -- how bad
    the bad days are, not just how often they come. Positive, like VaR."""
    _check_confidence("expected_shortfall", confidence)
    values = _finite_returns(returns, "expected_shortfall", at_least=1)
    threshold = np.percentile(values, (1.0 - confidence) * 100.0)
    return float(-np.mean(values[values <= threshold]))


def beta(asset: SeriesValue, market: SeriesValue) -> float:
    """Sensitivity of `asset`'s returns to `market`'s:
    `cov(asset, market) / var(market)`, over the bars where both have a
    value."""
    a, m = present_pairs(asset, market, "beta")
    market_variance = float(np.var(m, ddof=1))
    if market_variance == 0.0:
        raise AllieValueError("beta: undefined when the market series doesn't vary")
    return float(np.cov(a, m, ddof=1)[0, 1]) / market_variance


register("log_returns", FunctionType(params=(_SERIES_FLOAT,), ret=_SERIES_FLOAT), log_returns)
register(
    "cumulative_returns",
    FunctionType(params=(_SERIES_FLOAT,), ret=_SERIES_FLOAT),
    cumulative_returns,
)
register("drawdown", FunctionType(params=(_SERIES_FLOAT,), ret=_SERIES_FLOAT), drawdown)
register("max_drawdown", FunctionType(params=(_SERIES_FLOAT,), ret=FLOAT), max_drawdown)
register("volatility", FunctionType(params=(_SERIES_FLOAT, INT), ret=FLOAT), volatility)
register("sortino", FunctionType(params=(_SERIES_FLOAT, INT), ret=FLOAT), sortino)
register("value_at_risk", FunctionType(params=(_SERIES_FLOAT, FLOAT), ret=FLOAT), value_at_risk)
register(
    "expected_shortfall",
    FunctionType(params=(_SERIES_FLOAT, FLOAT), ret=FLOAT),
    expected_shortfall,
)
register("beta", FunctionType(params=(_SERIES_FLOAT, _SERIES_FLOAT), ret=FLOAT), beta)
