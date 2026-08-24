"""Backtest engine -- Phase 7.4. The single entry point every execution
path funnels into: the tree-walking interpreter, the IR interpreter, and
the LLVM/JIT backend all call `run_backtest`, so a `MovingAverage`
strategy produces identical metrics regardless of backend (verified in
tests/unit/test_backtest_cross_validation.py).

## How `when` clauses drive trades

The spec described the engine in operator terms ("on `crosses_above`
true and flat -> enter long"). Implemented one level more general, and
here's the reconciliation: each `when` clause contributes a boolean
signal series (from `crosses_above`/`crosses_below`) plus an *action*
-- its body. On each bar, every clause whose signal is True at that bar
has its action executed, and the `buy()`/`sell()` builtins called inside
that action are what actually move the position.

For the canonical strategy this is exactly the specced behavior:

    strategy MA {
        when fast crosses_above slow: buy();
        when fast crosses_below slow: sell();
    }

...but it avoids parsing, resolving, and type-checking the clause body
only to then ignore it, and it makes `buy()`/`sell()` real builtins
rather than decorative ones. A body that calls neither simply produces
no trade.

## Position and accounting model (v1)

Flat-or-long only, all-in, no leverage or shorting: `buy()` while flat
converts all cash to shares at that bar's close; `sell()` while long
converts all shares back to cash at that bar's close. `buy()` while
already long and `sell()` while flat are both no-ops rather than
errors -- a crossing signal firing twice before the opposite signal is
normal, not a program bug.

Equity at bar i is `cash + shares * close[i]`, so the curve marks to
market on every bar, not just on trade bars.

## Metrics

- **CAGR**: `(final/initial)^(1/years) - 1`, `years = bars /
  periods_per_year`.
- **Volatility**: `std(returns, ddof=1) * sqrt(periods_per_year)`.
- **Sharpe**: `mean(returns)/std(returns) * sqrt(periods_per_year)`.
  **Risk-free rate is 0 for v1** -- documented here rather than left
  implicit, since a nonzero rate changes the number materially and
  "Sharpe" alone doesn't say which convention is in use.
- **Max drawdown**: largest running-peak-to-trough decline over the
  equity curve, as a positive fraction.
- **Trades**: count of completed *round trips* (an entry still open at
  the final bar is held, marked to market in the equity curve, but not
  counted as a trade -- it hasn't produced a realized result yet).

`periods_per_year` defaults to 252 (trading days). Daily bars are what
`load` produces and what the fixture contains.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, PriceSeriesValue, SeriesValue

DEFAULT_INITIAL_CASH = 10_000.0
DEFAULT_PERIODS_PER_YEAR = 252


@dataclass
class Trade:
    entry_index: int
    entry_price: float
    exit_index: int
    exit_price: float

    @property
    def return_pct(self) -> float:
        return (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class BacktestMetrics:
    initial_equity: float
    final_equity: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    num_trades: int


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    metrics: BacktestMetrics
    trades: list[Trade]
    equity_curve: list[float]

    def render(self) -> str:
        """Fixed-width results table."""
        m = self.metrics
        rows = [
            ("Strategy", self.strategy_name),
            ("Symbol", self.symbol),
            ("Bars", f"{len(self.equity_curve)}"),
            ("Initial Equity", f"{m.initial_equity:,.2f}"),
            ("Final Equity", f"{m.final_equity:,.2f}"),
            ("CAGR", f"{m.cagr * 100:.2f}%"),
            ("Volatility", f"{m.volatility * 100:.2f}%"),
            ("Sharpe", f"{m.sharpe:.2f}"),
            ("Max Drawdown", f"{m.max_drawdown * 100:.2f}%"),
            ("Trades", f"{m.num_trades}"),
        ]
        width = 16
        lines = ["=" * 34, f"{'BACKTEST RESULTS':^34}", "=" * 34]
        lines.extend(f"{label:<{width}}{value:>18}" for label, value in rows)
        lines.append("=" * 34)
        return "\n".join(lines)


@dataclass
class WhenSignal:
    """One `when` clause: the bars it fires on, and what to do there."""

    signal: BoolSeriesValue
    action: Callable[[], None]


@dataclass
class _Portfolio:
    cash: float
    shares: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    _entry_index: int | None = None
    _entry_price: float | None = None

    @property
    def is_long(self) -> bool:
        return self.shares > 0.0

    def equity(self, price: float) -> float:
        return self.cash + self.shares * price

    def buy(self, index: int, price: float) -> None:
        if self.is_long or price <= 0.0:
            return
        self.shares = self.cash / price
        self.cash = 0.0
        self._entry_index = index
        self._entry_price = price

    def sell(self, index: int, price: float) -> None:
        if not self.is_long or price <= 0.0:
            return
        self.cash = self.shares * price
        self.shares = 0.0
        assert self._entry_index is not None and self._entry_price is not None
        self.trades.append(
            Trade(
                entry_index=self._entry_index,
                entry_price=self._entry_price,
                exit_index=index,
                exit_price=price,
            )
        )
        self._entry_index = None
        self._entry_price = None


class BacktestEngine:
    """Bar-by-bar driver. `buy()`/`sell()` reach the *active* engine
    through the module-level context below, since those builtins are
    plain zero-argument functions with no way to receive it directly."""

    def __init__(
        self,
        prices: PriceSeriesValue,
        *,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ) -> None:
        self.prices = prices
        self.initial_cash = initial_cash
        self.periods_per_year = periods_per_year
        self.portfolio = _Portfolio(cash=initial_cash)
        self._current_index = 0

    # -- called by the buy()/sell() builtins --

    def buy(self) -> None:
        self.portfolio.buy(self._current_index, self._close_at(self._current_index))

    def sell(self) -> None:
        self.portfolio.sell(self._current_index, self._close_at(self._current_index))

    def _close_at(self, index: int) -> float:
        return float(self.prices.close.data[index])

    def run(self, strategy_name: str, symbol: str, signals: list[WhenSignal]) -> BacktestResult:
        closes = self.prices.close.data
        n = len(closes)
        for signal in signals:
            if len(signal.signal) != n:
                raise AllieValueError(
                    f"backtest: signal series length {len(signal.signal)} does not match "
                    f"{n} price bars"
                )

        equity_curve: list[float] = []
        _push_engine(self)
        try:
            for index in range(n):
                self._current_index = index
                for signal in signals:
                    if bool(signal.signal.data[index]):
                        signal.action()
                equity_curve.append(self.portfolio.equity(self._close_at(index)))
        finally:
            _pop_engine()

        metrics = compute_metrics(
            equity_curve,
            num_trades=len(self.portfolio.trades),
            periods_per_year=self.periods_per_year,
        )
        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            metrics=metrics,
            trades=list(self.portfolio.trades),
            equity_curve=equity_curve,
        )


def compute_metrics(
    equity_curve: list[float],
    *,
    num_trades: int,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> BacktestMetrics:
    if not equity_curve:
        raise AllieValueError("backtest: cannot compute metrics over an empty equity curve")

    equity = np.asarray(equity_curve, dtype=np.float64)
    initial = float(equity[0])
    final = float(equity[-1])

    # Zero-variance and single-bar curves are normal (a strategy that
    # never trades), not errors -- report 0.0 rather than dividing by
    # zero or emitting a NaN that would silently poison downstream math.
    returns = (equity[1:] - equity[:-1]) / equity[:-1] if len(equity) > 1 else np.empty(0)
    returns = returns[np.isfinite(returns)]

    years = len(equity) / periods_per_year
    cagr = (final / initial) ** (1.0 / years) - 1.0 if years > 0 and initial > 0 else 0.0

    if len(returns) > 1:
        std = float(np.std(returns, ddof=1))
        mean = float(np.mean(returns))
    else:
        std = 0.0
        mean = 0.0

    volatility = std * math.sqrt(periods_per_year)
    # Risk-free rate = 0 for v1 -- see this module's docstring.
    sharpe = (mean / std) * math.sqrt(periods_per_year) if std > 0 else 0.0

    running_peak = np.maximum.accumulate(equity)
    drawdowns = (running_peak - equity) / running_peak
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0.0

    return BacktestMetrics(
        initial_equity=initial,
        final_equity=final,
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        num_trades=num_trades,
    )


def run_backtest(
    strategy_name: str,
    symbol: str,
    prices: PriceSeriesValue,
    clauses: list[WhenSignal],
    start: int,
    end: int,
    *,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> BacktestResult:
    """**The** entry point. Every execution path -- tree-walking
    interpreter, IR interpreter, LLVM/JIT -- calls exactly this, so no
    backend can drift into its own accounting.

    `start`/`end` are bar indices from `backtest S on SYM from N to M`.
    `end` is clamped to the number of available bars, so `from 0 to 100`
    against a 60-bar file runs the 60 bars rather than erroring -- a
    deliberately generous upper bound is a normal way to say "to the
    end". A `start` past the end, or `start >= end`, IS an error: those
    are mistakes, not shorthand.

    Signals are expected to have been computed over the *full* series
    and sliced here, not recomputed on the slice -- otherwise an
    indicator's warmup period would restart inside the test window and
    silently suppress early signals.
    """
    n_bars = len(prices.close)
    clamped_end = min(end, n_bars)
    if start < 0:
        raise AllieValueError(f"backtest: start bar {start} is negative")
    if start >= n_bars:
        raise AllieValueError(f"backtest: start bar {start} is past the last bar ({n_bars - 1})")
    if start >= clamped_end:
        raise AllieValueError(
            f"backtest: empty bar range (start={start}, end={clamped_end})"
        )

    sliced_prices = PriceSeriesValue(
        open=SeriesValue(prices.open.data[start:clamped_end]),
        high=SeriesValue(prices.high.data[start:clamped_end]),
        low=SeriesValue(prices.low.data[start:clamped_end]),
        close=SeriesValue(prices.close.data[start:clamped_end]),
        volume=SeriesValue(prices.volume.data[start:clamped_end]),
        dates=prices.dates[start:clamped_end] if len(prices.dates) else prices.dates,
    )
    sliced_clauses = [
        WhenSignal(
            signal=BoolSeriesValue(clause.signal.data[start:clamped_end]),
            action=clause.action,
        )
        for clause in clauses
    ]

    engine = BacktestEngine(
        sliced_prices, initial_cash=initial_cash, periods_per_year=periods_per_year
    )
    return engine.run(strategy_name, symbol, sliced_clauses)


# ---- IR / LLVM strategy registry ----
#
# The tree-walking interpreter holds a StrategyValue in its Environment
# and evaluates clauses directly. The IR and LLVM paths can't: their
# `strategy` lowering is a sequence of Calls (allie.ir.lowering's
# `_lower_strategy`), so the pieces arrive one bridge call at a time and
# need somewhere to accumulate. Both use exactly this registry, and both
# ultimately call `run_backtest` above -- the single shared entry point.
#
# `_when_body_dispatch` maps a lowered `when` body's function name to a
# zero-argument callable that runs it. Each backend fills it in its own
# way (the IR interpreter with a closure over `run_ir_program`, the JIT
# with a ctypes wrapper around the compiled function's address) and the
# engine just calls it, so neither backend needs its own engine.

_strategy_clauses: dict[str, list[tuple[BoolSeriesValue, str]]] = {}
_when_body_dispatch: dict[str, Callable[[], None]] = {}


def reset_strategy_registry() -> None:
    _strategy_clauses.clear()
    _when_body_dispatch.clear()


def register_when_body(name: str, action: Callable[[], None]) -> None:
    _when_body_dispatch[name] = action


def declare_strategy(name: str) -> None:
    _strategy_clauses[name] = []


def add_strategy_clause(spec: str, left: SeriesValue, right: SeriesValue) -> None:
    """`spec` is "strategy|operator|body_fn" -- packed into one value so
    the whole call fits the LLVM bridge's fixed 4-argument signature
    (see `_lower_when_clause`)."""
    from allie.stdlib.series_ops import crosses_above, crosses_below

    strategy_name, operator, body_fn = spec.split("|")
    if not isinstance(left, SeriesValue) or not isinstance(right, SeriesValue):
        raise AllieValueError("'when' clause operands must be Series<Float>")
    crossing = crosses_above if operator == "crosses_above" else crosses_below
    _strategy_clauses.setdefault(strategy_name, []).append((crossing(left, right), body_fn))


def run_registered_backtest(
    spec: str, prices: PriceSeriesValue, start: float, end: float
) -> None:
    """Bridge entry point for `__backtest_run`. `spec` is
    "strategy|symbol"."""
    strategy_name, symbol = spec.split("|")
    if strategy_name not in _strategy_clauses:
        raise AllieValueError(f"backtest: '{strategy_name}' is not a declared strategy")
    if not isinstance(prices, PriceSeriesValue):
        raise AllieValueError(
            f"backtest: '{symbol}' must be a PriceSeries (from load(...)), "
            f"got {type(prices).__name__}"
        )

    clauses = [
        WhenSignal(signal=signal, action=_resolve_when_body(body_fn))
        for signal, body_fn in _strategy_clauses[strategy_name]
    ]
    result = run_backtest(
        strategy_name=strategy_name,
        symbol=symbol,
        prices=prices,
        clauses=clauses,
        start=int(start),
        end=int(end),
    )
    print(result.render())


def _resolve_when_body(body_fn: str) -> Callable[[], None]:
    action = _when_body_dispatch.get(body_fn)
    if action is None:
        raise AllieValueError(f"backtest: no compiled body registered for '{body_fn}'")
    return action


# ---- active-engine context ----
#
# `buy()`/`sell()` are zero-argument stdlib builtins (Phase 7.2), so they
# have no parameter through which to receive the running engine. A
# module-level stack gives them one. It's a stack rather than a single
# slot so a nested `backtest` (a strategy body that itself runs one)
# can't strand the outer engine -- not reachable from today's grammar,
# but the stack costs nothing and removes the failure mode entirely.

_engine_stack: list[BacktestEngine] = []


def _push_engine(engine: BacktestEngine) -> None:
    _engine_stack.append(engine)


def _pop_engine() -> None:
    if _engine_stack:
        _engine_stack.pop()


def current_engine() -> BacktestEngine | None:
    return _engine_stack[-1] if _engine_stack else None
