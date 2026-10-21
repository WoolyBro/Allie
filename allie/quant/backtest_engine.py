"""Backtest engine. The single entry point every execution path funnels
into: the tree-walking interpreter, the IR interpreter, and the LLVM/JIT
backend all call `run_backtest`, so a strategy produces identical
results regardless of backend (tests/unit/test_backtest_cross_validation.py).

## How `when` clauses drive trades

Each `when` clause contributes a boolean signal series -- its condition,
any Series<Bool>, evaluated once over the full price history when the
strategy is declared -- plus an *action*, its body. On each bar, every
clause whose signal is true at that bar runs its action, and the order
builtins called there -- `buy()`, `sell()`, `short()`, `cover()`,
`target(f)` -- are what move the position:

    strategy Trend {
        when fast crosses_above slow and rsi < 70.0: buy();
        when fast crosses_below slow or rsi > 80.0: sell();
    }

The condition is evaluated at the declaration on every backend, so
reassigning `fast` afterwards doesn't change a strategy already declared.

## When an order fills

A condition at bar i is computed from bar i's close, so the earliest a
real trader could act on it is the *next* bar's open. That is the
default (`fill: "open"`): an order placed at bar i fills at
`open[i + 1]`, and an order placed on the last bar never fills. Filling
at the same close the signal was computed from (`fill: "close"`) is
available, because it is a common convention, but it assumes you traded
at a price you only knew once it was too late to trade at it.

Within one bar the actions of every firing clause run in order, and the
last order placed is the one that fills -- `buy()` then `sell()` in the
same bar nets out to no trade. A bar with no usable price (NaN, or not
positive) can't fill an order; the order waits for the next one that can.

## Costs are on by default

docs/spec.md: costs must be first-class, not opt-in, because "if it's
opt-in, people forget it, and backtests look unrealistically good". Every
fill pays `slippage` -- a buy fills `slippage` above the reference price
and a sell below it -- and `commission`, a fraction of the traded
notional. Slippage defaults to 5 bps per fill and commission to 0; both
are printed with every result, and turning them off is a visible
`with slippage: 0.0` in the source.

## Positions

`target(f)` holds `f` times the account's equity in the symbol: 1.0 is
all-in long, 0.5 half, 0.0 flat, -1.0 short. The other orders are
target() with a guard, so a repeated signal is a no-op:

- `buy()`: if not long, target(1.0) -- from a short this reverses.
- `sell()`: if long, target(0.0).
- `short()`: if not short, target(-1.0) -- from a long this reverses.
- `cover()`: if short, target(0.0).

A fill's costs come out of the equity it is sized against, so after it
the position is exactly the fraction asked for: an all-in `buy()`
spends exactly the cash, never borrows, and a repeated `target(0.5)` at
an unchanged price trades nothing. A reversal fills as a close then an
open, at the same price. Equity is `cash + shares * close`, marked to market every
bar; a short adds its proceeds to cash. If equity reaches zero the
position is closed out and the account stops trading, rather than
carrying a negative balance.

## Stops

`stop_loss` and `take_profit`, fractions of the entry price, are checked
on every bar a position is open, using that bar's open, high and low --
including the bar a next-open fill entered on. A bar that opens past a
level exits at the open (a gap gives no better price); otherwise the
level itself is the exit price. If one bar's range reaches both, the
stop is assumed to have been hit first: the high and low don't say which
came first, and the pessimistic reading is the safe one.

## Metrics

- **Total return**, `final / initial - 1`; **Buy & Hold**, the same for
  holding the symbol over the window (no costs).
- **CAGR**: `(final/initial)^(1/years) - 1`, `years = (bars - 1) /
  periods_per_year` -- n bars span n - 1 periods.
- **Volatility**: `std(returns, ddof=1) * sqrt(periods_per_year)`.
- **Sharpe**: `mean(returns)/std(returns) * sqrt(periods_per_year)`,
  **risk-free rate 0** -- stated, since "Sharpe" alone doesn't say which
  convention is in use. **Sortino** divides by the downside deviation
  instead (root mean square of the negative returns, target 0).
- **Max drawdown**: largest running-peak-to-trough decline of the equity
  curve, as a positive fraction.
- **Exposure**: the fraction of bars that end holding a position.
- **Trades** are completed round trips; a position still open at the last
  bar is marked to market but not counted -- it has no realized result.
  Each trade's return is its P&L, after costs, over its entry notional.
  **Win rate**, **profit factor** (gross profit / gross loss) and the
  **average trade** are over those; NaN when there are no trades.

`periods_per_year` defaults to 252 (trading days): daily bars are what
`load` produces and what the fixture contains.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, PriceSeriesValue, SeriesValue

DEFAULT_INITIAL_CASH = 10_000.0
DEFAULT_PERIODS_PER_YEAR = 252
DEFAULT_SLIPPAGE = 0.0005  # 5 bps per fill

# A rebalance smaller than this fraction of equity is float noise, not a trade.
_DUST = 1e-9

# The builtins that place an order (allie/stdlib/trading_fns.py). The type
# checker allows calls to them only where a strategy can run them.
ORDER_FUNCTIONS = frozenset({"buy", "sell", "short", "cover", "target"})


# ---- configuration: `backtest S on SYM from N to M with name: value, ...` ----


@dataclass(frozen=True)
class BacktestConfig:
    cash: float = DEFAULT_INITIAL_CASH
    commission: float = 0.0
    slippage: float = DEFAULT_SLIPPAGE
    fill: str = "open"
    stop_loss: float | None = None
    take_profit: float | None = None
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR


# Validity checks for option values: the reason a value is rejected, or
# None. Values arrive type-checked, so a Float option's value is a float.


def _positive(name: str, value: Any) -> str | None:
    return None if 0 < value < math.inf else f"'{name}' must be positive, got {value}"


def _slippage_fraction(name: str, value: Any) -> str | None:
    if 0.0 <= value < 1.0:
        return None
    return f"'{name}' is a fraction of the price from 0 up to 1 -- 0.0005 is 5 bps -- got {value}"


def _commission_fraction(name: str, value: Any) -> str | None:
    if 0.0 <= value < 1.0:
        return None
    return (
        f"'{name}' is a fraction of the value traded from 0 up to 1 -- 0.001 is 10 bps -- "
        f"got {value}"
    )


def _stop_fraction(name: str, value: Any) -> str | None:
    if 0.0 < value < 1.0:
        return None
    return f"'{name}' is a fraction of the entry price between 0 and 1 -- 0.05 is 5% -- got {value}"


def _profit_fraction(name: str, value: Any) -> str | None:
    if 0 < value < math.inf:
        return None
    return f"'{name}' is a positive fraction of the entry price -- 0.1 is 10% -- got {value}"


def _fill_point(name: str, value: Any) -> str | None:
    if value in ("open", "close"):
        return None
    return (
        f'\'{name}\' is "open" (the next bar\'s open) or "close" '
        f'(the signal bar\'s close), got "{value}"'
    )


def _at_least_one(name: str, value: Any) -> str | None:
    return None if value >= 1 else f"'{name}' must be at least 1, got {value}"


# The options a `with` clause accepts: name -> (Allie type, validity check).
# The type checker reads this table too (allie/typechecker/checker.py), so
# the names, types and rules exist once.
OPTIONS: dict[str, tuple[str, Callable[[str, Any], str | None]]] = {
    "cash": ("Float", _positive),
    "commission": ("Float", _commission_fraction),
    "slippage": ("Float", _slippage_fraction),
    "fill": ("String", _fill_point),
    "stop_loss": ("Float", _stop_fraction),
    "take_profit": ("Float", _profit_fraction),
    "periods_per_year": ("Int", _at_least_one),
}


def option_error(name: str, value: object) -> str | None:
    """Why `name: value` isn't a valid option, or None if it is. The type
    checker calls this on literal values; the engine on every value."""
    if name not in OPTIONS:
        return f"unknown backtest option '{name}'"
    if isinstance(value, float) and math.isnan(value):
        return f"'{name}' must be a number, got nan"
    return OPTIONS[name][1](name, value)


def config_from_options(options: Mapping[str, object]) -> BacktestConfig:
    for name, value in options.items():
        error = option_error(name, value)
        if error is not None:
            raise AllieValueError(f"backtest: {error}")
    return BacktestConfig(**options)  # type: ignore[arg-type]  # checked above


# ---- results ----


@dataclass
class Trade:
    """One round trip. Prices are fill prices, slippage included; `pnl`
    is in account currency, after every cost of the trade."""

    entry_index: int
    entry_price: float
    exit_index: int
    exit_price: float
    direction: int = 1  # 1 long, -1 short
    pnl: float = 0.0
    entry_notional: float = 1.0
    exit_reason: str = "signal"  # or "stop_loss", "take_profit", "ruin"

    @property
    def return_pct(self) -> float:
        return self.pnl / self.entry_notional


@dataclass
class BacktestMetrics:
    initial_equity: float
    final_equity: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    num_trades: int
    total_return: float = 0.0
    sortino: float = 0.0


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    metrics: BacktestMetrics
    trades: list[Trade]
    equity_curve: list[float]
    positions: list[float] = field(default_factory=list)
    costs: float = 0.0
    buy_hold_return: float = math.nan
    config: BacktestConfig = field(default_factory=BacktestConfig)
    period: tuple[str, str] | None = None
    ruined_at: int | None = None
    closes: list[float] = field(default_factory=list)  # the window's, for backtest_pvalue

    # -- trade statistics --

    @property
    def trade_returns(self) -> np.ndarray:
        return np.array([t.return_pct for t in self.trades], dtype=np.float64)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return math.nan
        return sum(t.pnl > 0 for t in self.trades) / len(self.trades)

    @property
    def profit_factor(self) -> float:
        if not self.trades:
            return math.nan
        gains = sum(t.pnl for t in self.trades if t.pnl > 0)
        losses = -sum(t.pnl for t in self.trades if t.pnl < 0)
        if losses == 0:
            return math.inf if gains > 0 else math.nan
        return gains / losses

    @property
    def avg_trade_return(self) -> float:
        return float(np.mean(self.trade_returns)) if self.trades else math.nan

    @property
    def exposure(self) -> float:
        if not self.positions:
            return 0.0
        return float(np.mean(np.asarray(self.positions) != 0.0))

    @property
    def final_position(self) -> str:
        last = self.positions[-1] if self.positions else 0.0
        return "long" if last > 0 else "short" if last < 0 else "flat"

    def drawdown(self) -> np.ndarray:
        equity = np.asarray(self.equity_curve, dtype=np.float64)
        peak = np.maximum.accumulate(equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(peak > 0, (peak - equity) / peak, 0.0)

    # -- presentation --

    def render(self) -> str:
        """Fixed-width results table. The assumptions -- when orders fill
        and what they cost -- are part of the result, so they're in it."""
        m = self.metrics
        c = self.config
        execution = "next bar open" if c.fill == "open" else "signal bar close"
        setup = [
            ("Strategy", self.strategy_name),
            ("Symbol", self.symbol),
            *([("Period", f"{self.period[0]} to {self.period[1]}")] if self.period else []),
            ("Bars", f"{len(self.equity_curve)}"),
            ("Fills", execution),
            ("Slippage", f"{c.slippage * 1e4:.1f} bps"),
            ("Commission", f"{c.commission * 1e4:.1f} bps"),
            *([("Stop Loss", _pct(c.stop_loss))] if c.stop_loss is not None else []),
            *([("Take Profit", _pct(c.take_profit))] if c.take_profit is not None else []),
        ]
        performance = [
            ("Initial Equity", f"{m.initial_equity:,.2f}"),
            ("Final Equity", f"{m.final_equity:,.2f}"),
            ("Total Return", _pct(m.total_return)),
            ("Buy & Hold", _pct(self.buy_hold_return)),
            ("CAGR", _pct(m.cagr)),
            ("Volatility", _pct(m.volatility)),
            ("Sharpe", _number(m.sharpe)),
            ("Sortino", _number(m.sortino)),
            ("Max Drawdown", _pct(m.max_drawdown)),
            ("Exposure", _pct(self.exposure)),
        ]
        trading = [
            ("Trades", f"{m.num_trades}"),
            ("Win Rate", _pct(self.win_rate)),
            ("Profit Factor", _number(self.profit_factor)),
            ("Avg Trade", _pct(self.avg_trade_return)),
            ("Costs Paid", f"{self.costs:,.2f}"),
            ("Position at End", self.final_position),
            *([("Ruined at Bar", f"{self.ruined_at}")] if self.ruined_at is not None else []),
        ]
        width = 40
        rule, thin = "=" * width, "-" * width
        lines = [rule, f"{'BACKTEST RESULTS':^{width}}".rstrip(), rule]
        for i, section in enumerate((setup, performance, trading)):
            if i:
                lines.append(thin)
            lines.extend(f"{label:<16}{value:>{width - 16}}" for label, value in section)
        lines.append(rule)
        return "\n".join(lines)

    def field(self, name: str) -> object:
        """A result field as Allie source reads it: `r.sharpe`,
        `r.returns`, ... -- the fields the type checker knows
        (allie.typechecker.checker._BACKTEST_RESULT_FIELDS)."""
        m = self.metrics
        equity = np.asarray(self.equity_curve, dtype=np.float64)
        fields: dict[str, Callable[[], object]] = {
            "sharpe": lambda: m.sharpe,
            "sortino": lambda: m.sortino,
            "cagr": lambda: m.cagr,
            "total_return": lambda: m.total_return,
            "buy_hold_return": lambda: self.buy_hold_return,
            "volatility": lambda: m.volatility,
            "max_drawdown": lambda: m.max_drawdown,
            "initial_equity": lambda: m.initial_equity,
            "final_equity": lambda: m.final_equity,
            "exposure": lambda: self.exposure,
            "costs": lambda: self.costs,
            "trades": lambda: m.num_trades,
            "win_rate": lambda: self.win_rate,
            "profit_factor": lambda: self.profit_factor,
            "avg_trade_return": lambda: self.avg_trade_return,
            "equity": lambda: SeriesValue(equity),
            # Per-bar returns of the equity curve; NaN at the first bar,
            # like pct_change, so the series lines up with the prices.
            # The same numbers `compute_metrics` takes the Sharpe of, so
            # `sharpe(r.returns, 252) == r.sharpe` holds exactly.
            "returns": lambda: SeriesValue(np.concatenate(([np.nan], bar_returns(equity)))),
            "drawdown": lambda: SeriesValue(self.drawdown()),
            "position": lambda: SeriesValue(np.asarray(self.positions, dtype=np.float64)),
            "trade_returns": lambda: SeriesValue(self.trade_returns),
        }
        if name not in fields:
            raise AllieValueError(f"a backtest result has no field '{name}'")
        return fields[name]()

    def __str__(self) -> str:
        return self.render()


def _pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value * 100:.2f}%"


def _number(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.2f}"


# ---- the engine ----


@dataclass
class WhenSignal:
    """One `when` clause: the bars it fires on, and what to do there."""

    signal: BoolSeriesValue
    action: Callable[[], None]


@dataclass
class _OpenTrade:
    direction: int
    entry_index: int
    entry_price: float
    entry_equity: float  # before the opening fill, so the P&L includes its costs
    entry_notional: float


def _sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def _tradable(price: float) -> bool:
    return math.isfinite(price) and price > 0.0


class BacktestEngine:
    """Bar-by-bar driver. The order builtins reach the *active* engine
    through the module-level context below, since they are plain
    functions with no way to receive it directly."""

    def __init__(self, prices: PriceSeriesValue, config: BacktestConfig | None = None) -> None:
        self.prices = prices
        self.config = config if config is not None else BacktestConfig()
        self.cash = self.config.cash
        self.shares = 0.0
        self.trades: list[Trade] = []
        self.orders: list[tuple[int, float]] = []  # (bar, target exposure), in the order placed
        self.costs = 0.0
        self.ruined_at: int | None = None
        self._index = 0
        self._pending: float | None = None  # target exposure waiting to fill
        self._open_trade: _OpenTrade | None = None
        self._mark = math.nan  # the last usable close

    # -- orders: what buy()/sell()/short()/cover()/target() call --

    def buy(self) -> None:
        if self._intended_direction() <= 0:
            self._order(1.0)

    def sell(self) -> None:
        if self._intended_direction() > 0:
            self._order(0.0)

    def short(self) -> None:
        if self._intended_direction() >= 0:
            self._order(-1.0)

    def cover(self) -> None:
        if self._intended_direction() < 0:
            self._order(0.0)

    def target(self, fraction: float) -> None:
        if not math.isfinite(fraction):
            raise AllieValueError(f"target: the fraction must be a finite number, got {fraction}")
        self._order(float(fraction))

    def _intended_direction(self) -> int:
        """Where the account is headed: the order waiting to fill if
        there is one, else the position held."""
        return _sign(self._pending if self._pending is not None else self.shares)

    def _order(self, fraction: float) -> None:
        if self.ruined_at is None:
            self._pending = fraction  # the last order placed in a bar wins
            self.orders.append((self._index, fraction))

    # -- the loop --

    def run(self, strategy_name: str, symbol: str, signals: list[WhenSignal]) -> BacktestResult:
        p = self.prices
        opens, highs, lows, closes = p.open.data, p.high.data, p.low.data, p.close.data
        n = len(closes)
        for signal in signals:
            if len(signal.signal) != n:
                raise AllieValueError(
                    f"backtest: signal series length {len(signal.signal)} does not match "
                    f"{n} price bars"
                )

        fill_at_open = self.config.fill == "open"
        has_stops = self.config.stop_loss is not None or self.config.take_profit is not None
        # Plain Python lists: indexing a numpy array bar by bar boxes a
        # numpy scalar each time, which was most of this loop's cost.
        open_, high, low, close = opens.tolist(), highs.tolist(), lows.tolist(), closes.tolist()
        clauses = [(signal.signal.data.tolist(), signal.action) for signal in signals]
        equity_curve: list[float] = []
        positions: list[float] = []
        _push_engine(self)
        try:
            for index in range(n):
                self._index = index
                if fill_at_open and self._pending is not None:
                    self._fill_pending(open_[index])
                if has_stops and self._open_trade is not None:
                    self._check_exits(open_[index], high[index], low[index])
                for fires, action in clauses:
                    if fires[index]:
                        action()
                if not fill_at_open and self._pending is not None:
                    self._fill_pending(close[index])
                equity, position = self._mark_to_market(close[index])
                equity_curve.append(equity)
                positions.append(position)
        finally:
            _pop_engine()

        metrics = compute_metrics(
            equity_curve,
            num_trades=len(self.trades),
            periods_per_year=self.config.periods_per_year,
        )
        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            metrics=metrics,
            trades=list(self.trades),
            equity_curve=equity_curve,
            positions=positions,
            costs=self.costs,
            buy_hold_return=_buy_and_hold(closes),
            config=self.config,
            period=_period(p.dates, n),
            ruined_at=self.ruined_at,
            closes=close,
        )

    def _fill_pending(self, price: float) -> None:
        if self._pending is None or not _tradable(price):
            return  # nothing to do, or no price to do it at: the order waits
        fraction, self._pending = self._pending, None
        self._rebalance(fraction, price, "signal")

    def _check_exits(self, open_: float, high: float, low: float) -> None:
        trade = self._open_trade
        stop_loss, take_profit = self.config.stop_loss, self.config.take_profit
        if trade is None or (stop_loss is None and take_profit is None):
            return
        d, entry = trade.direction, trade.entry_price
        stop = entry * (1 - d * stop_loss) if stop_loss is not None else None
        take = entry * (1 + d * take_profit) if take_profit is not None else None

        def beyond(price: float, level: float | None, adverse: bool) -> bool:
            # "Past the level" in the direction that triggers it.
            if level is None:
                return False
            moving_down = (d > 0) == adverse
            return price <= level if moving_down else price >= level

        exit_price, reason = None, ""
        if beyond(open_, stop, adverse=True):
            exit_price, reason = open_, "stop_loss"  # gapped through: the open is the price
        elif beyond(open_, take, adverse=False):
            exit_price, reason = open_, "take_profit"
        elif beyond(low if d > 0 else high, stop, adverse=True):
            exit_price, reason = stop, "stop_loss"  # checked first: the pessimistic reading
        elif beyond(high if d > 0 else low, take, adverse=False):
            exit_price, reason = take, "take_profit"
        if exit_price is not None and _tradable(exit_price):
            self._rebalance(0.0, exit_price, reason)

    def _mark_to_market(self, close: float) -> tuple[float, float]:
        if _tradable(close):
            self._mark = close
        equity = self.cash + (self.shares * self._mark if self.shares else 0.0)
        if equity <= 0.0 and self.shares and self.ruined_at is None:
            # The account is wiped out: close it out and stop trading.
            self._rebalance(0.0, self._mark, "ruin")
            self.ruined_at = self._index
            self._pending = None
            equity = self.cash
        equity = max(equity, 0.0)
        position = self.shares * self._mark / equity if self.shares and equity > 0 else 0.0
        return equity, position

    # -- fills --

    def _rebalance(self, fraction: float, price: float, reason: str) -> None:
        """Moves the position to `fraction` of equity at `price`. A
        reversal is two fills: the close, then the open."""
        if self.shares and _sign(fraction) != _sign(self.shares):
            self._fill(0.0, price, reason)
        if fraction != 0.0:
            self._fill(fraction, price, reason)

    def _fill(self, fraction: float, price: float, reason: str) -> None:
        """Trades to exactly `fraction` of equity *after* the fill's own
        costs, valued at `price`. Solving for that -- rather than sizing
        on the equity before costs -- means an all-in buy spends exactly
        the cash, and repeating an order at an unchanged price trades
        nothing instead of topping up the sliver the costs took."""
        if not math.isfinite(self._mark):
            self._mark = price  # marks a position opened before the first usable close
        equity = self.cash + self.shares * price
        if fraction != 0.0 and equity <= 0.0:
            return  # nothing left to trade with
        gap = fraction * equity - self.shares * price  # value to buy (+) or sell (-)
        if fraction != 0.0 and abs(gap) <= _DUST * equity:
            return  # already there
        slippage, commission = self.config.slippage, self.config.commission
        buying = gap > 0
        fill_price = price * (1 + slippage) if buying else price * (1 - slippage)
        # Cash per share traded, commission included: paid when buying,
        # received when selling.
        per_share = fill_price * (1 + commission) if buying else fill_price * (1 - commission)
        if fraction == 0.0:
            delta = -self.shares
        else:
            # Each share traded moves equity by (price - per_share), so
            # (shares + delta) * price = fraction * (equity + delta * (price - per_share)).
            denominator = price + fraction * (per_share - price)
            if denominator <= 0.0:
                raise AllieValueError(
                    f"target: {fraction} times equity is more leverage than these costs allow"
                )
            delta = gap / denominator
        before = self.shares
        self.cash -= delta * per_share
        self.shares = before + delta if fraction != 0.0 else 0.0
        self.costs += abs(delta) * (fill_price * commission + price * slippage)
        self._record(before, fill_price, equity, reason)

    def _record(self, before: float, fill_price: float, equity_before: float, reason: str) -> None:
        if not before and self.shares:
            self._open_trade = _OpenTrade(
                direction=_sign(self.shares),
                entry_index=self._index,
                entry_price=fill_price,
                entry_equity=equity_before,
                entry_notional=abs(self.shares) * fill_price,
            )
        elif before and not self.shares:
            opened = self._open_trade
            assert opened is not None
            self.trades.append(
                Trade(
                    entry_index=opened.entry_index,
                    entry_price=opened.entry_price,
                    exit_index=self._index,
                    exit_price=fill_price,
                    direction=opened.direction,
                    # Flat now, so equity is cash -- and cash earns nothing,
                    # so the change is exactly the trade's P&L after costs.
                    pnl=self.cash - opened.entry_equity,
                    entry_notional=opened.entry_notional,
                    exit_reason=reason,
                )
            )
            self._open_trade = None


def _buy_and_hold(closes: np.ndarray) -> float:
    usable = closes[np.isfinite(closes) & (closes > 0)]
    return float(usable[-1] / usable[0] - 1.0) if len(usable) >= 2 else math.nan


def _period(dates: np.ndarray, n: int) -> tuple[str, str] | None:
    if n == 0 or len(dates) != n:
        return None
    first, last = np.datetime_as_string(dates[[0, -1]], unit="D")
    return str(first), str(last)


# ---- metrics ----


def effective_std(returns: np.ndarray) -> float:
    """Sample standard deviation (`ddof=1`), rounded down to exactly 0
    when it is only float noise.

    `np.std` of a *constant* series is not exactly zero -- float64 has
    about 1e-16 of relative precision, so accumulated rounding leaves
    something like 1e-18 behind. Testing `std > 0` therefore passes for
    a series with no real variance at all, and dividing by 1e-18
    produced a **Sharpe ratio of 1.6e15** for a strategy with perfectly
    steady growth. That number went straight into the results table.

    The threshold is relative to the magnitude of the data rather than
    absolute, so it behaves the same for returns quoted as 0.01 or as
    1%. 1e-12 sits far above float64's noise floor and far below any
    real signal.
    """
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    scale = float(np.max(np.abs(returns)))
    return 0.0 if std <= scale * 1e-12 else std


def bar_returns(equity: np.ndarray) -> np.ndarray:
    """Simple returns between consecutive bars of an equity curve."""
    if len(equity) < 2:
        return np.empty(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns: np.ndarray = (equity[1:] - equity[:-1]) / equity[:-1]
    return returns


def annualized_sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    """Annualized Sharpe ratio, risk-free rate 0.

    The single definition -- `compute_metrics` and the `sharpe` builtin
    (allie.stdlib.significance) both call this, so a Sharpe computed in
    Allie source is guaranteed comparable to one from a `backtest`
    block rather than merely intended to be.
    """
    std = effective_std(returns)
    if std == 0.0:
        return 0.0
    return float(np.mean(returns)) / std * math.sqrt(periods_per_year)


def annualized_sortino(returns: np.ndarray, periods_per_year: int) -> float:
    """Like Sharpe, over the downside deviation: the root mean square of
    the returns below 0, so upside volatility isn't penalized. No
    downside at all is infinite for a positive mean, and 0.0 for none."""
    if len(returns) < 2:
        return 0.0
    scale = float(np.max(np.abs(returns)))
    mean = float(np.mean(returns))
    downside = math.sqrt(float(np.mean(np.minimum(returns, 0.0) ** 2)))
    if downside <= scale * 1e-12:
        return math.inf if mean > scale * 1e-12 else 0.0
    return mean / downside * math.sqrt(periods_per_year)


def compute_metrics(
    equity_curve: list[float],
    *,
    num_trades: int = 0,
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
    returns = bar_returns(equity)
    returns = returns[np.isfinite(returns)]

    years = (len(equity) - 1) / periods_per_year
    if years <= 0 or initial <= 0:
        cagr = 0.0
    elif final <= 0:
        cagr = -1.0
    else:
        cagr = (final / initial) ** (1.0 / years) - 1.0

    running_peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(running_peak > 0, (running_peak - equity) / running_peak, 0.0)

    return BacktestMetrics(
        initial_equity=initial,
        final_equity=final,
        cagr=cagr,
        volatility=effective_std(returns) * math.sqrt(periods_per_year),
        sharpe=annualized_sharpe(returns, periods_per_year),
        max_drawdown=float(np.max(drawdowns)),
        num_trades=num_trades,
        total_return=final / initial - 1.0 if initial > 0 else 0.0,
        sortino=annualized_sortino(returns, periods_per_year),
    )


def run_backtest(
    strategy_name: str,
    symbol: str,
    prices: PriceSeriesValue,
    clauses: list[WhenSignal],
    start: int,
    end: int,
    *,
    config: BacktestConfig | None = None,
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
        raise AllieValueError(f"backtest: empty bar range (start={start}, end={clamped_end})")
    for number, clause in enumerate(clauses, start=1):
        # Checked before slicing, so the message gives the lengths the
        # user wrote, not the lengths of two windows of them.
        if len(clause.signal) != n_bars:
            raise AllieValueError(
                f"backtest: the condition of 'when' clause {number} of '{strategy_name}' "
                f"has {len(clause.signal)} bars, but {symbol} has {n_bars}"
            )

    window = slice(start, clamped_end)
    sliced_prices = PriceSeriesValue(
        open=SeriesValue(prices.open.data[window]),
        high=SeriesValue(prices.high.data[window]),
        low=SeriesValue(prices.low.data[window]),
        close=SeriesValue(prices.close.data[window]),
        volume=SeriesValue(prices.volume.data[window]),
        dates=prices.dates[window] if len(prices.dates) else prices.dates,
    )
    sliced_clauses = [
        WhenSignal(signal=BoolSeriesValue(clause.signal.data[window]), action=clause.action)
        for clause in clauses
    ]
    engine = BacktestEngine(sliced_prices, config)
    result = engine.run(strategy_name, symbol, sliced_clauses)
    if _observers:
        record = BacktestRecord(
            strategy_name=strategy_name,
            symbol=symbol,
            start=start,
            end=clamped_end,
            signals=tuple(clause.signal.data for clause in clauses),
            orders=tuple(engine.orders),
            result=result,
        )
        for observe in list(_observers):
            observe(record)
    return result


# ---- observing backtests ----


@dataclass(frozen=True)
class BacktestRecord:
    """What one backtest decided, for an observer. Bars in `orders` and
    in the result's trades count from `start`; `signals` -- each `when`
    clause's condition -- cover every bar of the data."""

    strategy_name: str
    symbol: str
    start: int
    end: int
    signals: tuple[np.ndarray, ...]
    orders: tuple[tuple[int, float], ...]
    result: BacktestResult


_observers: list[Callable[[BacktestRecord], None]] = []


@contextmanager
def observe_backtests() -> Iterator[list[BacktestRecord]]:
    """Collects a record of every backtest run inside the `with`, on any
    backend -- they all come through `run_backtest`. The look-ahead audit
    (allie/semantic/audit.py) compares these between runs."""
    records: list[BacktestRecord] = []
    observe = records.append
    _observers.append(observe)
    try:
        yield records
    finally:
        _observers.remove(observe)


# ---- IR / LLVM strategy registry ----
#
# The IR interpreter and LLVM backend lower `strategy`/`when`/`backtest`
# to calls (`__strategy_declare`, `__strategy_clause`, `__backtest_option`,
# `__backtest_run`) rather than dedicated instructions, and those calls
# land here. They register clauses by strategy name and ultimately call
# `run_backtest` above -- the single shared entry point.
#
# `_when_body_dispatch` maps a lowered `when` body's function name to a
# zero-argument callable that runs it. Each backend fills it in its own
# way (the IR interpreter with a closure over `run_ir_program`, the JIT
# with a ctypes wrapper around the compiled function's address) and the
# engine just calls it, so neither backend needs its own engine.

_strategy_clauses: dict[str, list[tuple[BoolSeriesValue, str]]] = {}
_when_body_dispatch: dict[str, Callable[[], None]] = {}
# A `with` clause's options, set one call at a time just before the
# `__backtest_run` they belong to, which takes them -- one call per
# option keeps each within the LLVM bridge's fixed arity.
_pending_options: dict[str, object] = {}


def reset_strategy_registry() -> None:
    _strategy_clauses.clear()
    _when_body_dispatch.clear()
    _pending_options.clear()


def register_when_body(name: str, action: Callable[[], None]) -> None:
    _when_body_dispatch[name] = action


def declare_strategy(name: str) -> None:
    _strategy_clauses[name] = []


def add_strategy_clause(spec: str, condition: object) -> None:
    """`spec` is "strategy|body_fn" -- packed into one value so the call
    fits the LLVM bridge's fixed-arity signature. `condition` is the
    clause's Series<Bool>, already evaluated at the declaration."""
    from allie.runtime.elementwise import as_signal

    strategy_name, body_fn = spec.split("|")
    _strategy_clauses.setdefault(strategy_name, []).append((as_signal(condition), body_fn))


def set_backtest_option(name: str, value: object) -> None:
    """Bridge entry point for `__backtest_option`."""
    _pending_options[name] = value


def run_registered_backtest(
    spec: str, prices: PriceSeriesValue, start: float, end: float
) -> BacktestResult:
    """Bridge entry point for `__backtest_run`. `spec` is
    "strategy|symbol". Returns the result; the statement form of
    `backtest` prints it, the expression form keeps it."""
    options = dict(_pending_options)
    _pending_options.clear()
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
    return run_backtest(
        strategy_name=strategy_name,
        symbol=symbol,
        prices=prices,
        clauses=clauses,
        start=int(start),
        end=int(end),
        config=config_from_options(options),
    )


def _resolve_when_body(body_fn: str) -> Callable[[], None]:
    action = _when_body_dispatch.get(body_fn)
    if action is None:
        raise AllieValueError(f"backtest: no compiled body registered for '{body_fn}'")
    return action


# ---- active-engine context ----
#
# The order builtins take no engine argument, so a module-level stack
# gives them one. It's a stack rather than a single slot so a nested
# `backtest` (a strategy body that itself runs one) can't strand the
# outer engine -- not reachable from today's grammar, but the stack costs
# nothing and removes the failure mode entirely.

_engine_stack: list[BacktestEngine] = []


def _push_engine(engine: BacktestEngine) -> None:
    _engine_stack.append(engine)


def _pop_engine() -> None:
    if _engine_stack:
        _engine_stack.pop()


def current_engine() -> BacktestEngine | None:
    return _engine_stack[-1] if _engine_stack else None
