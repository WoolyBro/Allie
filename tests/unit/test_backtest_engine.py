"""Tests for allie/quant/backtest_engine.py.

Every number here is computed by hand in the test -- from the
definitions in the engine's docstring -- not copied from whatever the
engine happens to produce.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from allie.quant.backtest_engine import (
    OPTIONS,
    BacktestConfig,
    BacktestEngine,
    WhenSignal,
    compute_metrics,
    config_from_options,
    current_engine,
    observe_backtests,
    option_error,
    run_backtest,
)
from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, PriceSeriesValue, SeriesValue
from allie.typechecker.checker import _BACKTEST_RESULT_FIELDS
from allie.typechecker.types import INT, SeriesType

# Frictionless, filling at the signal bar's close: the old v1 model, and
# the easiest to compute by hand.
_AT_CLOSE = BacktestConfig(fill="close", slippage=0.0, cash=1000.0)


def _prices(
    closes: list[float],
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> PriceSeriesValue:
    return PriceSeriesValue(
        open=SeriesValue(opens if opens is not None else closes),
        high=SeriesValue(highs if highs is not None else closes),
        low=SeriesValue(lows if lows is not None else closes),
        close=SeriesValue(closes),
        volume=SeriesValue([1000.0] * len(closes)),
        dates=np.array([], dtype="datetime64[ns]"),
    )


def _on(n: int, *bars: int) -> BoolSeriesValue:
    return BoolSeriesValue([i in bars for i in range(n)])


def _run(
    prices: PriceSeriesValue,
    config: BacktestConfig,
    *orders: tuple[str, tuple[int, ...]],
):  # noqa: ANN202
    """Runs `orders` -- (method, bars it fires on) -- in clause order."""
    engine = BacktestEngine(prices, config)
    n = len(prices.close)
    signals = [WhenSignal(_on(n, *bars), getattr(engine, method)) for method, bars in orders]
    return engine.run("S", "T", signals)


# ---- the v1 model: one trade, fully hand-computed ----


def test_single_known_trade_hand_computed() -> None:
    # 5 bars. Buy at bar 1 (close 100), sell at bar 3 (close 110).
    result = _run(
        _prices([90.0, 100.0, 105.0, 110.0, 120.0]), _AT_CLOSE, ("buy", (1,)), ("sell", (3,))
    )
    #  bar 0: flat, cash=1000                      -> 1000
    #  bar 1: buy at 100 -> shares = 1000/100 = 10 -> 10*100 = 1000
    #  bar 2: long, 10 shares at 105               -> 1050
    #  bar 3: sell at 110 -> cash = 10*110 = 1100  -> 1100
    #  bar 4: flat, cash=1100                      -> 1100
    assert result.equity_curve == [1000.0, 1000.0, 1050.0, 1100.0, 1100.0]
    assert result.positions == [0.0, 1.0, 1.0, 0.0, 0.0]

    (trade,) = result.trades
    assert (trade.entry_index, trade.entry_price) == (1, 100.0)
    assert (trade.exit_index, trade.exit_price) == (3, 110.0)
    assert trade.pnl == pytest.approx(100.0)
    assert trade.return_pct == pytest.approx(0.10)

    returns = [0.0, 0.05, 50.0 / 1050.0, 0.0]
    mean = sum(returns) / len(returns)
    std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))
    m = result.metrics
    assert (m.initial_equity, m.final_equity) == (1000.0, 1100.0)
    assert m.total_return == pytest.approx(0.10)
    # 5 bars span 4 periods: years = 4 / 252.
    assert m.cagr == pytest.approx(1.1 ** (252.0 / 4.0) - 1.0)
    assert m.volatility == pytest.approx(std * math.sqrt(252))
    assert m.sharpe == pytest.approx(mean / std * math.sqrt(252))
    assert m.sortino == math.inf  # no negative return at all
    assert m.max_drawdown == pytest.approx(0.0)
    assert result.exposure == pytest.approx(2 / 5)
    assert result.buy_hold_return == pytest.approx(120.0 / 90.0 - 1.0)
    assert (result.win_rate, result.profit_factor) == (1.0, math.inf)


def test_max_drawdown_hand_computed() -> None:
    # Peak 120 at index 1, trough 90 at index 3 -> (120-90)/120 = 0.25.
    metrics = compute_metrics([100.0, 120.0, 100.0, 90.0, 110.0])
    assert metrics.max_drawdown == pytest.approx(0.25)


def test_sortino_divides_by_downside_deviation() -> None:
    equity = [100.0, 110.0, 99.0, 108.9]  # returns +10%, -10%, +10%
    returns = [0.1, -0.1, 0.1]
    downside = math.sqrt((0.1**2) / 3)  # root mean square of min(r, 0)
    expected = (sum(returns) / 3) / downside * math.sqrt(252)
    assert compute_metrics(equity).sortino == pytest.approx(expected)


# ---- when orders fill ----


def test_by_default_an_order_fills_at_the_next_bars_open() -> None:
    # A signal at bar 1 is computed from close[1]; the earliest anyone
    # could trade on it is open[2].
    closes = [100.0, 100.0, 104.0, 108.0, 110.0]
    opens = [100.0, 100.0, 102.0, 106.0, 109.0]
    config = BacktestConfig(slippage=0.0, cash=1000.0)
    result = _run(_prices(closes, opens), config, ("buy", (1,)), ("sell", (3,)))
    shares = 1000.0 / 102.0  # bought at open[2]
    assert result.equity_curve == pytest.approx(
        [1000.0, 1000.0, shares * 104.0, shares * 108.0, shares * 109.0]  # sold at open[4]
    )
    (trade,) = result.trades
    assert (trade.entry_index, trade.entry_price) == (2, 102.0)
    assert (trade.exit_index, trade.exit_price) == (4, 109.0)


def test_an_order_placed_on_the_last_bar_never_fills() -> None:
    result = _run(_prices([100.0, 110.0]), BacktestConfig(cash=1000.0), ("buy", (1,)))
    assert result.equity_curve == [1000.0, 1000.0]
    assert result.costs == 0.0


def test_the_last_order_placed_in_a_bar_is_the_one_that_fills() -> None:
    # buy() then sell() at the same bar nets out to nothing -- not a
    # round trip that pays costs for no position.
    prices = _prices([100.0, 101.0, 102.0])
    for config in (BacktestConfig(cash=1000.0), _AT_CLOSE):
        result = _run(prices, config, ("buy", (0,)), ("sell", (0,)))
        assert (result.trades, result.costs, result.exposure) == ([], 0.0, 0.0)


def test_an_order_waits_for_a_bar_with_a_price() -> None:
    opens = [100.0, math.nan, 0.0, 105.0]
    result = _run(
        _prices([100.0, 101.0, 102.0, 106.0], opens),
        BacktestConfig(slippage=0.0, cash=1000.0),
        ("buy", (0,)),
    )
    # open[1] is missing and open[2] isn't a price: it fills at open[3].
    assert result.positions[:3] == [0.0, 0.0, 0.0]
    assert result.equity_curve[3] == pytest.approx(1000.0 / 105.0 * 106.0)


def test_a_missing_close_carries_the_last_mark_forward() -> None:
    result = _run(_prices([100.0, math.nan, 120.0]), _AT_CLOSE, ("buy", (0,)))
    assert result.equity_curve == pytest.approx([1000.0, 1000.0, 1200.0])


# ---- costs ----


def test_slippage_and_commission_hand_computed() -> None:
    config = BacktestConfig(fill="close", slippage=0.01, commission=0.001, cash=1000.0)
    result = _run(_prices([100.0, 110.0, 110.0]), config, ("buy", (0,)), ("sell", (1,)))
    buy_price = 100.0 * 1.01  # slippage against you
    shares = 1000.0 / (buy_price * 1.001)  # spends exactly the cash, commission included
    sell_price = 110.0 * 0.99
    cash = shares * sell_price * (1 - 0.001)
    assert result.equity_curve == pytest.approx([shares * 100.0, cash, cash])
    (trade,) = result.trades
    assert (trade.entry_price, trade.exit_price) == pytest.approx((buy_price, sell_price))
    assert trade.pnl == pytest.approx(cash - 1000.0)
    slippage_paid = shares * 100.0 * 0.01 + shares * 110.0 * 0.01
    commission_paid = 0.001 * (shares * buy_price + shares * sell_price)
    assert result.costs == pytest.approx(slippage_paid + commission_paid)


def test_costs_are_on_by_default() -> None:
    prices = _prices([100.0, 100.0, 110.0, 110.0])
    free = _run(prices, BacktestConfig(slippage=0.0), ("buy", (0,)), ("sell", (2,)))
    default = _run(prices, BacktestConfig(), ("buy", (0,)), ("sell", (2,)))
    assert BacktestConfig().slippage == 0.0005
    assert default.costs > 0.0
    assert default.metrics.final_equity < free.metrics.final_equity


# ---- positions ----


def test_buy_while_already_long_is_a_noop() -> None:
    result = _run(_prices([100.0, 110.0, 120.0]), _AT_CLOSE, ("buy", (0, 1)))
    assert result.equity_curve == [1000.0, 1100.0, 1200.0]
    assert result.metrics.num_trades == 0  # never exited


@pytest.mark.parametrize("method", ["sell", "cover"])
def test_closing_while_flat_is_a_noop(method: str) -> None:
    result = _run(_prices([100.0, 110.0]), _AT_CLOSE, (method, (0,)))
    assert result.equity_curve == [1000.0, 1000.0]


def test_a_short_profits_when_the_price_falls() -> None:
    result = _run(_prices([100.0, 95.0, 90.0, 90.0]), _AT_CLOSE, ("short", (0,)), ("cover", (2,)))
    # Short 10 shares at 100: cash 2000, shares -10. Equity = 2000 - 10 * price.
    assert result.equity_curve == [1000.0, 1050.0, 1100.0, 1100.0]
    assert result.positions[:2] == pytest.approx([-1.0, -950.0 / 1050.0])
    (trade,) = result.trades
    assert (trade.direction, trade.pnl, trade.return_pct) == (-1, pytest.approx(100.0), 0.1)


def test_short_while_short_is_a_noop() -> None:
    result = _run(_prices([100.0, 90.0, 80.0]), _AT_CLOSE, ("short", (0, 1)))
    assert result.equity_curve == [1000.0, 1100.0, 1200.0]


def test_a_reversal_closes_the_long_and_opens_a_short() -> None:
    result = _run(_prices([100.0, 110.0, 100.0]), _AT_CLOSE, ("buy", (0,)), ("short", (1,)))
    (trade,) = result.trades  # the long; the short is still open
    assert (trade.direction, trade.exit_index, trade.pnl) == (1, 1, pytest.approx(100.0))
    # Short 1100/110 = 10 shares at 110; at 100 equity is 1100 + 10 * 10.
    assert result.equity_curve == pytest.approx([1000.0, 1100.0, 1200.0])
    assert result.final_position == "short"


def test_target_holds_a_fraction_of_equity() -> None:
    engine = BacktestEngine(_prices([100.0, 120.0, 120.0]), _AT_CLOSE)
    signals = [
        WhenSignal(_on(3, 0), lambda: engine.target(0.5)),
        WhenSignal(_on(3, 2), lambda: engine.target(0.0)),
    ]
    result = engine.run("S", "T", signals)
    # 5 shares at 100, 500 cash. At 120: 600 in shares, equity 1100.
    assert result.equity_curve == pytest.approx([1000.0, 1100.0, 1100.0])
    assert result.positions == pytest.approx([0.5, 600.0 / 1100.0, 0.0])
    (trade,) = result.trades
    assert trade.pnl == pytest.approx(100.0)
    assert trade.return_pct == pytest.approx(0.2)  # on the 500 invested


def test_target_rejects_a_non_finite_fraction() -> None:
    engine = BacktestEngine(_prices([100.0]), _AT_CLOSE)
    with pytest.raises(AllieValueError, match="target: the fraction must be a finite number"):
        engine.target(math.inf)


def test_a_wiped_out_account_is_closed_out_and_stops_trading() -> None:
    # Short at 100, the price triples: equity 2000 - 10 * 300 < 0.
    result = _run(_prices([100.0, 300.0, 50.0, 50.0]), _AT_CLOSE, ("short", (0, 2)))
    assert result.ruined_at == 1
    assert result.equity_curve[1:] == [0.0, 0.0, 0.0]  # and no new short at bar 2
    assert result.trades[0].exit_reason == "ruin"
    assert "Ruined at Bar" in result.render()


# ---- stops ----


def test_a_stop_loss_exits_at_the_stop_inside_the_bar() -> None:
    config = BacktestConfig(fill="close", slippage=0.0, cash=1000.0, stop_loss=0.05)
    result = _run(_prices([100.0, 97.0, 96.0], lows=[100.0, 94.0, 96.0]), config, ("buy", (0,)))
    (trade,) = result.trades
    assert (trade.exit_index, trade.exit_price, trade.exit_reason) == (1, 95.0, "stop_loss")
    assert result.equity_curve == pytest.approx([1000.0, 950.0, 950.0])


def test_a_gap_through_the_stop_exits_at_the_open() -> None:
    config = BacktestConfig(fill="close", slippage=0.0, cash=1000.0, stop_loss=0.05)
    result = _run(
        _prices([100.0, 91.0], opens=[100.0, 90.0], lows=[100.0, 89.0]), config, ("buy", (0,))
    )
    assert result.trades[0].exit_price == 90.0  # no fill at 95: it was never offered


def test_a_take_profit_exits_at_its_level() -> None:
    config = BacktestConfig(fill="close", slippage=0.0, cash=1000.0, take_profit=0.10)
    result = _run(_prices([100.0, 105.0], highs=[100.0, 111.0]), config, ("buy", (0,)))
    trade = result.trades[0]
    assert (trade.exit_price, trade.exit_reason) == (pytest.approx(110.0), "take_profit")


def test_a_bar_that_reaches_both_levels_is_read_as_the_stop() -> None:
    config = BacktestConfig(
        fill="close", slippage=0.0, cash=1000.0, stop_loss=0.05, take_profit=0.05
    )
    result = _run(
        _prices([100.0, 100.0], highs=[100.0, 106.0], lows=[100.0, 94.0]),
        config,
        ("buy", (0,)),
    )
    assert result.trades[0].exit_reason == "stop_loss"


def test_a_short_stop_is_above_the_entry() -> None:
    config = BacktestConfig(fill="close", slippage=0.0, cash=1000.0, stop_loss=0.05)
    result = _run(_prices([100.0, 103.0], highs=[100.0, 106.0]), config, ("short", (0,)))
    trade = result.trades[0]
    assert (trade.exit_price, trade.pnl) == (105.0, pytest.approx(-50.0))


def test_a_next_open_entry_is_protected_from_its_first_bar() -> None:
    config = BacktestConfig(slippage=0.0, cash=1000.0, stop_loss=0.05)
    result = _run(
        _prices([100.0, 96.0], opens=[100.0, 100.0], lows=[100.0, 94.0]), config, ("buy", (0,))
    )
    trade = result.trades[0]
    assert (trade.entry_index, trade.exit_index, trade.exit_price) == (1, 1, 95.0)


# ---- trade statistics ----


def test_trade_statistics_hand_computed() -> None:
    # Three round trips: +10, -5, +10 (frictionless, at the close).
    closes = [100.0, 110.0, 100.0, 95.0, 100.0, 110.0]
    result = _run(_prices(closes), _AT_CLOSE, ("buy", (0, 2, 4)), ("sell", (1, 3, 5)))
    returns = [t.return_pct for t in result.trades]
    assert returns == pytest.approx([0.10, -0.05, 0.10])
    assert result.win_rate == pytest.approx(2 / 3)
    gains = 100.0 + 0.10 * 1045.0
    losses = 0.05 * 1100.0
    assert result.profit_factor == pytest.approx(gains / losses)
    assert result.avg_trade_return == pytest.approx(sum(returns) / 3)


def test_a_strategy_that_never_trades_reports_zeros_and_no_trade_statistics() -> None:
    result = _run(_prices([100.0, 110.0, 120.0]), _AT_CLOSE)
    m = result.metrics
    assert (m.final_equity, m.volatility, m.sharpe, m.sortino, m.max_drawdown) == (
        1000.0, 0.0, 0.0, 0.0, 0.0,
    )  # fmt: skip
    assert m.num_trades == 0
    assert math.isnan(result.win_rate) and math.isnan(result.profit_factor)
    assert "Win Rate                               -" in result.render()


def test_an_open_position_at_the_end_is_marked_to_market_but_not_counted() -> None:
    result = _run(_prices([100.0, 200.0]), _AT_CLOSE, ("buy", (0,)))
    assert result.equity_curve[-1] == pytest.approx(2000.0)
    assert result.metrics.num_trades == 0
    assert result.final_position == "long"


# ---- run_backtest: windows ----


def test_end_is_clamped_to_available_bars() -> None:
    result = run_backtest("S", "T", _prices([100.0, 110.0, 120.0]), [], start=0, end=1000)
    assert len(result.equity_curve) == 3


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (5, 10, "start bar 5 is past the last bar"),
        (2, 2, "empty bar range"),
        (-1, 2, "start bar -1 is negative"),
    ],
)
def test_bad_windows_raise(start: int, end: int, message: str) -> None:
    with pytest.raises(AllieValueError, match=message):
        run_backtest("S", "T", _prices([100.0, 110.0, 120.0]), [], start=start, end=end)


def test_a_signal_length_mismatch_raises() -> None:
    clause = WhenSignal(BoolSeriesValue([True, False]), lambda: None)
    with pytest.raises(AllieValueError, match="has 2 bars, but T has 3"):
        run_backtest("S", "T", _prices([100.0, 110.0, 120.0]), [clause], start=0, end=3)


def test_slicing_realigns_signals_with_prices() -> None:
    # The signal fires at *absolute* bar 1; running bars 1..4 must line
    # that up with the window's bar 0.
    fired: list[int] = []
    signals = [WhenSignal(_on(4, 1), lambda: fired.append(1))]
    result = run_backtest("S", "T", _prices([50.0, 100.0, 110.0, 120.0]), signals, 1, 4)
    assert len(result.equity_curve) == 3
    assert fired == [1]


# ---- configuration ----


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("slippage", 1.5, "'slippage' is a fraction of the price from 0 up to 1"),
        ("commission", -0.001, "'commission' is a fraction of the value traded"),
        ("cash", 0.0, "'cash' must be positive"),
        ("fill", "midday", '\'fill\' is "open" (the next bar\'s open) or "close"'),
        ("stop_loss", 1.0, "'stop_loss' is a fraction of the entry price between 0 and 1"),
        ("take_profit", 0.0, "'take_profit' is a positive fraction"),
        ("periods_per_year", 0, "'periods_per_year' must be at least 1"),
        ("slippage", math.nan, "'slippage' must be a number, got nan"),
        ("leverage", 2.0, "unknown backtest option 'leverage'"),
    ],
)
def test_invalid_options_are_explained(name: str, value: object, message: str) -> None:
    assert message in str(option_error(name, value))
    with pytest.raises(AllieValueError, match="backtest: "):
        config_from_options({name: value})


def test_valid_options_build_the_config() -> None:
    config = config_from_options({"fill": "close", "stop_loss": 0.05, "periods_per_year": 52})
    assert config == BacktestConfig(fill="close", stop_loss=0.05, periods_per_year=52)
    assert all(option_error(n, getattr(BacktestConfig(), n)) is None for n in ("cash", "fill"))
    assert set(OPTIONS) <= set(BacktestConfig.__dataclass_fields__)


def test_render_states_its_assumptions() -> None:
    config = BacktestConfig(commission=0.001, stop_loss=0.05)
    text = run_backtest("MyStrat", "AAPL", _prices([1.0, 2.0]), [], 0, 2, config=config).render()
    for row in (
        "MyStrat",
        "Fills                      next bar open",
        "Slippage                         5.0 bps",
        "Commission                      10.0 bps",
        "Stop Loss                          5.00%",
        "Sharpe",
        "Max Drawdown",
        "Profit Factor",
        "Costs Paid",
    ):
        assert row in text


def test_the_period_comes_from_the_dates() -> None:
    prices = _prices([1.0, 2.0, 3.0])
    prices.dates = np.array(["2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[ns]")
    result = run_backtest("S", "T", prices, [], 1, 3)
    assert result.period == ("2024-01-03", "2024-01-04")


def test_every_field_the_type_checker_knows_is_served_with_its_type() -> None:
    result = _run(_prices([100.0, 110.0, 105.0]), _AT_CLOSE, ("buy", (0,)), ("sell", (1,)))
    for name, field_type in _BACKTEST_RESULT_FIELDS.items():
        value = result.field(name)
        if isinstance(field_type, SeriesType):
            assert isinstance(value, SeriesValue), name
        elif field_type == INT:
            assert isinstance(value, int), name
        else:
            assert isinstance(value, float), name
    assert result.field("trade_returns").data.tolist() == pytest.approx([0.1])  # type: ignore[attr-defined]
    with pytest.raises(AllieValueError, match="no field 'sharp'"):
        result.field("sharp")


def test_a_gap_through_the_take_profit_exits_at_the_better_open() -> None:
    config = BacktestConfig(fill="close", slippage=0.0, cash=1000.0, take_profit=0.10)
    result = _run(
        _prices([100.0, 116.0], opens=[100.0, 115.0], highs=[100.0, 117.0]),
        config,
        ("buy", (0,)),
    )
    assert result.trades[0].exit_price == 115.0  # not the 110 level: the open was better


def test_a_target_already_held_places_no_trade() -> None:
    # At an unchanged price, target(0.5) twice is one fill, not two.
    engine = BacktestEngine(_prices([100.0, 100.0, 100.0]), BacktestConfig(fill="close"))
    signals = [WhenSignal(_on(3, 0, 1), lambda: engine.target(0.5))]
    once = engine.run("S", "T", signals).costs
    engine = BacktestEngine(_prices([100.0, 100.0, 100.0]), BacktestConfig(fill="close"))
    signals = [WhenSignal(_on(3, 0), lambda: engine.target(0.5))]
    twice = engine.run("S", "T", signals).costs
    assert once == pytest.approx(twice)
    assert once > 0.0


def test_after_a_fill_the_position_is_exactly_the_target_costs_included() -> None:
    config = BacktestConfig(fill="close", slippage=0.001, commission=0.002)
    for fraction in (1.0, 0.5, -1.0, -0.25):
        engine = BacktestEngine(_prices([100.0, 100.0]), config)
        signals = [WhenSignal(_on(2, 0), lambda e=engine, f=fraction: e.target(f))]
        result = engine.run("S", "T", signals)
        assert result.positions[0] == pytest.approx(fraction), fraction
    # All-in long spends exactly the cash: none left, none borrowed.
    engine = BacktestEngine(_prices([100.0, 100.0]), config)
    engine.run("S", "T", [WhenSignal(_on(2, 0), engine.buy)])
    assert engine.cash == pytest.approx(0.0, abs=1e-9)


def _target(fraction: float) -> Callable[[], None]:
    def order() -> None:
        engine = current_engine()
        assert engine is not None
        engine.target(fraction)

    return order


def test_every_order_placed_is_recorded_and_observers_see_each_backtest() -> None:
    signal = _on(5, 1, 3)
    clauses = [WhenSignal(signal, _target(0.5)), WhenSignal(_on(5, 3), _target(1.0))]
    with observe_backtests() as records:
        run_backtest("S", "X", _prices([10.0] * 5), clauses, 1, 5, config=_AT_CLOSE)
    (record,) = records
    assert (record.strategy_name, record.start, record.end) == ("S", 1, 5)
    # Bars count from the backtest's start; both orders at bar 3 are kept.
    assert record.orders == ((0, 0.5), (2, 0.5), (2, 1.0))
    assert record.signals[0] is signal.data  # the whole series, not the window
    run_backtest("S", "X", _prices([10.0] * 5), clauses, 0, 5, config=_AT_CLOSE)
    assert len(records) == 1  # observing ended with the `with`
