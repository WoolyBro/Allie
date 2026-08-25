"""Tests for allie/quant/backtest_engine.py -- Phase 7.4/7.5.

The headline test (`test_single_known_trade_hand_computed`) engineers a
price series with exactly one crossover and one trade, and asserts every
metric against a value computed by hand below -- not against whatever
the engine happens to produce.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from allie.quant.backtest_engine import (
    BacktestEngine,
    WhenSignal,
    compute_metrics,
    run_backtest,
)
from allie.runtime.errors import AllieValueError
from allie.runtime.values import BoolSeriesValue, PriceSeriesValue, SeriesValue


def _prices(closes: list[float]) -> PriceSeriesValue:
    series = SeriesValue(closes)
    return PriceSeriesValue(
        open=series,
        high=series,
        low=series,
        close=series,
        volume=SeriesValue([1000.0] * len(closes)),
        dates=np.array([], dtype="datetime64[ns]"),
    )


def _signal(flags: list[bool]) -> BoolSeriesValue:
    return BoolSeriesValue(flags)


def test_single_known_trade_hand_computed() -> None:
    # 5 bars. Buy at bar 1 (close 100), sell at bar 3 (close 110).
    closes = [90.0, 100.0, 105.0, 110.0, 120.0]
    prices = _prices(closes)

    engine = BacktestEngine(prices, initial_cash=1000.0, periods_per_year=252)
    buy_signal = WhenSignal(signal=_signal([False, True, False, False, False]), action=engine.buy)
    sell_signal = WhenSignal(
        signal=_signal([False, False, False, True, False]), action=engine.sell
    )
    result = engine.run("Hand", "TEST", [buy_signal, sell_signal])

    # Hand-computed equity curve:
    #  bar 0: flat, cash=1000                      -> 1000
    #  bar 1: buy at 100 -> shares = 1000/100 = 10 -> 10*100 = 1000
    #  bar 2: long, 10 shares at 105               -> 1050
    #  bar 3: sell at 110 -> cash = 10*110 = 1100  -> 1100
    #  bar 4: flat, cash=1100                      -> 1100
    assert result.equity_curve == [1000.0, 1000.0, 1050.0, 1100.0, 1100.0]

    assert result.metrics.num_trades == 1
    trade = result.trades[0]
    assert (trade.entry_index, trade.entry_price) == (1, 100.0)
    assert (trade.exit_index, trade.exit_price) == (3, 110.0)
    assert trade.return_pct == pytest.approx(0.10)

    # returns = [0/1000, 50/1000, 50/1050, 0/1100]
    returns = [0.0, 0.05, 50.0 / 1050.0, 0.0]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance)

    m = result.metrics
    assert m.initial_equity == 1000.0
    assert m.final_equity == 1100.0
    # years = 5 bars / 252
    assert m.cagr == pytest.approx((1100.0 / 1000.0) ** (252.0 / 5.0) - 1.0)
    assert m.volatility == pytest.approx(std * math.sqrt(252))
    assert m.sharpe == pytest.approx((mean / std) * math.sqrt(252))
    # Equity never declines from a running peak here.
    assert m.max_drawdown == pytest.approx(0.0)


def test_max_drawdown_hand_computed() -> None:
    # Peak 120 at index 1, trough 90 at index 3 -> (120-90)/120 = 0.25.
    metrics = compute_metrics([100.0, 120.0, 100.0, 90.0, 110.0], num_trades=0)
    assert metrics.max_drawdown == pytest.approx(0.25)


def test_buy_while_already_long_is_a_noop() -> None:
    prices = _prices([100.0, 110.0, 120.0])
    engine = BacktestEngine(prices, initial_cash=1000.0)
    clause = WhenSignal(signal=_signal([True, True, False]), action=engine.buy)
    result = engine.run("S", "T", [clause])
    # Entered once at bar 0; the second buy must not re-enter at 110.
    assert result.equity_curve == [1000.0, 1100.0, 1200.0]
    assert result.metrics.num_trades == 0  # never exited


def test_sell_while_flat_is_a_noop() -> None:
    prices = _prices([100.0, 110.0])
    engine = BacktestEngine(prices, initial_cash=1000.0)
    clause = WhenSignal(signal=_signal([True, False]), action=engine.sell)
    result = engine.run("S", "T", [clause])
    assert result.equity_curve == [1000.0, 1000.0]
    assert result.metrics.num_trades == 0


def test_open_position_at_final_bar_is_marked_to_market_but_not_counted() -> None:
    prices = _prices([100.0, 200.0])
    engine = BacktestEngine(prices, initial_cash=1000.0)
    clause = WhenSignal(signal=_signal([True, False]), action=engine.buy)
    result = engine.run("S", "T", [clause])
    assert result.equity_curve[-1] == pytest.approx(2000.0)  # marked to market
    assert result.metrics.num_trades == 0  # not a completed round trip


def test_strategy_that_never_trades_reports_zeros_not_nan() -> None:
    prices = _prices([100.0, 110.0, 120.0])
    engine = BacktestEngine(prices, initial_cash=1000.0)
    clause = WhenSignal(signal=_signal([False, False, False]), action=engine.buy)
    m = engine.run("S", "T", [clause]).metrics
    assert m.final_equity == 1000.0
    assert m.volatility == 0.0
    assert m.sharpe == 0.0
    assert m.max_drawdown == 0.0
    assert m.num_trades == 0


def test_signal_length_mismatch_raises() -> None:
    prices = _prices([100.0, 110.0, 120.0])
    engine = BacktestEngine(prices)
    clause = WhenSignal(signal=_signal([True, False]), action=engine.buy)
    with pytest.raises(AllieValueError):
        engine.run("S", "T", [clause])


# ---- run_backtest slicing ----


def test_end_is_clamped_to_available_bars() -> None:
    prices = _prices([100.0, 110.0, 120.0])
    result = run_backtest("S", "T", prices, [], start=0, end=1000)
    assert len(result.equity_curve) == 3


def test_start_past_end_of_data_raises() -> None:
    prices = _prices([100.0, 110.0])
    with pytest.raises(AllieValueError):
        run_backtest("S", "T", prices, [], start=5, end=10)


def test_empty_range_raises() -> None:
    prices = _prices([100.0, 110.0, 120.0])
    with pytest.raises(AllieValueError):
        run_backtest("S", "T", prices, [], start=2, end=2)


def test_negative_start_raises() -> None:
    prices = _prices([100.0, 110.0])
    with pytest.raises(AllieValueError):
        run_backtest("S", "T", prices, [], start=-1, end=2)


def test_slicing_realigns_signals_with_prices() -> None:
    # The signal fires at *absolute* bar 1. Running bars 1..4 must line
    # that up with the window's bar 0 -- if prices were sliced but
    # signals weren't, the trade would land on the wrong bar (or vanish).
    prices = _prices([50.0, 100.0, 110.0, 120.0])
    fired_at: list[int] = []
    signals = [
        WhenSignal(
            signal=_signal([False, True, False, False]),
            action=lambda: fired_at.append(len(fired_at)),
        )
    ]
    result = run_backtest("S", "T", prices, signals, start=1, end=4)
    assert len(result.equity_curve) == 3
    assert len(fired_at) == 1  # fired exactly once, inside the window


def test_render_produces_a_table_with_every_metric() -> None:
    prices = _prices([100.0, 110.0, 120.0])
    result = run_backtest("MyStrat", "AAPL", prices, [], start=0, end=3)
    text = result.render()
    for label in (
        "BACKTEST RESULTS",
        "Strategy",
        "Symbol",
        "Initial Equity",
        "Final Equity",
        "CAGR",
        "Volatility",
        "Sharpe",
        "Max Drawdown",
        "Trades",
    ):
        assert label in text
    assert "MyStrat" in text
    assert "AAPL" in text
    assert "%" in text  # percentage formatting
