"""The look-ahead audit: look-ahead bias checked by experiment.

allie/semantic/lookahead.py proves from the source text that no trading
decision can see a later bar. This module checks the same property
independently, at run time, straight from its definition:

    A program looks ahead if a decision it makes at bar t changes when
    the bars after t are removed from its data.

`audit` runs the program on its data, then again for every t with each
`load(...)` cut to bars 0..t, and compares what every backtest decided at
the bars both runs had:

- whether each `when` condition held,
- the orders its actions placed, and their sizes,
- the trades that closed -- by a signal, a stop or a take-profit.

A difference anywhere is a decision that knew about a bar it shouldn't
have. So is a difference in *which* backtests run.

It is the static analysis's test oracle: tests/unit/test_lookahead_audit.py
generates programs and checks that every one the analysis accepts passes
the audit (and how many it rejects that pass too). `allie check --audit`
runs it on a file, which is also how to see whether a rejection is a real
leak on your data.

Limits, stated plainly. It sees only the data it's given: a program that
looks ahead only on other data passes, and data shifted before Allie read
it can't be told apart. It runs the program once per bar, on the
tree-walking interpreter. A backtest whose range depends on how many
bars there are (`from 0 to len(c) / 2`) covers different bars in the
shorter runs, and isn't compared there; nor is a run that fails, such as
one cut before a backtest's first bar.
"""

from __future__ import annotations

import contextlib
import io
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

import allie.stdlib as stdlib
from allie.ast.nodes import Program
from allie.quant.backtest_engine import BacktestRecord, Trade, observe_backtests
from allie.runtime.errors import AllieRuntimeError
from allie.runtime.interpreter import run_program
from allie.runtime.values import BuiltinFunction, PriceSeriesValue, SeriesValue
from allie.stdlib.registry import BUILTINS


@dataclass(frozen=True)
class Leak:
    """One decision that changed when the later bars were removed."""

    subject: str  # what was decided: "'when' clause 1", "orders placed", ...
    full: str  # the decision with all the bars
    cut: str  # ... and with only the first `kept`
    bars: int
    kept: int
    backtest: int = 0  # which one, from 1, in the order the program ran them; 0: the program
    strategy: str = ""
    bar: int = 0  # the bar the decision was made at, counted in the data

    def __str__(self) -> str:
        where = (
            f"backtest {self.backtest} ('{self.strategy}'), bar {self.bar} -- "
            if self.backtest
            else ""
        )
        return (
            f"{where}{self.subject}: {self.full} with all {self.bars} bars, "
            f"{self.cut} with only the first {self.kept}"
        )


@dataclass(frozen=True)
class AuditReport:
    bars: int  # in the data
    backtests: int  # the program runs, on all of it
    compared: int  # shorter runs compared with the full one
    skipped: int  # shorter runs that failed, so couldn't be
    leaks: tuple[Leak, ...]

    @property
    def clean(self) -> bool:
        return not self.leaks

    def render(self) -> str:
        if self.leaks:
            lines = [
                f"look-ahead audit: FAILED -- {len(self.leaks)} decision"
                f"{'s' if len(self.leaks) != 1 else ''} changed when later bars were removed"
            ]
            lines += [f"  {leak}" for leak in self.leaks]
            return "\n".join(lines)
        if not self.backtests:
            return "look-ahead audit: nothing to compare -- the program runs no backtest"
        skipped = f", {self.skipped} skipped (the shorter run failed)" if self.skipped else ""
        return (
            f"look-ahead audit: passed -- {self.backtests} backtest"
            f"{'s' if self.backtests != 1 else ''}, {self.compared} shorter runs"
            f"{skipped}: no decision changed when the bars after it were removed"
        )


def audit(
    program: Program, *, keep: Sequence[int] | None = None, first_leak: bool = False
) -> AuditReport:
    """Audits a type-checked `program`. `keep` is the bar counts to cut
    the data to (default: every one from 1 to all but one); `first_leak`
    stops at the first leak found. A runtime error in the full run is
    raised: there's nothing to compare with."""
    loader = _Loader()
    full = _run(program, loader, None)
    assert full is not None
    cuts = range(1, loader.bars) if keep is None else keep
    leaks: dict[tuple[int, str], Leak] = {}
    compared = skipped = 0
    for kept in cuts:
        short = _run(program, loader, kept)
        if short is None:
            skipped += 1
            continue
        compared += 1
        for leak in _compare(full, short, loader.bars, kept):
            leaks.setdefault((leak.backtest, leak.subject), leak)
        if first_leak and leaks:
            break
    ordered = sorted(leaks.values(), key=lambda leak: (leak.backtest, leak.bar, leak.subject))
    return AuditReport(loader.bars, len(full), compared, skipped, tuple(ordered))


# ---- running ----


class _Loader:
    """`load`, with the files read once and every series cut to the first
    `kept` bars on request."""

    def __init__(self) -> None:
        self._load = BUILTINS["load"][1]
        self._loaded: dict[str, PriceSeriesValue] = {}
        self.bars = 0

    def load(self, path: str, kept: int | None) -> PriceSeriesValue:
        prices = self._loaded.get(path)
        if prices is None:
            prices = self._loaded[path] = PriceSeriesValue(**vars(self._load(path)))
            self.bars = max(self.bars, len(prices.close))
        if kept is None:
            return prices
        window = slice(0, kept)
        return PriceSeriesValue(
            open=SeriesValue(prices.open.data[window]),
            high=SeriesValue(prices.high.data[window]),
            low=SeriesValue(prices.low.data[window]),
            close=SeriesValue(prices.close.data[window]),
            volume=SeriesValue(prices.volume.data[window]),
            dates=prices.dates[window] if len(prices.dates) else prices.dates,
        )


def _run(program: Program, loader: _Loader, kept: int | None) -> list[BacktestRecord] | None:
    """The program's backtests, run with its data cut to `kept` bars --
    None if that run fails. Output is discarded."""
    env = stdlib.create_global_environment()
    env.define("load", BuiltinFunction(name="load", fn=lambda path: loader.load(path, kept)))
    with observe_backtests() as records, contextlib.redirect_stdout(io.StringIO()):
        try:
            run_program(program, env)
        except (AllieRuntimeError, RecursionError):
            if kept is None:
                raise
            return None
    return records


# ---- comparing ----


def _compare(
    full: list[BacktestRecord], short: list[BacktestRecord], bars: int, kept: int
) -> list[Leak]:
    ran_full = [record.strategy_name for record in full]
    ran_short = [record.strategy_name for record in short]
    if ran_full != ran_short:
        return [
            Leak(
                "the backtests run",
                ", ".join(ran_full) or "none",
                ", ".join(ran_short) or "none",
                bars,
                kept,
            )
        ]
    leaks: list[Leak] = []
    for number, (f, s) in enumerate(zip(full, short, strict=True), start=1):
        if f.start != s.start or f.symbol != s.symbol:
            continue  # it covers other bars in the shorter run
        for subject, bar, a, b in _differences(f, s, kept):
            leaks.append(Leak(subject, a, b, bars, kept, number, f.strategy_name, bar))
    return leaks


def _differences(
    f: BacktestRecord, s: BacktestRecord, kept: int
) -> Iterable[tuple[str, int, str, str]]:
    """(what, at which bar, with all the bars, with `kept`) for each kind
    of decision that differs: the first difference of each."""
    # The last bar both runs decided at. A backtest's end can move with the
    # bar count (`to floor(len(c) * 0.7)`): only the bars both covered count.
    last = min(kept, f.end, s.end) - 1
    for clause, (a, b) in enumerate(zip(f.signals, s.signals, strict=True), start=1):
        differs = np.flatnonzero(a[f.start : last + 1] != b[f.start : last + 1])
        if differs.size:
            bar = f.start + int(differs[0])
            yield f"'when' clause {clause}", bar, _truth(a[bar]), _truth(b[bar])

    upto = last - f.start  # in the backtest's own bars, as orders and trades count them
    orders_full = _by_bar((bar, x) for bar, x in f.orders if bar <= upto)
    orders_short = _by_bar((bar, x) for bar, x in s.orders if bar <= upto)
    for bar in sorted(orders_full.keys() | orders_short.keys()):
        a_orders, b_orders = orders_full.get(bar, []), orders_short.get(bar, [])
        if not _same_orders(a_orders, b_orders):
            yield "orders placed", f.start + bar, _orders(a_orders), _orders(b_orders)
            break

    trades_full = _by_bar((t.exit_index, t) for t in f.result.trades if t.exit_index <= upto)
    trades_short = _by_bar((t.exit_index, t) for t in s.result.trades if t.exit_index <= upto)
    for bar in sorted(trades_full.keys() | trades_short.keys()):
        a_trades, b_trades = trades_full.get(bar, []), trades_short.get(bar, [])
        if not _same_trades(a_trades, b_trades):
            yield (
                "trades closed",
                f.start + bar,
                _trades(a_trades, f.start),
                _trades(b_trades, f.start),
            )
            break


_T = TypeVar("_T")


def _by_bar(items: Iterable[tuple[int, _T]]) -> dict[int, list[_T]]:
    grouped: dict[int, list[_T]] = {}
    for bar, item in items:
        grouped.setdefault(bar, []).append(item)
    return grouped


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))


def _same_orders(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))


def _same_trades(a: list[Trade], b: list[Trade]) -> bool:
    return len(a) == len(b) and all(
        (x.entry_index, x.direction, x.exit_reason) == (y.entry_index, y.direction, y.exit_reason)
        and _close(x.entry_price, y.entry_price)
        and _close(x.exit_price, y.exit_price)
        for x, y in zip(a, b, strict=True)
    )


def _truth(value: object) -> str:
    return "true" if bool(value) else "false"


def _orders(fractions: list[float]) -> str:
    return ", ".join(f"target {fraction:g}" for fraction in fractions) or "none"


def _trades(trades: list[Trade], start: int) -> str:
    return (
        ", ".join(
            f"{'long' if t.direction > 0 else 'short'} from bar {start + t.entry_index}"
            f" ({t.exit_reason})"
            for t in trades
        )
        or "none"
    )


__all__ = ["AuditReport", "Leak", "audit"]
