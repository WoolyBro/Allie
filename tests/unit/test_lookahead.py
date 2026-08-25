"""Tests for allie/semantic/lookahead.py -- look-ahead bias as a
compile-time guarantee.

The motivating case, measured before the pass existed: a strategy using
`shift(close, -1)` (tomorrow's price, available today) compiled fine and
backtested at Sharpe 3.34 / CAGR +41.56%, against the honest version's
-0.16 / -2.89%. Nothing anywhere indicated the number was meaningless.
"""

from __future__ import annotations

import pytest

import allie.stdlib as stdlib
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.semantic.lookahead import TIME_SHIFTING_FUNCTIONS, check_lookahead
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker

_PRELUDE = """
let AAPL = load("tests/fixtures/AAPL_sample.csv");
let slow: Series[Float] = SMA(AAPL.close, 20);
"""


def _violations(body: str) -> list[str]:
    """Runs the full static pipeline, returning look-ahead messages only."""
    program = Parser(Lexer(_PRELUDE + body).scan_tokens()).parse()
    Resolver().resolve(program)
    checker = TypeChecker()
    for name, signature in stdlib.builtin_function_types().items():
        checker.env.define(name, signature)
    checker.check(program)

    bag = check_lookahead(program)
    return [d.message for d in bag.diagnostics if "look-ahead" in d.message]


# ---- the sink rejects future-tainted operands ----


def test_negative_shift_used_as_a_signal_is_rejected() -> None:
    found = _violations(
        """
        let tomorrow: Series[Float] = shift(AAPL.close, -1);
        strategy Clairvoyant { when tomorrow crosses_above slow: buy(); }
        """
    )
    assert len(found) == 1
    assert "future bar" in found[0]
    assert "tomorrow" in found[0]  # names the offending variable


def test_negative_shift_inline_in_the_when_clause_is_rejected() -> None:
    found = _violations(
        "strategy S { when shift(AAPL.close, -1) crosses_above slow: buy(); }"
    )
    assert len(found) == 1


def test_taint_propagates_through_an_intervening_indicator() -> None:
    """Smoothing future data does not make it past data -- SMA over a
    forward-shifted series still embeds later bars."""
    found = _violations(
        """
        let peek: Series[Float] = shift(AAPL.close, -1);
        let smoothed: Series[Float] = SMA(peek, 5);
        strategy S { when smoothed crosses_above slow: buy(); }
        """
    )
    assert len(found) == 1


def test_taint_propagates_through_a_user_function() -> None:
    found = _violations(
        """
        func cheat(s: Series[Float]) -> Series[Float] { return shift(s, -1); }
        let peek: Series[Float] = cheat(AAPL.close);
        strategy S { when peek crosses_above slow: buy(); }
        """
    )
    assert len(found) == 1


def test_taint_propagates_through_reassignment() -> None:
    found = _violations(
        """
        let s: Series[Float] = SMA(AAPL.close, 5);
        s = shift(AAPL.close, -1);
        strategy S { when s crosses_above slow: buy(); }
        """
    )
    assert len(found) == 1


def test_both_operands_are_checked() -> None:
    found = _violations(
        """
        let peek: Series[Float] = shift(AAPL.close, -1);
        strategy S { when slow crosses_above peek: buy(); }
        """
    )
    assert len(found) == 1
    assert "right operand" in found[0]


def test_every_clause_is_reported_not_just_the_first() -> None:
    """Collect-and-report-all, like the parser and resolver -- fixing one
    clause should not reveal the next one only on the next compile."""
    found = _violations(
        """
        let peek: Series[Float] = shift(AAPL.close, -1);
        strategy S {
            when peek crosses_above slow: buy();
            when peek crosses_below slow: sell();
        }
        """
    )
    assert len(found) == 2


# ---- honest programs are unaffected ----


def test_honest_crossover_is_accepted() -> None:
    assert (
        _violations(
            """
            let fast: Series[Float] = SMA(AAPL.close, 5);
            strategy S { when fast crosses_above slow: buy(); }
            """
        )
        == []
    )


def test_positive_shift_is_accepted() -> None:
    """`shift(s, 1)` is *lagging* -- yesterday's value today. That is
    the safe direction and a completely normal thing to want."""
    assert (
        _violations(
            """
            let lagged: Series[Float] = shift(AAPL.close, 1);
            strategy S { when lagged crosses_above slow: buy(); }
            """
        )
        == []
    )


def test_zero_shift_is_accepted() -> None:
    assert (
        _violations(
            """
            let same: Series[Float] = shift(AAPL.close, 0);
            strategy S { when same crosses_above slow: buy(); }
            """
        )
        == []
    )


def test_future_data_outside_a_when_clause_stays_legal() -> None:
    """Look-ahead bias is *deciding* on future data, not touching it.
    Post-hoc analysis is legitimate and must not be blocked."""
    assert (
        _violations(
            """
            let peek: Series[Float] = shift(AAPL.close, -1);
            print(mean(peek));
            """
        )
        == []
    )


def test_a_clean_series_is_not_tainted_by_a_sibling() -> None:
    """Taint is per-value, not per-program: declaring a tainted series
    must not poison an unrelated clean one."""
    assert (
        _violations(
            """
            let peek: Series[Float] = shift(AAPL.close, -1);
            let fast: Series[Float] = SMA(AAPL.close, 5);
            strategy S { when fast crosses_above slow: buy(); }
            """
        )
        == []
    )


# ---- soundness: unprovable means rejected ----


def test_non_literal_shift_period_is_rejected_as_unprovable() -> None:
    """The pass cannot prove `shift(s, k)` is non-negative for a variable
    `k`, so it rejects. False positives are acceptable for a safety
    property; false negatives are silently wrong backtests."""
    found = _violations(
        """
        let k: Int = 0 - 1;
        let peek: Series[Float] = shift(AAPL.close, k);
        strategy S { when peek crosses_above slow: buy(); }
        """
    )
    assert len(found) == 1


def test_the_shifting_function_table_covers_the_stdlib() -> None:
    """Guards the assumption the whole analysis rests on: `shift` is the
    only registered builtin that can move data backwards in time. A new
    forward-looking function added without updating
    TIME_SHIFTING_FUNCTIONS would silently create an unchecked hole.
    """
    suspicious = {
        name
        for name in stdlib.BUILTINS
        if any(word in name.lower() for word in ("shift", "lead", "future", "ahead", "next"))
    }
    assert suspicious <= set(TIME_SHIFTING_FUNCTIONS), (
        f"builtins that may move data backwards in time but are not declared "
        f"in TIME_SHIFTING_FUNCTIONS: {sorted(suspicious - set(TIME_SHIFTING_FUNCTIONS))}"
    )


@pytest.mark.parametrize("period", ["-1", "-5", "-100"])
def test_any_negative_literal_is_rejected(period: str) -> None:
    found = _violations(
        f"""
        let peek: Series[Float] = shift(AAPL.close, {period});
        strategy S {{ when peek crosses_above slow: buy(); }}
        """
    )
    assert len(found) == 1
