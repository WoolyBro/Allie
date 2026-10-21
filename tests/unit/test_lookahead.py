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
from allie.errors import Severity
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.semantic.lookahead import (
    POINT_IN_TIME_FUNCTIONS,
    TIME_SHIFTING_FUNCTIONS,
    WHOLE_HISTORY_FUNCTIONS,
    check_lookahead,
)
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
    found = _violations("strategy S { when shift(AAPL.close, -1) crosses_above slow: buy(); }")
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
    assert "'peek' is derived from a backwards time shift" in found[0]


def _culprit_spans(body: str) -> list[tuple[str, int]]:
    """(message subject, column) per look-ahead error."""
    program = Parser(Lexer(_PRELUDE + body).scan_tokens()).parse()
    Resolver().resolve(program)
    checker = TypeChecker()
    for name, signature in stdlib.builtin_function_types().items():
        checker.env.define(name, signature)
    checker.check(program)
    return [
        (d.message.split("at that bar. ")[1].split(" ")[0], d.span.column)
        for d in check_lookahead(program).diagnostics
        if d.severity is Severity.ERROR
    ]


def test_a_compound_condition_points_at_the_part_that_sees_the_future() -> None:
    source = (
        "let tomorrow: Series[Float] = shift(AAPL.close, -1);\n"
        "strategy S { when AAPL.close > slow and tomorrow > AAPL.close: buy(); }\n"
    )
    column = source.splitlines()[1].index("tomorrow") + 1
    assert _culprit_spans(source) == [("'tomorrow'", column)]


def test_a_shift_inside_an_indicator_is_blamed_not_its_input() -> None:
    # Used to say "'c' is derived from a backwards time shift" -- naming
    # the honest input instead of the shift applied to it.
    line = "strategy S { when RSI(shift(AAPL.close, -2), 14) crosses_below 30.0: buy(); }"
    assert _culprit_spans(line) == [("'shift(...)'", line.index("shift") + 1)]


def test_each_leaking_part_of_a_condition_is_reported() -> None:
    line = (
        "strategy S { when shift(AAPL.close, -1) > slow or "
        "!(slow < shift(AAPL.close, -3)): buy(); }"
    )
    assert [subject for subject, _ in _culprit_spans(line)] == ["'shift(...)'", "'shift(...)'"]


def test_an_honest_compound_condition_is_accepted() -> None:
    assert (
        _violations(
            """
            let fast: Series[Float] = SMA(AAPL.close, 5);
            let rsi: Series[Float] = RSI(AAPL.close, 14);
            strategy S {
                when fast crosses_above slow and rsi < 70.0: buy();
                when rsi crosses_above 80.0 or AAPL.close < shift(slow, 1): sell();
            }
            """
        )
        == []
    )


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


# ---- whole-history builtins ----


def test_every_builtin_has_a_temporal_classification() -> None:
    """Like the purity table: a new builtin must be declared trailing,
    whole-history or time-shifting on purpose. Left unclassified, a
    whole-series statistic would pass through a `when` unnoticed."""
    tables = {
        "point-in-time": POINT_IN_TIME_FUNCTIONS,
        "whole-history": WHOLE_HISTORY_FUNCTIONS,
        "time-shifting": frozenset(TIME_SHIFTING_FUNCTIONS),
    }
    names = set(stdlib.builtin_function_types())
    for name in sorted(names):
        homes = [table for table, members in tables.items() if name in members]
        assert len(homes) == 1, f"{name} is classified as {homes or 'nothing'}"
    classified = set().union(*tables.values())
    assert classified <= names, f"classified but not builtins: {sorted(classified - names)}"


@pytest.mark.parametrize(
    ("condition", "culprit"),
    [
        ("zscore(AAPL.close) < -1.0", "'zscore(...)'"),
        ("AAPL.close > mean(AAPL.close)", "'mean(...)'"),
        ("AAPL.close > slow + 0.5 * std(AAPL.close)", "'std(...)'"),
        ("RSI(AAPL.close, 14) < percentile(RSI(AAPL.close, 14), 10.0)", "'percentile(...)'"),
    ],
)
def test_a_whole_series_statistic_in_a_condition_is_look_ahead(
    condition: str, culprit: str
) -> None:
    found = _violations(f"strategy S {{ when {condition}: buy(); }}")
    assert len(found) == 1
    assert f"{culprit} is computed from the whole series" in found[0]
    assert "rolling_zscore" in found[0]  # and names the trailing alternative


def test_a_statistic_stored_first_is_still_caught() -> None:
    found = _violations(
        """
        let average: Float = mean(AAPL.close);
        strategy S { when AAPL.close > average: buy(); }
        """
    )
    assert len(found) == 1 and "'average' is computed from the whole series" in found[0]


def test_a_whole_series_statistic_inside_a_function_is_caught() -> None:
    found = _violations(
        """
        func standardized(s: Series[Float]) -> Series[Float] { return zscore(s); }
        strategy S { when standardized(AAPL.close) < -1.0: buy(); }
        """
    )
    assert len(found) == 1


def test_trailing_versions_are_accepted() -> None:
    assert (
        _violations(
            """
            strategy S {
                when rolling_zscore(AAPL.close, 20) < -1.0: buy();
                when AAPL.close > SMA(AAPL.close, 20) + 2.0 * rolling_std(AAPL.close, 20): sell();
                when drawdown(AAPL.close) > 0.1 or log_returns(AAPL.close) < -0.02: sell();
            }
            """
        )
        == []
    )


def test_statistics_on_plain_numbers_are_not_look_ahead() -> None:
    # bonferroni of two numbers reads no bars; the threshold is a constant.
    assert _violations("strategy S { when AAPL.close > bonferroni(0.01, 5) * 100.0: buy(); }") == []


def test_statistics_outside_a_condition_stay_legal() -> None:
    assert _violations("print(mean(AAPL.close)); print(zscore(AAPL.close));") == []


# ---- more sources: a single bar, a backtest's statistics, the bar count ----


def _diagnostics(body: str) -> list[str]:
    """Every look-ahead diagnostic, errors and notes, rendered as text."""
    program = Parser(Lexer(_PRELUDE + body).scan_tokens()).parse()
    Resolver().resolve(program)
    checker = TypeChecker()
    for name, signature in stdlib.builtin_function_types().items():
        checker.env.define(name, signature)
    checker.check(program)
    return [f"{d.severity.name.lower()}: {d.message}" for d in check_lookahead(program).diagnostics]


_A_BACKTEST = "strategy A { when AAPL.close > slow: buy(); }\n"


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            "strategy S { when AAPL.close > AAPL.close[len(AAPL.close) - 1]: buy(); }",
            "'AAPL.close[...]' is a single bar of a series",
        ),
        (
            "let last: Float = AAPL.close[59];\nstrategy S { when AAPL.close > last: buy(); }",
            "'last' is a single bar of a series",
        ),
        (
            _A_BACKTEST + "let r = backtest A on AAPL from 0 to 60;\n"
            "strategy S { when AAPL.close > slow and r.sharpe > 1.0: buy(); }",
            "'r.sharpe' is a statistic of a whole backtest",
        ),
        (
            "strategy S { when AAPL.close > SMA(AAPL.close, len(AAPL.close) - 40): buy(); }",
            "'len(...)' is the number of bars in the whole series",
        ),
    ],
    ids=["last-bar", "stored-bar", "backtest-statistic", "bar-count"],
)
def test_more_sources_of_future_data(body: str, reason: str) -> None:
    found = _violations(body)
    assert len(found) == 1
    assert reason in found[0]


def test_what_is_known_at_every_bar_is_accepted() -> None:
    body = (
        _A_BACKTEST + "let r = backtest A on AAPL from 0 to 60;\n"
        "strategy S { when AAPL.close > AAPL.close[0] and r.equity > 9000.0: buy(); }\n"
        "backtest S on AAPL from 0 to len(AAPL.close);\n"  # a range, not a decision
    )
    assert _violations(body) == []


# ---- more decisions: inside an action, order sizes, backtest options ----


@pytest.mark.parametrize(
    ("body", "decision"),
    [
        (
            "strategy S { when AAPL.close > slow: { if mean(AAPL.close) > 150.0 { buy(); } } }",
            "this condition inside a 'when' action of strategy 'S'",
        ),
        (
            "strategy S { when AAPL.close > slow: { target(AAPL.close[0] / mean(AAPL.close)); } }",
            "the size of this order in strategy 'S'",
        ),
        (
            _A_BACKTEST + "backtest A on AAPL from 0 to 60 "
            "with stop_loss: volatility(pct_change(AAPL.close), 252) / 10.0;",
            "backtest option 'stop_loss'",
        ),
        (
            _A_BACKTEST + "if mean(AAPL.close) > 150.0 { backtest A on AAPL from 0 to 60; }",
            "this condition, which decides whether strategy 'A' is backtested,",
        ),
    ],
    ids=["condition-in-action", "order-size", "backtest-option", "which-backtest-runs"],
)
def test_more_decisions_are_checked(body: str, decision: str) -> None:
    found = _violations(body)
    assert len(found) == 1
    assert found[0].startswith(f"look-ahead bias: {decision} may contain data from a future bar")


def test_honest_decisions_inside_actions_are_accepted() -> None:
    body = (
        "strategy S { when AAPL.close > slow: { if AAPL.close[0] > 100.0 { target(0.5); } } }\n"
        "backtest S on AAPL from 0 to 60 with stop_loss: 0.05;"
    )
    assert _violations(body) == []


# ---- flow: the analysis follows the program's control flow ----


@pytest.mark.parametrize(
    "body",
    [
        # A value chosen by a branch on future data.
        "let s: Series[Float] = slow;\n"
        "if mean(AAPL.close) > 150.0 { s = SMA(AAPL.close, 5); } else { s = slow; }\n"
        "strategy S { when AAPL.close > s: buy(); }",
        # Tainted at the end of one iteration, read at the start of the next.
        "let s: Series[Float] = slow;\n"
        "for let k: Int = 0; k < 2; k = k + 1 {\n"
        "    strategy S { when s > AAPL.close: buy(); }\n"
        "    s = shift(AAPL.close, -1);\n"
        "}",
        # How many times the loop ran depends on a future bar.
        "let w: Int = 2;\n"
        "for let i: Int = 0; i < 10; i = i + 1 { if AAPL.close[i] > 160.0 { break; } w = w + 1; }\n"
        "strategy S { when AAPL.close > SMA(AAPL.close, w): buy(); }",
        # Possibly tainted on one branch is possibly tainted.
        "let s: Series[Float] = slow;\n"
        "if true { s = shift(AAPL.close, -1); }\n"
        "strategy S { when s > AAPL.close: buy(); }",
    ],
    ids=["chosen-by-a-branch", "loop-carried", "loop-count", "one-branch"],
)
def test_flow_that_lets_the_future_in_is_rejected(body: str) -> None:
    assert len(_violations(body)) == 1


@pytest.mark.parametrize(
    "body",
    [
        # Overwritten before it's read.
        "let s: Series[Float] = shift(AAPL.close, -1);\ns = slow;\n"
        "strategy S { when s > AAPL.close: buy(); }",
        # Tainted after the strategy read it.
        "let s: Series[Float] = slow;\nstrategy S { when s > AAPL.close: buy(); }\n"
        "s = shift(AAPL.close, -1);",
        # A branch on a plain number.
        "let w: Int = 5;\nif true { w = 10; }\n"
        "strategy S { when AAPL.close > SMA(AAPL.close, w): buy(); }",
        # A loop that counts to the number of bars, then is done.
        "let total: Float = 0.0;\n"
        "for let i: Int = 0; i < len(AAPL.close); i = i + 1 { total = total + 1.0; }\n"
        "strategy S { when AAPL.close > slow: buy(); }",
    ],
    ids=["overwritten", "tainted-later", "honest-branch", "honest-loop"],
)
def test_flow_that_keeps_the_future_out_is_accepted(body: str) -> None:
    assert _violations(body) == []


# ---- functions are analyzed where they're called ----


def test_a_function_returning_a_tainted_global_is_caught_with_a_trail() -> None:
    found = _diagnostics(
        "let tomorrow: Series[Float] = shift(AAPL.close, -1);\n"
        "func peek() -> Series[Float] { return tomorrow; }\n"
        "strategy S { when peek() > AAPL.close: buy(); }"
    )
    assert found[0].endswith(
        "'peek(...)' is derived from a backwards time shift "
        "(shift(series, k) with a negative k places bar i+k at bar i)."
    )
    assert found[1:] == [
        "note: 'peek' returns data from a future bar here",
        "note: the future enters here: shift by -1 gives each bar the value from 1 bar later",
    ]


def test_a_function_assigning_a_global_taints_it_from_the_call_on() -> None:
    spoil = (
        "let s: Series[Float] = slow;\n"
        "func spoil() -> Int { s = shift(AAPL.close, -1); return 0; }\n"
    )
    before = "strategy S { when s > AAPL.close: buy(); }\nlet z: Int = spoil();"
    after = "let z: Int = spoil();\nstrategy S { when s > AAPL.close: buy(); }"
    assert _violations(spoil + before) == []
    assert len(_violations(spoil + after)) == 1


@pytest.mark.parametrize(
    "function",
    [
        # Declared, never called.
        "func spoil() -> Int { s = shift(AAPL.close, -1); return 0; }",
        # A local it never returns.
        "func f(x: Series[Float]) -> Series[Float] { let ahead = shift(x, -1); return x; }\n"
        "let unused: Series[Float] = f(AAPL.close);",
    ],
    ids=["never-called", "unused-local"],
)
def test_what_a_function_does_not_do_is_not_held_against_it(function: str) -> None:
    body = f"let s: Series[Float] = slow;\n{function}\nstrategy S {{ when s > AAPL.close: buy(); }}"
    assert _violations(body) == []


def test_recursion_is_analyzed_to_a_fixpoint() -> None:
    body = (
        "func f(n: Int) -> Series[Float] {\n"
        "    if n <= 0 { return shift(AAPL.close, -1); }\n"
        "    return f(n - 1);\n"
        "}\n"
        "strategy S { when f(3) > AAPL.close: buy(); }"
    )
    assert len(_violations(body)) == 1


# ---- sign analysis: a shift's period, proven 0 or more ----


@pytest.mark.parametrize(
    "setup",
    [
        "let lag: Int = 1;",
        "let lag: Int = 2;\nlag = lag * 3 + 1;",
        "let base: Int = 0 - 3;\nlet lag: Int = base % 5;",  # floored: takes 5's sign
    ],
    ids=["variable", "arithmetic", "modulo"],
)
def test_a_period_proven_non_negative_is_accepted(setup: str) -> None:
    body = f"{setup}\nstrategy S {{ when shift(AAPL.close, lag) > slow: buy(); }}"
    assert _violations(body) == []


def test_a_loop_counter_that_only_grows_is_non_negative() -> None:
    body = (
        "for let w: Int = 1; w < 4; w = w + 1 {\n"
        "    strategy S { when shift(AAPL.close, w) > slow: buy(); }\n"
        "}"
    )
    assert _violations(body) == []


def test_an_unproven_period_says_so() -> None:
    found = _violations(
        "let lag: Int = 1;\nlag = lag - 2;\n"
        "strategy S { when shift(AAPL.close, lag) > slow: buy(); }"
    )
    assert len(found) == 1
    assert "'shift(...)' is derived from a time shift whose period can't be proven" in found[0]


# ---- state a backtest's actions leave behind ----


_COUNTING = (
    "let n: Float = 0.0;\n"
    "strategy A { when AAPL.close > slow: { n = n + 1.0; if n > 3.0 { buy(); } } }\n"
    "backtest A on AAPL from 0 to 60;\n"
)


def test_state_an_action_kept_is_the_future_after_its_backtest() -> None:
    found = _violations(_COUNTING + "strategy B { when slow > 150.0 + n: buy(); }")
    assert len(found) == 1
    assert "'n' is changed by the 'when' actions of a strategy" in found[0]


def test_a_second_backtest_starts_from_the_first_ones_state() -> None:
    found = _violations(_COUNTING + "backtest A on AAPL from 0 to 60;")
    assert len(found) == 1
    assert "inside a 'when' action of strategy 'A'" in found[0]


def test_state_read_bar_by_bar_or_reset_is_accepted() -> None:
    assert _violations(_COUNTING + "n = 0.0;\nbacktest A on AAPL from 0 to 60;") == []


def test_a_backtests_significance_is_the_future_too() -> None:
    body = (
        _A_BACKTEST + "let r = backtest A on AAPL from 0 to 60;\n"
        "strategy S { when AAPL.close > slow and backtest_pvalue(r) < 0.05: buy(); }"
    )
    found = _violations(body)
    assert len(found) == 1
    assert "'backtest_pvalue(...)' is computed from the whole series" in found[0]
