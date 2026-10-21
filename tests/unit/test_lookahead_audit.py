"""The look-ahead audit (allie/semantic/audit.py), and the static
analysis (allie/semantic/lookahead.py) checked against it.

The audit defines look-ahead by experiment -- a decision at bar t that
changes when the bars after t are removed -- independently of how the
analysis reasons about source text. So it can test the analysis:
generated programs mix honest and leaky constructs (backward shifts,
whole-series statistics, single bars, the number of bars, values picked
by a branch on future data, order sizes, stops, state left by a
backtest's actions), and every program the analysis accepts must pass
the audit. That is the soundness claim the analysis makes, tested on
programs nobody chose.

Seeded, so a failure reproduces exactly; the failing program is in the
message.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

import allie.stdlib as stdlib
from allie.ast.nodes import Program
from allie.cli import main
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.runtime.errors import AllieRuntimeError
from allie.semantic.audit import audit
from allie.semantic.lookahead import check_lookahead
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker

_ROOT = Path(__file__).resolve().parents[2]
_PRELUDE = """\
let AAPL = load("tests/fixtures/AAPL_sample.csv");
let c: Series[Float] = AAPL.close;
"""


def _checked(source: str) -> Program:
    program = Parser(Lexer(source).scan_tokens()).parse()
    Resolver().resolve(program)
    checker = TypeChecker()
    for name, signature in stdlib.builtin_function_types().items():
        checker.env.define(name, signature)
    checker.check(program)
    return program


def _accepted(program: Program) -> bool:
    return not check_lookahead(program).has_errors()


@pytest.fixture(autouse=True)
def _in_the_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_ROOT)  # the programs load a fixture by relative path


# ---- the audit itself ----


def test_the_cheat_demo_fails_the_audit_where_it_cheats() -> None:
    report = audit(_checked((_ROOT / "examples" / "cheat_demo.qs").read_text()))
    assert not report.clean
    subjects = {leak.subject for leak in report.leaks}
    assert "'when' clause 1" in subjects
    assert "orders placed" in subjects
    first = report.leaks[0]
    assert first.strategy == "Clairvoyant"
    assert first.kept == first.bar + 1  # found by cutting the data right after the bar
    assert "FAILED" in report.render()


def test_an_honest_strategy_passes_on_every_cut() -> None:
    report = audit(_checked((_ROOT / "examples" / "moving_average.qs").read_text()))
    assert report.clean
    assert (report.bars, report.backtests, report.compared, report.skipped) == (60, 1, 59, 0)
    assert "passed" in report.render()


@pytest.mark.parametrize(
    ("body", "subject"),
    [
        ("strategy S { when zscore(c) > 0.5: buy(); when zscore(c) < 0.0: sell(); }", "when"),
        (
            "strategy S { when c > SMA(c, 3): { target(c[0] / mean(c)); } "
            "when c < SMA(c, 3): sell(); }",
            "orders",
        ),
        (
            "strategy S { when c > SMA(c, 3): buy(); }",
            "trades",
        ),
    ],
    ids=["whole-series-condition", "order-size", "stop-calibrated-on-the-period"],
)
def test_each_kind_of_decision_is_compared(body: str, subject: str) -> None:
    stop = "with stop_loss: std(pct_change(c)) * 1.5" if subject == "trades" else ""
    source = _PRELUDE + body + f"\nbacktest S on AAPL from 0 to 60 {stop};\n"
    report = audit(_checked(source))
    assert any(subject in leak.subject for leak in report.leaks), report.render()


def test_which_backtests_run_is_a_decision_too() -> None:
    source = _PRELUDE + (
        "strategy A { when c > SMA(c, 3): buy(); }\n"
        "strategy B { when c < SMA(c, 3): buy(); }\n"
        "if c[len(c) - 1] > c[0] { backtest A on AAPL from 0 to 60; }\n"
        "else { backtest B on AAPL from 0 to 60; }\n"
    )
    report = audit(_checked(source))
    assert [leak.subject for leak in report.leaks] == ["the backtests run"]
    assert report.leaks[0].backtest == 0  # the program's, not one backtest's


@pytest.mark.parametrize("example", sorted(path.name for path in (_ROOT / "examples").glob("*.qs")))
def test_every_example_the_analysis_accepts_passes_the_audit(example: str) -> None:
    program = _checked((_ROOT / "examples" / example).read_text())
    assert audit(program).clean == _accepted(program)  # only cheat_demo is rejected


def test_a_range_that_moves_with_the_bar_count_is_compared_where_it_overlaps() -> None:
    # Cut to 28 bars, `to floor(len(c) * 0.7)` ends at bar 19, not 42.
    source = _PRELUDE + (
        "strategy S { when c > SMA(c, 3): buy(); when c < SMA(c, 3): sell(); }\n"
        "let split: Int = floor(len(c) * 0.7);\n"
        "backtest S on AAPL from 0 to split;\n"
    )
    assert audit(_checked(source)).clean


def test_a_cut_the_program_cannot_run_on_is_skipped_not_failed() -> None:
    source = _PRELUDE + (
        "strategy S { when c > SMA(c, 3): buy(); }\nbacktest S on AAPL from 30 to 60;\n"
    )
    report = audit(_checked(source))
    assert report.clean
    assert report.skipped == 30  # cut to 30 bars or fewer, there's no bar 30
    assert "30 skipped" in report.render()


def test_a_program_that_fails_outright_is_an_error_not_a_report() -> None:
    source = _PRELUDE + "let y: Int = 1 % 0;\n"
    with pytest.raises(AllieRuntimeError):
        audit(_checked(source))


def test_allie_check_audit(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", "--audit", "examples/moving_average.qs"]) == 0
    assert "look-ahead audit: passed" in capsys.readouterr().err
    assert main(["check", "--audit", "examples/cheat_demo.qs"]) == 1
    err = capsys.readouterr().err
    assert "look-ahead bias" in err  # the static analysis rejects it...
    assert "look-ahead audit: FAILED" in err  # ...and the audit shows it's real


def test_allie_check_audit_on_a_program_that_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    program = tmp_path / "fails.qs"
    program.write_text(_PRELUDE + "let y: Int = 1 % 0;\n")
    assert main(["check", "--audit", str(program)]) == 1
    err = capsys.readouterr().err
    assert "modulo by zero" in err
    assert "look-ahead audit: not run -- the program fails on its data" in err


# ---- the analysis, checked against the audit ----

_END = "\nbacktest S on AAPL from 0 to 60;\n"
_TRADES_ON_SMA = "when c < SMA(c, 3): sell();"


@pytest.mark.parametrize(
    "body",
    [
        "strategy S { when shift(c, -1) > c: buy(); when shift(c, -1) < c: sell(); }" + _END,
        "strategy S { when c > mean(c): buy(); when c < mean(c): sell(); }" + _END,
        "strategy S { when c > c[len(c) - 1]: buy(); when c < c[len(c) - 1]: sell(); }" + _END,
        "strategy S { when c > shift(c, len(c) - 58): buy(); "
        "when c < shift(c, len(c) - 58): sell(); }" + _END,
        "strategy A { when c > SMA(c, 3): buy(); " + _TRADES_ON_SMA + " }\n"
        "let r = backtest A on AAPL from 0 to 60;\n"
        "strategy S { when c > SMA(c, 3) - 1.0 + r.trades * 0.2: buy(); "
        + _TRADES_ON_SMA
        + " }"
        + _END,
        "let s: Series[Float] = c;\n"
        "if mean(c) > 157.0 { s = SMA(c, 2); } else { s = SMA(c, 8); }\n"
        "strategy S { when c > s: buy(); when c < s: sell(); }" + _END,
        "strategy S { when c > SMA(c, 3): { target(c[0] / mean(c)); } "
        + _TRADES_ON_SMA
        + " }"
        + _END,
        "strategy S { when c > SMA(c, 3): { if mean(c) > 158.5 { buy(); } } "
        + _TRADES_ON_SMA
        + " }"
        + _END,
        "strategy S { when c > SMA(c, 3): buy(); }\n"
        "backtest S on AAPL from 0 to 60 with stop_loss: std(pct_change(c)) * 1.5;\n",
        "let fired: Float = 0.0;\n"
        "strategy A { when c > SMA(c, 3): { fired = fired + 1.0; buy(); } "
        + _TRADES_ON_SMA
        + " }\n"
        "backtest A on AAPL from 0 to 60;\n"
        "strategy S { when SMA(c, 3) > 145.0 + fired: buy(); "
        "when SMA(c, 3) < 145.0 + fired: sell(); }" + _END,
    ],
    ids=[
        "backward-shift",
        "whole-series",
        "single-bar",
        "number-of-bars",
        "backtest-result",
        "value-chosen-by-a-branch",
        "order-size",
        "condition-in-an-action",
        "stop-loss",
        "state-left-by-actions",
    ],
)
def test_each_rule_the_analysis_enforces_is_a_real_leak(body: str) -> None:
    """Not over-strict: for every kind of source and decision the
    analysis rejects, a program where it drives a decision changes that
    decision when later bars are removed."""
    program = _checked(_PRELUDE + body)
    assert not _accepted(program)
    assert not audit(program).clean


class _Generator:
    """Random, type-correct strategies. Each program draws how often it
    takes a leaky option at a choice point -- from never to often -- so
    the suite holds honest programs, subtly leaky ones and blatant ones."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.leak_rate = rng.choice([0.0, 0.0, 0.02, 0.05, 0.1, 0.25])
        self.series = ["c", "AAPL.open", "AAPL.high", "AAPL.low"]
        self.ints: list[str] = []  # each holds 1 or more
        self.functions: list[str] = []

    def leaky(self) -> bool:
        return self.rng.random() < self.leak_rate

    def pick(self, honest: list[str], leaky: list[str]) -> str:
        return self.rng.choice(leaky if self.leaky() else honest)

    def window(self) -> str:
        return self.rng.choice(["2", "3", "5", *self.ints])

    def period(self) -> str:
        return self.pick(["0", "1", "1", "2", "3", *self.ints], ["-1", "-2", "len(c) - 58"])

    def scalar(self) -> str:
        return self.pick(
            ["150.0", "160.0", "c[0]", "AAPL.open[0] * 1.02"],
            ["mean(c)", "median(AAPL.high)", "c[5]", "c[len(c) - 1]", "std(c) + 150.0"],
        )

    def series_expr(self, depth: int) -> str:
        atom = self.rng.choice(self.series)
        if depth <= 0 or self.rng.random() < 0.25:
            return atom
        inner = self.series_expr(depth - 1)
        honest = [
            f"SMA({inner}, {self.window()})",
            f"EMA({inner}, {self.window()})",
            f"rolling_max({inner}, {self.window()})",
            f"rolling_min({inner}, {self.window()})",
            f"rolling_zscore({inner}, {self.window()}) + 150.0",
            f"shift({inner}, {self.period()})",
            f"{inner} + {self.series_expr(depth - 1)} - 150.0",
            f"{inner} * 1.01",
            f"{inner} / {self.scalar()} * 150.0",
            *(f"{name}({inner})" for name in self.functions),
        ]
        leaky = [f"{inner} - mean({inner}) + 150.0", f"zscore({inner}) + 150.0"]
        return self.pick(honest, leaky)

    def condition(self) -> str:
        left, right = self.series_expr(2), self.series_expr(2)
        test = self.rng.choice(
            [
                f"{left} > {right}",
                f"{left} < {right}",
                f"{left} crosses_above {right}",
                f"{left} crosses_below {right}",
                f"{left} > {self.scalar()}",
            ]
        )
        if self.rng.random() < 0.2:
            test += f" and {self.series_expr(1)} > {self.series_expr(1)}"
        return test

    def scalar_test(self) -> str:
        return self.rng.choice(["true", f"{self.scalar()} > 155.0", f"c[0] < {self.scalar()}"])

    def action(self, verb: str) -> str:
        roll = self.rng.random()
        if roll < 0.6:
            return f"{verb}();"
        if roll < 0.8:
            if verb != "buy":
                return "{ target(0.0); }"
            size = self.pick(["0.5", "1.0", "c[0] / 150.0"], ["c[0] / mean(c)", "c[3] / c[0]"])
            return "{ target(" + size + "); }"
        return "{ if " + self.scalar_test() + " { " + verb + "(); } }"

    def program(self) -> str:
        lines = [_PRELUDE.rstrip()]
        for number in range(self.rng.randint(0, 2)):
            value = self.rng.choice(["1", "2", "3", *(f"{k} + 1" for k in self.ints)])
            lines.append(f"let k{number}: Int = {value};")
            self.ints.append(f"k{number}")
        if self.rng.random() < 0.3:
            # Reads its parameter, and maybe a module-level series.
            module_level, self.series = self.series, ["s", "s", "c"]
            body = self.series_expr(2)
            self.series = module_level
            lines.append(f"func f0(s: Series[Float]) -> Series[Float] {{ return {body}; }}")
            self.functions.append("f0")
        for number in range(self.rng.randint(1, 3)):
            lines.append(f"let s{number}: Series[Float] = {self.series_expr(2)};")
            self.series.append(f"s{number}")
        if self.rng.random() < 0.3:
            chosen = self.rng.choice(self.series[4:])
            lines.append(
                f"if {self.scalar_test()} {{ {chosen} = {self.series_expr(1)}; }} "
                f"else {{ {chosen} = {self.series_expr(1)}; }}"
            )
        counts = self.rng.random() < 0.25  # an action keeps state, which T reads
        if counts:
            lines.append("let fired: Float = 0.0;")
        lines.append("strategy S {")
        buy = "{ fired = fired + 1.0; buy(); }" if counts else self.action("buy")
        lines.append(f"    when {self.condition()}: {buy}")
        lines.append(f"    when {self.condition()}: {self.action('sell')}")
        lines.append("}")
        stop = self.pick(["", " with stop_loss: 0.04"], [" with stop_loss: std(pct_change(c))"])
        lines.append(f"backtest S on AAPL from 0 to 60{stop};")
        if counts:
            if not self.leaky():
                lines.append("fired = 0.0;")
            lines.append("strategy T { when SMA(c, 3) > 150.0 + fired: buy(); }")
            lines.append("backtest T on AAPL from 0 to 60;")
        return "\n".join(lines) + "\n"


_SEEDS = range(300)


def test_every_program_the_analysis_accepts_passes_the_audit() -> None:
    accepted = rejected = rejected_leaky = 0
    for seed in _SEEDS:
        source = _Generator(random.Random(seed)).program()
        program = _checked(source)  # the generator writes type-correct programs
        try:
            if _accepted(program):
                report = audit(program)
                assert report.clean, f"seed {seed}: accepted, but\n{report.render()}\n{source}"
                accepted += 1
            else:
                leaky = not audit(program, first_leak=True).clean
                rejected += 1
                rejected_leaky += leaky
        except AllieRuntimeError:
            continue  # e.g. a negative window: not a program to judge
    # Not vacuous: plenty of both, and many rejections change a decision
    # on this very data. (The rest carry a leak this data doesn't
    # trigger -- a zscore in a condition that is never true -- so they
    # aren't a measure of over-strictness; the test above is.)
    assert accepted >= 150, (accepted, rejected)
    assert rejected >= 90, (accepted, rejected)
    assert rejected_leaky >= 40, (rejected_leaky, rejected)
