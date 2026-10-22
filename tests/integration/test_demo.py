"""The demo works, and says what is true.

docs/demo-script.md is spoken in front of judges, so a stale number or a
broken command there costs more than anywhere else. These run the demo
straight through (scripts/demo.sh, as `make demo` does) and check each
number the script and the README quote against the output it comes from.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

import allie.stdlib as stdlib
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.quant.backtest_engine import observe_backtests
from allie.runtime.interpreter import run_program
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = (_ROOT / "docs" / "demo-script.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def demo_output() -> str:
    """Every beat of scripts/demo.sh, run without pauses."""
    result = subprocess.run(
        ["bash", str(_ROOT / "scripts" / "demo.sh")],
        cwd=_ROOT,
        env={**os.environ, "DEMO_PAUSE": "0", "PYTHON": sys.executable},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr


def test_every_beat_the_script_names_is_the_one_the_demo_runs(demo_output: str) -> None:
    for command in (
        "allie run examples/blended.qs",
        "python main.py tests/golden/programs/for_loop.qs --debug-ir",
        "allie check --audit examples/cheat_demo.qs",
        "allie run examples/rigor.qs",
    ):
        assert f"`{command}`" in _SCRIPT
        assert f"$ {command}" in demo_output
    assert "Traceback" not in demo_output


def test_the_ir_beat_shows_what_the_script_points_at(demo_output: str) -> None:
    ir = demo_output.split("=== ALLIE IR ===")[1].split("=== LLVM IR ===")[0]
    assert "JumpIfFalse" in ir and "Jump(" in ir and "Label(" in ir
    assert "for" not in {line.split("(")[0] for line in ir.splitlines()}  # no for instruction
    optimized = demo_output.split("=== LLVM IR (after LLVM -O2) ===")[1].split("==")[0]
    assert "box_int(i64 8)" in optimized  # the loop, computed at compile time
    assert "call_builtin_bridge" in optimized


def test_the_audit_beat_finds_the_bar_the_script_names(demo_output: str) -> None:
    assert "look-ahead bias" in demo_output
    assert "bar 37 -- 'when' clause 1: true with all 60 bars, false with only the first 38" in (
        demo_output
    )
    assert "bar 37 -- orders placed: target 1 with all 60 bars, none with only the first 38" in (
        demo_output
    )
    assert "At bar 37 it buys with all 60 bars, and\n> doesn't with only the first 38." in _SCRIPT


def test_the_rigor_beat_prints_the_numbers_the_script_quotes(demo_output: str) -> None:
    rigor = demo_output.split("$ allie run examples/rigor.qs")[1]
    lines = rigor.splitlines()
    best = lines.index(
        "best Sharpe, and the chance its true Sharpe is above 0 -- alone, then deflated:"
    )
    sharpe, alone, deflated = (float(lines[best + k]) for k in (1, 2, 3))
    assert (round(sharpe, 2), round(alone * 100), round(deflated * 100)) == (1.84, 81, 73)
    assert "Sharpe of 1.84" in _SCRIPT and "81% likely" in _SCRIPT and "it's 73%" in _SCRIPT


def _run_without_lookahead(source: str) -> tuple[float, float]:
    """(Sharpe, CAGR) of the program's one backtest, with every check
    but the look-ahead one -- what the cheat would report if it ran."""
    program = Parser(Lexer(source).scan_tokens()).parse()
    Resolver().resolve(program)
    checker = TypeChecker()
    for name, signature in stdlib.builtin_function_types().items():
        checker.env.define(name, signature)
    checker.check(program)
    with observe_backtests() as records, contextlib.redirect_stdout(io.StringIO()):
        run_program(program, stdlib.create_global_environment())
    (record,) = records
    return record.result.metrics.sharpe, record.result.metrics.cagr


def test_the_cheats_numbers_are_the_ones_the_docs_quote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_ROOT)
    cheat = (_ROOT / "examples" / "cheat_demo.qs").read_text(encoding="utf-8")
    honest = cheat.replace("when tomorrow", "when AAPL.close")
    assert honest != cheat
    sharpe, cagr = _run_without_lookahead(cheat)
    assert (round(sharpe, 2), round(cagr * 100, 2)) == (3.31, 41.78)
    sharpe, cagr = _run_without_lookahead(honest)
    assert (round(sharpe, 2), round(cagr * 100, 2)) == (1.14, 13.39)
    for doc in (
        "README.md",
        "docs/demo-script.md",
        "docs/architecture.md",
        "examples/cheat_demo.qs",
    ):
        text = (_ROOT / doc).read_text(encoding="utf-8")
        assert "3.31" in text and "1.14" in text, doc
