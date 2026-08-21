"""Phase 0 smoke test: proves the package layout imports cleanly and the
CLI entry point is wired up. Real lexer/parser tests replace this once
Phase 2 lands (the evaluator itself already has its own tests -- see
test_phase1_eval.py)."""

from __future__ import annotations

import subprocess
import sys

from qubit.errors import Diagnostic, DiagnosticBag, Severity, SourceSpan


def test_diagnostic_bag_collects_multiple_errors() -> None:
    bag = DiagnosticBag()
    bag.error("first problem", SourceSpan(line=1, column=1, offset=0))
    bag.error("second problem", SourceSpan(line=2, column=5, offset=10))

    assert bag.has_errors()
    assert len(bag.diagnostics) == 2


def test_diagnostic_render_includes_caret() -> None:
    diagnostic = Diagnostic(
        severity=Severity.ERROR,
        message="unexpected token",
        span=SourceSpan(line=3, column=5, offset=20, length=3),
        source_line="let x == 1;",
    )
    rendered = diagnostic.render()

    assert "3:5" in rendered
    assert "^^^" in rendered


def test_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "qubit", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "qubit" in result.stdout.lower()
