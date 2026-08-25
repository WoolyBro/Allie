"""Central error type for the type checker (Phase 3.1) -- every check in
checker.py raises this instead of formatting its own string, so the
message format stays exactly consistent.

Deliberately fail-fast (raised as a Python exception, not collected like
allie.errors.DiagnosticBag): the checker stops at the first type error
rather than accumulating several, unlike the lexer/parser's
collect-and-report-all approach (docs/spec.md "Diagnostics strategy").
That's a real difference between phases, not an oversight -- a cascade
of type errors from one root cause is rarely useful to see all at once.
"""

from __future__ import annotations


class QSTypeError(Exception):
    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return f"TypeError: {self.message}. Line {self.line}, Column {self.column}"
