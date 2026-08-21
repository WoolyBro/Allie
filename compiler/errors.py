"""Diagnostics infrastructure, shared by every compiler phase.

Design decision (Phase 0.4): collect-and-report-all, not fail-fast.
A user iterating on a strategy file wants every syntax/type error in
one pass, not a whack-a-mole loop. This means the lexer/parser/semantic
phases must be structured to record a Diagnostic and attempt recovery
rather than raising on the first problem — see docs/spec.md.

Every token and AST node carries a SourceSpan from the moment it is
created, so a diagnostic raised in semantic analysis (phase 3+) can
still point at the original source text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A half-open range of source positions: [start, end)."""

    line: int
    column: int
    offset: int
    length: int = 1

    def __str__(self) -> str:
        return f"{self.line}:{self.column}"


class Severity(Enum):
    ERROR = auto()
    WARNING = auto()
    NOTE = auto()


@dataclass(frozen=True, slots=True)
class Diagnostic:
    severity: Severity
    message: str
    span: SourceSpan
    source_line: str | None = None

    def render(self) -> str:
        header = f"{self.severity.name.lower()}: {self.message} ({self.span})"
        if self.source_line is None:
            return header
        caret_line = " " * (self.span.column - 1) + "^" * max(self.span.length, 1)
        return f"{header}\n  {self.source_line}\n  {caret_line}"


@dataclass(slots=True)
class DiagnosticBag:
    """Accumulates diagnostics across a compilation pass.

    Callers (lexer/parser/semantic) push diagnostics here instead of
    raising, so a single `quantlang run` invocation can report every
    problem it finds rather than stopping at the first one.
    """

    diagnostics: list[Diagnostic] = field(default_factory=list)

    def error(self, message: str, span: SourceSpan, source_line: str | None = None) -> None:
        self.diagnostics.append(Diagnostic(Severity.ERROR, message, span, source_line))

    def warning(self, message: str, span: SourceSpan, source_line: str | None = None) -> None:
        self.diagnostics.append(Diagnostic(Severity.WARNING, message, span, source_line))

    def note(self, message: str, span: SourceSpan, source_line: str | None = None) -> None:
        self.diagnostics.append(Diagnostic(Severity.NOTE, message, span, source_line))

    def has_errors(self) -> bool:
        return any(d.severity is Severity.ERROR for d in self.diagnostics)

    def extend(self, other: DiagnosticBag) -> None:
        self.diagnostics.extend(other.diagnostics)

    def print_all(self) -> None:
        for diagnostic in self.diagnostics:
            print(diagnostic.render())
