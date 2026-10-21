"""Diagnostics infrastructure, shared by every compiler phase.

Design decision: collect-and-report-all, not fail-fast. A user iterating
on a strategy file wants every syntax error in one pass, not a
whack-a-mole loop, so the lexer, parser and semantic passes record a
Diagnostic and recover rather than raising on the first problem.

Every token and AST node carries a SourceSpan from the moment it is
created, so a diagnostic raised in semantic analysis can still point at
the original source text. `render(source)` shows that text with a caret
under the span and, when the reporter knows the likely fix, a `help:`
line.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TextIO

_MAX_SHOWN_LINE = 100


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


def source_line(source: str, line: int) -> str | None:
    """Line `line` (1-based) of `source` without its line ending, or None
    if there is no such line. A leading byte-order mark is dropped, since
    the lexer doesn't count it as a column."""
    lines = source.split("\n")
    if not 1 <= line <= len(lines):
        return None
    text = lines[line - 1].rstrip("\r")
    return text.removeprefix("\ufeff") if line == 1 else text


def format_excerpt(text: str, column: int, length: int) -> str:
    """Two indented lines: `text`, then a caret underline starting at
    1-based `column` and `length` characters long.

    Tabs before the column are copied into the caret line rather than
    replaced with spaces, so the caret lines up under tab-indented code
    in any terminal. Lines longer than 100 characters are cut to a
    window around the caret, and the underline never runs past the end
    of the line.
    """
    start = max(column - 1, 0)
    if len(text) > _MAX_SHOWN_LINE:
        window_start = max(0, start - _MAX_SHOWN_LINE // 3)
        window_end = min(len(text), window_start + _MAX_SHOWN_LINE)
        shown = text[window_start:window_end]
        lead = "..." if window_start > 0 else ""
        text = lead + shown + ("..." if window_end < len(text) else "")
        start = start - window_start + len(lead)
        remaining = len(lead) + len(shown) - start
    else:
        remaining = len(text) - start

    prefix = "".join("\t" if ch == "\t" else " " for ch in text[:start])
    underline = "^" * max(1, min(length, remaining))
    return f"  {text}\n  {prefix}{underline}"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    severity: Severity
    message: str
    span: SourceSpan
    source_line: str | None = None
    hint: str | None = None

    def render(self, source: str | None = None) -> str:
        """`error: message (line:col)`, then -- when the source text is
        available, either stored on the diagnostic or passed here -- the
        offending line with a caret, then the hint if there is one."""
        rendered = f"{self.severity.name.lower()}: {self.message} ({self.span})"
        text = self.source_line
        if text is None and source is not None:
            text = source_line(source, self.span.line)
        if text is not None:
            rendered += "\n" + format_excerpt(text, self.span.column, self.span.length)
        if self.hint is not None:
            rendered += f"\n  help: {self.hint}"
        return rendered


@dataclass(slots=True)
class DiagnosticBag:
    """Accumulates diagnostics across a compilation pass.

    Callers (lexer/parser/semantic) push diagnostics here instead of
    raising, so a single `allie run` invocation can report every
    problem it finds rather than stopping at the first one.
    """

    diagnostics: list[Diagnostic] = field(default_factory=list)

    def error(
        self,
        message: str,
        span: SourceSpan,
        source_line: str | None = None,
        *,
        hint: str | None = None,
    ) -> None:
        self.diagnostics.append(Diagnostic(Severity.ERROR, message, span, source_line, hint))

    def warning(
        self,
        message: str,
        span: SourceSpan,
        source_line: str | None = None,
        *,
        hint: str | None = None,
    ) -> None:
        self.diagnostics.append(Diagnostic(Severity.WARNING, message, span, source_line, hint))

    def note(
        self,
        message: str,
        span: SourceSpan,
        source_line: str | None = None,
        *,
        hint: str | None = None,
    ) -> None:
        self.diagnostics.append(Diagnostic(Severity.NOTE, message, span, source_line, hint))

    def has_errors(self) -> bool:
        return any(d.severity is Severity.ERROR for d in self.diagnostics)

    def extend(self, other: DiagnosticBag) -> None:
        self.diagnostics.extend(other.diagnostics)

    def print_all(self, source: str | None = None, file: TextIO | None = None) -> None:
        """Prints every diagnostic -- to stderr unless `file` says
        otherwise, like every other compile and runtime error, so a
        program's own output on stdout is never mixed with them. Pass the
        program text to get source excerpts with carets."""
        for diagnostic in self.diagnostics:
            print(diagnostic.render(source), file=file if file is not None else sys.stderr)
