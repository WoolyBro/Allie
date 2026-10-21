"""Central error type for the type checker -- every check in checker.py
reports through this rather than formatting its own string, so the
message format stays exactly consistent.

The checker collects these rather than stopping at the first: see
`TypeChecker.errors`. Cascades -- one root cause producing a pile of
consequential errors, the usual argument for fail-fast -- are prevented
by the poison type instead (allie.typechecker.types.ERROR): an
expression that fails to type-check is given a type compatible with
everything, so nothing downstream of it complains again.
"""

from __future__ import annotations

from allie.errors import format_excerpt, source_line


class QSTypeError(Exception):
    def __init__(
        self,
        message: str,
        line: int,
        column: int,
        length: int = 1,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column
        self.length = max(length, 1)
        # A suggested fix, shown by `render` but kept out of `__str__` so
        # the one-line format below stays exactly as pinned.
        self.hint = hint

    def __str__(self) -> str:
        return f"TypeError: {self.message}. Line {self.line}, Column {self.column}"

    def render(self, source: str) -> str:
        """`str(self)` plus the offending source line with a caret under
        the expression, and the hint if there is one -- the same layout as
        lexer and parser errors."""
        text = source_line(source, self.line)
        rendered = str(self)
        if text is not None:
            rendered += "\n" + format_excerpt(text, self.column, self.length)
        if self.hint is not None:
            rendered += f"\n  help: {self.hint}"
        return rendered
