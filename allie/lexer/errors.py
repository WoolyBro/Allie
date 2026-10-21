"""LexError -- recorded by allie.lexer.lexer.Lexer.

The lexer never raises one. Every lexical problem -- a bad character,
an unterminated string, a malformed number -- is appended to
`Lexer.errors` and scanning resumes, so one `scan_tokens()` call reports
every problem in the file (docs/spec.md "Diagnostics strategy":
collect-and-report-all). It is still an `Exception` subclass so callers
that want fail-fast behavior can `raise lexer.errors[0]`.
"""

from __future__ import annotations

from allie.errors import format_excerpt, source_line


class LexError(Exception):
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
        # How many characters the caret underline spans, starting at
        # `column`. Never extends past the end of `line`.
        self.length = max(length, 1)
        # A suggested fix, shown under the caret by `render` but kept out
        # of `__str__` so the one-line format below stays uniform.
        self.hint = hint

    def __str__(self) -> str:
        # One format across every user-facing error type
        # (`{Kind}: {message}. Line {L}, Column {C}`). QSTypeError's
        # wording was pinned by an explicit spec requirement and a test
        # asserting the exact string, so it became the canonical shape
        # and this one was brought into line -- rather than leaving two
        # near-identical formats that differ only in punctuation.
        return f"LexError: {self.message}. Line {self.line}, Column {self.column}"

    def render(self, source: str) -> str:
        """`str(self)` plus the offending source line, a caret underline,
        and the hint if there is one:

            LexError: unexpected '&&'. Line 3, Column 11
              if a > 0 && b > 0 {
                       ^^
              help: logical AND is spelled 'and'

        The excerpt is formatted by allie.errors.format_excerpt, the same
        helper parser and semantic diagnostics use, so every error in the
        pipeline looks alike.
        """
        text = source_line(source, self.line)
        if text is None:
            return str(self)
        rendered = f"{self}\n{format_excerpt(text, self.column, self.length)}"
        if self.hint is not None:
            rendered += f"\n  help: {self.hint}"
        return rendered
