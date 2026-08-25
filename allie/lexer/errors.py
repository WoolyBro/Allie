"""LexError -- raised/recorded by allie.lexer.lexer.Lexer.

Two different failure modes, deliberately handled differently (see
allie.lexer.lexer docstring for the full rationale):

- An unterminated string hits EOF with nothing left to recover with, so
  it's raised immediately as a Python exception.
- An invalid character is recoverable -- the lexer records a LexError and
  keeps scanning, so later valid tokens still come back. Callers check
  `Lexer.errors` after `scan_tokens()` returns.
"""

from __future__ import annotations


class LexError(Exception):
    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self) -> str:
        # One format across every user-facing error type
        # (`{Kind}: {message}. Line {L}, Column {C}`). QSTypeError's
        # wording was pinned by an explicit spec requirement and a test
        # asserting the exact string, so it became the canonical shape
        # and this one was brought into line -- rather than leaving two
        # near-identical formats that differ only in punctuation.
        return f"LexError: {self.message}. Line {self.line}, Column {self.column}"
