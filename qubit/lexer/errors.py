"""LexError -- raised/recorded by qubit.lexer.lexer.Lexer.

Two different failure modes, deliberately handled differently (see
qubit.lexer.lexer docstring for the full rationale):

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
        return f"LexError: {self.message} (line {self.line}, col {self.column})"
