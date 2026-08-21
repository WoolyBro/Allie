"""Qubit's tokenizer -- Phase 2.1.

Design decisions written down here so they don't need rediscovering:

- **`and`/`or` are keywords, not symbolic `&&`/`||`.** Simpler: no extra
  two-character lookahead cases, and TokenType has no AMP_AMP/PIPE_PIPE
  to begin with. `&`/`|` alone are just invalid characters.
- **Numbers require a digit on both sides of the decimal point.** `.` is
  only consumed as part of a NUMBER when it's followed by a digit, so
  `1.` lexes as `NUMBER(1)` + `DOT` (no digit after the dot) and `.5`
  lexes as `DOT` + `NUMBER(5)` (no digit before it) -- never a malformed
  float. Same rule Crafting Interpreters' Lox uses, for the same reason:
  it's unambiguous and needs no backtracking.
- **Keyword lookup is case-sensitive on purpose** (plain dict lookup,
  no `.lower()`): `Ahhh` (while) and `ahhh` (an identifier) are
  different tokens. Easy to trip over in a live demo -- that's exactly
  why it's called out here instead of silently "fixed".
- **Unterminated string vs. invalid character are handled differently.**
  Hitting EOF mid-string leaves nothing left to recover with, so
  `_scan_string` raises `LexError` immediately. An invalid character is
  recoverable -- it's recorded in `self.errors` and scanning continues,
  so a single `scan_tokens()` call can report every bad character in one
  pass (docs/spec.md "Diagnostics strategy": collect-and-report-all)
  instead of stopping at the first one.
"""

from __future__ import annotations

from qubit.lexer.errors import LexError
from qubit.lexer.keywords import KEYWORD_ALIASES
from qubit.lexer.tokens import Token, TokenType

_SIMPLE_SINGLE_CHAR: dict[str, TokenType] = {
    "+": TokenType.PLUS,
    "*": TokenType.STAR,
    "%": TokenType.PERCENT,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    ",": TokenType.COMMA,
    ".": TokenType.DOT,
    ":": TokenType.COLON,
    ";": TokenType.SEMICOLON,
}

_STRING_ESCAPES: dict[str, str] = {
    "n": "\n",
    "t": "\t",
    '"': '"',
    "\\": "\\",
}


class Lexer:
    def __init__(self, source: str) -> None:
        self.source = source
        self.start = 0
        self.current = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []
        self.errors: list[LexError] = []
        self._start_line = 1
        self._start_column = 1

    def scan_tokens(self) -> list[Token]:
        while not self._is_at_end():
            self.start = self.current
            self._start_line = self.line
            self._start_column = self.column
            self.scan_token()

        self.tokens.append(Token(TokenType.EOF, "", None, self.line, self.column))
        return self.tokens

    def scan_token(self) -> None:
        c = self._advance()

        if c in (" ", "\t", "\r"):
            return
        if c == "\n":
            self.line += 1
            self.column = 1
            return

        if c == "/":
            if self._match("/"):
                while self._peek() != "\n" and not self._is_at_end():
                    self._advance()
            else:
                self._add_token(TokenType.SLASH)
            return

        if c in _SIMPLE_SINGLE_CHAR:
            self._add_token(_SIMPLE_SINGLE_CHAR[c])
            return

        if c == "-":
            self._add_token(TokenType.ARROW if self._match(">") else TokenType.MINUS)
            return
        if c == "=":
            self._add_token(TokenType.EQUALS_EQUALS if self._match("=") else TokenType.EQUALS)
            return
        if c == "!":
            self._add_token(TokenType.BANG_EQUALS if self._match("=") else TokenType.BANG)
            return
        if c == "<":
            self._add_token(TokenType.LESS_EQUALS if self._match("=") else TokenType.LESS)
            return
        if c == ">":
            self._add_token(TokenType.GREATER_EQUALS if self._match("=") else TokenType.GREATER)
            return

        if c == '"':
            self._scan_string()
            return

        if c.isdigit():
            self._scan_number()
            return

        if c.isalpha() or c == "_":
            self._scan_identifier()
            return

        self.errors.append(
            LexError(f"unexpected character {c!r}", self._start_line, self._start_column)
        )

    def _scan_number(self) -> None:
        while self._peek().isdigit():
            self._advance()

        is_float = False
        if self._peek() == "." and self._peek_next().isdigit():
            is_float = True
            self._advance()
            while self._peek().isdigit():
                self._advance()

        lexeme = self.source[self.start : self.current]
        literal: int | float = float(lexeme) if is_float else int(lexeme)
        self._add_token(TokenType.NUMBER, literal)

    def _scan_string(self) -> None:
        chars: list[str] = []

        while self._peek() != '"' and not self._is_at_end():
            c = self._advance()

            if c == "\n":
                self.line += 1
                self.column = 1
                chars.append(c)
                continue

            if c == "\\":
                if self._is_at_end():
                    break
                escape = self._advance()
                if escape in _STRING_ESCAPES:
                    chars.append(_STRING_ESCAPES[escape])
                else:
                    self.errors.append(
                        LexError(
                            f"unknown escape sequence '\\{escape}'",
                            self._start_line,
                            self._start_column,
                        )
                    )
                continue

            chars.append(c)

        if self._is_at_end():
            raise LexError("unterminated string", self._start_line, self._start_column)

        self._advance()  # closing quote
        self._add_token(TokenType.STRING, "".join(chars))

    def _scan_identifier(self) -> None:
        while self._peek().isalnum() or self._peek() == "_":
            self._advance()

        lexeme = self.source[self.start : self.current]
        self._add_token(KEYWORD_ALIASES.get(lexeme, TokenType.IDENTIFIER))

    def _add_token(self, token_type: TokenType, literal: object = None) -> None:
        lexeme = self.source[self.start : self.current]
        self.tokens.append(Token(token_type, lexeme, literal, self._start_line, self._start_column))

    def _is_at_end(self) -> bool:
        return self.current >= len(self.source)

    def _advance(self) -> str:
        c = self.source[self.current]
        self.current += 1
        self.column += 1
        return c

    def _match(self, expected: str) -> bool:
        if self._is_at_end() or self.source[self.current] != expected:
            return False
        self.current += 1
        self.column += 1
        return True

    def _peek(self) -> str:
        if self._is_at_end():
            return "\0"
        return self.source[self.current]

    def _peek_next(self) -> str:
        if self.current + 1 >= len(self.source):
            return "\0"
        return self.source[self.current + 1]
