"""Token types and the Token record produced by allie.lexer.lexer.

TokenType enumerates every terminal the parser sees. Control-
flow keywords support slang aliases (allie.lexer.keywords); quant
keywords (STRATEGY, WHEN, CROSSES_ABOVE, ...) are canonical-only in v1 --
see docs/keywords.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class TokenType(Enum):
    LET = auto()
    FUNC = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    BREAK = auto()
    CONTINUE = auto()
    RETURN = auto()
    IMPORT = auto()
    TRUE = auto()
    FALSE = auto()
    STRATEGY = auto()
    WHEN = auto()
    CROSSES_ABOVE = auto()
    CROSSES_BELOW = auto()
    BACKTEST = auto()
    ON = auto()
    FROM = auto()
    TO = auto()
    WITH = auto()

    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()

    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    EQUALS = auto()
    EQUALS_EQUALS = auto()
    BANG = auto()
    BANG_EQUALS = auto()
    LESS = auto()
    LESS_EQUALS = auto()
    GREATER = auto()
    GREATER_EQUALS = auto()
    AND = auto()
    OR = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    DOT = auto()
    COLON = auto()
    SEMICOLON = auto()
    ARROW = auto()

    EOF = auto()


@dataclass(frozen=True, slots=True)
class Token:
    """One lexeme and where it came from.

    `lexeme` is always the exact source slice -- quotes, underscores and
    escapes included -- so `source[offset : offset + len(lexeme)] ==
    lexeme` holds for every token (tests/unit/test_lexer.py checks this
    against every example program). `literal` carries the decoded
    value. `line`/`column` are 1-based and count code points; `offset`
    is the 0-based index into the source string.
    """

    type: TokenType
    lexeme: str
    literal: Any
    line: int
    column: int
    offset: int = 0
