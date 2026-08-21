"""Phase 2: tokenizer for Qubit's brace-based syntax (docs/spec.md)."""

from qubit.lexer.errors import LexError
from qubit.lexer.keywords import KEYWORD_ALIASES
from qubit.lexer.lexer import Lexer
from qubit.lexer.tokens import Token, TokenType

__all__ = ["KEYWORD_ALIASES", "LexError", "Lexer", "Token", "TokenType"]
