"""Tests for qubit/lexer/ (Phase 2.1). No parser exists yet -- these only
check the token stream Lexer produces."""

from __future__ import annotations

from pathlib import Path

import pytest

from qubit.lexer.errors import LexError
from qubit.lexer.keywords import KEYWORD_ALIASES, render_keywords_markdown
from qubit.lexer.lexer import Lexer
from qubit.lexer.tokens import TokenType

_TOKEN_SAMPLES: dict[TokenType, str] = {
    TokenType.LET: "let",
    TokenType.FUNC: "func",
    TokenType.IF: "if",
    TokenType.ELSE: "else",
    TokenType.WHILE: "while",
    TokenType.FOR: "for",
    TokenType.BREAK: "break",
    TokenType.CONTINUE: "continue",
    TokenType.RETURN: "return",
    TokenType.IMPORT: "import",
    TokenType.TRUE: "true",
    TokenType.FALSE: "false",
    TokenType.STRATEGY: "strategy",
    TokenType.WHEN: "when",
    TokenType.CROSSES_ABOVE: "crosses_above",
    TokenType.CROSSES_BELOW: "crosses_below",
    TokenType.BACKTEST: "backtest",
    TokenType.ON: "on",
    TokenType.FROM: "from",
    TokenType.TO: "to",
    TokenType.IDENTIFIER: "myVar",
    TokenType.NUMBER: "42",
    TokenType.STRING: '"hi"',
    TokenType.PLUS: "+",
    TokenType.MINUS: "-",
    TokenType.STAR: "*",
    TokenType.SLASH: "/",
    TokenType.PERCENT: "%",
    TokenType.EQUALS: "=",
    TokenType.EQUALS_EQUALS: "==",
    TokenType.BANG: "!",
    TokenType.BANG_EQUALS: "!=",
    TokenType.LESS: "<",
    TokenType.LESS_EQUALS: "<=",
    TokenType.GREATER: ">",
    TokenType.GREATER_EQUALS: ">=",
    TokenType.AND: "and",
    TokenType.OR: "or",
    TokenType.LPAREN: "(",
    TokenType.RPAREN: ")",
    TokenType.LBRACE: "{",
    TokenType.RBRACE: "}",
    TokenType.LBRACKET: "[",
    TokenType.RBRACKET: "]",
    TokenType.COMMA: ",",
    TokenType.DOT: ".",
    TokenType.COLON: ":",
    TokenType.SEMICOLON: ";",
    TokenType.ARROW: "->",
    TokenType.EOF: "",
}


@pytest.mark.parametrize("token_type", list(TokenType), ids=lambda t: t.name)
def test_every_token_type_in_isolation(token_type: TokenType) -> None:
    tokens = Lexer(_TOKEN_SAMPLES[token_type]).scan_tokens()
    assert tokens[0].type == token_type


def test_exact_token_sequence_for_indicator_assignment() -> None:
    tokens = Lexer("let sma20 = SMA(prices.close, 20)").scan_tokens()
    assert [t.type for t in tokens] == [
        TokenType.LET,
        TokenType.IDENTIFIER,
        TokenType.EQUALS,
        TokenType.IDENTIFIER,
        TokenType.LPAREN,
        TokenType.IDENTIFIER,
        TokenType.DOT,
        TokenType.IDENTIFIER,
        TokenType.COMMA,
        TokenType.NUMBER,
        TokenType.RPAREN,
        TokenType.EOF,
    ]


@pytest.mark.parametrize(
    ("source", "expected"),
    [("42", 42), ("3.14", 3.14), ("0", 0)],
)
def test_number_literals(source: str, expected: object) -> None:
    token = Lexer(source).scan_tokens()[0]
    assert token.type == TokenType.NUMBER
    assert token.literal == expected
    assert type(token.literal) is type(expected)


def test_trailing_dot_is_number_then_dot() -> None:
    # No digit after the dot -- "1." is NUMBER(1) + DOT, not a malformed float.
    tokens = Lexer("1.").scan_tokens()
    assert [t.type for t in tokens] == [TokenType.NUMBER, TokenType.DOT, TokenType.EOF]
    assert tokens[0].literal == 1
    assert type(tokens[0].literal) is int


def test_leading_dot_is_dot_then_number() -> None:
    # No digit before the dot -- ".5" is DOT + NUMBER(5), not a malformed float.
    tokens = Lexer(".5").scan_tokens()
    assert [t.type for t in tokens] == [TokenType.DOT, TokenType.NUMBER, TokenType.EOF]
    assert tokens[1].literal == 5


def test_string_literal_basic() -> None:
    token = Lexer('"hello"').scan_tokens()[0]
    assert token.type == TokenType.STRING
    assert token.literal == "hello"


def test_string_literal_with_escapes() -> None:
    token = Lexer(r'"line1\nline2\ttab\"quote\\backslash"').scan_tokens()[0]
    assert token.literal == 'line1\nline2\ttab"quote\\backslash'


def test_unterminated_string_raises() -> None:
    with pytest.raises(LexError):
        Lexer('"never closed').scan_tokens()


def test_full_line_comment_produces_no_tokens() -> None:
    tokens = Lexer("// just a comment").scan_tokens()
    assert [t.type for t in tokens] == [TokenType.EOF]


def test_trailing_comment_ignored() -> None:
    tokens = Lexer("let x = 1; // set x").scan_tokens()
    assert [t.type for t in tokens] == [
        TokenType.LET,
        TokenType.IDENTIFIER,
        TokenType.EQUALS,
        TokenType.NUMBER,
        TokenType.SEMICOLON,
        TokenType.EOF,
    ]


def test_comment_containing_operators_not_tokenized() -> None:
    tokens = Lexer("// == != <= this is not code").scan_tokens()
    assert [t.type for t in tokens] == [TokenType.EOF]


def test_equals_equals_vs_equals_equals_separate() -> None:
    assert [t.type for t in Lexer("==").scan_tokens()] == [TokenType.EQUALS_EQUALS, TokenType.EOF]
    assert [t.type for t in Lexer("= =").scan_tokens()] == [
        TokenType.EQUALS,
        TokenType.EQUALS,
        TokenType.EOF,
    ]


def test_less_equals_vs_less_equals_separate() -> None:
    assert [t.type for t in Lexer("<=").scan_tokens()] == [TokenType.LESS_EQUALS, TokenType.EOF]
    assert [t.type for t in Lexer("< =").scan_tokens()] == [
        TokenType.LESS,
        TokenType.EQUALS,
        TokenType.EOF,
    ]


def test_invalid_character_records_error_and_continues() -> None:
    lexer = Lexer("let x @ = 1;")
    tokens = lexer.scan_tokens()

    assert len(lexer.errors) == 1
    assert lexer.errors[0].line == 1
    assert lexer.errors[0].column == 7  # position of '@'

    assert [t.type for t in tokens] == [
        TokenType.LET,
        TokenType.IDENTIFIER,
        TokenType.EQUALS,
        TokenType.NUMBER,
        TokenType.SEMICOLON,
        TokenType.EOF,
    ]


def test_multiple_invalid_characters_all_collected_in_one_pass() -> None:
    lexer = Lexer("@ $ ~")
    lexer.scan_tokens()
    assert [e.column for e in lexer.errors] == [1, 3, 5]


def test_line_and_column_tracking_multiline() -> None:
    tokens = Lexer("let x = 1;\nlet y = 2;").scan_tokens()

    assert (tokens[0].type, tokens[0].line, tokens[0].column) == (TokenType.LET, 1, 1)
    assert (tokens[1].type, tokens[1].line, tokens[1].column) == (TokenType.IDENTIFIER, 1, 5)
    assert (tokens[2].type, tokens[2].line, tokens[2].column) == (TokenType.EQUALS, 1, 7)
    assert (tokens[3].type, tokens[3].line, tokens[3].column) == (TokenType.NUMBER, 1, 9)
    assert (tokens[4].type, tokens[4].line, tokens[4].column) == (TokenType.SEMICOLON, 1, 10)
    assert (tokens[5].type, tokens[5].line, tokens[5].column) == (TokenType.LET, 2, 1)
    assert (tokens[6].type, tokens[6].line, tokens[6].column) == (TokenType.IDENTIFIER, 2, 5)


def test_keyword_lookup_is_case_sensitive() -> None:
    # "Ahhh" (capital A) is WHILE; "ahhh" (lowercase) is just an identifier.
    # Easy live-demo mistake -- this is why it's asserted explicitly.
    assert Lexer("Ahhh").scan_tokens()[0].type == TokenType.WHILE
    assert Lexer("ahhh").scan_tokens()[0].type == TokenType.IDENTIFIER


@pytest.mark.parametrize("spelling", sorted(KEYWORD_ALIASES))
def test_keyword_alias_matches_canonical_token_type(spelling: str) -> None:
    tokens = Lexer(spelling).scan_tokens()
    assert tokens[0].type == KEYWORD_ALIASES[spelling]


def test_mixed_canonical_goofy_and_quant_keywords_coexist() -> None:
    source = "ts x be lowkey Ahhh strategy when crosses_above backtest on from to"
    tokens = Lexer(source).scan_tokens()
    assert [t.type for t in tokens] == [
        TokenType.LET,  # ts
        TokenType.IDENTIFIER,  # x
        TokenType.FUNC,  # be
        TokenType.IF,  # lowkey
        TokenType.WHILE,  # Ahhh
        TokenType.STRATEGY,
        TokenType.WHEN,
        TokenType.CROSSES_ABOVE,
        TokenType.BACKTEST,
        TokenType.ON,
        TokenType.FROM,
        TokenType.TO,
        TokenType.EOF,
    ]


def test_keywords_doc_matches_generated_output() -> None:
    docs_path = Path(__file__).resolve().parents[2] / "docs" / "keywords.md"
    assert docs_path.read_text(encoding="utf-8") == render_keywords_markdown()
