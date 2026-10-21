"""Tests for allie/lexer/: the token stream, every lexical error path,
position bookkeeping, error rendering, and the CLI's handling of lex
errors. The lexer's contract is that `scan_tokens()` never raises, so
most error tests assert both the recorded LexError *and* the recovery
token the parser will see."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from allie.cli import run_pipeline
from allie.lexer.errors import LexError
from allie.lexer.keywords import KEYWORD_ALIASES, render_keywords_markdown
from allie.lexer.lexer import Lexer
from allie.lexer.tokens import Token, TokenType

_REPO = Path(__file__).resolve().parents[2]

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
    TokenType.WITH: "with",
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


def _lex(source: str) -> tuple[list[Token], list[LexError]]:
    lexer = Lexer(source)
    tokens = lexer.scan_tokens()
    return tokens, lexer.errors


def _types(source: str) -> list[TokenType]:
    return [t.type for t in Lexer(source).scan_tokens()]


def _only_error(source: str) -> LexError:
    _, errors = _lex(source)
    assert len(errors) == 1, [str(e) for e in errors]
    return errors[0]


# ---- tokens ----


@pytest.mark.parametrize("token_type", list(TokenType), ids=lambda t: t.name)
def test_every_token_type_in_isolation(token_type: TokenType) -> None:
    tokens = Lexer(_TOKEN_SAMPLES[token_type]).scan_tokens()
    assert tokens[0].type == token_type


def test_exact_token_sequence_for_indicator_assignment() -> None:
    assert _types("let sma20 = SMA(prices.close, 20)") == [
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


def test_equals_equals_vs_equals_equals_separate() -> None:
    assert _types("==") == [TokenType.EQUALS_EQUALS, TokenType.EOF]
    assert _types("= =") == [TokenType.EQUALS, TokenType.EQUALS, TokenType.EOF]


def test_less_equals_vs_less_equals_separate() -> None:
    assert _types("<=") == [TokenType.LESS_EQUALS, TokenType.EOF]
    assert _types("< =") == [TokenType.LESS, TokenType.EQUALS, TokenType.EOF]


def test_scan_tokens_is_idempotent() -> None:
    lexer = Lexer("let x = @;")
    first = lexer.scan_tokens()
    second = lexer.scan_tokens()
    assert second is first
    assert len(first) == 5
    assert len(lexer.errors) == 1


# ---- numbers ----


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("42", 42),
        ("0", 0),
        ("3.14", 3.14),
        ("1e5", 1e5),
        ("1E5", 1e5),
        ("2.5e-3", 0.0025),
        ("1e+2", 100.0),
        ("10_000", 10_000),
        ("1_000.000_1", 1000.0001),
        ("1_0e1_0", 1e11),
        ("9223372036854775807", 2**63 - 1),
    ],
)
def test_number_literals(source: str, expected: object) -> None:
    tokens, errors = _lex(source)
    assert errors == []
    assert tokens[0].type == TokenType.NUMBER
    assert tokens[0].literal == expected
    assert type(tokens[0].literal) is type(expected)
    assert tokens[0].lexeme == source


def test_trailing_dot_is_number_then_dot() -> None:
    # No digit after the dot -- "1." is NUMBER(1) + DOT, not a malformed float.
    tokens, errors = _lex("1.")
    assert errors == []
    assert [t.type for t in tokens] == [TokenType.NUMBER, TokenType.DOT, TokenType.EOF]
    assert tokens[0].literal == 1
    assert type(tokens[0].literal) is int


def test_leading_dot_float_is_an_error_that_recovers_as_the_intended_number() -> None:
    tokens, errors = _lex(".5")
    assert [t.type for t in tokens] == [TokenType.NUMBER, TokenType.EOF]
    assert tokens[0].literal == 0.5
    assert len(errors) == 1
    assert "needs a digit before the '.'" in errors[0].message
    assert errors[0].hint == "write 0.5"


def test_field_access_is_not_mistaken_for_a_leading_dot_float() -> None:
    tokens, errors = _lex("AAPL.close")
    assert errors == []
    assert [t.type for t in tokens] == [
        TokenType.IDENTIFIER,
        TokenType.DOT,
        TokenType.IDENTIFIER,
        TokenType.EOF,
    ]


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("1__0", "misplaced '_'"),
        ("1_", "misplaced '_'"),
        ("1e", "malformed exponent in '1e'"),
        ("10px", "invalid suffix 'px'"),
        ("2.5f", "invalid suffix 'f'"),
        ("0x1F", "unsupported number literal '0x1F'"),
        ("0b101", "unsupported number literal '0b101'"),
        ("9223372036854775808", "too large for Int"),
        ("1e999", "too large for Float"),
    ],
)
def test_malformed_number_reports_one_error_spanning_the_literal(
    source: str, fragment: str
) -> None:
    tokens, errors = _lex(source)
    assert len(errors) == 1, [str(e) for e in errors]
    assert fragment in errors[0].message
    assert (errors[0].line, errors[0].column, errors[0].length) == (1, 1, len(source))
    # The literal still becomes one NUMBER token, so the parser sees a
    # number where the author wrote one.
    assert [t.type for t in tokens] == [TokenType.NUMBER, TokenType.EOF]


def test_exponent_without_digits_leaves_the_sign_as_its_own_token() -> None:
    tokens, errors = _lex("1e+")
    assert len(errors) == 1
    assert [t.type for t in tokens] == [TokenType.NUMBER, TokenType.PLUS, TokenType.EOF]


def test_at_most_one_error_per_number_literal() -> None:
    error = _only_error("1__0px")
    assert "invalid suffix" in error.message


def test_enormous_integer_literal_is_reported_not_crashed_on() -> None:
    # CPython refuses int() on digit strings past sys.get_int_max_str_digits
    # (4300 by default) with a ValueError.
    error = _only_error("9" * 5000)
    assert "too large for Int" in error.message


@pytest.mark.parametrize("digit", ["\u00b2", "\u0663", "\uff15"])
def test_non_ascii_digits_are_reported_not_crashed_on(digit: str) -> None:
    # str.isdigit() accepts these, and int() then rejects them: the
    # original lexer crashed with a raw ValueError on `let x = \u00b2;`.
    tokens, errors = _lex(f"let x = {digit};")
    assert len(errors) == 1
    assert f"U+{ord(digit):04X}" in errors[0].message
    assert TokenType.NUMBER not in [t.type for t in tokens]


# ---- strings ----


def test_string_literal_basic() -> None:
    token = Lexer('"hello"').scan_tokens()[0]
    assert token.type == TokenType.STRING
    assert token.literal == "hello"
    assert token.lexeme == '"hello"'


def test_string_literal_with_escapes() -> None:
    token = Lexer(r'"line1\nline2\ttab\"quote\\backslash\rcr"').scan_tokens()[0]
    assert token.literal == 'line1\nline2\ttab"quote\\backslash\rcr'


def test_non_ascii_is_allowed_inside_strings() -> None:
    tokens, errors = _lex('"h\u00e9llo \u2013 \u00fc \u201cquoted\u201d \u00b2"')
    assert errors == []
    assert tokens[0].literal == "h\u00e9llo \u2013 \u00fc \u201cquoted\u201d \u00b2"


def test_unterminated_string_is_recorded_not_raised() -> None:
    tokens, errors = _lex('"never closed')
    assert len(errors) == 1
    assert "unterminated string" in errors[0].message
    assert (errors[0].line, errors[0].column) == (1, 1)
    assert tokens[0].type == TokenType.STRING
    assert tokens[0].literal == "never closed"


def test_unterminated_string_stops_at_end_of_line() -> None:
    # A missing quote must not swallow the rest of the file.
    tokens, errors = _lex('let s = "abc;\nlet y = 2;')
    assert len(errors) == 1
    assert (errors[0].line, errors[0].column) == (1, 9)
    assert [t.type for t in tokens][-6:] == [
        TokenType.LET,
        TokenType.IDENTIFIER,
        TokenType.EQUALS,
        TokenType.NUMBER,
        TokenType.SEMICOLON,
        TokenType.EOF,
    ]
    assert (tokens[-6].line, tokens[-6].column) == (2, 1)


def test_unknown_escape_points_at_the_backslash_and_keeps_the_character() -> None:
    tokens, errors = _lex(r'"a\qb"')
    assert len(errors) == 1
    assert errors[0].message == "unknown escape sequence '\\q'"
    assert (errors[0].column, errors[0].length) == (3, 2)
    assert tokens[0].literal == "aqb"


def test_errors_come_back_in_source_order() -> None:
    # The bad escape is found first, but the unterminated string starts first.
    _, errors = _lex('"a\\qb')
    assert [e.column for e in errors] == [1, 3]


def test_single_quoted_string_recovers_as_a_string() -> None:
    tokens, errors = _lex("let s = 'abc';")
    assert len(errors) == 1
    assert "double quotes" in errors[0].message
    assert [t.type for t in tokens][3:] == [TokenType.STRING, TokenType.SEMICOLON, TokenType.EOF]
    assert tokens[3].literal == "abc"


def test_unclosed_single_quote_reports_once_and_continues() -> None:
    tokens, errors = _lex("let s = 'abc\nlet y = 1;")
    assert len(errors) == 1
    assert TokenType.LET in [t.type for t in tokens[1:]]


@pytest.mark.parametrize(
    ("opener", "closer"), [("\u201c", "\u201d"), ("\u2018", "\u2019"), ("\u201e", "\u201c")]
)
def test_curly_quotes_are_named_and_recover_as_a_string(opener: str, closer: str) -> None:
    tokens, errors = _lex(f"print({opener}hi{closer});")
    assert len(errors) == 1
    assert f"U+{ord(opener):04X}" in errors[0].message
    assert tokens[2].type == TokenType.STRING
    assert tokens[2].literal == "hi"


# ---- comments ----


def test_full_line_comment_produces_no_tokens() -> None:
    assert _types("// just a comment") == [TokenType.EOF]


def test_trailing_comment_ignored() -> None:
    assert _types("let x = 1; // set x") == [
        TokenType.LET,
        TokenType.IDENTIFIER,
        TokenType.EQUALS,
        TokenType.NUMBER,
        TokenType.SEMICOLON,
        TokenType.EOF,
    ]


def test_comment_containing_operators_not_tokenized() -> None:
    assert _types("// == != <= this is not code") == [TokenType.EOF]


def test_non_ascii_is_allowed_inside_comments() -> None:
    _, errors = _lex("// prix en \u20ac, \u00b2  \n/* na\u00efve \u201cquotes\u201d */ let x = 1;")
    assert errors == []


def test_block_comment() -> None:
    assert _types("a /* ignored == */ b") == [
        TokenType.IDENTIFIER,
        TokenType.IDENTIFIER,
        TokenType.EOF,
    ]


def test_block_comments_nest() -> None:
    tokens, errors = _lex("/* outer /* inner */ still comment */ x")
    assert errors == []
    assert [t.type for t in tokens] == [TokenType.IDENTIFIER, TokenType.EOF]


def test_multiline_block_comment_keeps_line_numbers_right() -> None:
    tokens = Lexer("/* one\ntwo\nthree */ x").scan_tokens()
    assert (tokens[0].lexeme, tokens[0].line, tokens[0].column) == ("x", 3, 10)


def test_unterminated_block_comment_points_at_its_opening() -> None:
    _, errors = _lex("let x = 1;\n/* never\nclosed")
    assert len(errors) == 1
    assert "unterminated block comment" in errors[0].message
    assert (errors[0].line, errors[0].column, errors[0].length) == (2, 1, 2)


def test_lone_slash_is_still_division() -> None:
    assert _types("a/b") == [
        TokenType.IDENTIFIER,
        TokenType.SLASH,
        TokenType.IDENTIFIER,
        TokenType.EOF,
    ]


# ---- invalid characters ----


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


def test_adjacent_invalid_characters_are_one_error() -> None:
    error = _only_error("x @@@ y")
    assert error.message == "unexpected characters '@@@'"
    assert (error.column, error.length) == (3, 3)


@pytest.mark.parametrize(
    ("source", "shown", "word"),
    [("a && b", "&&", "and"), ("a || b", "||", "or"), ("a & b", "&", "and"), ("a | b", "|", "or")],
)
def test_c_style_logical_operators_get_a_hint(source: str, shown: str, word: str) -> None:
    error = _only_error(source)
    assert error.message == f"unexpected {shown!r}"
    assert error.hint is not None and f"'{word}'" in error.hint
    assert error.length == len(shown)


def test_hash_comment_is_reported_and_skipped_like_a_comment() -> None:
    tokens, errors = _lex("# note: not a comment\nlet x = 1;")
    assert len(errors) == 1
    assert errors[0].hint == "comments start with '//'"
    assert [t.type for t in tokens][0] == TokenType.LET


def test_look_alike_letter_is_named() -> None:
    # The second letter is CYRILLIC SMALL LETTER IE, not Latin 'e'.
    error = _only_error("let pric\u0435 = 1;")
    assert "U+0435 CYRILLIC SMALL LETTER IE" in error.message
    assert error.column == 9


@pytest.mark.parametrize(
    ("char", "name"),
    [
        ("\u00a0", "NO-BREAK SPACE"),
        ("\u200b", "ZERO WIDTH SPACE"),
        ("\ufeff", "ZERO WIDTH NO-BREAK SPACE"),
    ],
)
def test_invisible_characters_are_named_by_code_point(char: str, name: str) -> None:
    error = _only_error(f"let{char}x = 1;")
    assert f"U+{ord(char):04X} {name}" in error.message
    assert error.hint is not None and "ASCII" in error.hint


def test_word_in_another_script_is_one_error() -> None:
    error = _only_error("\u043f\u0440\u0438\u0432\u0435\u0442 = 1;")
    assert "'\u043f\u0440\u0438\u0432\u0435\u0442'" in error.message
    assert error.length == 6


def test_byte_order_mark_at_start_is_skipped() -> None:
    tokens, errors = _lex("\ufefflet x = 1;")
    assert errors == []
    assert (tokens[0].type, tokens[0].line, tokens[0].column) == (TokenType.LET, 1, 1)


# ---- positions ----


def test_line_and_column_tracking_multiline() -> None:
    tokens = Lexer("let x = 1;\nlet y = 2;").scan_tokens()

    assert (tokens[0].type, tokens[0].line, tokens[0].column) == (TokenType.LET, 1, 1)
    assert (tokens[1].type, tokens[1].line, tokens[1].column) == (TokenType.IDENTIFIER, 1, 5)
    assert (tokens[2].type, tokens[2].line, tokens[2].column) == (TokenType.EQUALS, 1, 7)
    assert (tokens[3].type, tokens[3].line, tokens[3].column) == (TokenType.NUMBER, 1, 9)
    assert (tokens[4].type, tokens[4].line, tokens[4].column) == (TokenType.SEMICOLON, 1, 10)
    assert (tokens[5].type, tokens[5].line, tokens[5].column) == (TokenType.LET, 2, 1)
    assert (tokens[6].type, tokens[6].line, tokens[6].column) == (TokenType.IDENTIFIER, 2, 5)


def test_crlf_line_endings_give_the_same_positions_as_lf() -> None:
    lf = Lexer("let x = 1;\nlet y = 2;\n").scan_tokens()
    crlf = Lexer("let x = 1;\r\nlet y = 2;\r\n").scan_tokens()
    assert [(t.type, t.line, t.column) for t in lf] == [(t.type, t.line, t.column) for t in crlf]


def _assert_positions_consistent(source: str, tokens: list[Token]) -> None:
    """Every token's lexeme is the exact source slice at its offset, and
    its line/column agree with that offset."""
    bom = 1 if source.startswith("\ufeff") else 0
    for token in tokens:
        assert source[token.offset : token.offset + len(token.lexeme)] == token.lexeme
        line_start = source.rfind("\n", 0, token.offset) + 1
        expected_line = source.count("\n", 0, token.offset) + 1
        expected_column = token.offset - line_start + 1 - (bom if expected_line == 1 else 0)
        assert (token.line, token.column) == (expected_line, expected_column), token
    offsets = [t.offset for t in tokens]
    assert offsets == sorted(offsets)
    assert tokens[-1].type == TokenType.EOF
    assert tokens[-1].offset == len(source)


_PROGRAMS = sorted(
    [*_REPO.glob("examples/*.qs"), *_REPO.glob("tests/**/*.qs")], key=lambda p: p.as_posix()
)


@pytest.mark.parametrize("path", _PROGRAMS, ids=lambda p: p.name)
def test_every_program_in_the_repo_lexes_cleanly_with_exact_positions(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tokens, errors = _lex(source)
    assert errors == []
    _assert_positions_consistent(source, tokens)


def test_parser_spans_carry_real_source_offsets() -> None:
    from allie.ast.nodes import LetStmt
    from allie.parser.parser import Parser

    source = "let a = 1;\nlet total = 2;"
    program = Parser(Lexer(source).scan_tokens()).parse()
    second = program.statements[1]
    assert isinstance(second, LetStmt)
    assert source[second.span.offset : second.span.offset + second.span.length] == "let"
    assert second.span.offset == source.index("let total")


# ---- robustness ----

_FUZZ_PIECES = [
    *"abcxyzAEe_019.+-*/%=!<>(){}[],:;",
    '"', "'", "\\", "\n", "\r\n", "\t", " ", "#", "&", "|", "@", "$",
    "\u00b2", "\u00e9", "\u00a0", "\u200b", "\ufeff", "\u0435",
    "\u201c", "\u201d", "\u2018", "\u2019",
    "//", "/*", "*/", "1e", "1e-", "1_", "__", "0x", ".5", "\\n", "\\q",
    "let", "Ahhh", "crosses_above", "9223372036854775808",
]


def test_random_input_never_raises_and_keeps_every_invariant() -> None:
    rng = random.Random(20261020)
    for _ in range(3000):
        source = "".join(rng.choice(_FUZZ_PIECES) for _ in range(rng.randint(0, 40)))
        tokens, errors = _lex(source)  # must not raise
        _assert_positions_consistent(source, tokens)
        line_count = source.count("\n") + 1
        for error in errors:
            assert 1 <= error.line <= line_count, (source, str(error))
            assert error.column >= 1 and error.length >= 1
            error.render(source)  # must not raise either
        assert [(e.line, e.column) for e in errors] == sorted((e.line, e.column) for e in errors)


# ---- rendering ----


def test_render_puts_the_caret_under_the_error() -> None:
    source = "let ok = 1;\nif a && b { }"
    error = _only_error(source)
    assert error.render(source).split("\n") == [
        "LexError: unexpected '&&'. Line 2, Column 6",
        "  if a && b { }",
        "       ^^",
        "  help: logical AND is spelled 'and'",
    ]


def test_render_keeps_tabs_so_the_caret_lines_up() -> None:
    source = "\t\tlet x = @;"
    caret_line = _only_error(source).render(source).split("\n")[2]
    assert caret_line == "  \t\t        ^"


def test_render_ignores_a_leading_byte_order_mark() -> None:
    source = "\ufefflet x = @;"
    lines = _only_error(source).render(source).split("\n")
    assert lines[1] == "  let x = @;"
    assert lines[2] == "          ^"


def test_render_windows_a_very_long_line() -> None:
    source = "let x = " + "a + " * 100 + "@" + " + b" * 100 + ";"
    lines = _only_error(source).render(source).split("\n")
    assert len(lines[1]) < 120
    assert lines[1][lines[2].index("^")] == "@"


def test_render_without_a_matching_source_line_falls_back_to_str() -> None:
    error = LexError("boom", 99, 1)
    assert error.render("one line") == str(error)


# ---- keywords ----


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
    assert _types(source) == [
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
    docs_path = _REPO / "docs" / "keywords.md"
    assert docs_path.read_text(encoding="utf-8") == render_keywords_markdown(), (
        "docs/keywords.md is stale -- run python scripts/generate_keywords_doc.py"
    )


def test_language_reference_keyword_block_matches_generated_output() -> None:
    """The reference doc's keyword table is generated from the same dict
    (Phase 9's checklist: generated, not hand-typed). Two documents
    rendering the same source of truth can still drift if only one is
    regenerated -- this pins both."""
    import sys

    sys.path.insert(0, str(_REPO / "scripts"))
    from generate_keywords_doc import render_reference_block

    reference = (_REPO / "docs" / "language-reference.md").read_text(encoding="utf-8")
    assert render_reference_block() in reference, (
        "docs/language-reference.md's generated keyword block is stale -- "
        "run python scripts/generate_keywords_doc.py"
    )


# ---- through the CLI ----


def _run(tmp_path: Path, data: bytes, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    path = tmp_path / "p.qs"
    path.write_bytes(data)
    code = run_pipeline(path)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.mark.parametrize(
    "source",
    ['let s: String = "abc;\nprint(s);\n', "let x: Int = \u00b2;\n", "let x = 1e;\n"],
    ids=["unterminated-string", "unicode-digit", "bad-exponent"],
)
def test_cli_reports_lex_errors_with_context_not_a_traceback(
    tmp_path: Path, source: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = _run(tmp_path, source.encode(), capsys)
    assert code == 1
    assert "LexError: " in err
    assert "^" in err
    assert "Traceback" not in err


def test_cli_rejects_a_file_that_is_not_utf8(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = _run(tmp_path, b"let x = 1;\n\xff\xfe\n", capsys)
    assert code == 2  # can't be read: a usage error, like a missing file
    assert "not valid UTF-8 (byte 0xff at offset 11)" in err


def test_cli_caps_the_number_of_lex_errors_shown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = _run(tmp_path, b"@ " * 50, capsys)
    assert code == 1
    assert err.count("LexError: ") == 20
    assert "... and 30 more lexical error(s)" in err


def test_cli_runs_a_windows_file_with_bom_and_crlf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = "\ufefflet n: Int = 10_000;\r\nprint(n);\r\nprint(2.5e-3);\r\n"
    code, out, err = _run(tmp_path, source.encode("utf-8"), capsys)
    assert (code, err) == (0, "")
    assert out.split() == ["10000", "0.0025"]
