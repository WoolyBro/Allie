"""Allie's tokenizer.

## Lexical structure

The complete rules (docs/grammar.md "Lexical structure" restates them
for people writing Allie rather than reading this file):

- **Whitespace** is space, tab, CR and LF. A byte-order mark at the very
  start of the file -- Windows Notepad writes one -- is skipped.
- **Comments** are `// to end of line` and `/* block */`. Block comments
  nest, so commenting out a region that already contains one works.
- **Identifiers** are ASCII: `[A-Za-z_][A-Za-z0-9_]*`. Keyword lookup is
  a single case-sensitive dict lookup in `KEYWORD_ALIASES` (`Ahhh` is
  `while`, `ahhh` is an identifier), and the lexer never branches on
  which spelling of a keyword it saw -- nothing downstream can tell.
- **Numbers**: `42`, `3.14`, `1e-4`, `2.5E6`, `10_000`. A fractional
  part or an exponent makes a Float. `_` may only sit between two
  digits. A digit is required on both sides of `.`: `1.` lexes as
  NUMBER + DOT, and `.5` is reported with a "write 0.5" hint. An Int
  literal must fit in a signed 64-bit integer, because `Int` is `i64`
  under LLVM and a larger literal would silently mean different things
  on different backends.
- **Strings** are double-quoted and end on the line they start on.
  Escapes: `\\n \\t \\r \\" \\\\`. A missing closing quote is reported at
  the opening quote and stops at the end of that line, so one typo
  can't swallow the rest of the file.
- **`and`/`or` are keywords**, not `&&`/`||` -- no two-character
  lookahead cases, and TokenType has no AMP_AMP/PIPE_PIPE. Writing `&&`
  is an error whose hint says to use `and`.

Everything outside strings and comments must be ASCII. That rules out
look-alike letters (a Cyrillic `\u0430` in place of a Latin `a`) and
invisible characters pasted from a document or chat, which would
otherwise surface as a baffling "undefined name" several stages later.
Error messages name such characters by code point, since the raw
character is either invisible or indistinguishable from the real one.

## Errors

`scan_tokens()` never raises. Each problem is appended to `self.errors`
as a LexError -- with a caret length and, where the mistake is
predictable, a hint -- and scanning resumes. Where the intent is
obvious the lexer also emits the token that was meant (a STRING for an
unterminated or single-quoted string, a NUMBER for `.5` or `1_000_`),
so the parser does not stack follow-on errors on top of the real one.
A run of adjacent invalid characters is reported once, not once per
character. At most one error is reported per number literal.
"""

from __future__ import annotations

import math
import re
import string
import unicodedata

from allie.lexer.errors import LexError
from allie.lexer.keywords import KEYWORD_ALIASES
from allie.lexer.tokens import Token, TokenType

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
    "r": "\r",
    '"': '"',
    "\\": "\\",
}

_DIGITS = frozenset(string.digits)
_IDENT_START = frozenset(string.ascii_letters + "_")
_IDENT_CONTINUE = _IDENT_START | _DIGITS
_WHITESPACE = frozenset(" \t\r\n")
# ASCII-only by construction: `\w` would also match non-ASCII letters.
_IDENT_TAIL = re.compile(r"[A-Za-z0-9_]*")

_INT64_MAX = 2**63 - 1

# Strings written with the wrong quote characters, mapped to the closers
# accepted when recovering the STRING token that was meant. Curly quotes
# are what word processors and chat apps substitute for `"`.
_WRONG_QUOTES: dict[str, str] = {
    "'": "'",
    "\u2018": "\u2018\u2019'",
    "\u2019": "\u2018\u2019'",
    "\u201c": '\u201c\u201d"',
    "\u201d": '\u201c\u201d"',
    "\u201e": '\u201c\u201d"',
}

# Characters that can begin a token, plus whitespace.
_VALID_START = (
    _WHITESPACE | _IDENT_START | _DIGITS | frozenset(_SIMPLE_SINGLE_CHAR) | frozenset('-=!<>/"')
)
# Invalid characters with a known cause, handled individually so they get
# a specific hint instead of the generic "unexpected character".
_EXPLAINED_INVALID = frozenset("&|#") | frozenset(_WRONG_QUOTES)

_NON_ASCII_HINT = (
    "names, numbers and operators must be ASCII; non-ASCII text is fine inside strings and comments"
)
_ESCAPES_HINT = 'valid escapes are \\n \\t \\r \\" \\\\'


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
        self._scanned = False

    def scan_tokens(self) -> list[Token]:
        """Tokenizes the whole source; the list always ends with EOF.

        Never raises -- check `self.errors` afterwards. Calling it again
        returns the same list rather than appending a second copy.
        """
        if self._scanned:
            return self.tokens
        self._scanned = True

        if self.source.startswith("\ufeff"):
            self.current = 1  # BOM: invisible, so the column stays 1

        while not self._is_at_end():
            self._begin_token()
            self.scan_token()

        self._begin_token()
        self._add_token(TokenType.EOF)
        # Source order, for the reader. Scanning order differs in one case:
        # a bad escape inside a string is found before the string turns
        # out to be unterminated, but the string's error starts earlier.
        self.errors.sort(key=lambda error: (error.line, error.column))
        return self.tokens

    def scan_token(self) -> None:
        c = self._advance()

        if c in _WHITESPACE:
            return

        if c == "/":
            if self._match("/"):
                self._skip_line_comment()
            elif self._match("*"):
                self._skip_block_comment()
            else:
                self._add_token(TokenType.SLASH)
            return

        if c == "." and self._peek() in _DIGITS:
            self._scan_number(leading_dot=True)
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

        if c in _DIGITS:
            self._scan_number()
            return

        if c in _IDENT_START:
            self._scan_identifier()
            return

        self._invalid_character(c)

    # ---- comments ----

    def _skip_line_comment(self) -> None:
        newline = self.source.find("\n", self.current)
        self._advance_within_line(len(self.source) if newline == -1 else newline)

    def _skip_block_comment(self) -> None:
        """The opening `/*` has been consumed. Nests: each inner `/*`
        needs its own `*/`."""
        depth = 1
        while depth > 0:
            if self._is_at_end():
                self._error(
                    "unterminated block comment",
                    length=2,
                    hint="this '/*' never closes with '*/' (block comments nest)",
                )
                return
            c = self._advance()
            if c == "/" and self._match("*"):
                depth += 1
            elif c == "*" and self._match("/"):
                depth -= 1

    # ---- numbers ----

    def _scan_number(self, *, leading_dot: bool = False) -> None:
        """The first character -- a digit, or the `.` of `.5` -- has been
        consumed."""
        separators_ok = self._scan_digits()
        is_float = leading_dot
        if not leading_dot and self._peek() == "." and self._peek_next() in _DIGITS:
            is_float = True
            self._advance()
            separators_ok = self._scan_digits() and separators_ok
        if self._peek() in ("e", "E") and (
            self._peek_next() in _DIGITS
            or (self._peek_next() in ("+", "-") and self._peek_at(2) in _DIGITS)
        ):
            is_float = True
            self._advance()
            if self._peek() in ("+", "-"):
                self._advance()
            separators_ok = self._scan_digits() and separators_ok

        number_text = self.source[self.start : self.current]
        digits = number_text.replace("_", "")
        literal: int | float
        if is_float:
            literal = float(digits)
        else:
            # Checked by length before converting: past 19 digits it can't
            # fit i64 anyway, and CPython refuses int() on very long digit
            # strings (sys.get_int_max_str_digits) with a ValueError.
            significant = digits.lstrip("0") or "0"
            literal = int(significant) if len(significant) <= 19 else _INT64_MAX + 1

        # A letter straight after the digits (`10px`, `1e`, `0x1F`) is
        # never valid; swallow it into this token so it isn't re-lexed as
        # a separate identifier and reported a second time by the parser.
        suffix_start = self.current
        while self._peek() in _IDENT_CONTINUE:
            self._advance()
        suffix = self.source[suffix_start : self.current]

        if suffix:
            message, hint = _describe_number_suffix(number_text, suffix)
            self._error(message, hint=hint)
        elif not separators_ok:
            self._error(
                f"misplaced '_' in number literal {number_text!r}",
                hint="'_' can only separate two digits, as in 10_000",
            )
        elif leading_dot:
            self._error(
                f"number literal {number_text!r} needs a digit before the '.'",
                hint=f"write 0{number_text}",
            )
        elif isinstance(literal, int) and literal > _INT64_MAX:
            self._error(
                f"integer literal is too large for Int (maximum {_INT64_MAX})",
                hint="Int is a signed 64-bit integer; use a Float literal such as 1e19",
            )
        elif isinstance(literal, float) and math.isinf(literal):
            self._error(f"number literal {number_text!r} is too large for Float")

        self._add_token(TokenType.NUMBER, literal)

    def _scan_digits(self) -> bool:
        """Consumes a run of digits and digit separators. Returns False if
        an `_` was not directly between two digits."""
        well_formed = True
        while self._peek() in _DIGITS or self._peek() == "_":
            if self._peek() == "_" and (
                self.source[self.current - 1] not in _DIGITS or self._peek_next() not in _DIGITS
            ):
                well_formed = False
            self._advance()
        return well_formed

    # ---- strings ----

    def _scan_string(self) -> None:
        """The opening quote has been consumed. Always emits a STRING
        token, even when unterminated, so the parser sees the shape the
        author meant."""
        chars: list[str] = []
        while True:
            if self._is_at_end() or self._peek() == "\n":
                self._error(
                    "unterminated string literal",
                    hint="close it with '\"' on the same line; write \\n for a line break",
                )
                break
            c = self._advance()
            if c == '"':
                break
            if c == "\\":
                self._scan_escape(chars)
                continue
            chars.append(c)
        self._add_token(TokenType.STRING, "".join(chars))

    def _scan_escape(self, chars: list[str]) -> None:
        """The backslash has been consumed."""
        if self._is_at_end() or self._peek() in ("\r", "\n"):
            return  # reported by _scan_string as an unterminated string
        line, column = self.line, self.column - 1
        escape = self._advance()
        if escape in _STRING_ESCAPES:
            chars.append(_STRING_ESCAPES[escape])
            return
        self.errors.append(
            LexError(
                f"unknown escape sequence '\\{escape}'", line, column, length=2, hint=_ESCAPES_HINT
            )
        )
        chars.append(escape)

    def _scan_wrong_quote(self, opener: str) -> None:
        """A string opened with `'` or a curly quote. If it closes on the
        same line, emit the STRING it was obviously meant to be."""
        if opener == "'":
            message = "strings use double quotes, not single quotes"
            hint = "write \"...\" instead of '...'"
        else:
            message = f"curly quote {_describe_char(opener)} cannot delimit a string"
            hint = (
                'replace it with a straight double quote (") -- '
                "word processors and chat apps swap them in"
            )

        closers = _WRONG_QUOTES[opener] + "\n"
        end = self.current
        while end < len(self.source) and self.source[end] not in closers:
            end += 1
        if end == len(self.source) or self.source[end] == "\n":
            self._error(message, hint=hint)
            return

        content = self.source[self.current : end]
        while self.current <= end:
            self._advance()
        self._error(message, hint=hint)
        self._add_token(TokenType.STRING, content)

    # ---- identifiers ----

    def _scan_identifier(self) -> None:
        tail = _IDENT_TAIL.match(self.source, self.current)
        assert tail is not None  # a `*` pattern always matches, possibly empty
        self._advance_within_line(tail.end())
        lexeme = self.source[self.start : self.current]
        self._add_token(KEYWORD_ALIASES.get(lexeme, TokenType.IDENTIFIER))

    # ---- invalid input ----

    def _invalid_character(self, c: str) -> None:
        if c in ("&", "|"):
            word = "and" if c == "&" else "or"
            shown = c * 2 if self._match(c) else c
            self._error(f"unexpected {shown!r}", hint=f"logical {word.upper()} is spelled '{word}'")
            return
        if c == "#":
            self._error("'#' does not start a comment", hint="comments start with '//'")
            self._skip_line_comment()
            return
        if c in _WRONG_QUOTES:
            self._scan_wrong_quote(c)
            return

        # Unexplained: report the whole run of them as one error.
        while (
            not self._is_at_end()
            and _is_unexplained_invalid(self._peek())
            and self._peek().isascii() == c.isascii()
        ):
            self._advance()
        text = self.source[self.start : self.current]
        noun = "character" if len(text) == 1 else "characters"
        if c.isascii():
            self._error(f"unexpected {noun} {text!r}")
        else:
            self._error(
                f"non-ASCII {noun} {_describe(text)} outside a string or comment",
                hint=_NON_ASCII_HINT,
            )

    # ---- helpers ----

    def _begin_token(self) -> None:
        self.start = self.current
        self._start_line = self.line
        self._start_column = self.column

    def _add_token(self, token_type: TokenType, literal: object = None) -> None:
        lexeme = self.source[self.start : self.current]
        self.tokens.append(
            Token(token_type, lexeme, literal, self._start_line, self._start_column, self.start)
        )

    def _error(self, message: str, *, length: int | None = None, hint: str | None = None) -> None:
        """Records an error at the start of the current token, by default
        underlining everything consumed for it so far."""
        self.errors.append(
            LexError(
                message,
                self._start_line,
                self._start_column,
                length=self.current - self.start if length is None else length,
                hint=hint,
            )
        )

    def _is_at_end(self) -> bool:
        return self.current >= len(self.source)

    def _advance(self) -> str:
        c = self.source[self.current]
        self.current += 1
        if c == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return c

    def _advance_within_line(self, end: int) -> None:
        """Jumps to `end` in one step. Only for spans known to contain no
        newline (identifier tails, the rest of a `//` comment) -- the fast
        path for the two things most source characters belong to."""
        self.column += end - self.current
        self.current = end

    def _match(self, expected: str) -> bool:
        if self._is_at_end() or self.source[self.current] != expected:
            return False
        self._advance()
        return True

    def _peek(self) -> str:
        return self._peek_at(0)

    def _peek_next(self) -> str:
        return self._peek_at(1)

    def _peek_at(self, distance: int) -> str:
        index = self.current + distance
        return self.source[index] if index < len(self.source) else "\0"


def _is_unexplained_invalid(c: str) -> bool:
    return c not in _VALID_START and c not in _EXPLAINED_INVALID


def _describe_number_suffix(number_text: str, suffix: str) -> tuple[str, str]:
    """(message, hint) for a number immediately followed by letters."""
    if number_text == "0" and suffix[0] in "xXbBoO":
        return (
            f"unsupported number literal '0{suffix}'",
            "only decimal literals are supported",
        )
    if suffix[0] in "eE":
        return (
            f"malformed exponent in '{number_text}{suffix}'",
            "an exponent needs digits, as in 1e5 or 2.5e-3",
        )
    return (
        f"invalid suffix {suffix!r} on number literal {number_text!r}",
        "names cannot start with a digit; put an operator or space between a number and a name",
    )


def _describe(text: str) -> str:
    """Names non-ASCII characters so look-alikes and invisible ones can
    be identified. Long visible runs (a word in another script) are just
    quoted."""
    if len(text) > 3 and all(ch.isprintable() and not ch.isspace() for ch in text):
        return repr(text)
    return ", ".join(_describe_char(ch) for ch in dict.fromkeys(text))


def _describe_char(ch: str) -> str:
    """`'\u0435' (U+0435 CYRILLIC SMALL LETTER IE)`, or just the code point and
    name for a character that would print as nothing."""
    name = unicodedata.name(ch, "")
    code = f"U+{ord(ch):04X}" + (f" {name}" if name else "")
    if ch.isprintable() and not ch.isspace():
        return f"'{ch}' ({code})"
    return code
