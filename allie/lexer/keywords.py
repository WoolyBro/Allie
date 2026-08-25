"""Single source of truth for keyword resolution -- both canonical and
slang spellings map to the identical TokenType. allie.lexer.lexer.Lexer
never branches on "goofy vs canonical"; it's one dict lookup, so the
parser and everything downstream stays unaware of which spelling was
used.

Quant keywords (strategy/when/crosses_above/crosses_below/backtest/on/
from/to) are canonical-only in v1 -- the mechanism supports slang
aliases for them uniformly later without touching the lexer.

`and`/`or` are added here beyond the originally pasted table: TokenType
defines AND/OR, and the grammar's logic_and/logic_or productions rely on
them as keyword tokens, not identifiers -- canonical-only for now, same
as the quant keywords, since no slang spelling was specified for them.

docs/keywords.md is generated from this dict (see
scripts/generate_keywords_doc.py) -- tests/unit/test_lexer.py fails if
the two drift apart.
"""

from __future__ import annotations

from allie.lexer.tokens import TokenType

KEYWORD_ALIASES: dict[str, TokenType] = {
    "let": TokenType.LET,
    "ts": TokenType.LET,
    "func": TokenType.FUNC,
    "be": TokenType.FUNC,
    "if": TokenType.IF,
    "lowkey": TokenType.IF,
    "else": TokenType.ELSE,
    "nahhh": TokenType.ELSE,
    "while": TokenType.WHILE,
    "Ahhh": TokenType.WHILE,
    "for": TokenType.FOR,
    "grindset": TokenType.FOR,
    "break": TokenType.BREAK,
    "bruh": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "pullup": TokenType.CONTINUE,
    "return": TokenType.RETURN,
    "yeet": TokenType.RETURN,
    "import": TokenType.IMPORT,
    "fax": TokenType.IMPORT,
    "true": TokenType.TRUE,
    "fr": TokenType.TRUE,
    "false": TokenType.FALSE,
    "pmo": TokenType.FALSE,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "strategy": TokenType.STRATEGY,
    "when": TokenType.WHEN,
    "crosses_above": TokenType.CROSSES_ABOVE,
    "crosses_below": TokenType.CROSSES_BELOW,
    "backtest": TokenType.BACKTEST,
    "on": TokenType.ON,
    "from": TokenType.FROM,
    "to": TokenType.TO,
}


def render_keywords_table() -> str:
    """Just the table. Shared by docs/keywords.md and the generated
    block in docs/language-reference.md, so the two documents can never
    disagree about which spellings exist."""
    by_token: dict[TokenType, list[str]] = {}
    for spelling, token_type in KEYWORD_ALIASES.items():
        by_token.setdefault(token_type, []).append(spelling)

    lines = ["| Token | Canonical | Also accepted |", "|---|---|---|"]
    for token_type, spellings in by_token.items():
        canonical, *aliases = spellings
        alias_cell = ", ".join(f"`{a}`" for a in aliases) if aliases else "—"
        lines.append(f"| `{token_type.name}` | `{canonical}` | {alias_cell} |")
    return "\n".join(lines) + "\n"


def render_keywords_markdown() -> str:
    """Render docs/keywords.md's content from KEYWORD_ALIASES -- the
    single source of truth. See scripts/generate_keywords_doc.py."""
    lines = [
        "# Keywords",
        "",
        "Generated from `allie/lexer/keywords.py`'s `KEYWORD_ALIASES` -- do not "
        "hand-edit; run `python scripts/generate_keywords_doc.py` after changing "
        "that dict. `tests/unit/test_lexer.py` fails if this file drifts from the "
        "code.",
        "",
        "Lookup is case-sensitive: `Ahhh` (while) and `ahhh` are different tokens.",
        "",
        "Quant keywords (`strategy`, `when`, `crosses_above`, `crosses_below`, "
        "`backtest`, `on`, `from`, `to`) are canonical-only in v1 -- the mechanism "
        "supports slang aliases for them uniformly later without touching the "
        "lexer.",
        "",
    ]
    return "\n".join(lines) + "\n" + render_keywords_table()
