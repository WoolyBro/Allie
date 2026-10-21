# Keywords

Generated from `allie/lexer/keywords.py`'s `KEYWORD_ALIASES` -- do not hand-edit; run `python scripts/generate_keywords_doc.py` after changing that dict. `tests/unit/test_lexer.py` fails if this file drifts from the code.

Lookup is case-sensitive: `Ahhh` (while) and `ahhh` are different tokens.

Quant keywords (`strategy`, `when`, `crosses_above`, `crosses_below`, `backtest`, `on`, `from`, `to`, `with`) are canonical-only in v1 -- the mechanism supports slang aliases for them uniformly later without touching the lexer.

| Token | Canonical | Also accepted |
|---|---|---|
| `LET` | `let` | `ts` |
| `FUNC` | `func` | `be` |
| `IF` | `if` | `lowkey` |
| `ELSE` | `else` | `nahhh` |
| `WHILE` | `while` | `Ahhh` |
| `FOR` | `for` | `grindset` |
| `BREAK` | `break` | `bruh` |
| `CONTINUE` | `continue` | `pullup` |
| `RETURN` | `return` | `yeet` |
| `IMPORT` | `import` | `fax` |
| `TRUE` | `true` | `fr` |
| `FALSE` | `false` | `pmo` |
| `AND` | `and` | — |
| `OR` | `or` | — |
| `STRATEGY` | `strategy` | — |
| `WHEN` | `when` | — |
| `CROSSES_ABOVE` | `crosses_above` | — |
| `CROSSES_BELOW` | `crosses_below` | — |
| `BACKTEST` | `backtest` | — |
| `ON` | `on` | — |
| `FROM` | `from` | — |
| `TO` | `to` | — |
| `WITH` | `with` | — |
