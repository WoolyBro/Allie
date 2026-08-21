# Keywords

Generated from `qubit/lexer/keywords.py`'s `KEYWORD_ALIASES` -- do not hand-edit; run `python scripts/generate_keywords_doc.py` after changing that dict. `tests/unit/test_lexer.py` fails if this file drifts from the code.

Lookup is case-sensitive: `Ahhh` (while) and `ahhh` are different tokens.

Quant keywords (`strategy`, `when`, `crosses_above`, `crosses_below`, `backtest`, `on`, `from`, `to`) are canonical-only in v1 -- the mechanism supports slang aliases for them uniformly later without touching the lexer.

| Token | Spellings |
|---|---|
| `LET` | `let` (aka `ts`) |
| `FUNC` | `func` (aka `be`) |
| `IF` | `if` (aka `lowkey`) |
| `ELSE` | `else` (aka `nahhh`) |
| `WHILE` | `while` (aka `Ahhh`) |
| `FOR` | `for` (aka `grindset`) |
| `BREAK` | `break` (aka `bruh`) |
| `CONTINUE` | `continue` (aka `pullup`) |
| `RETURN` | `return` (aka `yeet`) |
| `IMPORT` | `import` (aka `fax`) |
| `TRUE` | `true` (aka `fr`) |
| `FALSE` | `false` (aka `pmo`) |
| `AND` | `and` |
| `OR` | `or` |
| `STRATEGY` | `strategy` |
| `WHEN` | `when` |
| `CROSSES_ABOVE` | `crosses_above` |
| `CROSSES_BELOW` | `crosses_below` |
| `BACKTEST` | `backtest` |
| `ON` | `on` |
| `FROM` | `from` |
| `TO` | `to` |
