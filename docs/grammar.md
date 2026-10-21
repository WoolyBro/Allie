# Allie Grammar

The complete syntax, as `allie/parser/parser.py` implements it — one
parser method per production below, in the same order, so the two can be
read side by side. The rules for individual tokens are in
[Lexical structure](#lexical-structure) at the end.

## Grammar

```ebnf
program        → statement* EOF

statement      → letStmt | funcDecl | ifStmt | whileStmt | forStmt
               | breakStmt | continueStmt | returnStmt | importStmt
               | strategyDecl | backtestStmt | exprStmt | block

letStmt        → "let" IDENTIFIER (":" type)? "=" expression ";"
funcDecl       → "func" IDENTIFIER "(" params? ")" ("->" type)? block
params         → param ("," param)* ","?
param          → IDENTIFIER (":" type)?

ifStmt         → "if" expression block ("else" (ifStmt | block))?
whileStmt      → "while" expression block
forStmt        → "for" letStmt expression ";" expression block
breakStmt      → "break" ";"
continueStmt   → "continue" ";"
returnStmt     → "return" expression? ";"
importStmt     → "import" STRING ";"

strategyDecl   → "strategy" IDENTIFIER "{" whenClause* "}"
whenClause     → "when" expression ":" statement
backtestStmt   → backtest ";"
backtest       → "backtest" IDENTIFIER "on" IDENTIFIER "from" term "to" term
                 ("with" option ("," option)*)?
option         → IDENTIFIER ":" term

exprStmt       → expression ";"
block          → "{" statement* "}"

expression     → assignment
assignment     → IDENTIFIER "=" assignment | logic_or
logic_or       → logic_and ("or" logic_and)*
logic_and      → equality ("and" equality)*
equality       → comparison (("==" | "!=") comparison)?
comparison     → term (("<" | ">" | "<=" | ">=" | "crosses_above" | "crosses_below") term)?
term           → factor (("+" | "-") factor)*
factor         → unary (("*" | "/" | "%") unary)*
unary          → ("-" | "!") unary | call
call           → primary ( "(" arguments? ")" | "." IDENTIFIER | "[" expression "]" )*
arguments      → expression ("," expression)* ","?

primary        → NUMBER | STRING | "true" | "false" | IDENTIFIER
               | "(" expression ")" | arrayLiteral | backtest
arrayLiteral   → "[" (expression ("," expression)* ","?)? "]"

type           → genericType | baseType
genericType    → genericName "[" type "]"
genericName    → "Series" | "Matrix" | "Vector"
baseType       → "Int" | "Float" | "Bool" | "String" | "Void"
               | "Date" | "DateRange" | "PriceSeries" | "BacktestResult"
```

Notes on the productions:

- **Comparisons and equality don't chain** — note the `?` rather than
  `*`. `0 < x < 10` is an error whose hint says to write
  `0 < x and x < 10`, instead of silently meaning `(0 < x) < 10` and
  failing later with a confusing "compare Bool with Int" type error. To
  compare the result of a comparison, parenthesize: `(a < b) == c`.
- **Indexing** `s[i]` reads one element of a series, counting from 0.
  Series are immutable values, so `s[i] = x` is an error, and so is
  assigning to a field (`p.close = x`): the only assignment target is a
  variable name.
- **Trailing commas** are allowed in parameter lists, argument lists and
  array literals, so multi-line lists can end every line with one.
- **A `when` condition is any expression** — the type checker requires
  a `Series<Bool>`, one decision per bar: `fast crosses_above slow`,
  `rsi < 30.0`, `fast crosses_above slow and rsi < 70.0`.
- **`crosses_above` / `crosses_below`** are comparison operators: same
  precedence as `<`, and they don't chain either. They work anywhere,
  not only in a `when`: `let up = fast crosses_above slow;`.
- **`backtest`** is a statement, which prints the results table, or an
  expression, which evaluates to a `BacktestResult`
  (`let r = backtest S on AAPL from 0 to 100;`). The statement form is
  what a `backtest` at the start of a statement means.
- **`backtest` bounds** are bar indices, `from` inclusive and `to`
  exclusive, written at `+`/`-` precedence: `from 0 to 100`,
  `from split to len(AAPL.close)`. A literal bound that is fractional or
  negative, or two literals with `to <= from`, is a compile error; a
  computed bound must be an `Int` (a type error otherwise), and is
  checked against the data when the backtest runs. A `to` past the last
  bar is fine — it's clamped to the data.
- **`with` options** (`with slippage: 0.001, fill: "close"`) are parsed
  for any name; which names exist and their types is the type checker's
  rule, from the one table the engine also reads
  (`allie.quant.backtest_engine.OPTIONS`).
- **`=` used as an `if`/`while`/`for` condition** is reported; it is
  almost always a typo for `==`.
- **Type annotations** are optional on `let` (the type checker infers
  them) and required on function parameters and return types — a
  type-checker rule, not a grammatical one.
- **Generics use brackets** (`Series[Float]`), which avoids the
  `<`/`>`-versus-comparison ambiguity angle brackets would bring.
  `Series`, `Matrix` and `Vector` always take a type argument; a bare
  `Series` is an unknown type.

## Syntax errors

The parser reports every syntax error in a file in a single run, each
with the offending line, a caret, and often a `help:` line:

- Messages say what was found: `found keyword 'let'`, `found ')'`,
  `found end of file`.
- A missing `;` or `)` at the end of a line is reported just after the
  last token on that line, where the fix goes — not at the start of the
  next line.
- An unclosed `{`, `(` or `[` is reported at the opening delimiter. For
  a `{`, the hint names an inner block whose `}` is indented less than
  the line that opened it, which is usually where a brace is actually
  missing.
- A keyword used as a name says so, including slang spellings:
  `let ts = 1;` → "'ts' is reserved as another spelling of 'let'".
- Recovery is local: after an error the parser resumes at the next
  statement, never skips past the `}` that closes the current block, and
  a broken `when` clause only costs that clause.
- Nesting deeper than 64 levels (parentheses, brackets, calls, blocks)
  or a statement nested more than 200 levels deep is an error, not a
  crash.

## Lexical structure

The terminals used above (`IDENTIFIER`, `NUMBER`, `INTEGER`, `STRING`)
and everything the lexer discards. Implemented in `allie/lexer/lexer.py`.

```ebnf
IDENTIFIER     → [A-Za-z_] [A-Za-z0-9_]*        (* minus keywords *)
NUMBER         → digits ("." digits)? exponent?
INTEGER        → digits                          (* a NUMBER with no "." or exponent *)
digits         → [0-9] ("_"? [0-9])*
exponent       → ("e" | "E") ("+" | "-")? digits
STRING         → '"' (char | escape)* '"'      (* on one line *)
escape         → "\n" | "\t" | "\r" | '\"' | "\\"
comment        → "//" (any but newline)* | "/*" (any | comment)* "*/"
```

- **Whitespace** is space, tab, CR and LF; CRLF files behave exactly
  like LF ones. A byte-order mark at the start of a file is ignored.
- **Comments** are `//` to end of line, or `/* ... */`. Block comments
  nest, so a region containing one can itself be commented out.
- **Identifiers and keywords** are ASCII and case-sensitive: `Ahhh` is
  `while`, `ahhh` is an identifier. Slang keyword aliases (`ts` for
  `let`, ... — `allie/lexer/keywords.py`) resolve to the same token as
  the canonical spelling, so every production above is
  spelling-agnostic by construction.
- **Numbers**: `42`, `3.14`, `1e-4`, `2.5E6`, `10_000`. A fractional part
  or an exponent makes the literal a `Float`; otherwise it's an `Int`,
  which must fit in a signed 64-bit integer. `_` separates digits and
  may only sit between two of them. A digit is needed on both sides of
  `.`: `1.` is `NUMBER` then `.`, and `.5` is an error suggesting `0.5`.
  Only decimal is supported (`0x1F` is an error).
- **Strings** use double quotes and end on the line they start on. Use
  `\n` for a line break inside one.
- `and`/`or` are keywords, not `&&`/`||`.
- **Outside strings and comments, source must be ASCII.** This catches
  look-alike letters (Cyrillic `а` for Latin `a`) and invisible
  characters (no-break space, zero-width space) pasted from documents or
  chat. The error names the character by code point.

Lexical errors never stop the lexer: every one in the file is reported
in a single run, each with the offending line, a caret, and usually a
`help:` line (`&&` → use `and`, `'...'` → use double quotes, `#` → use
`//`, curly quotes → use `"`).
