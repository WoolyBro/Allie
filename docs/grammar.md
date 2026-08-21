# Qubit Grammar (Phase 2, merged)

Written before `parser.py` exists, per Phase 2.2. Supersedes
`docs/grammar.ebnf` (the Phase 0 placeholder, which only covered
`let/func/if/else/while/return` and scalar expressions).

## Reconciliations against what's already committed

The pasted grammar conflicted with two things already locked in
elsewhere in this repo. Flagging both rather than silently picking a
side:

1. **Type annotations are optional, not mandatory.** The pasted
   `letStmt`/`funcDecl`/`params` productions had no `?` on the type
   clause, which would make every `let`, parameter, and return type
   mandatory. That contradicts the already-locked "gradual typing"
   decision in `docs/spec.md` ("dynamic by default; annotations enable a
   lightweight checker") and would invalidate `examples/hello_world.ql`,
   which already mixes annotated (`let prices: Series[float] = ...`) and
   unannotated (`let fast_ma = sma(...)`) `let`s. Treated the missing `?`
   as a paste omission, not a new decision — say so if mandatory typing
   was actually intended.
2. **Generics use `Series[float]`** (brackets), not `Series<float>`
   (angle brackets). `docs/grammar.ebnf`, `docs/spec.md`, and
   `examples/hello_world.ql` all already use bracket generics, and
   brackets avoid the classic `<`/`>`-vs-comparison-operator parsing
   ambiguity that angle-bracket generics reintroduce. Kept brackets.

Also fixed: `factor`'s multiplication alternative was dropped in the
paste (rendered as an empty string between the pipes) — restored as
`"*"`, matching the shape of `term`'s production right above it. An
`arguments` production was referenced by `call` but never defined in the
paste — added the obvious `expression ("," expression)*` shape, matching
`arrayLiteral`'s inner list.

## Grammar

```ebnf
program        → statement* EOF

statement      → letStmt | funcDecl | ifStmt | whileStmt | forStmt
               | breakStmt | continueStmt | returnStmt | importStmt
               | strategyDecl | backtestStmt | exprStmt | block

letStmt        → "let" IDENTIFIER (":" type)? "=" expression ";"
funcDecl       → "func" IDENTIFIER "(" params? ")" ("->" type)? block
params         → param ("," param)*
param          → IDENTIFIER (":" type)?

ifStmt         → "if" expression block ("else" (ifStmt | block))?
whileStmt      → "while" expression block
forStmt        → "for" letStmt expression ";" expression block
breakStmt      → "break" ";"
continueStmt   → "continue" ";"
returnStmt     → "return" expression? ";"
importStmt     → "import" STRING ";"

strategyDecl   → "strategy" IDENTIFIER "{" whenClause* "}"
whenClause     → "when" expression ("crosses_above" | "crosses_below") expression ":" statement
backtestStmt   → "backtest" IDENTIFIER "on" IDENTIFIER "from" NUMBER "to" NUMBER ";"

exprStmt       → expression ";"
block          → "{" statement* "}"

expression     → assignment
assignment     → IDENTIFIER "=" assignment | logic_or
logic_or       → logic_and ("or" logic_and)*
logic_and      → equality ("and" equality)*
equality       → comparison (("==" | "!=") comparison)*
comparison     → term (("<" | ">" | "<=" | ">=") term)*
term           → factor (("+" | "-") factor)*
factor         → unary (("*" | "/" | "%") unary)*
unary          → ("-" | "!") unary | call
call           → primary ("(" arguments? ")" | "." IDENTIFIER)*
arguments      → expression ("," expression)*

primary        → NUMBER | STRING | "true" | "false" | IDENTIFIER
               | "(" expression ")" | arrayLiteral
arrayLiteral   → "[" (expression ("," expression)*)? "]"

type           → baseType ("[" type "]")?
baseType       → "Int" | "Float" | "Bool" | "String" | "Void"
               | "Series" | "Matrix" | "Vector" | "Date" | "DateRange"
```

## Notes carried over from Phase 2.1 (lexer)

- `and`/`or` are keywords (`TokenType.AND`/`OR`), not symbolic `&&`/`||`
  — simpler, and matches `logic_and`/`logic_or` above using the word
  forms. They were missing from the pasted `KEYWORD_ALIASES` example
  even though `TokenType` defines them; added as canonical-only entries
  (see `docs/keywords.md`).
- Slang keyword aliases (`ts` for `let`, `Ahhh` for `while`, etc. —
  `qubit/lexer/keywords.py`) are transparent to this grammar: the lexer
  resolves every spelling to the same `TokenType` before the parser ever
  sees it, so every production above is spelling-agnostic by
  construction.
- Number literals require a digit on both sides of the decimal point
  (`1.` and `.5` each lex as two tokens, never a malformed float) — see
  `qubit/lexer/lexer.py`.

## Not yet covered

`assignment → IDENTIFIER "=" assignment` only supports assigning to a
bare identifier — no `arr[i] = x` or `obj.field = x` targets yet. Revisit
once the parser needs them; not blocking Phase 2.1/2.2.
