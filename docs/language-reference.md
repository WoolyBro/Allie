# Allie Language Reference

Filled in incrementally as each language feature actually lands — see
`docs/architecture.md` for pipeline status and `docs/grammar.ebnf` (Phase 0
placeholder) / `docs/grammar.md` (Phase 2+ merged grammar) for the formal
grammar.

## Types (Phase 3)

`Int`, `Float`, `Bool`, `String`, `Void`, and `Array<T>` (inferred from
array-literal expressions; there's no user-writable `Array[...]`
annotation yet). `Series`/`Matrix`/`Vector`/`Date`/`DateRange` are named in
`docs/grammar.md`'s `baseType` but not yet implemented as real types
(Phase 4) — writing one in a type annotation currently fails type-checking
with "unknown type".

### Assignability

`allie.typechecker.types.is_assignable(source, target)`:

| Source | Target | Assignable? | Why |
|---|---|---|---|
| `T` | `T` (same type) | Yes | exact match |
| `Int` | `Float` | Yes | widening |
| `Float` | `Int` | **No** | narrowing rejected — would silently truncate |
| anything else | anything else | No | no other conversions exist in v1 |

This governs `let` initializers (when annotated), function argument
passing, and `return` values. It's a plain lookup table keyed by each
type's `str()`, not a hardcoded enum switch — see `types.py`'s module
docstring for why that matters once Phase 4 adds more types.

### Known limitations (v1, documented rather than silently inferred)

- **Functions require explicit parameter and return types.** No return-
  type inference is attempted; a function with no `-> Type` (or a
  parameter with no `: Type`) fails type-checking rather than silently
  guessing. Simpler and more honest than partial inference — see
  `allie/typechecker/checker.py`'s module docstring.
- **"Every path returns" is a flat structural check, not full CFG
  reachability.** It recognizes a direct `return`, and an `if`/`else`
  where both branches return — it does not reason about loops that
  always execute at least once, exhaustive `while true` bodies, etc.
  Good enough to catch the common case; a function that provably always
  returns via a more complex path may still be flagged as incomplete.
- **`when` clauses don't yet check their operands are `Series<Float>`**
  (Phase 4, once `SeriesType` exists) — currently just type-checks them
  as ordinary expressions.
- **`backtest`'s strategy reference isn't type-checked** (Phase 7, once
  strategy type-checking exists) — the resolver already checks the name
  was *declared* somewhere; the checker doesn't yet re-verify it.
- **Field access (`a.b`) has no real type system behind it yet** — no
  record/module type exists to check property types against, so it's
  accepted as `Void` rather than claiming a type it can't verify.

## Grammar note

Nothing in this file is implemented via a lexer/parser without a real
AST — Phase 2.3 (`allie/parser/`) produces the dataclass-based AST in
`allie/ast/nodes.py`; Phase 1's hand-built-tuple evaluator has not yet
been adapted to walk it (see `ROADMAP.md`).
