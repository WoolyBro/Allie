# Roadmap

## Phase 0 — Foundations (done)

- [x] Directory skeleton (`allie/{lexer,parser,ast,semantic,typechecker,ir,optimizer,codegen,runtime,stdlib,quant}`, `tests/`, `examples/`, `docs/`, `benchmarks/`)
- [x] `.gitignore`, `LICENSE` (MIT), initial commit
- [x] CI (`.github/workflows/ci.yml`): ruff + mypy + pytest on every push, matrix over Python 3.11/3.12
- [x] Diagnostics infrastructure (`allie/errors.py`): `SourceSpan`, `Diagnostic`, collect-and-report-all `DiagnosticBag`
- [x] Runtime error hierarchy (`allie/runtime/errors/__init__.py`), distinct from compile-time diagnostics
- [x] CLI skeleton (`allie/cli.py`): `allie run FILE --dump-tokens --dump-ast --trace`
- [x] Canonical north-star example (`examples/hello_world.ql`)
- [x] Dependency management: `uv` + `uv.lock`, `pre-commit` wired to ruff/mypy
- [x] Design decisions locked in (`docs/spec.md`): execution model, typing, precision, syntax style, backtest loop, slippage model, calendar handling, determinism, concurrency, module system, statistical rigor, scope cut line

## Phase 1 — Tiny hand-built evaluator (done)

No lexer/parser yet — evaluation semantics validated on trees built by hand
in Python, so Phase 2's real parser output has a trusted evaluator waiting
for it.

- [x] Values (`allie/runtime/values/__init__.py`): raw `int`/`float`/`bool`/`str`, no wrapper class; `int op float -> float` promotion
- [x] `Environment` (`allie/runtime/environment.py`): parent chain, `define`/`get`/`assign`, shadowing, assign-to-outer-scope, assign-to-undefined raises rather than creating a global
- [x] Hand-built tree evaluator (`allie/runtime/interpreter/__init__.py`): `eval_expr`/`exec_stmt` over plain-tuple nodes
- [x] Short-circuit `and`/`or`, proven before Phase 2 has more moving parts
- [x] `for` loop, `break`/`continue` via internal `BreakSignal`/`ContinueSignal` exceptions
- [x] Functions as closures (params, body, defining env), `return` via `ReturnSignal`, closures capture by reference (proven in `test_closures_capture_environment_by_reference_not_by_value`)
- [x] Division/modulo by zero raise `AllieValueError`, not a bare Python exception
- [x] Tests (`tests/unit/test_phase1_eval.py`, `tests/unit/test_environment.py`): precedence, if/else, while, for, break/continue at known iterations, short-circuit, recursive factorial(5)/fibonacci(10), closures — 25 tests passing, ruff + mypy clean

## Project rename: QuantLang → Allie (done)

The language is called **Allie**. Flattened the package layout to match:
`compiler/*` and the `quantlang/` CLI wrapper merged into a single `allie/`
package (`allie.lexer`, `allie.runtime.interpreter`, `allie.cli`, etc.) —
no more `compiler/` nesting. `QuantLang*Error` classes renamed to
`Allie*Error`. CLI command is now `allie`. `pyproject.toml` project name,
entry point, and package list all updated; `uv.lock` regenerated.

## Phase 2 — Lexer & Parser (2.1 done, 2.2 grammar written, parser next)

### 2.1 Lexer (done)

- [x] `allie/lexer/tokens.py`: `TokenType` (50 members), `Token` dataclass
- [x] `allie/lexer/keywords.py`: `KEYWORD_ALIASES` (single lookup table, canonical + slang spellings -> identical `TokenType`; quant keywords canonical-only in v1), `render_keywords_markdown()`
- [x] `allie/lexer/lexer.py`: `Lexer.scan_tokens`/`scan_token` — numbers (digit required on both sides of `.`), strings with escapes, `//` comments, case-sensitive keyword lookup, invalid characters collected into `.errors` (scanning continues), unterminated string raises immediately
- [x] `allie/lexer/errors.py`: `LexError`
- [x] `scripts/generate_keywords_doc.py` + `docs/keywords.md` (generated, not hand-edited)
- [x] `tests/unit/test_lexer.py`: 104 tests — every `TokenType` in isolation, exact token sequence, number/string/comment edge cases, multi-char operators, invalid-character collection, line/column tracking, every `KEYWORD_ALIASES` entry parametrized, mixed canonical/goofy/quant keyword file, and a drift check against `docs/keywords.md`

Reconciled while building: `and`/`or` were in `TokenType` but missing from
the pasted `KEYWORD_ALIASES` example — added as canonical-only entries
(`logic_and`/`logic_or` need them as keyword tokens).

### 2.2 Grammar (done)

- [x] `docs/grammar.md` — full merged EBNF, written before `parser.py` exists. Reconciled two conflicts against already-committed decisions rather than silently picking a side: type annotations kept **optional** (pasted grammar had no `?`, which would contradict the locked gradual-typing decision and break `hello_world.ql`); generics kept as **`Series[float]`** (pasted grammar used `<float>`, but everything already committed uses brackets, which also avoid the `<`/`>`-vs-comparison ambiguity). Also fixed `factor`'s multiplication alternative, dropped in the paste, and added the `arguments` production referenced but never defined.
- `docs/grammar.ebnf` (Phase 0 placeholder) is now superseded by `docs/grammar.md`.

### 2.3 Parser + AST (done — designed here, no spec was ever sent)

Phase 3's typechecker spec assumed a parser, a real (mutable, attribute-
bearing) AST, and a resolver already existed. None of that had been
built or specified. Rather than guess piecemeal, this was designed from
scratch against `docs/grammar.md`:

- [x] `allie/ast/nodes.py` — dataclass AST (`LetStmt`, `FuncDecl`, `Binary`, `Call`, `StrategyDecl`, `WhenClause`, `BacktestStmt`, ...), each carrying a `SourceSpan`; expression nodes carry a mutable `inferred_type` field the typechecker annotates. **Departure from the Phase 1 plan**: the evaluator was meant to consume "the exact tuple shape" the parser produces, but Phase 3 needs mutable per-node type annotations tuples can't hold — `allie.runtime.interpreter` has *not* been adapted to walk these nodes yet. Running a real `.ql` file end to end through the Phase 1 evaluator is still blocked on that adaptation.
- [x] `allie/parser/parser.py` — recursive-descent, one method per grammar production, precedence climbing for expressions. Errors go through `DiagnosticBag` (collect-and-report-all) via an internal `_ParseError` signal + `_synchronize()`, the same shape as the interpreter's `BreakSignal`/`ContinueSignal`.
- [x] `tests/unit/test_parser.py` — 22 tests, one per grammar production plus error-recovery cases.

### 2.4 Resolver (done — also designed here)

Phase 3's spec repeatedly referenced "Phase 2's resolver" tracking
`loop_depth`/`inside_strategy`/`inside_when` — nothing matching existed.
Built `allie/semantic/resolver.py`: a `Scope` stack (name-existence only,
*not* types — kept deliberately separate from the typechecker's own
`TypeEnvironment`) plus the counters/flags Phase 3 assumed, validating
`break`/`continue` (loop only), `return` (function only), `when`
(strategy only), and `backtest`'s strategy reference. Deliberately does
*not* existence-check a bare identifier used as a call's callee (no
builtin registry exists yet — see Phase 3 notes below). 17 tests in
`tests/unit/test_resolver.py`.

## Phase 3 — Static typing (done)

- [x] `allie/typechecker/types.py`: `Type` base with `__str__`-derived structural `__eq__`/`__hash__` (works "for free" on `ArrayType`/`FunctionType` too); `IntType`/`FloatType`/`BoolType`/`StringType`/`VoidType` singletons; `is_assignable` (`Int -> Float` widens, everything else including narrowing rejected) keyed by a `(str, str)` table so Phase 4's `SeriesType` etc. slot in without reworking equality or assignability.
- [x] `allie/typechecker/checker.py`: `TypeChecker` walks the AST, annotates `expr.inferred_type`, raises `QSTypeError` fail-fast (not collect-and-report — a deliberate difference from the lexer/parser, explained in the module docstring) on: `let` assignability, arithmetic/comparison/equality/logical operator rules, unary `-`/`!`, call arity + arg-type checking, `if`/`while`/`for` conditions requiring exact `Bool`, functions requiring explicit param/return types (v1 limitation, documented not silently inferred), and "every path returns" for non-`Void` functions (flat structural check, not full CFG reachability). `when`-clause `Series<Float>` checking and `backtest` strategy-type validation are stubbed with `TODO` markers referencing the phases that unblock them (4 and 7), not silently skipped.
- [x] `allie/typechecker/errors.py`: `QSTypeError` — `"TypeError: {message}. Line {line}, Column {column}"`, exactly.
- [x] `docs/language-reference.md`: the assignability table, plus every v1 limitation written down (explicit types required, flat return-path check, `when`/`backtest`/field-access stubs).
- [x] `tests/unit/test_typechecker.py`: 31 tests matching every spec bullet (passing + failing pairs, exact-message assertion, arity/arg-type mismatches, recursive type-checking, `if`/`while`/`for` condition checks tested separately, missing-return vs. returns-on-every-path, and a regression net running every Phase-1/2 evaluator-style program — including `for`/`break`/`continue` — through the checker).
- [x] **Wire-in**: `allie/cli.py`'s `run` command now runs lex → parse → resolve → type-check, halting with diagnostics printed if any static pass fails. Execution itself stays stubbed (see the AST/tuple gap above) — kept the file named `cli.py` rather than the spec's suggested `main.py` rename, since nothing else in this repo uses that name and the rename wasn't confirmed.
- [x] Fixed `examples/hello_world.ql`, which had drifted from the Phase 2.2 grammar (lowercase `str`/`float` instead of `String`/`Float`, keyword arguments and `[index]` subscripting that were never added to the grammar, and an ad-hoc `backtest NAME { on_bar { ... } }` block that doesn't match `backtestStmt`'s actual `"backtest" NAME "on" SYMBOL "from" N "to" M ";"` shape). It now parses and produces a full AST (`--dump-ast` exits 0); full `run` still fails at the typechecker on `undefined name 'print'` since there's no builtin/stdlib function-type registry yet — expected, not a bug, and out of scope for what Phase 3 asked for.

199 tests total, ruff + mypy clean.

## Phases 4–13

Fill in from prior planning notes (the 5 indicators, `SeriesType` and the
rest of the quant type system, event-driven backtest engine, walk-forward
+ multiple-comparisons stats tooling, benchmarking, paper writeup, etc.) —
not reconstructed here since these phases weren't pasted into this
session. Two concrete follow-ups already known from building Phase 3:
adapt `allie.runtime.interpreter` to walk the real AST instead of
hand-built tuples, and give the typechecker/resolver a builtin function
registry so programs that call `print`/`sma`/`load_csv`/etc. can actually
pass type-checking.
