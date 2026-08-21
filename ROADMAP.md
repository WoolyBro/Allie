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

## Phase 4 — Quantitative types & standard library (done)

### 4.1 Type extensions

- [x] `SeriesType`/`MatrixType`/`VectorType`/`DateType`/`DateRangeType` added to `allie/typechecker/types.py`, exactly as planned back in Phase 3 (no rework of `is_assignable`/`__eq__`/`__hash__` needed). Also added, beyond the original list, because they turned out to be genuinely necessary: `PriceSeriesType` (stdlib `load`'s return type) and `AnyType` (stdlib `print`'s parameter type — deliberately *not* user-writable, see `docs/language-reference.md`).
- [x] `is_assignable`: scalar → Series is rejected (falls out of the existing logic with no new code — see the regression test guarding it), and `Series<Int>` does *not* widen to `Series<Float>` even though bare `Int` → `Float` does.
- [x] `ArrayLiteral` now type-checks to `SeriesType(elem)`, not `ArrayType(elem)` — array literals default to `Series`, documented in `docs/language-reference.md`. `ArrayType` stays in `types.py` for a hypothetical future `Array[T]` annotation nothing currently produces.
- [x] `WhenClause` operands now really require `Series<Float>` (Phase 3's `TODO` stub wired for real).
- [x] `tests/unit/test_typechecker_quant.py`: 15 tests.
- **Correction, not new work**: the spec framed generic syntax as `IDENTIFIER "<" type ">"`, "already done" in Phase 2.2/2.4. What's actually built uses brackets (`Series[Float]`) — a deliberate Phase 2.2 reconciliation (angle brackets reintroduce the `<`/`>`-vs-comparison ambiguity), unchanged here.

### 4.2 Runtime representation

- [x] `SeriesValue`/`MatrixValue`/`PriceSeriesValue`/`BuiltinFunction` added to `allie/runtime/values`. **Deviation from spec, documented**: `SeriesValue`/`MatrixValue` wrap plain numpy `float64` ndarrays, not `pandas.Series`/`DataFrame` — matches Phase 4's literal ask, but is a step back from `docs/spec.md`'s original "Series wraps pandas.Series" aspiration; revisit if real `DatetimeIndex` alignment is needed later. `PriceSeriesValue.dates` is a plain `datetime64[ns]` array rather than a `SeriesValue` (which is float64-only by construction) — a documented deviation from a literal "each a SeriesValue" reading.
- [x] **The big one**: `allie/runtime/interpreter` rewritten to walk the real AST (`allie.ast.nodes`) instead of Phase 1's hand-built tuples — every semantic property Phase 1 proved (short-circuit `and`/`or`, closures by reference, recursion, `for`/`break`/`continue`, division-by-zero raising a custom error) is unchanged, just re-proved against source text through the real lexer/parser in the new `tests/unit/test_interpreter.py` (17 tests), which **replaces** `tests/unit/test_phase1_eval.py` (deleted — its hand-built tuples can no longer run against the new node-based contract). `ArrayLiteral` evaluates to a `SeriesValue`; `FieldAccess` reads `PriceSeriesValue`'s named fields (`prices.close`); `Call` dispatches to either a user `Closure` or a stdlib `BuiltinFunction` through the same path; `StrategyDecl`/`BacktestStmt` raise a clear "not executable yet" error (Phase 7 builds the actual backtest engine).

### 4.3 Standard library

- [x] `allie/stdlib/registry.py`: `BUILTINS` (name → `(FunctionType, Callable)`), `create_global_environment()`, `builtin_function_types()`. Importing `allie.stdlib` registers all 22 builtins as a side effect.
- [x] `allie/stdlib/math_fns.py`: `mean`/`median`/`std`/`variance`/`covariance`/`correlation` — `ddof=1`, empty/single-element/mismatched-length series raise `AllieValueError` rather than returning `NaN`.
- [x] `allie/stdlib/series_ops.py`: `SMA`/`EMA`/`rolling_mean`/`rolling_std`/`shift`/`diff`/`pct_change` — `SMA`/`rolling_mean` NaN-pad the first `window - 1` points (pandas convention). Generic `rolling(series, window, fn)` isn't built (no first-class function values from source yet); `rolling_mean`/`rolling_std` are the concrete stand-ins, as the spec allowed. `crosses_above`/`crosses_below` deliberately **not** built here — Phase 7's responsibility.
- [x] `allie/stdlib/matrix_ops.py`: `transpose`/`inverse`/`determinant`/`dot` — singular matrix, non-square `determinant`, and dimension-mismatched `dot` all raise `AllieValueError`, never a raw numpy error.
- [x] `allie/stdlib/stats_fns.py`: `zscore`/`percentile`, and `normal` — **with an added, required `seed` parameter** beyond the literal spec (`normal(Float, Float, Int) -> Series<Float>`), to honor the already-locked "explicit seed, no hidden global RNG" decision in `docs/spec.md`.
- [x] `allie/stdlib/io.py`: `load` (CSV → `PriceSeriesValue`, case-insensitive column matching, names the missing column) and `print` (compact `SeriesValue` formatting, `Void`/`Unit` return type).
- [x] `tests/unit/test_stdlib_{math,series,matrix,stats,io}.py`: 46 tests, hand-computed values cross-checked against pandas where useful (test-only dependency, never a runtime one).
- [x] `tests/fixtures/AAPL_sample.csv`: 60-row synthetic OHLCV fixture (seeded random walk, reproducible).
- [x] `tests/integration/test_quant_pipeline.py`: the `load` → `SMA` → `print` reference example end to end through the full pipeline, plus a second test combining it with a `for` loop over several window sizes — 2 tests.
- [x] **Wire-in, finally real**: `allie/cli.py`'s `run` command now executes lex → parse → resolve → type-check → **run**, with `allie.stdlib`'s builtins visible to both the type checker and the interpreter. `allie run some_file.ql` actually computes and prints things now.

262 tests total, ruff + mypy clean. (mypy's target bumped to `python_version = "3.12"` — numpy 2.5's bundled stubs use PEP 695 syntax mypy only parses under a 3.12+ target; our own code still supports 3.11, nothing here uses 3.12-only syntax.)

## Phases 5–13

Fill in from prior planning notes (optimizer/`PURE_FUNCTIONS`, the
backtest engine and `crosses_above`/`crosses_below`, walk-forward +
multiple-comparisons stats tooling, benchmarking, paper writeup, etc.) —
not reconstructed here since these phases weren't pasted into this
session. Known follow-ups from building Phase 4: `examples/hello_world.ql`
still doesn't run end to end (`load_csv`/`sma` naming mismatches with the
registry's `load`/`SMA`, and `buy`/`sell`/strategy execution wait on
Phase 7) — not worth chasing further until those land.
