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

Fill in from prior planning notes (the backtest engine and
`crosses_above`/`crosses_below`, walk-forward + multiple-comparisons
stats tooling, benchmarking, paper writeup, etc.) — not reconstructed
here since these phases weren't pasted into this session. Known
follow-ups from building Phase 4: `examples/hello_world.ql` still
doesn't run end to end (`load_csv`/`sma` naming mismatches with the
registry's `load`/`SMA`, and `buy`/`sell`/strategy execution wait on
Phase 7) — not worth chasing further until those land.

## Phase 5 — IR & optimizer (done)

**Scope note, flagged before building**: Phase 0 decided "tree-walking
interpreter stays what actually runs; `ir`/`optimizer`/`codegen` are
aspirational placeholders" (`docs/spec.md`). Phase 5 explicitly asked
for a real second execution path with a cross-validation suite against
the tree-walker, so that's what got built — `allie run` still executes
via the tree-walking interpreter; the IR path exists for the optimizer
to operate on and be demonstrated against.

### 5.1 IR instruction set

- [x] `allie/ir/instructions.py`: every op from the spec (`LoadConst`/`LoadVar`/`StoreVar`/`LoadSeries`/`BinOp`/`UnaryOp`/`Call`/`MemberGet`/`Label`/`Jump`/`JumpIfFalse`/`Return`), plus one addition: `Move(dest, src)`. Confirmed, not assumed: `for`/`break`/`continue`/`strategy`/`when`/`backtest` need no new instruction — but short-circuit `and`/`or` genuinely does, since the given set has no `JumpIfTrue` and no copy/phi to converge two conditional branches into one result register. `LoadSeries` is defined but unused — array literals lower through `Call("__make_series", ...)` instead (the spec's own 5.2 decision).
- [x] `IRProgram`: flat `list[IRInstr]`, no basic-block/CFG structure, exactly as specced. `RegisterAllocator.new_reg()`/`new_label()` just increment counters.

### 5.2 AST → IR lowering

- [x] `allie/ir/lowering.py`. Two decisions the spec left open, made and documented: **separate `IRProgram` per function** (avoids call-stack/return-address machinery a flat-list-with-labels design would need); **functions are static, name-registered, not first-class closures** — `factorial`/`fib` calling themselves by name works identically to the tree-walker, but a function *returned from* another function and called later (real lexical closure capture) has no way to work when calls dispatch by static name. This is a real, bounded limitation, not an oversight — the Phase 5.3 cross-validation suite is scoped around it (see below).
- [x] `ForStmt` desugars to the identical `while`-shaped Label/Jump/JumpIfFalse pattern, exactly as asked — with one detail the spec didn't spell out and getting wrong would have silently broken `for`: `continue`'s target is a dedicated `increment_label`, *not* the same label `while`'s `continue` uses, because a `for` loop's `continue` must still run the increment before rechecking the condition.
- [x] `break`/`continue` via a `(continue_label, break_label)` stack, pushed/popped around loop lowering.
- [x] `strategy`/`when`/`backtest` lower to `Call`s into the not-yet-built Phase 7 backtest engine, invoked not inlined.
- [x] 24 tests in `tests/unit/test_ir_lowering.py`.

### 5.3 IR interpreter

- [x] `allie/ir/interpreter.py`: flat instruction-pointer loop, a register dict plus the existing `Environment` for named variables (scoping deliberately flat — no per-block child scope, safe for every program this phase's suite covers). Operator/field-access logic is independently re-implemented, not imported from the tree-walker, so a bug shared by both wouldn't hide from cross-validation.
- [x] **Cross-validation suite** (`tests/unit/test_ir_cross_validation.py`, 15 tests): every Phase 1 for/break/continue/recursion program plus the Phase 4 quant integration programs (`load`/`SMA`/`mean`, `for`-loop-over-window-sizes), run through both interpreters, asserting identical results — including a `SeriesValue`-aware comparison. Deliberately does **not** include the two closure-capture programs, per 5.2's documented limitation.
- [x] Found and fixed a real mypy gap while wiring this up: reassigning `promote_numeric`'s result to the same `Value`-annotated parameter (instead of a fresh local) keeps mypy checking the full `Value` union against `+`/`-`/`*`/`/` — same root cause the tree-walker's binop helper already had, silently masked by a stale `warn_return_any` override comment that no longer matched why it was needed. Fixed both, and updated the override's justification to the real reason.

### 5.4 Optimizer passes

- [x] `allie/optimizer/pure_functions.py`: `PURE_FUNCTIONS`, written once, imported by both `dce.py` and `cse.py` — a hand-picked allowlist (not "everything except a denylist"), so a new builtin defaults to *not* being optimized across. Explicitly excludes `load`/`print` (I/O), `normal` (technically deterministic given a seed, but deduplicating a "random sample" call would be a surprising thing to do silently), and `buy`/`sell` (Phase 7, not registered yet, but listed now per the merge notes).
- [x] `constant_fold.py`: folds a `BinOp` whose both operands trace to `LoadConst` into one `LoadConst`, removing the original operand constants if they end up unused — but never the fold's *own* result, regardless of whether anything downstream reads it (that's DCE's job). Leaves a folded `1/0` as a real `BinOp` rather than raising at "compile" time.
- [x] `dce.py`: removes a `StoreVar`/`LoadConst`/`LoadVar`/`BinOp`/`UnaryOp`/`MemberGet`/`Move` whose destination is never read *anywhere in the whole program* (not "read strictly after this point") — the simpler, whole-program check that sidesteps the loop-carried-liveness trap by construction, at the cost of being more conservative than a real forward-flow analysis would be (documented trade-off). Runs to a **fixpoint**, not a single pass: found via a failing test that removing one dead `StoreVar` can strand a `LoadConst` that only ever fed it, which a single pass would leave behind. Never removes a `Call` regardless of liveness.
- [x] `cse.py`: `const_cache` (keyed by `(type, value)`, so `Int(1)` and `Float(1.0)` don't collide), `var_reg_cache` (variable name → register currently holding its value, updated on both `LoadVar` and `StoreVar` — this is what makes reassignment invalidate CSE with no separate "clear everything" step), `expr_cache`/`call_cache` for `BinOp`/pure `Call`s. **A second control-flow-join bug found by actually running optimized code, not just structural assertions**: none of the spec's CSE tests exercise a loop or a branch, so the positive/negative tests alone passed while the pass was still wrong — lowering and optimizing a real `for` loop produced IR that silently computed the wrong answer, because `var_reg_cache`/`expr_cache`/`call_cache` weren't being invalidated at `Label`s, so a register loaded *before* a loop or `if`/`else` got reused for a read *inside* it. Fixed by clearing those three caches (not `const_cache`, which is path-independent) at every `Label`; added regression tests for both the loop and the if/else case.
- [x] `pass_manager.py`: `run_passes(program, passes, enabled)`.
- [x] `tests/unit/test_optimizer.py`: 16 tests — every case the spec asked for (constant-fold collapse, DCE removing a dead store but never a `Call`, the loop-increment negative test, CSE's positive/negative reassignment cases, CSE's impure-call negative tests) plus the two control-flow-join regression tests above.

317 tests total, ruff + mypy clean. Verified beyond the test suite: lowered + fully optimized (constant-fold → DCE → CSE) versions of a `for` loop, a recursive `factorial`, and an `if`/`else` all execute to the correct answer through the IR interpreter.

## Phase 6 — LLVM backend + MCJIT (done)

A **third** execution backend. `allie run` still defaults to the
tree-walking interpreter; `main.py` defaults to LLVM/JIT.

### 6.1 Setup

- [x] `llvmlite` added as a project dependency (0.49.0, bundling LLVM 22.1.0), `uv.lock` regenerated. Verified a minimal hello-IR + MCJIT example standalone before writing any project code, per the spec's own first ask.
- [x] **The spec's warning about API drift was justified** — three things had changed from what any older reference would say, each caught by testing rather than assumed: `llvm.initialize()` is not just deprecated but *removed* (calling it raises `RuntimeError`); `add_global_mapping` needs the `ValueRef` from the parsed backing module, not the `ir.Function` builder object (passing the latter gives an opaque `ctypes.ArgumentError`); and `get_function` raises `NameError` for an unknown symbol rather than returning `None`.

### 6.2 Type mapping

- [x] `Int -> i64`, `Float -> double`, `Bool -> i1`, `Void -> void`, documented in `docs/architecture.md`.
- [x] **Deviation from spec, documented**: `String -> i8*` (pointer to a global constant) was specified on the premise that no dynamic string ops were needed — but Allie's `+` concatenates strings at runtime, already proven in both interpreters. Every non-native-scalar type instead uses one uniform boxed-`i64`-handle scheme (`allie/codegen/runtime_bridge.py`), rather than special-casing strings. One mechanism, not two.
- [x] Series/Matrix/Vector native codegen explicitly out of scope; stdlib calls bridge back into the Python/numpy runtime.

### 6.3 IR → LLVM IR lowering (`allie/codegen/llvm_lowering.py`)

- [x] Native LLVM instructions for scalar `LoadConst`/`BinOp`/`UnaryOp` (including `%` — `srem`/`frem`), `Jump`/`JumpIfFalse`/`Label` → basic blocks and `br`/`cbranch`, `Call` → native call for scalar user functions or an external bridge call otherwise, function declarations from the type table.
- [x] **Required self-check passed**: `for`/`break`/`continue` need zero dedicated code here — `Label`/`Jump`/`JumpIfFalse` handle `if`/`while`/`for` uniformly, exactly because Phase 5.2's desugar was applied consistently. Traced `sum_to_n` through by hand to confirm; no `for`-specific branch had crept in.
- [x] Retrofitted `allie/ir/lowering.py` to record `register_types` and `FunctionInfo` (param/return types) from the type checker's existing annotations — LLVM codegen needs each register's static type, which the untyped IR didn't carry. Additive: the IR interpreter and its tests are unaffected.

### 6.4 JIT execution (`allie/codegen/jit.py`) + CLI

- [x] `compile_and_run(module, entry_fn_name)`: init, parse/verify, bind bridge symbols, finalize, invoke.
- [x] `main.py` implementing the specified interface exactly — `python main.py <file> [--debug-tokens] [--debug-ast] [--debug-ir] [--no-optimize]`, all flags independent and combinable, `--debug-ir` printing the Allie IR and the LLVM IR as two labeled sections, `--no-optimize` toggling the Phase 5.4 pass manager. Defaults to LLVM/JIT per the spec's pipeline.
- [x] Kept `allie run` too (installed console script, referenced by README/docs/tests) with its tree-walking default and a new `--backend {tree,ir,llvm}`; both entry points share `allie.cli.run_pipeline`. `--dump-tokens`/`--dump-ast` remain as aliases.

### 6.5 Tests

- [x] `tests/unit/test_llvm_lowering.py` (16 tests): LLVM backend on its own — native arithmetic, promotion, if/else, while, for with break+continue, short-circuit, recursion, string concat, quant calls via the bridge, and the documented non-scalar-user-function limitation.
- [x] `tests/unit/test_llvm_cross_validation.py` (52 tests): **all three backends agree** across 22 programs (scalar, quant, and `for`/`break`/`continue`, per the spec's explicit ask); exact iteration counts asserted through the JIT path specifically (8 cases, including zero-iteration and inclusive-bound edges); plus optimized-vs-unoptimized JIT agreement on every program, verifying the optimizer's register rewrites don't invalidate the `register_types` map LLVM lowering depends on.
- [x] External-call bridge exercised end-to-end through JIT for `SMA`/`mean`/`load`.

### Three real bugs found by testing beyond the spec's own list

1. **Exceptions silently vanish across the ctypes/JIT boundary.** Verified directly: a Python exception raised in a callback invoked from JIT code never reaches the Python caller — ctypes prints "Exception ignored" and returns garbage. Every bridge function now records the exception in a pending slot and returns a sentinel, with subsequent calls short-circuiting so one failure can't cascade; `compile_and_run` re-raises once control returns to Python.
2. **Native division by zero was undefined behavior.** `sdiv`/`srem` by zero traps and kills the process. The lowering now emits an explicit zero-check that calls the error bridge and returns early, never reaching the division. Caught because the div-by-zero test *didn't raise* — the first version silently produced garbage.
3. **`Move` broke LLVM's SSA dominance rule.** Short-circuit `and`/`or` writes the same destination register from two different branches; aliasing the value directly produced "Instruction does not dominate all uses". Fixed with entry-block stack slots — allocated up front, since an `alloca` emitted lazily lands *after* the block's terminator and is invalid IR.

**Also fixed, pre-existing:** the type checker rejected `+` on two `String`s, even though both interpreters have always concatenated them correctly — a real checker/interpreter gap, invisible until something ran typed string concatenation through the full pipeline for the first time. Regression tests added for both the fix and the still-rejected `"a" + 1` case.

365+ tests total, ruff + mypy clean.
