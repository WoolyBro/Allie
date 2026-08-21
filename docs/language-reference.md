# Allie Language Reference

Filled in incrementally as each language feature actually lands — see
`docs/architecture.md` for pipeline status and `docs/grammar.ebnf` (Phase 0
placeholder) / `docs/grammar.md` (Phase 2+ merged grammar) for the formal
grammar.

## Types

`Int`, `Float`, `Bool`, `String`, `Void`, `Date`, `DateRange`, `PriceSeries`
(the type of stdlib `load`'s return value), and the generic quant types
`Series[T]`, `Matrix[T]`, `Vector[T]` (bracket syntax in source;
`Series<T>` etc. is how these render in error messages and `str()` —
see `docs/grammar.md`'s reconciliation notes for why source syntax uses
brackets, not angle brackets). `Array<T>` still exists but is no longer
produced by anything user-writable — see "Array literals default to
Series" below.

`Any` exists internally (`is_assignable(source, ANY)` is always `True`)
but is deliberately **not** a user-writable annotation — it exists only
so stdlib `print` can accept any value. Writing `let x: Any` fails with
"unknown type", by design: exposing a general escape hatch would
undermine gradual typing's whole point for one narrow builtin need.

### Array literals default to Series

`[1, 2, 3]` type-checks as `Series<Int>`, not `Array<Int>` — this
language operates on series by default, so an array literal should be
usable anywhere a series is expected (fed straight into `SMA`, `mean`,
a `when` clause, ...) without an extra conversion step. `ArrayType` is
kept in `types.py` for a hypothetical future user-writable `Array[T]`
annotation (there's no `Array` keyword in the grammar's `baseType` yet),
but nothing currently produces one.

### Assignability

`allie.typechecker.types.is_assignable(source, target)`:

| Source | Target | Assignable? | Why |
|---|---|---|---|
| `T` | `T` (same type) | Yes | exact match |
| anything | `Any` | Yes | `Any` is a top type — see above |
| `Int` | `Float` | Yes | widening |
| `Float` | `Int` | **No** | narrowing rejected — would silently truncate |
| `Float` | `Series<Float>` | **No** | a scalar is never assignable to a series — see below |
| `Series<Int>` | `Series<Float>` | **No** | no per-element widening; the container types must match exactly |
| anything else | anything else | No | no other conversions exist in v1 |

**Why scalar → Series is rejected**: Series operations (`SMA`, `mean`, a
`when` clause's crossover check, ...) are defined over an actual sequence
of observations. Silently accepting a scalar where a series was expected
would be exactly the class of bug this type distinction exists to catch
— e.g. passing today's single closing price where a whole price history
was expected. This falls out of `is_assignable`'s existing logic with no
special-casing (there's simply no widening row for it) — see
`test_typechecker_quant.py`'s `test_scalar_not_assignable_to_series` for
the regression test guarding against a future change accidentally adding
one.

This table governs `let` initializers (when annotated), function
argument passing, and `return` values. It's a plain lookup table keyed
by each type's `str()`, not a hardcoded enum switch — see `types.py`'s
module docstring for why that matters when a new type is added.

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
- **`backtest`'s strategy reference isn't type-checked** (Phase 7, once
  strategy type-checking exists) — the resolver already checks the name
  was *declared* somewhere; the checker doesn't yet re-verify it.
- **Field access (`a.b`) is only properly typed for `PriceSeries`**
  (`.open`/`.high`/`.low`/`.close`/`.volume` → `Series<Float>`). Anything
  else — including `PriceSeries.dates` (there's no `Series<Date>` type
  to give it yet) and field access on any other type — is accepted as
  `Void` rather than claiming a type that can't be verified. No general
  record/module type system exists.
- **`strategy`/`backtest` declarations type-check but don't execute.**
  The interpreter raises a clear "not executable yet" error if a program
  actually reaches one at runtime (Phase 7 builds the backtest engine).
- **No first-class function values from source.** `rolling(series,
  window, fn)` as a generic higher-order stdlib function isn't built —
  `BuiltinFunction`/`Closure` are runtime-only wrapper types, not
  something a `.ql` program can pass around as data yet. `rolling_mean`/
  `rolling_std` are the concrete stand-ins.

## Runtime representation (Phase 4.2)

`SeriesValue`/`MatrixValue` are thin wrappers around plain numpy
`float64` ndarrays — **not** `pandas.Series`/`DataFrame`. This is a
deliberate simplification versus `docs/spec.md`'s original "Series wraps
`pandas.Series`, don't reimplement `DatetimeIndex`" aspiration: Phase 4
asked for a numpy wrapper specifically, so pandas stays a test-only
dependency (cross-checking stdlib functions against a known-good
reference), not a runtime one, for now. Revisit the pandas-backed
version if/when real `DatetimeIndex` alignment (holiday calendars,
resampling, ...) is actually needed.

`PriceSeriesValue.dates` is a plain numpy `datetime64[ns]` array, not a
`SeriesValue` — `SeriesValue` is float64-only by construction, so
forcing dates into one would either break that invariant or silently
encode a date as a meaningless float. A documented, deliberate deviation
from a literal reading of "each [OHLCV field, including dates] a
SeriesValue".

## Standard library (Phase 4.3)

All 22 builtins are registered in `allie.stdlib.registry.BUILTINS`
(name → `(FunctionType, Callable)`) — see `allie/stdlib/{registry,
math_fns,series_ops,matrix_ops,stats_fns,io}.py`. Implementations use
plain numpy, not pandas, matching the runtime-representation decision
above; pandas is used only in `tests/unit/test_stdlib_*.py` to
cross-check results against a known-good reference.

- **`normal(mean, std, n, seed)` takes an explicit `seed`**, beyond what
  was literally specified (`normal(Float, Float, Int)`). Honors the
  already-locked determinism decision in `docs/spec.md`: "All stochastic
  stdlib functions ... take an explicit seed; no hidden global RNG
  state." A seedless `normal` would be exactly the case that decision
  rules out.
- **`SMA`/`rolling_mean` NaN-pad the first `window - 1` points**,
  matching pandas' `.rolling(window).mean()` convention.
- **`std`/`variance`/`covariance`/`correlation` use `ddof=1`** (sample,
  not population) and raise rather than returning `NaN` on an empty
  series, a single-element series (undefined for `ddof=1`), or
  mismatched-length series for the two-argument functions.
- **`inverse`/`determinant`/`dot` raise `AllieValueError`** (not a raw
  numpy `LinAlgError` or a bare dimension-mismatch message) on a
  singular matrix, a non-square `determinant` target, or a dimension
  mismatch in `dot`.
- **`load` matches CSV columns case-insensitively** against
  `date/open/high/low/close/volume` and names exactly which column is
  missing, rather than surfacing a raw `KeyError`.

## Grammar note

Phase 2.3 (`allie/parser/`) produces the dataclass-based AST in
`allie/ast/nodes.py`; Phase 4 adapted `allie/runtime/interpreter` to walk
that AST directly (superseding Phase 1's hand-built-tuple evaluator —
see `ROADMAP.md` and `allie/runtime/interpreter/__init__.py`'s module
docstring). A full `.ql` program can now run end to end through
`allie run`.
