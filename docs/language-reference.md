# Allie Language Reference

Complete reference for types, the standard library, and keywords. For
the formal grammar see [`grammar.md`](grammar.md); for how the compiler
is put together see [`how-it-works.md`](how-it-works.md).

---

## Types

| Type | Notes |
|---|---|
| `Int` | 64-bit signed. Native `i64` under LLVM. |
| `Float` | Double precision. Native `double`. |
| `Bool` | Native `i1`. No truthiness — conditions must be exactly `Bool`. |
| `String` | Immutable. `+` concatenates. Boxed at the LLVM boundary. |
| `Void` | A function returning nothing. Must be written explicitly (`-> Void`). |
| `Series[T]` | 1-D numeric series. `Series[Float]` is the workhorse; `Series[Bool]` comes from the crossing operators. |
| `Matrix[T]` | 2-D. Usable via the matrix stdlib, but **no literal syntax constructs one** — see Known limitations. |
| `Vector[T]` | Declared in the grammar; no stdlib operates on it yet. |
| `PriceSeries` | OHLCV bundle returned by `load`. Fields: `.open .high .low .close .volume` → `Series[Float]`, plus `.dates`. |
| `Date`, `DateRange` | Declared in the grammar; not yet produced by anything. |

Two types exist internally but **cannot be written in source**:

- `Any` — only so `print` accepts anything. Writing `let x: Any` fails
  with "unknown type", deliberately: a general escape hatch would
  undermine the point of the type system.
- `Strategy` — the type a `strategy` declaration binds, so `backtest`
  can verify its target through ordinary name lookup.

### Assignability

`is_assignable(source, target)`:

| Source | Target | OK? | Why |
|---|---|---|---|
| `T` | `T` | Yes | exact match |
| anything | `Any` | Yes | top type, internal only |
| `Int` | `Float` | Yes | widening |
| `Float` | `Int` | **No** | narrowing would silently truncate |
| `Float` | `Series[Float]` | **No** | a scalar is not a one-element series |
| `Series[Int]` | `Series[Float]` | **No** | no per-element widening; containers must match exactly |
| anything else | anything else | No | no other conversions exist |

The scalar → `Series` rejection is the one worth calling out. Both carry
`Float`, so it looks like it could work — but series operations are
defined over a sequence of observations. Accepting a scalar there is
exactly the bug the distinction exists to catch (passing today's close
where a whole price history was expected).

Governs `let` initializers with annotations, function arguments, and
`return` values.

### Gradual typing

Annotations are optional; where present they're enforced. An unannotated
`let` infers from its initializer.

**Functions are the exception** — every parameter and the return type
must be annotated, including `-> Void`. No inference is attempted. That
is a deliberate v1 limitation: partial inference that works sometimes is
worse to use than none.

---

## Operators

| Category | Operators | Rules |
|---|---|---|
| Arithmetic | `+ - * / %` | Both operands numeric. Result is `Float` if either is. `+` also concatenates two `String`s. |
| Comparison | `< > <= >=` | Both operands numeric. Result `Bool`. |
| Equality | `== !=` | Operands must be the *same* type. Result `Bool`. |
| Logical | `and or` | Both operands `Bool`. Short-circuit. |
| Unary | `-` `!` | `-` numeric; `!` `Bool`. |

`and`/`or` are keywords, not `&&`/`||`. Division or modulo by zero
raises `AllieValueError` at runtime on all three backends — including
the JIT, where an explicit zero-check is emitted before the divide
(hardware division by zero would otherwise kill the process rather than
raise).

---

## Statements

```allie
let name: Type = expr;          // annotation optional
func f(a: Int) -> Int { ... }   // annotations required
if cond { ... } else { ... }    // cond must be exactly Bool
while cond { ... }
for let i: Int = 0; i < n; i = i + 1 { ... }
break; continue; return expr;
import "path";                  // parses, resolves nothing (see limitations)

strategy Name {
    when seriesA crosses_above seriesB: buy();
    when seriesA crosses_below seriesB: sell();
}
backtest Name on Symbol from 0 to 100;
```

`backtest ... on SYM` — `SYM` is an ordinary variable holding a
`PriceSeries` (typically from `load`). `from`/`to` are bar indices; `to`
is clamped to the available bar count, so `to 100` against 60 bars runs
60. A `from` past the end is an error, not a clamp.

---

## Standard library

Every function is registered once in `allie/stdlib/registry.py` and
resolved by name — none are special-cased anywhere else.

### Statistics

| Signature | |
|---|---|
| `mean(Series<Float>) -> Float` | |
| `median(Series<Float>) -> Float` | |
| `std(Series<Float>) -> Float` | sample std, `ddof=1` |
| `variance(Series<Float>) -> Float` | sample, `ddof=1` |
| `covariance(Series<Float>, Series<Float>) -> Float` | |
| `correlation(Series<Float>, Series<Float>) -> Float` | |

```allie
let prices = load("data.csv");
print(mean(prices.close));
print(std(prices.close));
```

**Edge cases.** Empty series raises rather than returning `NaN` — an
empty series is nearly always a bug (bad load, off-by-one window), and
`NaN` would propagate silently through everything downstream.
`std`/`variance` need ≥2 elements (`ddof=1` is undefined for one).
`covariance`/`correlation` raise on mismatched lengths rather than
truncating.

Note these do *not* skip `NaN`: `mean` of a series containing `SMA`'s
warmup padding is `NaN`. Correct, but surprising the first time.

### Series operations

| Signature | |
|---|---|
| `SMA(Series<Float>, Int) -> Series<Float>` | simple moving average |
| `EMA(Series<Float>, Int) -> Series<Float>` | `alpha = 2/(span+1)` |
| `rolling_mean(Series<Float>, Int) -> Series<Float>` | alias for `SMA` |
| `rolling_std(Series<Float>, Int) -> Series<Float>` | |
| `shift(Series<Float>, Int) -> Series<Float>` | positive shifts forward |
| `diff(Series<Float>) -> Series<Float>` | first difference |
| `pct_change(Series<Float>) -> Series<Float>` | |
| `crosses_above(Series<Float>, Series<Float>) -> Series<Bool>` | |
| `crosses_below(Series<Float>, Series<Float>) -> Series<Bool>` | |

```allie
let fast: Series[Float] = SMA(prices.close, 5);
let slow: Series[Float] = SMA(prices.close, 20);
let signal: Series[Bool] = crosses_above(fast, slow);
```

**Edge cases.** `SMA`/`rolling_mean`/`rolling_std` NaN-pad the first
`window - 1` bars (pandas convention). `EMA` does **not** — it seeds
from the first observation, which is why the `blended.qs` example uses
`EMA` where it needs a `NaN`-free mean. `shift`/`diff`/`pct_change`
NaN-fill vacated positions. Crossings are never true on the first bar
(no prior bar to cross from) and use strict inequality, so touching
without crossing doesn't fire; moving from *equal* to above does.
Mismatched lengths raise.

`rolling_std` is vectorized with `sliding_window_view` and is
bit-identical to the per-window loop it replaced (~190× faster than
that loop). It is deliberately *not* implemented with the cumulative
sum-of-squares trick, which is faster still but loses accuracy to
catastrophic cancellation on price-like data. `EMA` remains a Python
loop on purpose — it is a first-order recurrence whose vectorized
closed form overflows to non-finite values on long series.

### Matrix

| Signature | |
|---|---|
| `transpose(Matrix<Float>) -> Matrix<Float>` | |
| `inverse(Matrix<Float>) -> Matrix<Float>` | raises on singular |
| `determinant(Matrix<Float>) -> Float` | raises on non-square |
| `dot(Matrix<Float>, Matrix<Float>) -> Matrix<Float>` | raises on dimension mismatch |

All raise `AllieValueError` with a readable message rather than leaking
a raw numpy `LinAlgError`. **No source syntax constructs a `Matrix`** —
see Known limitations.

### Sampling

| Signature | |
|---|---|
| `normal(Float, Float, Int, Int) -> Series<Float>` | `(mean, std, n, seed)` |
| `zscore(Series<Float>) -> Series<Float>` | |
| `percentile(Series<Float>, Float) -> Float` | `p` on 0–100 |

`normal` takes an explicit **seed** — there is no hidden global RNG, so
reruns are reproducible. `zscore` raises on zero standard deviation.
`percentile` rejects `p` outside 0–100.

### I/O and trading

| Signature | |
|---|---|
| `load(String) -> PriceSeries` | CSV → OHLCV |
| `print(Any) -> Void` | compact `Series` formatting |
| `buy() -> Void` | strategy bodies only |
| `sell() -> Void` | strategy bodies only |

`load` matches `date,open,high,low,close,volume` case-insensitively and
names the missing column if one is absent. `print` truncates long series
(`Series[1, 2, 3, ..., 8, 9, 10]`). `buy`/`sell` raise unless a backtest
is running — they're ordinary builtins, not keywords, so nothing earlier
in the pipeline can tell they're special.

---

## Keywords

Every control-flow keyword has a second, informal spelling. The lexer
resolves both to the same token, so they are fully interchangeable
everywhere — verified end to end on all three backends in
`tests/unit/test_keyword_alias_transparency.py`.

Lookup is **case-sensitive**: `Ahhh` is `while`; `ahhh` is an identifier.

Quant keywords are canonical-only in v1. The alias mechanism would
support them unchanged if that ever changes.

<!-- BEGIN GENERATED KEYWORDS -- do not hand-edit; run scripts/generate_keywords_doc.py -->

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

<!-- END GENERATED KEYWORDS -->

---

## Known limitations

- **`import` parses but resolves nothing.** No module loader, no path
  resolution. Deliberately inert rather than half-working.
- **No `Matrix` literal syntax.** Array literals produce `Series` (a
  nested one types as `Series[Series[Float]]`), and every matrix builtin
  requires a `Matrix` argument — so nothing in the language can
  construct the first one.
- **`break`/`continue` cannot escape a `when` action.** It's a per-bar
  callback; there's no enclosing loop to leave. A loop written *inside*
  the action is fine.
- **Assignment targets only bare names** — no `arr[i] = x` or
  `obj.field = x`.
- **Field access is typed only for `PriceSeries`.** Anything else is
  accepted as `Void` rather than claiming a type that can't be verified.
- **User-defined functions must be fully scalar to compile.** One taking
  a `Series` parameter is rejected by the LLVM backend with a clear
  error; it still runs on the other two backends.
- **"Every path returns" is a structural check**, not full reachability
  analysis. It sees direct `return`s and `if`/`else` where both branches
  return.
- **Most runtime errors lack line numbers**, because they originate
  inside stdlib functions operating on data rather than at a source
  position.
