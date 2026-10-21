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
| `Series[T]` | 1-D series, one value per bar. `Series[Float]` is the workhorse; `Series[Bool]` — one decision per bar — comes from comparing series (`close > SMA(close, 20)`, `fast crosses_above slow`). |
| `Matrix[T]` | 2-D, built from series with `matrix(values, rows, cols)` and `with_column` — see [Matrices](#matrices). |
| `Vector[T]` | Declared in the grammar; no stdlib operates on it yet. |
| `PriceSeries` | OHLCV bundle returned by `load`. Fields: `.open .high .low .close .volume` → `Series[Float]`, plus `.dates` → `Series[Date]`. Any other field is a compile error. |
| `BacktestResult` | What a `backtest` expression evaluates to; fields in [Results](#results). |
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
| `Series[Int]` | `Series[Float]` | Yes | element-wise widening — sound because series are immutable |
| `Series[Float]` | `Series[Int]` | **No** | narrowing |
| anything else | anything else | No | no other conversions exist |

The scalar → `Series` rejection is the one worth calling out. Both carry
`Float`, so it looks like it could work — but series operations are
defined over a sequence of observations. Accepting a scalar there is
exactly the bug the distinction exists to catch (passing today's close
where a whole price history was expected).

Governs `let` initializers with annotations, function arguments, and
`return` values.

### Static typing with local inference

Allie is statically typed: every expression has one type, fixed at
compile time, and there is no dynamic type to fall back on. An
unannotated `let` takes its initializer's type; an annotation, where
written, is enforced.

**Functions are the exception** — every parameter and the return type
must be annotated, including `-> Void`. No inference is attempted. That
is a deliberate v1 limitation: partial inference that works sometimes is
worse to use than none. Functions are hoisted within their scope, so a
function can be called above its declaration and two functions can call
each other.

### What type-checks runs the same everywhere

A program that type-checks behaves identically on all three backends.
Some rules exist only for that reason — each closes a case where one
backend used to accept a program another crashed on or ran differently:

| Rule | Why |
|---|---|
| `/` always produces a `Float` (`7 / 2` is `3.5`) | no silently truncating integer division |
| Functions and strategies aren't values (`let g = f;` is an error) | compiled code calls functions by name |
| A nested function can't use its enclosing function's local variables | there are no closures; pass the value as a parameter |
| Function names are unique across the program | the same reason — two nested `h`s used to call each other's code |
| A `Void` result isn't a value (`let x = print(1);` is an error) | there is no value to store |
| `==`/`!=` compare numbers (Int and Float mix), Bools or Strings — not series | series equality had no defined meaning |
| A series literal holds numbers or Bools | that's all the runtime can store |
| An empty literal needs a numeric series annotation: `let found: Series[Float] = [];` | `[]` alone says nothing about what it will hold |

### Errors

Every type error in the program is reported in one run, each with the
source line, a caret, and a hint where the mistake is a common one —
`int` → did you mean `Int`?, `AAPL.clsoe` → did you mean `close`?, a
Float stored into an Int, a single value passed where a series is
expected. An expression that fails to type-check gets an internal error
type that every later check accepts, so one mistake is one error, not a
cascade.

A runtime error names where it happened, on every backend, with the
same caret — at the innermost expression that failed:

```
AllieValueError: series index 5 is out of range for a series of length 2. Line 2, Column 13
      return s[i];
              ^
```

---

## Literals and comments

```allie
let capital: Float = 1_000_000.0;    // `_` separates digits
let threshold: Float = 2.5e-4;       // an exponent always makes a Float
let bars: Int = 252;                 // Int literals must fit in 64 bits
let label: String = "fast\tslow\n";  // escapes: \n \t \r \" \\
/* block comments /* nest */ too */
```

Strings use double quotes and stay on one line. `.5` must be written
`0.5`. Outside strings and comments the source must be ASCII, which
catches look-alike letters and invisible characters pasted from a
document. The full rules are in [`grammar.md`](grammar.md) "Lexical
structure".

---

## Operators

| Category | Operators | Rules |
|---|---|---|
| Arithmetic | `+ - * %` | Both operands numeric. Result is `Float` if either is, else `Int`. `+` also concatenates two `String`s. |
| Division | `/` | Both operands numeric. Result is always `Float`: `7 / 2` is `3.5`. |
| Comparison | `< > <= >=` | Both operands numeric. Result `Bool`. |
| Equality | `== !=` | Two numbers (Int and Float mix), two Bools or two Strings. Result `Bool`. |
| Logical | `and or` | Both operands `Bool`. Short-circuit. |
| Unary | `-` `!` | `-` numeric; `!` `Bool`. |
| Index | `s[i]` | `s` a `Series[Float]`, `Series[Int]` or `Series[Bool]`; `i` an `Int`. Result is the element type. |

Comparisons don't chain: `0 < x < 10` is a compile error — write
`0 < x and x < 10`. `and`/`or` are keywords, not `&&`/`||`.

### On series

Operators apply to whole series, bar by bar, so indicators and signals
are ordinary expressions. A number on either side applies to every bar.

| Operators | On series | Result |
|---|---|---|
| `+ - * /` | element-wise; `Series[Int]` operands widen | `Series[Float]` |
| `-` (unary) | element-wise | `Series[Float]` |
| `< > <= >=` | element-wise | `Series[Bool]` |
| `crosses_above` `crosses_below` | `a crosses_above b` is true exactly at the bars where `a` moves from at or below `b` to strictly above it; at least one side must be a series, so `rsi crosses_below 30.0` is a level crossing | `Series[Bool]` |
| `and or` | bar by bar; both sides always evaluated; a `Bool` on either side applies to every bar | `Series[Bool]` |
| `!` | bar by bar, on a `Series[Bool]` | `Series[Bool]` |

```allie
let macd: Series[Float] = EMA(close, 12) - EMA(close, 26);
let upper: Series[Float] = SMA(close, 20) + 2.0 * rolling_std(close, 20);
let breakout: Series[Bool] = close crosses_above upper and volume > SMA(volume, 20);
```

- **Warm-up bars are `NaN`, and every comparison with `NaN` is false**,
  so a signal can't fire before its indicator has data. A crossing is
  never true on the first bar (there is no previous bar to cross from),
  and touching without crossing doesn't fire.
- **Division by zero in a series is IEEE**, `inf` or `NaN` for that
  bar, not an error: one zero-volume bar shouldn't abort the whole
  computation. (Scalar division by zero is still an error.)
- **Two series of different lengths are a runtime error**, never
  silently aligned.
- `%`, `==` and `!=` aren't defined on series; `crosses_above` and
  `crosses_below` share `<`'s precedence and don't chain either.

One numpy implementation (`allie/runtime/elementwise.py`) serves all
three backends, the way stdlib functions do.

`%` takes the divisor's sign, as in Python: `-7 % 2` is `1`. `Int` is
64-bit on every backend: a result outside that range is a runtime error
("integer overflow"), never a silently wrapped or unbounded number.
Floats follow IEEE 754 — `1e308 * 10.0` is `inf`, and a `NaN` compares
unequal to everything, itself included. Division or modulo by zero
raises `AllieValueError` at runtime on all three backends — including
the JIT, where an explicit zero-check is emitted before the divide
(hardware division by zero would otherwise kill the process rather than
raise).

### Indexing

```allie
let close: Series[Float] = AAPL.close;
let last: Float = close[len(close) - 1];
for let i: Int = 1; i < len(close); i = i + 1 {
    if close[i] > close[i - 1] { up = up + 1; }
}
```

Indices count from 0 and must be in range — an out-of-range or negative
index is a runtime error on every backend. There is deliberately no
Python-style `s[-1]`: in a bar-by-bar language, an index that quietly
reads the *last* bar is a look-ahead trap. That's also why one bar
(`close[59]`, `close[len(close) - 1]`) is look-ahead inside a trading
decision: compared with every bar, it's the future for the ones before
it ([below](#look-ahead-bias-analysis)). `close[0]` is fine. Series are
immutable, so there is no `s[i] = x`.

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

strategy Name { when condition: action; ... }
backtest Name on Symbol from 0 to 100;          // prints the results table
let r = backtest Name on Symbol from 0 to 100;  // a BacktestResult
backtest Name on Symbol from 0 to 100 with slippage: 0.001, stop_loss: 0.05;
```

### Strategies and backtests

```allie
let fast: Series[Float] = SMA(AAPL.close, 5);
let slow: Series[Float] = SMA(AAPL.close, 20);
let rsi: Series[Float] = RSI(AAPL.close, 14);

strategy Trend {
    when fast crosses_above slow and rsi < 70.0: buy();
    when fast crosses_below slow or rsi > 80.0: sell();
}

let r = backtest Trend on AAPL from 0 to len(AAPL.close);
print(r.sharpe);
```

- **A `when` condition is any `Series[Bool]`**: a crossing, a
  comparison, or a combination of them with `and`/`or`/`!`. A single
  `Bool` (`close[0] > 100.0`) or a `Series[Float]` is a type error, each
  with a hint. On every bar, each clause whose condition is true runs its
  action. The condition must have one value per price bar.
- **A condition is evaluated when its strategy is declared**, on every
  backend. Reassigning `fast` afterwards doesn't change a strategy that
  already exists; declaring it again does. That's what makes a
  parameter sweep work — a `strategy` declared inside a loop captures
  each iteration's values:

  ```allie
  for let w: Int = 3; w <= 12; w = w + 3 {
      let fast: Series[Float] = SMA(AAPL.close, w);
      strategy Sweep { when fast crosses_above slow: buy(); when fast crosses_below slow: sell(); }
      print((backtest Sweep on AAPL from 0 to len(AAPL.close)).sharpe);
  }
  ```

- **`backtest S on SYM from N to M`** — `SYM` is an ordinary variable
  holding a `PriceSeries` (typically from `load`). The bounds are `Int`
  expressions: bar indices, `from` inclusive and `to` exclusive, so a
  train/test split is `from 0 to split` and `from split to len(c)`.
  Conditions are computed over the whole history and then windowed, so
  an indicator's warm-up never restarts inside the test window. A
  literal bound that is fractional or negative, or `to <= from` with two
  literals, is a compile error. `to` is clamped to the available bars,
  so `to 100` against 60 bars runs 60; a `from` past the end is an
  error, not a clamp.
- **As a statement**, `backtest` prints the results table. **As an
  expression** it evaluates to a `BacktestResult` (fields under
  [Types](#types)), and `print(r)` prints the same table.
  `r.returns` are the per-bar returns the Sharpe was computed from, so
  `sharpe(r.returns, 252) == r.sharpe` exactly; `r.returns[0]` is `NaN`,
  like `pct_change`.

### Orders

An action places orders with these builtins. Calling one anywhere else —
outside a `when` action or a function — is a compile error.

| Order | Effect |
|---|---|
| `buy()` | if not long, go all-in long — from a short this reverses |
| `sell()` | if long, go flat |
| `short()` | if not short, go all-in short — from a long this reverses |
| `cover()` | if short, go flat |
| `target(Float)` | hold that fraction of equity: `0.5` half in, `0.0` flat, `-1.0` all-in short |

The guards make a repeated signal a no-op, so `when rsi < 30.0: buy();`
buys once, not on every bar the condition holds. A fill's costs come
out of the equity it's sized against, so afterwards the position is
exactly the fraction asked for: `buy()` spends exactly the cash and
never borrows, and repeating `target(0.5)` at an unchanged price trades
nothing. Equity marks to market at every close; if it reaches zero, the
position is closed out and the strategy stops trading.

**When an order fills.** A condition at bar i is computed from bar i's
close, so by default an order placed there fills at bar i + 1's **open**
— the first price anyone could actually have traded at. An order placed
on the last bar never fills. When several clauses fire on one bar, the
last order placed is the one that fills (`buy()` then `sell()` is no
trade). A bar without a usable price (`NaN`, or not positive) can't fill
an order, which waits for the next one.

### Options

`with name: value, ...` after the bars sets how the backtest trades.
Values are expressions of the option's type, so a cost sweep is a loop.
Unknown names, the wrong type, a literal out of range, or the same name
twice are compile errors; a computed value is checked when it runs.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `slippage` | `Float` | `0.0005` | fraction of the price lost per fill — buys fill above it, sells below |
| `commission` | `Float` | `0.0` | fraction of the value traded, per fill |
| `fill` | `String` | `"open"` | `"open"`: the next bar's open; `"close"`: the signal bar's own close |
| `cash` | `Float` | `10000.0` | starting equity |
| `stop_loss` | `Float` | none | exit when the price moves this fraction against the entry |
| `take_profit` | `Float` | none | exit when it moves this fraction in favor |
| `periods_per_year` | `Int` | `252` | for annualizing CAGR, volatility, Sharpe and Sortino |

```allie
for let bps: Int = 0; bps <= 50; bps = bps + 25 {
    print((backtest Trend on AAPL from 0 to len(c) with slippage: bps / 10000.0).final_equity);
}
```

**Costs are on by default** — 5 bps of slippage per fill — because a
cost model that has to be opted into gets forgotten, and the backtest
looks better than trading would (`docs/spec.md`). The results table
states the fill point and both costs, so they're never implicit; a
frictionless run is a visible `with slippage: 0.0`.

**Stops** are checked on every bar a position is open, against the
bar's high and low — including the bar a next-open fill entered on. A
bar that opens past a level exits at the open, since a gap gives no
better price; otherwise the level is the exit price. If one bar's range
reaches both levels, it's read as the stop: the bar doesn't say which
came first, and the pessimistic reading is the safe one.

### Results

| Field | Type | |
|---|---|---|
| `final_equity`, `initial_equity` | `Float` | |
| `total_return` | `Float` | `final / initial - 1` |
| `buy_hold_return` | `Float` | holding the symbol over the window, no costs |
| `cagr` | `Float` | over `(bars - 1) / periods_per_year` years |
| `volatility` | `Float` | annualized, `ddof=1` |
| `sharpe` | `Float` | annualized, risk-free rate 0 |
| `sortino` | `Float` | annualized, over the downside deviation; `inf` with no losing bar |
| `max_drawdown` | `Float` | largest peak-to-trough fall, as a positive fraction |
| `exposure` | `Float` | fraction of bars that ended holding a position |
| `costs` | `Float` | slippage and commission paid, in currency |
| `trades` | `Int` | completed round trips; one still open at the end isn't counted |
| `win_rate`, `profit_factor`, `avg_trade_return` | `Float` | over completed trades, after costs; `NaN` with none |
| `equity`, `returns`, `drawdown`, `position` | `Series[Float]` | one value per bar of the window; `position` is the signed fraction of equity held |
| `trade_returns` | `Series[Float]` | one per completed trade: its P&L after costs over its entry value |

---

## Standard library

Every function is registered once in `allie/stdlib/registry.py` and
resolved by name — none are special-cased anywhere else. A builtin never
ends a program with a Python traceback: a numeric-library error inside
one becomes an `AllieValueError` naming the function, and floating-point
events inside one follow IEEE 754 silently (no numpy warnings on stderr).

### Missing values

A `NaN` in a series means *no value at this bar*: an indicator's warm-up,
a gap in the data file, a bar `shift` moved past the edge. Every function
treats it the same way, by kind (`allie/stdlib/missing.py`):

| Kind | Rule | e.g. |
|---|---|---|
| Reductions | skip missing bars, as pandas does; raise if none are left | `mean(SMA(c, 20))` is the mean of the bars SMA has |
| Paired reductions | use the bars where both series have a value | `correlation`, `covariance`, `beta` |
| Windows | missing where the window holds a missing bar, and only there | `SMA`, `rolling_*` |
| Recurrences | run over the bars that have a value; missing bars stay missing | `EMA(SMA(c, 20), 9)` starts with the SMA |
| Positional | missing exactly where an input they read is | `shift`, `diff`, `pct_change`, operators |

Only `NaN` is missing; `inf` is a value. Return statistics (`sharpe`,
`sortino`, `volatility`, the tail measures) and the significance tests
also drop infinite values, which only a zero price produces — so
`sortino(r.returns, 252)` equals the backtest's `r.sortino` exactly.

### Point in time, or the whole history

The **T** column below marks the functions whose value at bar i uses
only bar i and earlier — **trailing** ones, which a `when` condition can
use. The rest read the whole series, later bars included: `zscore(c)[i]`
is measured against the mean of every bar, and `c > mean(c)` compares
each bar with an average of the future. Using one of those in a
condition is a look-ahead error ([below](#look-ahead-bias-analysis)),
and the error names the trailing alternative. Outside a condition, for
analysis, they're fine.

### Statistics

| Signature | T | |
|---|---|---|
| `mean`, `median`, `sum`, `min`, `max` `(Series<Float>) -> Float` | | |
| `std`, `variance` `(Series<Float>) -> Float` | | sample, `ddof=1`; need 2 values |
| `covariance`, `correlation` `(Series<Float>, Series<Float>) -> Float` | | a series that doesn't vary has no correlation (an error) |
| `percentile(Series<Float>, Float) -> Float` | | `p` on 0–100, linear interpolation |
| `zscore(Series<Float>) -> Series<Float>` | | distance from the whole series' mean, in std |
| `count(Series<Bool>) -> Int` | | how many bars are true |
| `len(Series) -> Int` | | number of bars, for a series of any element type: fine as a loop bound or a backtest's range, but in a decision it counts the bars still to come |

An empty series raises rather than returning `NaN` — an empty series is
nearly always a bug (a bad load, an off-by-one window) — and so does one
whose bars are all missing.

### Moving averages and rolling statistics

| Signature | T | |
|---|---|---|
| `SMA(Series<Float>, Int) -> Series<Float>` | ✓ | simple moving average; `rolling_mean` is the same |
| `EMA(Series<Float>, Int) -> Series<Float>` | ✓ | `alpha = 2/(span+1)`, seeded with the first value |
| `rolling_sum`, `rolling_std`, `rolling_max`, `rolling_min` `(Series<Float>, Int) -> Series<Float>` | ✓ | `rolling_std` is `ddof=1` |
| `rolling_zscore(Series<Float>, Int) -> Series<Float>` | ✓ | `(x - SMA(x, w)) / rolling_std(x, w)`: the trailing `zscore` |
| `rolling_corr(Series<Float>, Series<Float>, Int) -> Series<Float>` | ✓ | correlation of the last `w` bars |
| `cumsum(Series<Float>) -> Series<Float>` | ✓ | running total |
| `shift(Series<Float>, Int) -> Series<Float>` | ✓ if ≥ 0 | positive shifts later; a negative shift is look-ahead |
| `diff`, `pct_change` `(Series<Float>) -> Series<Float>` | ✓ | first difference, simple return; `NaN` at bar 0 |
| `append(Series<Float>, Float) -> Series<Float>` | ✓ | a new series, one longer: how a loop collects a value per pass |

```allie
let fast: Series[Float] = SMA(prices.close, 5);
let slow: Series[Float] = SMA(prices.close, 20);
let signal: Series[Bool] = fast crosses_above slow;   // an operator, see Operators
```

Windowed functions are `NaN` for their first `window - 1` bars (pandas'
convention). Each is computed per window, so one missing bar affects
only the windows that hold it. `rolling_std` is vectorized and
deliberately *not* computed from running sums of `x` and `x²`, which is
faster but loses accuracy to cancellation on price-like data. `EMA` is
an O(n) loop on purpose: its vectorized closed form underflows to
garbage on long series.

### Indicators

Most indicators are one expression — MACD, Bollinger bands, a
volatility filter; see [On series](#on-series). These are the ones
operators can't express.

| Signature | T | |
|---|---|---|
| `RSI(Series<Float>, Int) -> Series<Float>` | ✓ | Relative Strength Index, 0–100, Wilder smoothing |
| `ATR(PriceSeries, Int) -> Series<Float>` | ✓ | Average True Range, Wilder smoothing; gaps count as range |

```allie
let rsi: Series[Float] = RSI(AAPL.close, 14);
print(count(rsi crosses_below 30.0));    // how often it became oversold
let channel_high: Series[Float] = rolling_max(AAPL.high, 20);
```

`RSI` and `ATR` are `NaN` for the first `period` bars, then follow
Wilder (and TA-Lib): the first value is the mean of the first `period`
changes, then `avg = (avg * (period - 1) + x) / period`. `RSI` is 100
over a window with no losses and 50 over one with no movement, rather
than dividing by zero. Across a missing bar the change is from the last
bar with a value. A period or window below 1 raises.

### Returns and risk

| Signature | T | |
|---|---|---|
| `log_returns(Series<Float>) -> Series<Float>` | ✓ | `log(p[i] / p[i-1])` |
| `cumulative_returns(Series<Float>) -> Series<Float>` | ✓ | compounded growth to each bar; a missing return counts as none |
| `drawdown(Series<Float>) -> Series<Float>` | ✓ | fall below the running high of a price or equity series |
| `max_drawdown(Series<Float>) -> Float` | | the largest of those |
| `volatility(Series<Float>, Int) -> Float` | | annualized std of returns |
| `sharpe(Series<Float>, Int) -> Float` | | annualized, risk-free rate 0 |
| `sortino(Series<Float>, Int) -> Float` | | annualized, over the downside deviation |
| `value_at_risk(Series<Float>, Float) -> Float` | | historical VaR at a confidence (`0.95`), as a positive loss |
| `expected_shortfall(Series<Float>, Float) -> Float` | | the mean loss beyond the VaR |
| `beta(Series<Float>, Series<Float>) -> Float` | | `cov(asset, market) / var(market)` of returns |

`sharpe`, `sortino`, `volatility` and `max_drawdown` are the backtest
engine's own definitions, so a figure computed in source is the one a
`backtest` reports.

### Math

| Signature | T | |
|---|---|---|
| `abs(x)` | ✓ | `x` an `Int`, `Float` or numeric series; an `Int` stays an `Int` |
| `sqrt(x)`, `log(x)`, `exp(x)` | ✓ | a number gives a `Float`, a series a `Series<Float>` |
| `floor`, `ceil`, `round` `(Float) -> Int` | ✓ | `round` sends ties to the even neighbor, as Python and numpy do |

On a series these follow IEEE per bar, as the operators do: `log` of 0
is `-inf`, of a negative `NaN`. On a single number a value outside the
domain is an error, as `1 / 0` is. `floor(len(c) * 0.7)` is how a split
point is computed.

### Significance testing

Is a backtest's result real, or the best of many tries? `docs/spec.md`
treats this as structural rather than optional: reporting the best Sharpe
across N strategies without correcting for N is p-hacking by
construction. Every test here is exact, with no sampling, so none takes
a seed and each gives the same answer on every run and every backend.

| Signature | T | |
|---|---|---|
| `backtest_pvalue(BacktestResult) -> Float` | | is the strategy's timing luck? (below) |
| `permutation_pvalue(Series<Float>, Series<Float>) -> Float` | | `(signal, returns)`: is the association luck? |
| `probabilistic_sharpe(Series<Float>, Float, Int) -> Float` | | `(returns, benchmark, periods_per_year)`: P(true Sharpe > benchmark) |
| `deflated_sharpe(Series<Float>, Series<Float>, Int) -> Float` | | `(returns, trial_sharpes, periods_per_year)`: the same, for the best of several |
| `holm(Series<Float>, Float) -> Series<Bool>` | | family-wise error control; True where rejected |
| `benjamini_hochberg(Series<Float>, Float) -> Series<Bool>` | | false-discovery-rate control; True where rejected |
| `bonferroni(Float, Int) -> Float` | ✓ | adjusted p-value, `min(1, p*n)` |

A sweep collects one Sharpe and one p-value per strategy, then judges
them together ([`examples/rigor.qs`](../examples/rigor.qs)):

```allie
let sharpes: Series[Float] = [];
let pvalues: Series[Float] = [];
for let w: Int = 4; w <= 20; w = w + 4 {
    let trend: Series[Float] = SMA(c, w);
    strategy Cross { when c crosses_above trend: buy(); when c crosses_below trend: sell(); }
    let r = backtest Cross on AAPL from 0 to len(c);
    sharpes = append(sharpes, r.sharpe);
    pvalues = append(pvalues, backtest_pvalue(r));
}
print(holm(pvalues, 0.05));                          // which survive five tries
print(deflated_sharpe(best_returns, sharpes, 252));  // is the winner's Sharpe real?
```

**Timing and association: every shift, not a shuffle.**
`backtest_pvalue(r)` asks how often the strategy's own positions,
shifted to every other point in time, would have earned at least as much
from the market's close-to-close moves, before costs. `permutation_pvalue`
asks the same of a signal's correlation with returns. A circular shift
keeps each series exactly as it is (its trends, its persistence, how long
a position is held) and breaks only the alignment. Shuffling bars one by
one instead destroys that persistence and makes chance look too
unlikely. On two unrelated series as persistent as a trend signal
(AR(1), φ = 0.9), a shuffle test calls the link significant at 5% in 48%
of trials; the shift test does so in 3%. With n bars the smallest
p-value is 1/n: that is all n bars can show. Test returns and spreads,
not price levels, since a trending series looks unlike its own shifted
copies whatever it is compared with. Buy-and-hold has no timing to test
and scores 1.

**Sharpe ratios are estimates.** `probabilistic_sharpe` (Bailey & López
de Prado) is the probability that the true Sharpe exceeds a benchmark,
given the number of bars and how skewed and fat-tailed the returns are.
`probabilistic_sharpe(r.returns, 0.0, 252) > 0.95` is "this strategy
makes money" at 95% confidence, stated properly. `deflated_sharpe` asks
the same of the *best* of several strategies. It sets the benchmark to
the Sharpe the best of that many zero-skill strategies would reach by
luck, from how many there were and how widely their Sharpes spread.
`trial_sharpes` holds every strategy tried, the winner included. Similar
trials (adjacent windows) spread less and cost less. It is conservative:
the best of 20 zero-skill strategies passes 0.95 about 0.1% of the time,
where `probabilistic_sharpe` alone passes it 64% of the time. A true
Sharpe of 3.5 over three years, among 19 without, is still found 96% of
the time.

**Several results at once.** `holm` and `bonferroni` bound the chance of
even one false positive, and `holm` rejects everything `bonferroni` would,
often more. `benjamini_hochberg` bounds the share of false ones among the
rejections, which is more lenient when many results are real. Both
series forms return a `Series<Bool>` aligned with the input.
`benjamini_hochberg` is a *step-up* procedure: it rejects all of the k
smallest p-values, not only those individually under the line.

Every one of these is a statistic of whole series or a whole backtest,
so none can drive a trade: `when backtest_pvalue(r) < 0.05` is a
look-ahead error. Whether each test holds its error rate is checked by
simulation in `tests/unit/test_rigor_calibration.py`.

### Matrices

A matrix is built from series, which is where the data is:

```allie
let X = with_column(with_column(matrix(ones, n, 1), market), size);  // 3 columns
let coefficients: Series[Float] = ols(stock, X);
```

| Signature | T | |
|---|---|---|
| `matrix(Series<Float>, Int, Int) -> Matrix<Float>` | | `(values, rows, cols)`, filled row by row |
| `with_column(Matrix<Float>, Series<Float>) -> Matrix<Float>` | ✓ | appends a column |
| `identity(Int) -> Matrix<Float>` | ✓ | |
| `rows`, `cols` `(Matrix<Float>) -> Int` | ✓ | |
| `row`, `column` `(Matrix<Float>, Int) -> Series<Float>` | `column` ✓ | one row or column, from 0 |
| `transpose`, `inverse` `(Matrix<Float>) -> Matrix<Float>` | | `inverse` of a singular matrix is an error |
| `determinant(Matrix<Float>) -> Float` | | |
| `dot(Matrix<Float>, Matrix<Float>) -> Matrix<Float>` | | matrix product |
| `covariance_matrix(Matrix<Float>) -> Matrix<Float>` | | covariance between columns; incomplete rows left out |
| `ols(Series<Float>, Matrix<Float>) -> Series<Float>` | | least-squares coefficients, one per column; no implicit intercept |

`print` shows a matrix as `Matrix[2x2: [1, 2], [3, 4]]`.

### Sampling, I/O and orders

| Signature | T | |
|---|---|---|
| `normal(Float, Float, Int, Int) -> Series<Float>` | ✓ | `(mean, std, n, seed)` — an explicit seed, so reruns reproduce |
| `load(String) -> PriceSeries` | ✓ | CSV → OHLCV; rules below |
| `print(Any) -> Void` | ✓ | compact series and matrices; a `PriceSeries` as its size and dates; a `BacktestResult` as its table |
| `buy()`, `sell()`, `short()`, `cover()` → `Void` | ✓ | orders; see [Orders](#orders) |
| `target(Float) -> Void` | ✓ | an order for a fraction of equity |

**`load`** finds `date, open, high, low, close, volume` by name, in any
case and order, ignoring other columns; a UTF-8 byte-order mark (as
Excel writes) is fine. An empty cell, or `NA`, `N/A`, `NaN`, `null`, `-`,
is a missing value. Anything else that isn't a number is an error naming
its line and column. Dates are ISO 8601 — `YYYY-MM-DD`, optionally with
a time, `/` allowed between the parts — and anything else is an error:
`01/02/2024` could be either of two days, and `20240102` would otherwise
be read as a date in the year 2256. Rows must be in time order: a file
that is entirely newest-first, as many exports are, is reversed, and any
other disorder, or a repeated date, is an error naming the line.

The orders are ordinary builtins, not keywords; the type checker allows
them only in a `when` action or a function, and a function that places
one raises if it's called while no backtest is running.

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
| `WITH` | `with` | — |

<!-- END GENERATED KEYWORDS -->

---

## Look-ahead bias analysis

Every trading decision must be provably free of future data, checked at
compile time. Look-ahead bias is *deciding* at bar t on data from after
t, so the checks sit where decisions are made:

- a `when` condition;
- a condition inside a `when` action (`if mean(c) > 150.0 { buy(); }`);
- the size of an order, `target(f)`;
- a backtest option (`with stop_loss: ...`);
- the condition of an `if` or loop that decides whether a backtest runs.

Future data enters at these sources:

| Source | Example | Why it sees the future |
|---|---|---|
| a backward time shift | `shift(c, -1)` | bar `i+1`'s value, placed at bar `i` |
| a shift whose period can't be proven ≥ 0 | `shift(c, k)` after `k = k - 2` | it might be backward |
| a whole-series builtin | `mean(c)`, `zscore(c)`, `percentile(c, 90.0)` | its value at any bar reflects the later ones (the unmarked rows in the [Standard library](#standard-library) tables) |
| the number of bars | `len(c)` | it counts the bars still to come |
| a single bar | `c[59]`, `c[len(c) - 1]` | it's the future for every bar before it (`c[0]` is fine) |
| a backtest's statistics | `r.sharpe`, `r.trades`, `backtest_pvalue(r)` | they cover the whole backtest (`r.equity`, `r.returns`, `r.drawdown`, `r.position` are per bar, and fine) |
| state a backtest's actions left | `n` after `when ...: { n = n + 1.0; }` ran | it holds what they did on every bar |
| a hand-built series | `[c[1], c[0]]` | which bar each element came from can't be checked |

A sign analysis proves a shift's period is 0 or more when it can: a
literal, a variable holding one (`let lag: Int = 1;`), sums and products
of those, `x % 5`, and a loop counter that starts at 0 or more and only
grows.

From a source, the taint travels through variables, operators,
indicators, field access, indexing and user functions. A function is
analyzed where it's called, with its arguments: one that returns a
tainted global taints its result, one that assigns a tainted value to a
global taints that global from the call on, and one that's never called
changes nothing. The analysis follows control flow. A later `s = slow`
clears an earlier `s = shift(c, -1)`. A value that a branch on future
data chose is tainted: after `if mean(c) > 150.0 { s = fast; }`, so is
`s`. Loops are analyzed until nothing changes, so a value tainted at the
end of one iteration counts at the start of the next.

```allie
let tomorrow: Series[Float] = shift(AAPL.close, -1);
strategy S { when tomorrow crosses_above slow: buy(); }   // compile error
strategy T { when rsi < 30.0 and shift(AAPL.close, -1) > AAPL.close: buy(); }  // also
strategy U { when zscore(AAPL.close) < -2.0: buy(); }   // also: use rolling_zscore(c, 20)
strategy V { when AAPL.close > slow: { target(0.1 / std(pct_change(AAPL.close))); } }  // also
```

The error points at the part of the decision that brings the future in
(`tomorrow`, or the `shift(...)` inside `T`'s condition), and its notes
trace it back to where the future entered:

```
error: look-ahead bias: this 'when' condition in strategy 'S' may contain data
from a future bar. ... 'tomorrow' is derived from a backwards time shift ... (2:19)
  help: a positive shift looks back: shift(x, 1) is the previous bar's value
note: 'tomorrow' is given data from a future bar here (1:1)
note: the future enters here: shift by -1 gives each bar the value from 1 bar later (1:31)
```

Future data outside a decision stays legal. `print(mean(shift(x, -1)))`
is post-hoc analysis, and `backtest S on AAPL from 0 to len(c)` is a
range, not a decision.

### Checking it by experiment: `allie check --audit`

The analysis is a proof from the source text. `--audit` checks the same
property by experiment, straight from its definition: a program looks
ahead if a decision it makes at bar t changes when the bars after t are
removed. It runs the program once more for every bar t, with each
`load(...)` cut to bars 0..t. It then compares what every backtest
decided at the bars both runs had: which `when` conditions held, the
orders placed and their sizes, and the trades closed, stops included.

```
$ allie check --audit examples/cheat_demo.qs
error: look-ahead bias: ...
look-ahead audit: FAILED -- 3 decisions changed when later bars were removed
  backtest 1 ('Clairvoyant'), bar 23 -- 'when' clause 2: true with all 60 bars, false with only the first 24
  backtest 1 ('Clairvoyant'), bar 37 -- 'when' clause 1: true with all 60 bars, false with only the first 38
  backtest 1 ('Clairvoyant'), bar 37 -- orders placed: target 1 with all 60 bars, none with only the first 38
```

It runs even on a program the analysis rejected, which shows whether the
leak is real on your data. It can only show what the data triggers: a
`zscore` inside a condition that is never true passes the audit and is
still rejected, because on other data it would decide trades.

### Soundness

The analysis is biased toward false positives, because a missed
violation is a silently wrong backtest while a spurious one is a compile
error you can see. Unprovable means rejected. The test suite holds it to
the audit's definition: it generates hundreds of programs that mix
honest and leaky constructs, and every program the analysis accepts must
pass the audit (`tests/unit/test_lookahead_audit.py`).

What it can't see:

- data already shifted outside the language, such as a pre-shifted CSV
  column. Nothing in the source reveals it, and the audit can't tell
  either;
- the scope a function or a `when` action is declared in. Its
  module-level variables are resolved where it is called or backtested,
  so a local there with the same name would be read instead.

Where it is stricter than it needs to be: choosing a strategy by a
backtest's results, such as the window with the best `r.sharpe`, and then
trading it is rejected even when the choice used only earlier bars, as
in a train/test split. Bar ranges are run-time values, so the analysis
can't tell which bars a statistic covered. Report the in-sample winner
through `deflated_sharpe` instead, which corrects for having picked it.

## Known limitations

- **`import` parses but resolves nothing.** No module loader, no path
  resolution. Deliberately inert rather than half-working.
- **No `Matrix` literal syntax.** A matrix is built from series with
  `matrix(values, rows, cols)` and `with_column`, not written out.
- **No series slicing.** A statistic over the first N bars only — a
  threshold fitted on a training window — can't be computed; and
  because of that, a whole-series statistic is rejected in a condition
  even where a careful author meant it to cover only earlier bars.
- **`break`/`continue` cannot escape a `when` action.** It's a per-bar
  callback; there's no enclosing loop to leave. A loop written *inside*
  the action is fine.
- **Assignment targets only bare names.** `s[i]` reads an element, but
  series are immutable, so there is no `s[i] = x`; fields are read-only
  too.
- **Field access is typed only for `PriceSeries` and `BacktestResult`.**
  Anything else is accepted as `Void` rather than claiming a type that
  can't be verified.
- **The backtest trades one symbol on daily-style bars.** No portfolio
  of several symbols, no volume limits or partial fills, no borrow cost
  for shorts or interest on cash, and the risk-free rate in Sharpe is 0.
  Stops are fractions of the entry price, not trailing.
- **A strategy's actions can't read the current bar's index.** Per-bar
  logic belongs in the condition, which can express it with series
  operators; the action decides only *what* to do.
- **Native recursion is capped at 10,000 nested calls** on the LLVM
  backend ("recursion too deep"); the interpreters hit their host's
  limit much sooner.
- **"Every path returns" is a structural check**, not full reachability
  analysis. It sees direct `return`s and `if`/`else` where both branches
  return.
