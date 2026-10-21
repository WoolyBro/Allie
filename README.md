# Allie

Allie is a statically typed programming language that compiles through a
custom IR to LLVM and JIT-executes. It's two languages merged into one:
a general-purpose core (functions, `for`/`while`, `break`/`continue`,
recursion) and a quantitative-finance DSL (`Series` types, a numpy-backed
indicator library, and first-class `strategy`/`when`/`backtest`
constructs). Neither half special-cases the other — they compose because
they lower to the same IR. It also accepts a second spelling for every
control-flow keyword (`ts` for `let`, `Ahhh` for `while`), which is a
joke, but a joke implemented as one lookup table rather than a hack.

Three things it does that a backtesting *library* structurally cannot:

- **Look-ahead bias is a compile error.** A strategy whose decisions
  depend on future data fails to compile, and `allie check --audit`
  confirms it by experiment. See below.
- **Multiple-comparisons correction is built in.** `bonferroni`,
  `benjamini_hochberg`, and a seeded `permutation_pvalue` are stdlib
  functions, so correcting a window sweep is a one-liner instead of an
  intention.
- **`--verify` cross-checks three independent backends** and refuses to
  report a result they disagree on.

```allie
let AAPL = load("tests/fixtures/AAPL_sample.csv");

func spread_bps(fast_avg: Float, slow_avg: Float) -> Float {
    return (fast_avg - slow_avg) / slow_avg * 10000.0;
}

// Ordinary control flow driving the quant stdlib.
let slow: Series[Float] = EMA(AAPL.close, 20);
for let w: Int = 4; w <= 16; w = w + 4 {
    let fast: Series[Float] = EMA(AAPL.close, w);
    if w == 12 { continue; }
    print(spread_bps(mean(fast), mean(slow)));
}

// The quant DSL: crossing signals drive a bar-by-bar backtest.
let fast_signal: Series[Float] = SMA(AAPL.close, 5);
let slow_signal: Series[Float] = SMA(AAPL.close, 20);

strategy Crossover {
    when fast_signal crosses_above slow_signal: buy();
    when fast_signal crosses_below slow_signal: sell();
}

backtest Crossover on AAPL from 0 to 100;
```

```
$ allie run examples/blended.qs
4
64.53966914603127
8
47.55619501084162
16
13.521536279216305
========================================
            BACKTEST RESULTS
========================================
Strategy                       Crossover
Symbol                              AAPL
Period          2024-01-02 to 2024-03-25
Bars                                  60
Fills                      next bar open
Slippage                         5.0 bps
Commission                       0.0 bps
----------------------------------------
Initial Equity                 10,000.00
Final Equity                    9,920.38
Total Return                      -0.80%
Buy & Hold                         1.25%
CAGR                              -3.36%
Volatility                        13.15%
Sharpe                             -0.19
Sortino                            -0.27
Max Drawdown                       6.75%
Exposure                          31.67%
----------------------------------------
Trades                                 1
Win Rate                           0.00%
Profit Factor                       0.00
Avg Trade                         -0.80%
Costs Paid                          9.96
Position at End                     flat
========================================
```

## Install

Requires Python 3.11+. [`uv`](https://docs.astral.sh/uv/) handles the
rest.

```bash
git clone <this-repo> && cd Lang
make install          # uv sync --group dev
make test             # 1234 tests
```

## Running programs

```bash
allie run examples/blended.qs                    # tree-walking interpreter
allie run examples/blended.qs --backend ir       # IR interpreter
python main.py examples/blended.qs               # LLVM + MCJIT

# Run on all three and refuse to report unless they agree
allie run examples/blended.qs --verify

allie check examples/*.qs                        # every static check, nothing run
allie check --audit examples/cheat_demo.qs       # ...plus look-ahead, by experiment
allie repl                                       # an interactive session
echo 'print(6 * 7);' | allie run -               # a program from stdin

# Seeing inside (independent, combinable)
allie run examples/blended.qs --trace --time
python main.py examples/blended.qs --debug-tokens --debug-ast --debug-ir --no-optimize
```

A runtime error points at the expression that failed, the same way on
every backend:

```
AllieValueError: modulo by zero. Line 3, Column 9
  print(w % z);
          ^
```

Exit codes: 0 success, 1 the program has an error, 2 a usage error, 3 the
backends disagreed under `--verify`, 70 an internal compiler error.

There are **three independent execution backends**, and they're
cross-validated against each other on every test program — same source,
same output, or the build fails. `--verify` exposes that as a property
of *your* run: a number three separately-written evaluators agree on is
much harder to get wrong than one a single implementation produced.

## Strategies are expressions

Operators work on whole series, so an indicator is a line and a signal
is any `Series[Bool]` — combined with `and`/`or`/`!`, crossing another
series or a level. `backtest` evaluates to a result whose fields feed
ordinary code:

```allie
let macd: Series[Float] = EMA(c, 12) - EMA(c, 26);
let rsi: Series[Float] = RSI(c, 14);

strategy Trend {
    when macd crosses_above EMA(macd, 9) and rsi < 70.0: buy();
    when rsi crosses_above 80.0 or c < SMA(c, 20) - 2.0 * rolling_std(c, 20): sell();
}

let train = backtest Trend on AAPL from 0 to split;          // bounds are Int expressions
let test = backtest Trend on AAPL from split to len(c);
print(test.sharpe);
print(sharpe(test.returns, 252) == test.sharpe);             // true: one definition
```

A strategy declared inside a loop captures that iteration's indicators,
so a parameter sweep is a `for` loop, and correcting it is `bonferroni`.
[`examples/strategies.qs`](examples/strategies.qs) runs all of it.

The engine errs against the strategy, not for it: orders fill at the
**next bar's open** (a signal computed from a close can't trade at that
close), every fill pays **5 bps of slippage unless the source says
otherwise**, and the results table states both. Long, short, `target(0.5)`
sizing, stop-loss/take-profit, and per-trade statistics are in
[`examples/execution.qs`](examples/execution.qs):

```allie
let r = backtest Trend on AAPL from 0 to len(c)
    with commission: 0.001, stop_loss: 0.02, take_profit: 0.05;
print(r.win_rate);
print(mean(r.trade_returns));
```

## Look-ahead bias is a compile error

`shift(s, -1)[i]` is `s[i+1]` — tomorrow's price, available today. Feed
that into a trading signal and the backtest reports Sharpe 3.31 / CAGR
+41.78%, against the honest strategy's −0.19 / −3.36%. Every number is
meaningless and nothing says so.

```allie
let tomorrow: Series[Float] = shift(AAPL.close, -1);
strategy Clairvoyant {
    when tomorrow crosses_above slow: buy();
}
```

```
$ allie check examples/cheat_demo.qs
error: look-ahead bias: this 'when' condition in strategy 'Clairvoyant' may
contain data from a future bar. A trading decision cannot depend on data the
strategy would not have had at that bar. 'tomorrow' is derived from a
backwards time shift (shift(series, k) with a negative k places bar i+k at
bar i). (8:10)
  help: a positive shift looks back: shift(x, 1) is the previous bar's value
note: 'tomorrow' is given data from a future bar here (5:1)
note: the future enters here: shift by -1 gives each bar the value from 1 bar later (5:31)
```

A taint analysis (`allie/semantic/lookahead.py`) follows future data
from where it enters to wherever a trading decision is made. The
sources are a backward shift, a whole-series statistic, a single bar,
the number of bars, a backtest's results, and state an earlier
backtest's actions left behind. Along the way it passes through
variables, branches and loops, and through user functions, each analyzed
where it's called. The decisions checked are `when` conditions,
conditions inside actions, order sizes, backtest options, and whether a
backtest runs at all. It follows control flow, so a value that a branch
on future data chose is caught too. Future data used for post-hoc
analysis stays legal: the bias is deciding on it, not touching it.

The subtler leak is a statistic of the whole series. `when zscore(c) <
-2.0` measures each bar against the mean of *every* bar, later ones
included, and `when c > mean(c)` does the same with an average. Every
builtin is classified as trailing or whole-history (a test fails if a new
one isn't), so both are compile errors that name the trailing fix —
`rolling_zscore(c, 20)`, `SMA(c, 20)`.

**And it's checked by experiment.** `allie check --audit` re-runs the
program once for every bar, with the data cut off after that bar, and
reports any decision that changes. That is the definition of look-ahead,
measured rather than proven:

```
look-ahead audit: FAILED -- 3 decisions changed when later bars were removed
  backtest 1 ('Clairvoyant'), bar 23 -- 'when' clause 2: true with all 60 bars, false with only the first 24
  backtest 1 ('Clairvoyant'), bar 37 -- 'when' clause 1: true with all 60 bars, false with only the first 38
  backtest 1 ('Clairvoyant'), bar 37 -- orders placed: target 1 with all 60 bars, none with only the first 38
```

The test suite holds the analysis to that definition. Across hundreds of
generated programs, every one the analysis accepts must pass the audit.
For each kind of source and decision the analysis rejects, there is a
program in which it changes a trade.

## A quant standard library that handles real data

One rule for missing values everywhere: a `NaN` bar is skipped by
reductions, local to the windows that hold it, and stepped over by
recurrences — so `mean(SMA(c, 20))` is a number, not `NaN`, and one gap no
longer erases an `SMA`. `load` reads what real exports contain (empty and
`NA` cells, byte-order marks, newest-first files) and rejects, by line,
what it can't interpret. The library covers rolling statistics, RSI/ATR,
returns and risk (volatility, Sortino, VaR, expected shortfall, drawdown,
beta), math on numbers and series alike, and matrices built from series
with least squares — checked against pandas and numpy in the tests.
[`examples/risk.qs`](examples/risk.qs) is a risk report, a regression and
a train/test split in 50 lines.

## Architecture

```
 source (.qs)
     │
     ▼  allie/lexer          tokens, with slang aliases resolved here and nowhere else
  Lexer
     │
     ▼  allie/parser         recursive descent → dataclass AST
  Parser ──────────────────► allie/ast
     │
     ▼  allie/semantic       scope stack; break/continue/return/when placement
  Resolver
     │
     ▼  allie/typechecker    static types, local inference; annotates every expression
  Type checker
     │
     ▼  allie/ir             flat instruction list; `for` desugars to `while` here
  IR lowering
     │
     ▼  allie/optimizer      constant folding, DCE, CSE (toggleable)
  Optimizer
     │
     ├─────────────► allie/runtime/interpreter   (tree-walking, off the AST)
     ├─────────────► allie/ir/interpreter        (IR interpreter)
     └─────────────► allie/codegen               (LLVM IR → MCJIT)
                         │
                         ▼  allie/codegen/runtime_bridge
                     Python/numpy for Series, Matrix, and all stdlib calls
```

## Documentation

| doc | what's in it |
|---|---|
| [`docs/how-it-works.md`](docs/how-it-works.md) | **start here** — a program traced through every stage, the interesting design calls, benchmarks, known limits |
| [`docs/language-reference.md`](docs/language-reference.md) | every type, assignability table, every stdlib function, every keyword |
| [`docs/grammar.md`](docs/grammar.md) | the full EBNF |
| [`docs/architecture.md`](docs/architecture.md) | per-stage design decisions, including what each stage deliberately does *not* do |
| [`docs/keywords.md`](docs/keywords.md) | generated from `KEYWORD_ALIASES` — never hand-edited |
| [`benchmarks/README.md`](benchmarks/README.md) | methodology and an honest reading of the numbers |

## Benchmarks

```bash
make bench          # regenerates benchmarks/results.{md,csv,json}
make bench-quick    # 3-trial smoke run, writes nothing
```

Every implementation is checked to compute the same answer before it's
timed, every ratio carries a 95% confidence interval, and the headline
is computed from the measurements. What the committed run
([`benchmarks/results.md`](benchmarks/results.md), Apple M5) shows:

- **Native code is fast; compiling it isn't free.** A JIT-compiled loop
  iteration takes ~0.4 ns against CPython's ~30 ns. But LLVM spends
  1–4 ms generating code, so on short programs the JIT's *total* time
  is several times CPython's, and `-O2` only pays for itself after
  about a million loop iterations.
- **On quant workloads Allie is 2–5× slower than pandas**, mostly in
  loading the CSV (pandas' parser is C): both sides do the same numpy
  work after that. The case there is the LOC columns and what Allie
  checks, not speed.

[`benchmarks/README.md`](benchmarks/README.md) has the methodology, and
what earlier versions of the harness got wrong.

## Development

```bash
make check            # lint + typecheck + tests with coverage floor
make lint             # ruff
make typecheck        # mypy
make cov              # pytest with the 85% floor enforced
make regen-fixtures   # regenerate integration expected-output files
make regen-golden     # regenerate IR/LLVM golden snapshots
```

CI runs lint, type check, and the full suite with the coverage floor on
Python 3.11 and 3.12.

## License

MIT — see [`LICENSE`](LICENSE).
