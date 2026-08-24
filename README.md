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
==================================
         BACKTEST RESULTS
==================================
Strategy                 Crossover
Symbol                        AAPL
Bars                            60
Initial Equity           10,000.00
Final Equity              9,930.31
CAGR                        -2.89%
Volatility                  13.15%
Sharpe                       -0.16
Max Drawdown                 6.71%
Trades                           1
==================================
```

## Install

Requires Python 3.11+. [`uv`](https://docs.astral.sh/uv/) handles the
rest.

```bash
git clone <this-repo> && cd Lang
make install          # uv sync --group dev
make test             # 531 tests
```

## Running programs

```bash
allie run examples/blended.qs                    # tree-walking interpreter
allie run examples/blended.qs --backend ir       # IR interpreter
python main.py examples/blended.qs               # LLVM + MCJIT

# Debug flags (independent, combinable)
python main.py examples/blended.qs --debug-tokens --debug-ast --debug-ir --no-optimize
```

There are **three independent execution backends**, and they're
cross-validated against each other on every test program — same source,
same output, or the build fails.

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
     ▼  allie/typechecker    gradual typing; annotates every expression node
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
make bench          # regenerates benchmarks/results.{md,csv}
make bench-quick    # 2-trial smoke run
```

Headline: **the JIT runs loop-heavy code ~127× faster than the
tree-walking interpreter, and ~11× faster than equivalent hand-written
Python.** On quant workloads Allie is *slower* than Python — both sides
bottom out in the same numpy calls, and Allie adds front-end overhead on
top. That trade-off, and why the LOC columns are the more interesting
comparison there, is discussed in
[`benchmarks/README.md`](benchmarks/README.md).

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
