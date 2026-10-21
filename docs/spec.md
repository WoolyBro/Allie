# Allie — Design Decisions (Phase 0)

This document is the north star for every phase after this one. If a later
phase wants to contradict something here, that's a deliberate revision —
update this file in the same commit, don't let code and doc drift apart.

## Resolved decisions

| Decision | Choice | Why |
|---|---|---|
| Execution model | Tree-walking interpreter (v1) | Fastest path to a correct end-to-end system. Bytecode VM and transpile-to-Python/numpy are documented alternatives (see below), not ruled out — but rewriting an interpreter's *semantics* later is far more expensive than rewriting its *execution strategy*, so this is the safe order. |
| Typing | Static, with local inference (`Series[Float]`) | `let` annotations optional (inferred), function signatures required; no dynamic type, not Hindley-Milner. Catches `Series[Float] - String` before a 10-year backtest runs. (Originally planned as gradual/dynamic-by-default; shipped as static so that every program that type-checks runs identically on all three backends.) |
| Numeric precision | `float64` for prices/returns/indicators; a distinct `Decimal`-backed `Money` type for the accounting ledger (cash, P&L) | Speed matters for indicator math and relative error is tolerable there. Compounding float error across thousands of trades in the ledger is a real correctness bug, not a rounding curiosity. |
| Pandas dependency | ~~Yes (`numpy` + `pandas`)~~ **Revised: numpy only** | Originally: don't reimplement `DatetimeIndex` / time-series alignment, wrap a `pandas.Series`. As built, a series is a numpy float64 array (`allie/runtime/values`), and the stdlib applies its own missing-bar rule. pandas is a dev dependency only: an oracle the tests check the stdlib against, and the benchmarks' hand-written baseline. |
| Syntax style | **Brace-based**, not indentation-significant | An indentation-based lexer needs stateful INDENT/DEDENT tracking on top of everything else the lexer already has to do (numeric literals, string interpolation, spans). Brace-based is strictly less lexer complexity for the same v1 deadline. Python-like indentation is a plausible v2 ergonomics improvement, not a v1 requirement — revisit once the grammar is stable. |
| Diagnostics strategy | Collect-and-report-all | See `allie/errors.py`. Harder to implement (every phase must record-and-recover instead of raise-and-die) but much better for a user iterating on a strategy file. |
| Backtest loop architecture | Event-driven (`on_bar`/`on_tick` callbacks, stateful) | Vectorized (apply-across-array) backtesting is fast but makes it easy to accidentally leak future data into a signal (look-ahead bias). Event-driven is the only model that structurally matches "what did the strategy know at time t." This is the actual "quant emphasis" of the language, not a detail. |
| Transaction costs / slippage | First-class language construct (`slippage_model: ...` inside a backtest declaration), not an opt-in stdlib call | If it's opt-in, people forget it, and backtests look unrealistically good — which directly undermines the "reliable under rigorous conditions" thesis of the paper this language exists to support. |
| Time & calendar handling | Structural point-in-time support: every `Series` carries a `DatetimeIndex`; the backtest engine only ever exposes data up to (and including) the current bar to `on_bar` | Look-ahead bias is the single biggest source of silent backtest bugs. Making "you cannot see the future" a property of the execution model (not a discipline the strategy author has to maintain) is the whole point. Trading-calendar/holiday awareness (`pandas_market_calendars` or similar) is planned as a stdlib dependency, not built from scratch. |
| Determinism | All stochastic stdlib functions (bootstrap, Monte Carlo, permutation tests) take an explicit seed; no hidden global RNG state | Reruns must be reproducible — this matters for the paper's credibility as much as for debugging. |
| Concurrency model | Sequential, single-strategy execution for v1. Each run gets a fresh `Interpreter` instance (isolated state) | Parameter sweeps / walk-forward windows can be parallelized *outside* the language (e.g. a process pool driving independent runs) without the interpreter itself needing to be thread-safe. Revisit only if sweep wall-clock time becomes a real bottleneck. |
| Module / import system | File-path-based imports for v1 (`import "path/to/file.qs"`), resolved relative to the importing file; `allie.stdlib` resolved by name | A package registry / versioned dependency system is out of scope for v1 — see Deferred. |
| Runtime error philosophy | Exceptions halt the running strategy (`allie.runtime.errors.AllieRuntimeError` hierarchy), distinct from `allie.errors.Diagnostic` | Simple, matches "this backtest run is now invalid, stop" semantics. Result-style propagation for partial-failure strategies is deferred. |
| Statistical rigor | Multiple-comparisons correction is a *built-in* stdlib evaluation function (`allie.stdlib.significance`), not something the paper's author has to remember to apply by hand | The project tests 5 indicators (MA, MACD, RSI, Bollinger, EMA). Reporting the best Sharpe ratio across 5 strategies without correction (Bonferroni, or a permutation/Monte Carlo null) is p-hacking by construction. If this isn't structural, the paper's central claim is methodologically weak regardless of interpreter quality. |
| Benchmarking | `benchmarks/` compares a Allie backtest against an equivalent hand-written pandas backtest | Gives the paper a concrete, falsifiable performance claim instead of a vibes-based one. |

## Considered alternatives (not chosen for v1, not ruled out forever)

- **Bytecode VM.** Better performance for tight backtesting loops than a
  tree-walking interpreter. Deferred until language semantics (typing,
  error handling, module system) are settled — compiling a moving target
  to bytecode is wasted work.
- **Transpile to Python/numpy.** Backtest loops become C-level numpy/pandas
  operations for free instead of a hand-rolled numeric runtime. The
  trade-off: Allie's semantics get pulled toward Python's, which fights
  the "the language structurally prevents look-ahead bias" goal above.
  Worth a paragraph in the paper's "design decisions considered" section
  even though it isn't v1.

## Scope cut line

This is a solo project with 13 planned phases and a fixed deadline. Written
down now so it's a decision, not something discovered under pressure in
week 6:

**Ship for v1:**
- Tree-walking interpreter, brace-based syntax, static typing with local inference
- The 5 target indicators (MA, MACD, RSI, Bollinger, EMA)
- Event-driven backtest loop with a first-class slippage/cost model
- Walk-forward / out-of-sample split tooling
- Multiple-comparisons correction (Bonferroni + permutation null) in `allie.stdlib.significance`
- CLI with `--dump-tokens` / `--dump-ast` / `--trace`
- One benchmark: Allie vs. hand-written pandas equivalent

**Defer (note as future work in the paper):**
- Bytecode VM / transpile-to-Python execution backends
- Live trading / broker integration
- LSP / editor tooling
- Package registry / versioned module system
- Parallel parameter-sweep execution inside the language itself
- Result-style (non-exception) runtime error propagation

## Open for Phase 2+

Grammar details (operator precedence, the exact `backtest { ... }` block
syntax, string interpolation) belong in `docs/grammar.ebnf` once the lexer
work starts, not here.
