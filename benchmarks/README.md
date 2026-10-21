# Benchmarks

Allie against hand-written Python/numpy/pandas, across five execution
paths and three workload families — with every result checked for
correctness first, every ratio given a confidence interval, and every
claim in [`results.md`](results.md) computed from the measurements
rather than written by hand.

```bash
make bench                                   # 10 trials: results.md, .csv, .json
make bench-quick                             # 3 trials, prints, writes nothing
python -m benchmarks.run --only fib_20,loop_50k
python -m benchmarks.run --out /tmp/new --compare benchmarks/results.json
```

`results.*` are generated — regenerate, never hand-edit. `results.json`
holds every raw sample and the machine they came from, so a later run on
the **same machine** can be compared against it: `--compare` gives each
workload and implementation a new/old ratio with a 95% interval, flags a
regression when the whole interval is more than 5% slower, and exits 1
if there is one. Comparing runs from different machines measures the
machines.

## What is measured

| key | implementation |
|---|---|
| `python` | the hand-written Python/numpy/pandas twin |
| `tree` | Allie's tree-walking interpreter |
| `ir` | Allie's IR interpreter, on optimized IR |
| `jit_O0` | LLVM + MCJIT with no optimization at all (`allie run --no-optimize`) |
| `jit_O2` | LLVM + MCJIT as `allie run --backend llvm` runs it: Allie's passes and LLVM `-O2` |

| workload | family | |
|---|---|---|
| `load_10k` | quant | reading the 10k-row OHLCV file, alone |
| `sma_10k`, `rolling_std_10k` | quant | a rolling statistic over it, repeatedly |
| `backtest_10k` | quant | a moving-average crossover backtest over 10k bars |
| `matmul_200` | quant | two random 200×200 matrices, their product, its norm |
| `loop_50k` | pure-cs | an integer loop, no quant content |
| `fib_20` | pure-cs | naive recursion: ~22k calls |
| `blended_sweep` | blended | a `for` loop sweeping SMA windows, statistics per window |

Plus a **scaling study**: `loop_50k`'s loop at three sizes per
implementation, fitted to *time = fixed + per-iteration × n*.

## Methodology

- **Correctness before speed.** Every workload ends by printing one
  number and its Python twin returns the same one. The harness checks
  every implementation against the twin (to 1e-9 relative) before any
  timing counts, and a mismatch fails the workload.
  `tests/unit/test_benchmarks.py` runs that check for every workload.
- **No instrumentation while timing.** Peak memory comes from a separate,
  untimed run under `tracemalloc`. Earlier versions of this harness
  measured both at once, and tracing every allocation slowed
  allocation-heavy Python several-fold. That's where the old "the JIT is
  11× faster than Python on a loop" came from: CPython was timed at 72 ms
  for a loop that takes about 1.3 ms.
- **Interleaved, warmed-up trials.** Each trial runs every implementation
  once, in an order rotated per trial, so drift falls on all of them.
  Round 0 is a warm-up and is discarded; garbage collection runs before
  and is off during each timed run, as `timeit` does.
- **Phases, not one number.** Front end (lex, parse, resolve, type-check,
  look-ahead), lowering (IR, passes, LLVM IR), code generation (LLVM
  parsing, optimizing and emitting machine code) and the native run are
  timed separately. Code generation happens at run time, but it is
  compilation, and the old harness counted it as running.
- **Uncertainty, stated.** Times are medians ± half the interquartile
  range. Ratios carry a seeded bootstrap 95% confidence interval on the
  ratio of medians (`stats.py`), and an interval that contains 1 is
  reported as *within noise*, not as a 0.4% gain.
- **No cherry-picking.** The headline covers every workload, by family,
  as geometric means — not the best one.
- **Context.** The report records the CPU, OS, Python, numpy, pandas,
  llvmlite and LLVM versions and the commit. Inputs are generated from a
  fixed seed (`docs/spec.md` "Determinism").

## What the committed numbers show

From the run in [`results.md`](results.md) (Apple M5, CPython 3.14), rounded to
what holds across runs on that machine:

**Natively compiled code is fast, and compiling it costs more than these
programs take to run.** The JIT runs a loop iteration in about 0.4 ns,
against CPython's ~30 ns and the tree interpreter's ~5 µs. But LLVM
spends ~1 ms generating code at `-O0` and ~3 ms at `-O2`, so on
`loop_50k` and `fib_20` the JIT's *total* is several times CPython's.
Counting only the native run, it's several times faster. Both are true,
and the report gives both.

**`-O2` pays off only for long-running code.** It costs ~2 ms more to
compile and saves ~1.9 ns per loop iteration, so it breaks even around a
million iterations (the fitted estimate moves between 1.0 and 1.5
million from run to run). Every workload here is shorter, and every one is
slower overall with optimization on — by a confidence interval that
excludes 1. A JIT that picked its optimization level per program would
win both ways; this one doesn't yet.

**On quant workloads the JIT can't help, and doesn't.** Both sides spend
their time in the same numpy code, so the native run equals the
interpreters' and code generation is pure overhead. The remaining gap to
pandas is mostly the loader: `load_10k` alone is ~11 ms for Allie against
~3.5 ms for pandas, whose CSV parser is C. Allie's case on this family
is the LOC columns — the backtest is 10 lines against 18 — and what the
language checks that pandas doesn't (look-ahead, missing data, costs).

**The interpreters cost ~25–30× CPython per operation.** ~800 ns per
loop iteration in the tree interpreter and ~900 ns in the IR interpreter,
against CPython's ~30 ns, on code CPython itself runs: each Allie
operation is several Python calls. They're the reference and the
cross-check, not the fast path.

## Findings that changed the code

The benchmark is a profiler with a baseline, and it has pointed at real
fixes more than once:

- **`rolling_std`** was a per-window Python loop, 112× slower than
  pandas. `sliding_window_view` made it vectorized and bit-identical. The
  faster running-sum formula was rejected for losing ~1e-08 accuracy to
  cancellation on price-like data.
- **`load`** parsed every cell with its own `float()` call. Parsing each
  column in one numpy call, and validating all dates with one regular
  expression, took it from 16.5 ms to ~10.7 ms on 10k rows. The per-cell
  path is kept, to name the line and column of a bad cell.
- **The backtest engine** indexed numpy arrays bar by bar, boxing a numpy
  scalar each time. Plain lists cut `backtest_10k` from 23 ms to 14 ms.
- **Both interpreters** re-inspected every node or instruction with an
  `isinstance` chain each time it ran. Compiling once to closures — with
  arithmetic specialized by type, control flow as return values, and
  constants preloaded — took the tree interpreter from ~4.9 µs to ~0.8 µs
  per loop iteration and the IR interpreter from ~3.6 µs to ~0.9 µs.

## Layout

| file | |
|---|---|
| `workloads.py` | each workload as Allie source and as its Python twin; the scaling study |
| `run.py` | the harness: verification, timing, memory, the report, `--compare` |
| `stats.py` | medians, interquartile ranges, bootstrap intervals, linear fits |
| `data/synthetic_10k.csv` | generated on demand from a fixed seed; not checked in |
| `results.md` / `.csv` / `.json` | generated output |
