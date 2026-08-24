# Benchmarks

Compares Allie against hand-written Python/numpy/pandas across three
workload families, and measures what the optimizer passes are actually
worth (Phase 8.2; `docs/spec.md` "Benchmarking").

```bash
make bench          # full run -> results.md + results.csv
make bench-quick    # 2 trials, prints to stdout, writes nothing
```

`results.md` and `results.csv` are **generated** — regenerate them,
never hand-edit, so the committed numbers always correspond to one
clean run.

## Layout

| file | what it is |
|---|---|
| `workloads.py` | each workload defined twice — as Allie source and as equivalent Python |
| `run.py` | the harness: trials, timing, memory, report generation |
| `data/synthetic_10k.csv` | generated on demand from a fixed seed; not checked in |
| `results.md` / `results.csv` | generated output |

`tests/unit/test_benchmarks.py` runs every workload's front end and one
real single-trial measurement in CI, so a workload that stops compiling
after a language change fails immediately rather than the next time
someone regenerates numbers for the write-up.

## The three families, and why they're separate

- **quant** — SMA, rolling std, the `MovingAverage` backtest, and a
  matrix multiply. What the language is ostensibly for.
- **pure-CS** — a 50k-iteration `for` loop with no quant content at all.
  This is the workload that shows whether the compiler does anything for
  control-flow-heavy code. Without it, any "optimization improved
  performance by X%" claim would only ever be a statement about numpy
  call overhead.
- **blended** — a `for` loop sweeping SMA windows. The only workload
  exercising both halves of the language *together* rather than side by
  side.

## What the current numbers show

Read `results.md` for the measured table. The short version, and the
honest reading of it:

**The JIT is dramatically faster than the tree-walking interpreter on
control-flow-heavy code** — roughly two orders of magnitude on
`loop_sum` (~127x), and about 11x faster than the equivalent
hand-written Python. That is the strongest performance claim this project can make,
and the pure-CS workload is the only reason it's visible.

**On the quant workloads, Allie is slower than hand-written Python.**
Both sides bottom out in the same numpy/pandas calls, and Allie adds
front-end and bridge overhead on top. Compiling the surrounding code
cannot move a number already dominated by library time. This is stated
plainly rather than buried: it's the expected outcome for a DSL that
delegates its numerics, and the language's argument is expressiveness
(see the LOC columns — the backtest is 8 lines of Allie against 20 of
Python) rather than raw speed on library-bound work.

**The optimizer's effect is small — low single-digit percentages, near
measurement noise.** `tests/unit/test_optimizer.py` proves the
transformations fire; this table shows how little that's worth on *this*
mix of workloads, because the passes act on scalar IR while these
programs spend their time inside bridged library calls the passes
deliberately never touch. Reporting a ~2% number honestly is more useful
than picking a workload engineered to flatter it.

**`rolling_std` was an outlier, and the benchmark is what caught it.**
It used to run ~112× slower than pandas because the implementation was a
per-window Python loop (O(n·window)) where pandas is vectorized.
Rewriting it with `sliding_window_view` cut that to ~9× with
bit-identical results — no compiler work involved, just a stdlib fix the
measurement pointed straight at. The faster-still cumulative
sum-of-squares formula was rejected: it loses ~1e-08 accuracy to
catastrophic cancellation on exactly the large-mean, small-variance data
this function is for.

## Methodology

- **N trials (default 10), first discarded**, median of the rest. Warmup
  cuts both ways: the JIT pays LLVM setup on its first run, CPython pays
  import and allocator warmup. Median rather than mean so one scheduler
  hiccup doesn't dominate.
- **Compile and run timed separately.** Compile = lex + parse +
  **resolve** + type-check + lower + optimize. Run = execution only.
- **Peak memory** via `tracemalloc`. Stated as a limitation, not glossed:
  this counts Python allocations only and does **not** see memory the
  LLVM JIT allocates natively, so the JIT rows undercount real usage.
- **Deterministic inputs** — the 10k-row dataset is generated from a
  fixed seed (`docs/spec.md` "Determinism"), so runs are comparable.

## Known gap

`matmul_200` has **no Allie implementation**, and is reported as such
rather than dropped. Allie cannot construct a `Matrix` from source: array
literals produce `Series` (a nested one types as `Series<Series<Float>>`),
and every `Matrix` builtin requires a `Matrix` argument, so there is no
constructor to start from. Verified directly, not assumed. Adding matrix
literal syntax — or a `matrix(...)` builtin — would close this.
