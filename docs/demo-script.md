# 60-second demo script

Keyed to a live terminal. Judges remember seeing real IR on screen far
better than they remember slides, so the whole point is that everything
below is *run*, not described.

**Setup before you start:** terminal at the repo root, font large enough
to read from a few feet away, `make install` already done. Then:

```bash
make demo
```

Each beat runs when you press Enter, so there is nothing to type and
nothing to find in shell history. To rehearse straight through, run
`DEMO_PAUSE=0 make demo`. The test suite does exactly that
(`tests/integration/test_demo.py`), and it also checks every number this
script quotes, so a demo that has stopped working fails the build first.

---

## 0:00–0:10 — What it is

> "This is Allie — a programming language I built. It's statically
> typed, it compiles through my own intermediate representation down to
> LLVM, and it JIT-compiles to machine code. What makes it a bit unusual
> is that it's really two languages in one: a normal general-purpose
> language, and a quantitative-finance DSL."

## 0:10–0:25 — Show it running

*Beat 1:* `allie run examples/blended.qs`

> "Same program, both halves. Up here" — *point at the loop output* —
> "that's an ordinary `for` loop calling a function I wrote. And down
> here" — *point at the results table* — "that's a `strategy` block with
> `crosses_above` signals, running a real bar-by-bar backtest, with
> costs, filled at the next bar's open. `strategy`, `when`, and
> `backtest` are keywords in the language, not a library."

## 0:25–0:45 — The actual compiler (the important part)

*Beat 2:* `python main.py tests/golden/programs/for_loop.qs --debug-ir`,
a five-line `for` loop with a `continue`.

> "This is my IR. Flat instruction list, virtual registers." — *point at
> the `Label`/`JumpIfFalse`/`Jump` cluster* —
>
> "Notice there's no `for` instruction. `for` desugars into the same
> labels-and-jumps shape a `while` produces, before code generation ever
> sees it. So when I wrote the LLVM backend, I needed **zero** extra
> code for `for`, `break`, or `continue`."

Scroll to the end: the LLVM IR after LLVM's own `-O2`.

> "And this is the LLVM IR after optimization. The loop is gone. LLVM
> ran it at compile time, and what's left is `box_int(i64 8)`: the
> answer, 0 + 1 + 3 + 4, as a constant in the machine code. The
> `call_builtin_bridge` is how compiled code calls back into Python,
> here for `print`. The series functions go the same way, into numpy."

## 0:45–1:00 — The part nobody else has, then close

*Beat 3:* `allie check --audit examples/cheat_demo.qs`, a strategy using
`shift(close, -1)`.

> "This strategy peeks one bar into the future: `shift(close, -1)` is
> tomorrow's price, available today. If it were allowed to run, it would
> report a Sharpe of 3.31, where the same rule on today's close scores
> 1.14. Nothing would say the number was garbage. Here it's a compile
> error: a taint analysis follows future data through variables,
> branches and functions, and refuses to let it reach a trading
> decision. That's the most common silent bug in backtesting, and it's a
> compile error here. And it's not just a claim: `--audit` re-runs the
> program with the data cut off after every bar, and shows the exact bar
> where a decision changes. At bar 37 it buys with all 60 bars, and
> doesn't with only the first 38. The test suite runs that experiment on
> hundreds of generated programs, and every one the compiler accepts
> passes it."

*Beat 4, if there's time:* `allie run examples/rigor.qs`

> "And this asks whether a sweep's winner is real. The best of five
> windows has a Sharpe of 1.84, and a plain Sharpe test says 81% likely
> real. Deflated for being the best of five, it's 73%, and nothing
> survives the multiple-testing correction. On 60 bars, nothing should."

## If you have longer — the best bug

> "The thing I'd point to as most interesting: my dead-code elimination
> pass had a real bug. It removed the loop-increment variable, because
> scanning forward, nothing *after* the increment reads the counter —
> the read happens at the top of the loop, reached through the
> back-edge. So the 'optimization' turned every loop into an infinite
> loop. Program order and execution order aren't the same thing once you
> have loops, so the optimizer now works on a real control-flow graph.
>
> Everything runs on three independent backends — tree-walking
> interpreter, IR interpreter, and the JIT — and they're cross-validated
> against each other on every test: over 1,200 tests, 97% coverage, and
> a fuzzer that generates programs and runs them five ways."

---

## If they ask questions

**"Is it faster than Python?"**
> "Per loop iteration, yes — compiled code runs an iteration in about
> half a nanosecond against CPython's thirty. But LLVM takes a few
> milliseconds to compile, so for short programs Python still wins on
> total time, and for the quant workloads both sides end up in the same
> numpy calls. Every number in the benchmark report has a confidence
> interval, and the first version of my harness was wrong — it timed
> Python with memory tracing on, which made the JIT look 11× faster than
> it is. I found that and fixed it."

**"Why the weird keywords?"** *(`ts`, `Ahhh`, `grindset`)*
> "Mostly for fun. But the implementation is the interesting part: it's
> one lookup table with two columns, so the lexer maps both spellings to
> the same token and nothing downstream can tell which you wrote. Adding
> one is a single line and it works through the whole compiler."

**"What would you do next?"**
> "The quant side, where it's thinnest. Several symbols at once, with
> portfolio accounting. Dates in the language, so a walk-forward test can
> be written. And letting the look-ahead check see bar ranges, so
> choosing a parameter on a training window and trading it on a test
> window stops being rejected."

**"How long did this take / did you use AI?"**
> Answer honestly. The early history is incremental and phase-tagged
> (`git log --oneline`, `git tag`). The later review-and-hardening
> commits were made with an AI assistant, and their trailers say so.

---

## Fallbacks

- **Laptop won't run it:** `tests/golden/snapshots/for_loop.ir.txt` and
  `for_loop.llvm.txt` are committed, from the same program beat 2 runs,
  so the IR is readable straight from the repo. `benchmarks/results.md`
  has the numbers.
- **Running long:** cut the questions, keep 0:25–0:45. The IR is the
  part that distinguishes this from a library.
- **Running short:** `allie run examples/blended.qs --verify` runs it on
  all three backends and prints `[verified: tree, ir, llvm agree]` — a
  fast way to show the three-backend cross-validation is real.
