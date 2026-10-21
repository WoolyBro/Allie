# 60-second demo script

Keyed to a live terminal. Judges remember seeing real IR on screen far
better than they remember slides — so the whole point is that everything
below is *run*, not described.

**Setup before you start:** terminal at repo root, font large enough to
read from a few feet, `make install` already done. Have these two
commands in shell history so you're not typing during the demo:

```bash
allie run examples/blended.qs
python main.py examples/blended.qs --debug-ir
```

---

## 0:00–0:10 — What it is

> "This is Allie — a programming language I built. It's statically
> typed, it compiles through my own intermediate representation down to
> LLVM, and it JIT-compiles to machine code. What makes it a bit unusual
> is that it's really two languages in one: a normal general-purpose
> language, and a quantitative-finance DSL."

## 0:10–0:25 — Show it running

Run: `allie run examples/blended.qs`

> "Same program, both halves. Up here" — *point at the loop output* —
> "that's an ordinary `for` loop calling a function I wrote. And down
> here" — *point at the results table* — "that's a `strategy` block with
> `crosses_above` signals, running a real bar-by-bar backtest with
> Sharpe and max drawdown. `strategy`, `when`, and `backtest` are
> keywords in the language, not a library."

## 0:25–0:45 — The actual compiler (the important part)

Run: `python main.py examples/blended.qs --debug-ir`

Scroll to the IR section.

> "This is my IR. Flat instruction list, virtual registers." — *point at
> a `Label`/`JumpIfFalse`/`Jump` cluster* —
>
> "Notice there's no `for` instruction. `for` loops desugar into the
> same labels-and-jumps shape a `while` produces, before code generation
> ever sees them. Which meant when I wrote the LLVM backend, I needed
> **zero** extra code for `for`, `break`, or `continue`."

Scroll to the LLVM section.

> "And that's real LLVM IR — actual basic blocks, `icmp`, `br`. This
> gets JIT-compiled and executed. The `call_builtin_bridge` line is the
> one place it calls back into Python, for the numpy-backed series
> functions."

## 0:45–1:00 — The part nobody else has, then close

Run: `allie check --audit examples/cheat_demo.qs` (a strategy using
`shift(close, -1)`).

> "This strategy peeks one bar into the future — `shift(close, -1)` is
> tomorrow's price, available today. Before I added a check for it, this
> compiled fine and backtested at Sharpe 3.3 while the honest version
> scored minus 0.16. Nothing anywhere said the number was garbage. Now
> it's a compile error: there's a taint analysis that tracks
> future-derived data through variables and indicators and refuses to
> let it reach a trading decision. That's the most common silent bug in
> backtesting, and it's a type error here. And it's not just a claim:
> `--audit` re-runs the program with the data cut off after every bar,
> and shows the exact bar where a decision changes. At bar 37 it buys
> with all 60 bars and doesn't with only the first 38. The test suite
> runs that experiment on hundreds of generated programs, and every one
> the compiler accepts passes it."

Optional, if time: `allie run examples/rigor.qs`

> "And this corrects a window sweep for multiple comparisons — window 10
> looks significant at p=0.02 until you admit you tried five windows,
> then it's 0.11 and it isn't."

## If you have longer — the best bug

> "The thing I'd point to as most interesting: my dead-code elimination
> pass had a real bug. It removed the loop-increment variable, because
> scanning forward, nothing *after* the increment reads the counter —
> the read happens at the top of the loop, reached through the
> back-edge. So the 'optimization' turned every loop into an infinite
> loop. Program order and execution order aren't the same thing once you
> have loops.
>
> Everything runs on three independent backends — tree-walking
> interpreter, IR interpreter, and the JIT — and they're cross-validated
> against each other on every test. 1237 tests, 97% coverage."

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
> "Replace the flat IR with a real control-flow graph. Both optimizer
> bugs I hit came from passes reasoning about text order instead of
> control flow — a CFG would make them both safer and more aggressive."

**"How long did this take / did you use AI?"**
> Answer honestly. The commit history is incremental and phase-tagged
> (`git log --oneline`, `git tag`), which is worth showing if asked.

---

## Fallbacks

- **Laptop won't run it:** `benchmarks/results.md` and
  `tests/golden/snapshots/for_loop.llvm.txt` are committed — the LLVM IR
  is readable straight from the repo.
- **Running long:** cut the questions, keep 0:25–0:45. The IR is the
  part that distinguishes this from a library.
- **Running short:** `allie run examples/blended.qs --backend ir` and
  `python main.py examples/blended.qs` produce byte-identical output —
  a fast way to show the three-backend cross-validation is real.
