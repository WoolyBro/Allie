# How Allie actually works

Written for someone curious about how the thing is built, not as a
defense of it. If you want the reference material instead, see
[`language-reference.md`](language-reference.md) and
[`grammar.md`](grammar.md).

---

## 1. What it is

Allie is a statically typed language that compiles through a custom IR
to LLVM and JIT-executes. Two things got merged into one language:

- **A general-purpose core** — functions, `for`/`while`,
  `break`/`continue`, recursion, `Int`/`Float`/`Bool`/`String`.
- **A quant DSL** — `Series[Float]`, a numpy-backed indicator stdlib
  (`SMA`, `EMA`, `rolling_std`, `zscore`, ...), and first-class
  `strategy`/`when`/`backtest` constructs with a real bar-by-bar
  backtest engine behind them.

Plus a second spelling for every control-flow keyword — `ts` for `let`,
`Ahhh` for `while`, `grindset` for `for`, `yeet` for `return`. That part
is a joke. The implementation isn't: it's one extra column in a lookup
table, which is the whole point (§3.1).

There are three execution backends — a tree-walking interpreter, an IR
interpreter, and LLVM/MCJIT — and every test program runs through all
three with output compared. Three independent implementations agreeing
is the strongest correctness evidence in the project.

---

## 2. How a program actually runs

Take this program:

```allie
let total: Int = 0;
for let i: Int = 0; i < 4; i = i + 1 {
    if i == 2 { continue; }
    total = total + i * 2;
}
print(total);
```

Answer: `0 + 2 + 6 = 8` (i=2 skipped). Here's what each stage does to it.

### Lex → tokens

`allie/lexer/`. Characters to tokens, each carrying line and column.
This is the *only* place slang aliases exist: `ts` and `let` both become
`TokenType.LET` right here, and nothing downstream can tell which was
written.

### Parse → AST

`allie/parser/parser.py`, recursive descent, one method per grammar
production. Produces dataclass nodes (`allie/ast/nodes.py`), each with a
`SourceSpan`. Parse errors accumulate in a `DiagnosticBag` and the
parser resynchronizes, so one bad line doesn't hide the next five.

### Resolve → scope and placement checks

`allie/semantic/resolver.py`. Walks a scope stack checking that names
exist and that jumps are legal: `break`/`continue` need an enclosing
loop, `return` needs a function, `when` needs a strategy. It tracks
`loop_depth` and resets it at boundaries a jump can't cross — a function
body, a `when` action. (That reset was missing until Phase 7; see §3.4.)

### Type-check → annotate every expression

`allie/typechecker/checker.py`. Static typing with local inference: annotations are
optional, but where present they're enforced. `Int` widens to `Float`;
nothing narrows; a scalar is never assignable to a `Series`. Every
expression node gets its resolved type written back onto it — which is
what makes LLVM codegen possible later, since it needs to know whether
`+` is an `add` or an `fadd`.

### Look-ahead → reject decisions that could see the future

`allie/semantic/lookahead.py`. A taint analysis over the typed AST.
Sources give a value a taint: backward shifts, whole-series statistics,
single bars, `len`, backtest statistics. The taint follows variables,
branches, loops and calls. A decision that may carry it is a compile
error: a `when` condition, a condition inside an action, an order size,
a backtest option. The error points at the culprit, with notes that
trace it back to the source. The pass runs last because it reads the
types the checker wrote: `c[3]` is a source on a series and on nothing
else. §3.5 covers how it's tested.

### Lower → IR

`allie/ir/lowering.py`. A flat list of instructions over virtual
registers. Here's the real output (`--debug-ir --no-optimize`):

```
LoadConst(dest='r0', value=0)
StoreVar(name='total', src='r0')
LoadConst(dest='r1', value=0)
StoreVar(name='i', src='r1')
Label(name='for_loop_0')
LoadVar(dest='r2', name='i')
LoadConst(dest='r3', value=4)
BinOp(dest='r4', op='<', left='r2', right='r3')
JumpIfFalse(cond='r4', label='for_end_2')
LoadVar(dest='r5', name='i')
LoadConst(dest='r6', value=2)
BinOp(dest='r7', op='==', left='r5', right='r6')
JumpIfFalse(cond='r7', label='if_end_4')
Jump(label='for_increment_1')          <- `continue` jumps here
Label(name='if_end_4')
LoadVar(dest='r8', name='total')
LoadVar(dest='r9', name='i')
LoadConst(dest='r10', value=2)
BinOp(dest='r11', op='*', left='r9', right='r10')
BinOp(dest='r12', op='+', left='r8', right='r11')
StoreVar(name='total', src='r12')
Label(name='for_increment_1')          <- ...so the increment still runs
LoadVar(dest='r13', name='i')
LoadConst(dest='r14', value=1)
BinOp(dest='r15', op='+', left='r13', right='r14')
StoreVar(name='i', src='r15')
Jump(label='for_loop_0')
Label(name='for_end_2')
LoadVar(dest='r16', name='total')
Call(dest='r17', fn_name='print', args=['r16'])
```

Notice there is no `For` instruction. The loop is `Label` +
`JumpIfFalse` + `Jump` — the same shape a `while` produces (§3.2).

### Optimize

`allie/optimizer/`. Constant propagation and folding, dead-code
elimination, common subexpression elimination. Each is a pure
`IRProgram -> IRProgram` function; the first two are dataflow analyses
over a control-flow graph built on demand. `--no-optimize` skips them;
`--debug-ir` verifies the IR after every pass and prints what each one
removed. §3.4 is the interesting part.

### LLVM → machine code

`allie/codegen/llvm_lowering.py`. Labels become basic blocks, jumps
become `br`/`cbranch`:

```llvm
define void @"__allie_main"()
{
entry:
  %"total_ptr" = alloca i64
  store i64 0, i64* %"total_ptr"
  %"i_ptr" = alloca i64
  store i64 0, i64* %"i_ptr"
  br label %"for_loop_0"
for_loop_0:
  %"r2" = load i64, i64* %"i_ptr"
  %".5" = icmp slt i64 %"r2", 4
  br i1 %".5", label %"cont", label %"for_end_2"
if_end_4:
  %"r8" = load i64, i64* %"total_ptr"
  %"r9" = load i64, i64* %"i_ptr"
  %".10" = call {i64, i1} @"llvm.smul.with.overflow.i64"(i64 %"r9", i64 2)
  %".11" = extractvalue {i64, i1} %".10", 1
  br i1 %".11", label %"runtime_error", label %"guard_ok"
for_increment_1:
  %"r13" = load i64, i64* %"i_ptr"
  %".24" = call {i64, i1} @"llvm.sadd.with.overflow.i64"(i64 %"r13", i64 1)
  %".25" = extractvalue {i64, i1} %".24", 1
  br i1 %".25", label %"runtime_error.2", label %"guard_ok.2"
for_end_2:
  %"r16" = load i64, i64* %"total_ptr"
  %".32" = call i64 @"box_int"(i64 %"r16")
  %".33" = call i64 @"call_builtin_bridge"(i64 4438292448, i64 %".32", i64 0, i64 0, i64 0, i64 1)
  ret void
cont:
  %"r5" = load i64, i64* %"i_ptr"
  %".7" = icmp eq i64 %"r5", 2
  br i1 %".7", label %"cont.1", label %"if_end_4"
cont.1:
  br label %"for_increment_1"
runtime_error:
  %".13" = call i64 @"raise_runtime_error_bridge"(i64 4438147232)
  ret void
guard_ok:
  %".15" = extractvalue {i64, i1} %".10", 0
  %".16" = call {i64, i1} @"llvm.sadd.with.overflow.i64"(i64 %"r8", i64 %".15")
  %".17" = extractvalue {i64, i1} %".16", 1
  br i1 %".17", label %"runtime_error.1", label %"guard_ok.1"
runtime_error.1:
  %".19" = call i64 @"raise_runtime_error_bridge"(i64 4438147232)
  ret void
guard_ok.1:
  %".21" = extractvalue {i64, i1} %".16", 0
  store i64 %".21", i64* %"total_ptr"
  br label %"for_increment_1"
runtime_error.2:
  %".27" = call i64 @"raise_runtime_error_bridge"(i64 4438147232)
  ret void
guard_ok.2:
  %".29" = extractvalue {i64, i1} %".24", 0
  store i64 %".29", i64* %"i_ptr"
  br label %"for_loop_0"
}
```

The arithmetic is native `i64` code — `icmp` for the comparisons, and
`mul`/`add` through LLVM's overflow-checking intrinsics
(`llvm.smul.with.overflow`, ...), which branch to a `runtime_error`
block instead of wrapping around: `Int` is 64-bit on every backend,
and overflowing it is an error everywhere. The `print` call isn't
native — it goes through `call_builtin_bridge` back into Python (§3.3).
That `4438292448` is a pointer to a Python object, which is why LLVM
dumps aren't byte-stable across runs and the golden tests normalize them.

### JIT

`allie/codegen/jit.py` parses the module, binds bridge symbols to their
Python implementations via ctypes, finalizes, and calls the entry point.

---

## 3. Design calls worth talking about

### 3.1 Slang keywords are one lookup table, not a preprocessing pass

The obvious way to add `ts` as an alias for `let` is to rewrite the
source before lexing, or to branch in the lexer. Both are bad: a
rewriter breaks column numbers in error messages, and a branch means
every downstream stage could in principle learn which spelling was used.

Instead `KEYWORD_ALIASES` maps *both* spellings to the same `TokenType`:

```python
"let": TokenType.LET,     "ts": TokenType.LET,
"while": TokenType.WHILE, "Ahhh": TokenType.WHILE,
```

The lexer does one dict lookup and never branches on which it found. By
the time the parser runs, the distinction doesn't exist. Adding an alias
is one line, and it works everywhere — parser, type checker, IR, LLVM —
for free.

`docs/keywords.md` is *generated* from that dict, with a test that fails
if the doc drifts from the code.

Lookup is deliberately case-sensitive: `Ahhh` is `while`, `ahhh` is an
ordinary identifier. That's an easy thing to trip over in a live demo,
which is exactly why there's an explicit test for it.

### 3.2 `for` loops get no IR instructions of their own

A `for` loop desugars into the same `Label`/`JumpIfFalse`/`Jump` shape a
`while` produces, at the IR-lowering stage — before codegen exists.

The payoff: **LLVM lowering needed zero `for`-specific code.** No
`for` branch, no special case for `break`, none for `continue`. It only
knows about labels and jumps. `if` and `while` came out the same way.

One detail that isn't obvious and would silently break things: `for`'s
`continue` must jump to the *increment*, not the condition check. Jump
to the condition and the counter never advances — infinite loop. So the
lowering emits a dedicated `for_increment_N` label, visible in the dump
above, and `while`'s `continue` (which has no increment step) targets
its loop label directly. There's a golden snapshot pinning this exact
shape, because it's the kind of thing a later refactor could quietly
get wrong.

### 3.3 Series/Matrix code isn't compiled — it calls back into Python

`SMA`, `mean`, `load`, and everything else touching a `Series` compile
to an external call into the Python/numpy runtime, not to LLVM IR.
Scalars (`Int`/`Float`/`Bool`) *are* native — `add`, `fadd`, `icmp`.
Everything else crosses the boundary as an opaque `i64` handle (the
`id()` of a live Python object, kept alive by a registry).

Is that a cop-out? Genuinely no, and the benchmarks say why: on the
quant workloads the JIT's native run takes exactly as long as the
interpreters' — the time is inside numpy, and compiling the thin Allie
wrapper around a `numpy.mean` call can't speed up `numpy.mean`.
Reimplementing vectorized rolling windows in LLVM IR to maybe match
numpy is a large amount of work for a likely-zero gain.

Meanwhile the part where compilation *does* pay — scalar arithmetic and
control flow — is fully native, and that's where it shows: a loop
iteration in ~0.4 ns against CPython's ~30 ns (§4).

User-defined functions can take and return series too: across a native
call, a `Series` is just its boxed handle, the same way it travels
everywhere else in compiled code.

Before code generation the module goes through LLVM's own `-O2` pipeline
(`allie/codegen/jit.py`). Lowering keeps every variable in a stack slot;
`mem2reg` turns those into registers, and the loop passes take it from
there. A 30-million-iteration Int loop runs 5.6× faster than at `-O0`,
`fib(30)` 1.7×, a Float recurrence 2.3×. The price is about 2 ms of
extra code generation per program, so `-O2` breaks even around a
million loop iterations, and every benchmark workload is *slower* with
optimization on: they finish before the faster code can pay for itself.
`--no-optimize` turns LLVM's passes off along with Allie's.

### 3.4 A dead-code elimination bug that took a while to see

DCE removes instructions whose results are never read. The obvious
implementation scans forward: for each write, is this register or
variable read *later*?

That's wrong for loops, and the failure is nasty. Consider the increment
in the IR dump above:

```
Label(name='for_increment_1')
...
StoreVar(name='i', src='r15')     <- writes i
Jump(label='for_loop_0')          <- back-edge
Label(name='for_end_2')
LoadVar(dest='r16', name='total') <- reads total, not i
```

Nothing *after* that `StoreVar` reads `i`. A forward-only scan concludes
it's dead and deletes it — and the loop counter never advances. Infinite
loop, from an "optimization."

The read is real, it just happens *earlier in the list* — at the
condition check — reached again through the back-edge `Jump`. Program
order and execution order aren't the same thing once you have loops.

The first fix was deliberately blunt: liveness asked "is this read
*anywhere* in the program," not "read after this point." Position
stopped mattering, so back-edges stopped being a special case — but it
couldn't see that the first write in `x = 1; x = 2; print(x);` is dead.

The real fix is a control-flow graph (`allie/ir/cfg.py`). The IR is
still stored as a flat list, but DCE and constant propagation now build
basic blocks with successor and predecessor edges and solve liveness
and constants as proper dataflow problems — so a loop's back-edge is
just an edge, and the dead first store is removed. There's a regression
test with that exact loop shape.

A related one turned up in CSE, which had the same blind spot from the
other direction: it was reusing a register computed *before* a loop for
a read *inside* it, which is only valid if nothing in the loop
reassigns it. Fixed by invalidating the caches at every label —
i.e. at every place control flow can merge.

Both bugs share a root cause: a flat instruction list makes it easy to
write a pass that reasons about text order when it should be reasoning
about control flow. A third one, the same shape again, survived until
the IR got a verifier: CSE kept sharing *constants* across labels,
reasoning that a literal's value doesn't depend on the path. The value
doesn't; the register holding it does. `if flag { x = x + 7; }
print(x + 7);` reused a register written only inside the `if`, and the
IR backend crashed when `flag` was false. The verifier
(`allie/ir/verify.py`) checks that every register is written before it
is read on every path, runs after every optimizer pass in the tests, and
flagged it the first time it ran.

What keeps this class of bug from coming back is a differential fuzzer
(`tests/unit/test_optimizer_fuzz.py`): random type-correct programs run
on the tree-walker, the IR interpreter and the JIT, with the optimizer
off and on, and all five must print the same thing — runtime errors
included. Its first runs found two more bugs, both in the LLVM backend
rather than the optimizer: a loop written after a `break` crashed
codegen, and a variable whose first store sat inside a branch produced
invalid LLVM.

### 3.5 The look-ahead analysis is tested against an experiment

A static analysis that claims soundness is only making a claim until
something independent checks it. Here the independent check is the
definition itself, run: a decision at bar t that changes when the bars
after t are removed. `allie/semantic/audit.py` runs the program, then
runs it again once per bar with `load` cut off at that bar. It compares
each backtest's signals, orders and closed trades between the runs. It
sees them through a hook in `run_backtest`, which every backend calls.

`tests/unit/test_lookahead_audit.py` generates programs, and each draws
how often it takes a leaky option, from never to often. The leaks
include backward shifts; whole-series statistics in conditions, order
sizes and stop levels; single bars; values picked by a branch on
`mean(c)`; and counters a backtest's actions left behind. Every program
the analysis accepts must pass the audit. Of the 300 seeded programs,
182 are accepted and all of them pass. 104 are rejected and run, and 54
of those change a decision on the sample data. The other 50 carry a
leak this data happens not to trigger, such as a `zscore` inside a
condition that is never true. Rejecting them is right, so the share of
rejections the audit confirms doesn't measure over-strictness. A
separate test does: one program per rule, in which the leak drives a
trade, and the audit must catch each one.

Building it found real gaps. Before the rewrite the analysis tracked
only data flow, and only into `when` conditions. It let through:

- a function returning a tainted global;
- a taint carried from one loop iteration to the next;
- `c > c[len(c) - 1]`;
- `r.sharpe` in a condition;
- `target(mean(c) / c[0])`;
- an `if` on `mean(c)` inside an action;
- a stop calibrated on the whole period's volatility.

It also rejected honest programs like `let lag: Int = 1; shift(c, lag)`.
Writing down what the audit would measure turned up three more leaks:

- a value chosen by a branch on future data (control dependence);
- `len(c)` inside a window;
- a counter that a backtest's actions left behind.

The REPL had a gap of its own: a submission rejected *before* it ran
still updated the analysis's state, so `s = c; <something rejected>`
made the analysis think `s` was clean when, at run time, it still held
tomorrow's prices.

---

## 4. Benchmarks

Methodology and the full report: [`benchmarks/README.md`](../benchmarks/README.md)
and [`benchmarks/results.md`](../benchmarks/results.md); regenerate with
`make bench`. Every implementation is checked to compute the same answer
as its Python twin before it's timed, and every ratio below has a 95%
confidence interval in the report. Figures from the committed run on an
Apple M5:

| | ns per loop iteration | fixed cost |
|---|---|---|
| Allie tree interpreter | ~800 | — |
| Allie IR interpreter | ~900 | — |
| CPython | ~30 | — |
| Allie JIT, `-O0` | ~2.3 | ~1.5 ms of compilation |
| Allie JIT, `-O2` | ~0.4 | ~3.9 ms of compilation |

Three things this shows.

**Native code is fast, and compiling it costs real time.** The JIT runs a
loop iteration roughly 75× faster than CPython and ~2,000× faster than
the tree interpreter. But LLVM's code generation takes longer than a
50k-iteration loop or `fib(20)` takes to run, so on those workloads the
JIT's total is several times CPython's. A scaling study — the same loop
at three sizes, fitted to *fixed + per-iteration × n* — is what tells
those two costs apart; a single size can't.

**Compilation is worth nothing for library-bound work.** On the quant
workloads the JIT's native run equals the interpreters', and code
generation is overhead on top. Allie is 2–5× slower than pandas there,
mostly in reading the CSV: `load_10k` alone is ~11 ms against pandas'
~3.5 ms, because pandas parses in C. The interesting columns are LOC —
the backtest is 10 lines of Allie against 18 of Python.

**The optimizer costs more than it saves at these sizes.** `-O2` (Allie's
passes and LLVM's) makes the native run up to 2× faster but costs ~2 ms
more to compile, so it pays off only past about a million loop iterations;
every workload here is slower overall with it on, by an interval that
excludes 1. A JIT that chose its optimization level per program would
take both.

**The first version of these numbers was wrong.** It timed with
`tracemalloc` running — slowing allocation-heavy Python several-fold —
and counted LLVM's code generation as run time. Its "127× faster than
the interpreter, 11× faster than Python" was mostly those two mistakes.
Measuring the benchmark's own methodology is part of the benchmark.

The benchmark has also pointed at real fixes: `rolling_std` was a
per-window Python loop 112× slower than pandas (now vectorized and
bit-identical; the faster running-sum formula was rejected for losing
~1e-08 accuracy to cancellation), `load` parsed every cell with its own
`float()` call, and the backtest engine indexed numpy arrays bar by bar.

---

## 5. What's not done

Stated as v2 work, not apologies.

**Allie's own optimizer matters little to the JIT.** Its passes are real
dataflow analyses (§3.4) and roughly halve the IR interpreter's time on
constant-heavy loops, but for compiled code LLVM's `-O2` does that work
and more. CSE is still local to each straight-line stretch.

**Series work isn't compiled.** Series/Matrix operations bridge back to
Python (§3.3); only scalar code is native.

**`import` parses but resolves nothing.** `import "foo";` type-checks and
is then ignored — there's no module loader, no path resolution, no
cross-file name binding. It's a parsed placeholder, deliberately inert
rather than half-working.

**`break`/`continue` can't escape a `when` action.** A `when` body is a
callback the backtest engine invokes per bar, so there's no enclosing
loop to jump out of. Rejected at resolve time with a specific message. A
loop written *inside* a `when` action is fine.

**No `Matrix` literal syntax.** A matrix is built from series —
`matrix(values, rows, cols)`, `with_column` — rather than written out.

**Assignment targets only bare names.** `s[i]` reads an element, but
series are immutable values, so there is no `s[i] = x`, and fields are
read-only.
