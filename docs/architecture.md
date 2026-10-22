# Architecture

Per-stage design notes: what each stage does, why it exists, and — the
part that's usually left out — what it deliberately does *not* do.

For a narrative walkthrough of one program through the whole pipeline,
read [`how-it-works.md`](how-it-works.md) instead.

## Pipeline

```
 source (.qs)
     │
     ▼  allie/lexer/              Lexer          → tokens (line/col on each)
     │                                             slang aliases resolved HERE, nowhere else
     ▼  allie/parser/parser.py    Parser         → allie/ast/nodes.py (dataclass AST)
     │                                             errors collect into a DiagnosticBag
     ▼  allie/semantic/           Resolver       → scope + jump-placement checks
     │                                             loop_depth, inside_strategy, inside_when
     ▼  allie/typechecker/        Type checker   → annotates expr.inferred_type in place
     │                                             static, local inference; reports every error
     ▼  allie/semantic/           Look-ahead     → rejects decisions that could see a later bar
     │                                             taint + sign analysis; audit.py tests it by experiment
     ▼  allie/ir/lowering.py      IR lowering    → flat instruction list, virtual registers
     │                                             `for` desugars to `while` shape HERE
     ▼  allie/optimizer/          Optimizer      → constant fold, DCE, CSE (toggleable)
     │
     ├──────────► allie/runtime/interpreter/   tree-walking, straight off the AST
     ├──────────► allie/ir/interpreter.py      IR interpreter
     └──────────► allie/codegen/               LLVM IR → MCJIT
                      │
                      ▼  allie/codegen/runtime_bridge.py
                  Python/numpy for Series, Matrix, strings, all stdlib calls
```

Three backends execute the same programs and are cross-validated against
each other. That redundancy is the project's main correctness argument:
a bug would have to appear identically in three independent
implementations to go unnoticed.

---

## Lexer (`allie/lexer/`)

**Does:** characters → tokens, each carrying line, column and source
offset (the lexeme is always the exact source slice, which a test checks
against every program in the repo). Resolves keyword spellings through
one `KEYWORD_ALIASES` dict, so `ts` and `let` both emit `TokenType.LET`.
The full lexical rules are in `docs/grammar.md` "Lexical structure".

**Why:** the alias table is the entire implementation of the slang
keywords feature. Putting it here — a single dict lookup with no
branching — means every later stage is automatically spelling-agnostic,
and adding an alias is one line.

**Does not:** branch on which spelling it saw, or expose that
information downstream. It also does not lowercase identifiers: lookup
is case-sensitive on purpose, so `Ahhh` is `while` and `ahhh` is an
identifier.

**Never raises.** Every lexical problem is recorded as a `LexError`
(position, caret length, optional hint) and scanning resumes, so one
run reports every problem in the file. Where the intent is obvious the
lexer also emits the token that was meant — a `STRING` for an
unterminated or single-quoted string, a `NUMBER` for `.5` — so the
parser doesn't stack follow-on errors on the real one. A string can't
span lines, which is what makes an unterminated one recoverable: the
error stops at the end of its line instead of swallowing the file.
A seeded fuzz test feeds it thousands of hostile inputs and checks that
it never raises and that every position stays consistent.

## Parser (`allie/parser/parser.py`)

**Does:** recursive descent, one method per grammar production, one
method per precedence level for expressions. Produces the dataclass AST
in `allie/ast/nodes.py`, with every span carrying its real source
offset.

**Why:** hand-written rather than generated — the grammar is small
enough that a parser generator would add a build step and a debugging
layer for no benefit.

**Does not:** type-check, resolve names, or evaluate anything. On error
it records a `Diagnostic` (with a hint where the mistake is
predictable) and recovers rather than raising, so one syntax error
doesn't mask the next five. Recovery is deliberately local: it resumes
at the next statement but never skips the `}` closing the current
block, and a broken `when` clause costs only that clause — so one
mistake produces one error, not a cascade (`test_parser_diagnostics.py`
pins this on a file with six separate mistakes).

**Limits:** nesting is capped at 64 levels and AST depth at 200, so a
pathological input is one error rather than a Python `RecursionError`
in the parser or in any later stage that walks the tree recursively. A
test runs the deepest allowed program on all three backends.

## Resolver (`allie/semantic/resolver.py`)

**Does:** walks a scope stack checking names exist, and checks jump
placement — `break`/`continue` need an enclosing loop, `return` needs a
function, `when` needs a strategy.

**Why:** separating "does this name exist" from "does this name have the
right type" keeps both passes simple. The resolver's `Scope` tracks
existence only; the type checker's `TypeEnvironment` tracks types. They
are deliberately different structures.

**Does not:** existence-check the callee of a call. Builtins have no
entry in the lexical scope chain, so treating every call target as a
local would misfire on every `print`/`SMA`. The type checker validates
call targets instead.

**Design decision — jumps can't cross an execution boundary.**
`loop_depth` resets to 0 when entering a function body or a `when`
action. A `break` inside a `when` action cannot target a loop enclosing
the *strategy declaration*: the action is a callback the backtest engine
invokes per bar, so that loop isn't on the stack at runtime. A loop
written *inside* the action is a perfectly valid target. `inside_when`
selects a more specific error message; it deliberately does not subsume
`loop_depth`.

## Type checker (`allie/typechecker/`)

**Does:** static typing with local inference — `let` annotations are
optional (inferred from the initializer) but enforced where present.
Annotates `inferred_type` on every expression node in place, which is
what makes LLVM codegen possible later (it needs to know whether `+` is
`add` or `fadd`). Rejects anything only some backends can run: function
values, closures over an enclosing function's locals, duplicate
function names, `Void` used as a value, series equality.

**Why:** collect-and-report, like the parser, but without the cascade
that usually argues for fail-fast: an expression that fails gets a
poison type (`ERROR`) that every later check accepts silently, so one
mistake is one error. `check()` still raises the first error once
everything has been checked.

**Does not:** infer function signatures. Every parameter and return type
must be annotated, including `-> Void`. Partial inference that works
sometimes is harder to use than none.

**Does not:** allow truthiness. Conditions must be exactly `Bool` —
`if 1 { }` is an error, not a shorthand.

## Look-ahead analysis (`allie/semantic/lookahead.py`)

**Does:** a taint analysis that proves no trading decision can see a
later bar. Sources (backward shifts, whole-series builtins, single bars,
`len`, backtest statistics, state left by a backtest's actions,
hand-built series) carry a taint. The taint flows through variables,
operators, calls and field access to the sinks, which are the places a
decision is made: `when` conditions, conditions inside actions,
`target(...)` sizes, backtest options, and the conditions that decide
whether a backtest runs.

**Why:** `docs/spec.md`'s founding claim is that the language should
make "you cannot see the future" structural rather than a discipline the
author maintains. The backtest engine enforces that at runtime: an order
fills at the next bar's open. But that constrains the *engine*, not the
*data* handed to it. A strategy built on `shift(close, -1)` would
backtest at Sharpe 3.31, against 1.14 for the same rule on today's
close. This pass closes the gap the runtime guarantee never covered.

**Design decision — flow-sensitive, with control dependence.** The state
is a scope stack mapping each variable to its taint, whether it's proven
non-negative (the sign analysis that accepts `shift(c, lag)`), and where
it got its taint (for the notes). At an `if`, both branches are analyzed
from the same state and joined. Loops run quietly to a fixpoint, then
once more to report. Assignments under a tainted condition take its
taint, so a value chosen by a branch on future data is caught. So is
code after an `if` that may `break`, `continue` or `return`, until the
loop or function ends.

**Design decision — functions analyzed per call site.** A function's
body is analyzed at each call, with its parameters bound to the
arguments' taints and signs. Assignments to module-level variables land
on them, a never-called function changes nothing, and recursion is
iterated to a fixpoint. A `when` action is analyzed at its declaration,
and again at each backtest of its strategy, where it really runs, as a
loop over the bars. Whatever it assigns then carries a "state left by a
backtest" taint.

**Design decision — unprovable means rejected.** For a safety property,
"cannot prove safe" has to mean "reject". A false positive is a compile
error the user can see and fix; a false negative is a silently wrong
backtest.

**Tested against an experiment (`allie/semantic/audit.py`).** The audit
defines look-ahead dynamically: a decision at bar t that changes when
the bars after t are removed. It re-runs the program once per cut. The
engine reports what each backtest decided (signals, orders, closed
trades) to observers, through `observe_backtests()` in
`run_backtest`, the entry point all three backends share. A fuzz test
generates honest and leaky programs and asserts that every program the
analysis accepts passes the audit. That turns soundness from a claim
into a tested property. `allie check --audit` runs the same experiment
on a user's file.

**Does not:** flag future data used outside a decision (post-hoc
analysis is legitimate), see data pre-shifted outside the language, or
resolve a function's module-level variables where it was declared
rather than where it's called.

## IR lowering (`allie/ir/`)

**Does:** flattens the AST into a linear instruction list over virtual
registers, with `Label`/`Jump`/`JumpIfFalse` for control flow.

**Design decision — `for` gets no instructions of its own.** It
desugars into exactly the shape `while` produces. The payoff is that
LLVM lowering needed *zero* `for`-specific code. One subtlety: `for`'s
`continue` targets a dedicated increment label, not the condition check
— jump to the condition and the counter never advances. A golden
snapshot pins this exact shape.

**Design decision — stored flat, analyzed as a graph.** The IR is one
instruction list — simple to emit, and sufficient for lowering to LLVM,
which builds real basic blocks from the labels. Passes that need control
flow build a graph on demand (`allie/ir/cfg.py`: basic blocks, edges, a
worklist dataflow solver) instead of reasoning about text order, which
is where the bugs in [`how-it-works.md`](how-it-works.md) §3.4 came from.

**Invariants, checked.** `allie/ir/verify.py` checks that labels are
unique and every jump target exists, that every register is written
exactly once (except `Move` targets), and that on every path a register
is written before it is read. The pass manager can run it after every
pass and name the one that broke the IR; the tests do, for every
program in the repo and every program the optimizer fuzzer generates.

**Lowered IR has no unreachable code.** Statements after a `break` /
`continue` / `return` are dropped at lowering, whether or not the
optimizer runs. Implicit Int → Float widening is an explicit
`UnaryOp("to_float")`, inserted where the type checker found it.

**Does not:** support first-class function values. Functions are static
and name-registered, so a function returned from another function and
called later has no way to work through this path. Documented, and the
cross-validation suite is scoped around it.

## Optimizer (`allie/optimizer/`)

**Does:** constant propagation and folding (through variables and
across branches, resolving constant conditions), dead-code elimination
(liveness: dead values, dead stores, unreachable blocks), and common
subexpression elimination (local to each straight-line stretch). Each is
a pure `IRProgram -> IRProgram` function. The pipeline is defined once
(`DEFAULT_PASSES`); the pass manager has an on/off switch for the
benchmarks, optional verification after each pass, and a per-pass
report that `--debug-ir` prints.

**Does not:** remove any `Call`, ever (DCE). It has no way to know
whether a call is side-effecting without consulting purity, so it
assumes every call matters. CSE *does* consult `PURE_FUNCTIONS` —
defined once, in one module, so the two passes cannot disagree. A test
asserts every registered builtin is classified pure or impure, so a new
builtin can't silently default to "never optimized".

**Does not:** assume anything about a user function's effect on
variables. A call to anything but a known builtin may read or assign
any module-level variable, so constant propagation forgets every
variable after one, DCE treats it as reading them all, and CSE drops
its variable cache. At a function's exit every variable counts as live.
CSE invalidates all its caches at every label.

**Does not:** remove a division or modulo whose divisor isn't a known
non-zero constant, even when the result is unused. Raising is an
observable side effect: `let x: Int = 1 / 0;` with `x` never read still
has to fail. This was found by `--verify` — with the optimizer on, the
tree-walker raised while the IR and LLVM backends exited 0, because only
they run this pass.

## Backends

### Tree-walking interpreter (`allie/runtime/interpreter/`)

Runs off the AST, no IR. The default for `allie run`, and the reference
implementation the other two are checked against.

**Design decision — compile to closures, then run.** The AST is walked
once, turning each node into a Python closure that does that node's
work; running the program is calling closures. Dispatch is decided at
compile time; arithmetic is specialized by the type the checker
inferred (`+` on two Ints is an add and a range check, on two Floats a
bare add); `break`/`continue`/`return` are return values rather than
exceptions; and a block that declares nothing runs in its enclosing
scope instead of allocating one per iteration. The first version
re-inspected every node with an `isinstance` chain each time it ran:
~4.9 µs per loop iteration against ~0.8 µs now
(`benchmarks/results.md`). A node with no type annotation falls back to
the generic, dynamically typed operator.

### IR interpreter (`allie/ir/interpreter.py`)

Translates each program once into one small closure per instruction,
returning the index of the next; labels become indices, and a register
written only by one `LoadConst` is filled before the program starts
rather than reloaded on every loop iteration. Registers are list slots.
~3.6 µs per loop iteration became ~0.9 µs. Scalar operator logic is
deliberately **re-implemented** rather than imported from the
tree-walker: sharing the helpers would let a bug common to both hide
from cross-validation, defeating the point of having two. Operators on
series are the exception: like stdlib functions, every backend calls
one numpy implementation (`allie/runtime/elementwise.py`), since the
arithmetic there is numpy's, not ours.

### LLVM + MCJIT (`allie/codegen/`)

Labels become basic blocks, jumps become `br`/`cbranch`. Scalar
arithmetic (`Int`/`Float`/`Bool`) compiles to native instructions.

**Design decision — Series/Matrix codegen is out of scope for v1.**
Everything touching a `Series`, `Matrix`, or `String` crosses into
Python as an opaque `i64` handle, and every stdlib call routes through
`runtime_bridge`. Reimplementing vectorized rolling windows in LLVM IR
to *maybe* match numpy is a large amount of work for a likely-zero gain
— on the quant benchmarks the JIT's native run takes exactly as long as
the interpreters' run: the time is in the library calls, not the code
around them.
A user-defined function can take and return a `Series` or `PriceSeries`
— the handle is just an `i64` — and an operator with a series operand
(`close - open`, `fast crosses_above slow`) is one `binop_bridge` call.

**Does not:** let exceptions cross the JIT boundary. A Python exception
raised inside a ctypes callback invoked from JIT code is swallowed by
ctypes, never reaching the caller. Bridge functions record the exception
in a pending slot and return a sentinel; `compile_and_run` re-raises it
the moment control returns to Python.

**Does not:** emit a bare `sdiv`/`srem`. Hardware division by zero traps
and kills the process rather than raising, so the lowering emits an
explicit zero-check first — matching both interpreters' behavior.

---

## Cross-cutting

**Errors.** `LexError`, `QSTypeError` and `AllieRuntimeError` carry
line/column and format identically: `{Kind}: {message}. Line {L}, Column
{C}`, rendered with the source line and a caret. A runtime error is
located at the innermost expression that failed, **on every backend**:
the tree-walker attaches the AST node's span; lowering stamps the same
span on each IR instruction that can fail (`IRInstr.span`, outside its
`repr` so golden snapshots don't change), and the IR interpreter attaches
it; the JIT puts it in the payload of a native guard (a zero divisor,
overflow) and, after a failed call into Python, attaches it from the
error-exit block — code that only runs on failure, so the happy path
pays nothing. An error raised inside a stdlib function gets the span of
the call. The differential fuzzer compares error text across backends,
locations included. `tests/unit/test_error_formats.py` pins the format.

**Generated docs.** `docs/keywords.md` and the keyword table in
`docs/language-reference.md` are both generated from `KEYWORD_ALIASES`
by `scripts/generate_keywords_doc.py`, with tests that fail if either
drifts from the code.

**`import` parses but resolves nothing.** There is no module loader, no
path resolution, no cross-file binding. `import "foo";` type-checks and
is then inert on every backend. Deliberately a parsed placeholder rather
than something half-working that appears to load files.

---

## Phase notes

Detailed notes from each phase's implementation, kept for
reference.

## Phase 6 — LLVM backend + MCJIT

A third execution backend, alongside the tree-walking interpreter
(`allie.runtime.interpreter`) and the IR interpreter (`allie.ir.
interpreter`). All three are cross-validated against each other in
`tests/unit/test_llvm_cross_validation.py` — three independent paths
agreeing on the same programs.

### Type mapping (Phase 6.2)

| Allie type | LLVM type |
|---|---|
| `Int` | `i64` |
| `Float` | `double` |
| `Bool` | `i1` |
| `Void` | `void` |
| `String` | `i64` — boxed handle (see below) |
| `Series[T]` / `Matrix[T]` / `Vector[T]` / `PriceSeries` / `Date` / `DateRange` / `Any` | `i64` — boxed handle |

**Deviation from the literal spec.** `String -> i8*` (a pointer to a
global constant) was specified, scoped to "no dynamic string ops needed
yet". But Allie's `+` does runtime string concatenation — proven in both
interpreters' test suites — so a string's value isn't always a
compile-time constant a fixed global could hold. Rather than special-case
strings, *every* non-native-scalar type uses one uniform boxed-handle
scheme: an opaque `i64` that is `id()` of a live Python object, kept
alive by a registry in `allie.codegen.runtime_bridge`. One mechanism
instead of two.

**Series/Matrix/Vector native codegen is explicitly out of scope for
v1**, per the spec. `SMA`, `mean`, `load`, etc. lower to external calls
back into the existing Python/numpy runtime through
`runtime_bridge.call_builtin_bridge` — never reimplemented in LLVM IR.
`buy()`/`sell()`, `__make_series` (array literals), and the
`strategy`/`when`/`backtest` stubs go through the same mechanism, with no
special-casing: by the time lowering runs they're all just `Call`
instructions.

### Native vs. bridged dispatch

Arithmetic/comparison on `Int`/`Float`/`Bool` compiles to direct LLVM
instructions (`fadd`/`fdiv`/`icmp_signed`/`fcmp_ordered`, Int `+ - *`
through LLVM's overflow-checking intrinsics, `%` floored like Python's). Anything else routes through the bridge, which calls the
*same* `_apply_binop`/`_apply_unary` the IR interpreter uses — so there
is no third reimplementation of operator semantics to drift out of sync.

A call to a user-defined function becomes a direct native LLVM `call`
(this is also how recursion works); a `Series` or other non-scalar
argument or result crosses it as its boxed handle. Main-program
variables that a function uses live in LLVM globals, each with an
"assigned yet?" flag, so a function reading one before its `let` ran
raises the interpreters' "undefined name" error.

**Ending a run.** Native code can't raise, so failures set a module-level
`__allie_error` byte (the bridge writes it from Python) that native code
checks after every call, returning immediately; a call-depth counter
turns runaway recursion into "recursion too deep" instead of a native
stack overflow.

**LLVM's optimizer.** The module runs through LLVM's standard `-O2`
pipeline before code generation (`--no-optimize` skips it). Lowering's
`alloca`s become SSA registers, and hot loops get 2–5× faster; it costs
about 1–1.5 ms of code generation per program.

### Confirming the Phase 5.2 desugar (Phase 6.3's required self-check)

`for`/`break`/`continue` need **zero** dedicated code in the LLVM
lowering. `Label`/`Jump`/`JumpIfFalse` become basic blocks and
`br`/`cbranch` uniformly for `if`/`while`/`for` alike — precisely because
Phase 5.2 already desugared `for` onto `while`'s shape. Traced
`sum_to_n` through by hand to confirm no `for`-specific branch crept in;
none did.

### Two things the JIT boundary forced, verified rather than assumed

**Exceptions can't cross a ctypes callback.** A Python exception raised
inside a callback invoked from JIT-compiled code does *not* propagate to
the Python caller — ctypes prints "Exception ignored" to stderr and the
callback returns garbage. Every bridge function is therefore wrapped in a
guard that records the exception in a module-level pending slot and
returns a sentinel; once set, subsequent bridge calls short-circuit so a
first failure can't cascade into confusing secondary errors.
`jit.compile_and_run` re-raises it the moment the JIT call returns —
the first point where a normal `raise` actually reaches the caller.

**Native division by zero is undefined behavior, not a catchable
error.** An `sdiv`/`srem` by zero traps at the hardware level and kills
the process. So the lowering emits an explicit zero-check *before* the
division: if the divisor is zero it calls the error bridge and returns
from the function immediately, never reaching the division instruction.
Both interpreters raise `AllieValueError` here, and now so does the JIT.

### CLI

```
allie run FILE|- [--backend tree|ir|llvm] [--verify] [--no-optimize] [--trace] [--time]
                 [--debug-tokens] [--debug-ast] [--debug-ir]
allie check FILE...
allie repl
allie --version
python main.py FILE ...        # allie run --backend llvm
```

`main.py` implements the Phase 6.4 interface; it takes its options from
the same definition as `allie run` (`add_run_arguments`), so the two
can't drift. `check` runs every static pass and nothing else. `repl`
checks each submission incrementally — the resolver, checker and
look-ahead analysis keep their state — so errors point at the line just
typed, and a submission that fails is forgotten. `--verify` compares
output, exit code *and* the error that ended the run across all three
backends. `--trace` prints each statement as it runs; `--time` prints
front end, lowering, LLVM code generation and run separately. LLVM is
imported only for the llvm backend.

Exit codes: 0 success, 1 the program has an error, 2 usage (bad
arguments, an unreadable file), 3 `--verify` disagreement, 70 an
internal compiler error — reported as such, with
`ALLIE_TRACEBACK=1` for the Python traceback, never a raw traceback.

## Phase 7 — Backtesting DSL

`strategy` / `when` / `backtest` execute for real. All three backends
funnel into one entry point, `allie.quant.backtest_engine.run_backtest`,
so no backend can drift into its own accounting — verified in
`tests/unit/test_backtest_cross_validation.py`, which asserts byte-identical
results tables from the tree-walking interpreter, the IR interpreter, and
the LLVM/JIT backend.

### How `when` clauses drive trades

The spec described the engine in operator terms ("on `crosses_above`
true and flat → enter long"). What's implemented is one level more
general: each `when` clause contributes a boolean signal series plus an
*action* (its body), and on each bar every clause whose signal is True
runs its action. The `buy()`/`sell()` builtins called inside that action
are what actually move the position.

For the canonical crossover strategy this is exactly the specced
behavior, but it avoids parsing, resolving, and type-checking a clause
body only to then ignore it — and it makes `buy()`/`sell()` real
builtins rather than decorative ones.

### Position and accounting model (revised after v1)

v1 was flat-or-long and all-in, filled every order at the close of the
bar whose close produced the signal, and charged nothing. Each of those
made results look better than trading would, which is the one thing a
backtest must not do, so the engine was revised:

- **Fills at the next bar's open by default.** A signal computed from
  bar i's close can't be traded at that close — by the time it's known,
  the bar is over. `with fill: "close"` keeps v1's convention, visibly.
  Every clause's actions run before anything fills, and the last order
  of a bar is the one that fills, so `buy()` then `sell()` on one bar is
  no trade rather than a round trip at one price.
- **Costs on by default.** 5 bps of slippage per fill; commission
  optional. docs/spec.md made costs first-class for exactly this reason
  ("if it's opt-in, people forget it"), and v1 had none. The results
  table prints the fill point and both costs.
- **Positions are a target fraction of equity.** `target(f)` is the
  primitive; `buy`/`sell`/`short`/`cover` are guarded targets, so a
  repeated signal stays a no-op as in v1. A reversal is a close and an
  open at one price, so every trade has a clean entry and exit.
- **Stops** use each bar's open, high and low: a gap exits at the open,
  and a bar reaching both levels is read as the stop.
- **Accounting** is cash plus shares at the last usable close, so a NaN
  close carries the mark forward rather than poisoning equity; an
  account that reaches zero is closed out instead of going negative. A
  trade's P&L is the change in equity from just before its entry to its
  exit — cash earns nothing, so that's exactly the position's P&L after
  costs.
- **Metrics** gained total return, buy-and-hold, Sortino, exposure, win
  rate, profit factor, average trade and costs; CAGR now counts
  `bars - 1` periods. **Risk-free rate is 0** for Sharpe and Sortino,
  stated since "Sharpe" alone doesn't say which convention is in use.

On the 60-bar fixture the canonical crossover went from Sharpe −0.16 to
−0.19 — costs, mostly: the fixture's opens equal the previous closes, so
there the fill point changes nothing. `tests/unit/test_backtest_engine.py`
checks every rule above against hand-computed numbers on prices that do
gap.

The options (`with slippage: 0.001, stop_loss: 0.05`) are lowered as
one `__backtest_option(name, value)` call each before `__backtest_run`,
which takes them: a single call carrying all of them wouldn't fit the
LLVM bridge's fixed arity, and the IR needed no new instruction. The
names, types and valid ranges are one table
(`allie.quant.backtest_engine.OPTIONS`) that both the type checker and
the engine read.

### `Series<Bool>` is a distinct runtime type

Comparisons and crossings on series, `and`/`or`/`!` on them, and a
`[true, false]` literal produce a `BoolSeriesValue`, not a float64
`SeriesValue` holding 1.0/0.0. `SeriesValue` enforces float64 by
construction, so a boolean series stored there would either break that
invariant or silently become numeric data that `mean`/`SMA` would
happily — and meaninglessly — accept.

### `when` takes any `Series<Bool>` (revised after v1)

v1's grammar required a crossing (`when a crosses_above b:`), on the
grounds that an arbitrary condition has no natural long/flat reading.
But entry and exit were never the condition's job: each clause has its
own action, and `buy()` or `sell()` in it says which. So the revision
made `crosses_above`/`crosses_below` comparison operators, gave series
the arithmetic, comparison and boolean operators, and made a `when`
condition any expression the type checker can prove is a
`Series<Bool>`: `when fast crosses_above slow and rsi < 70.0: buy();`.
A single `Bool` is still rejected, with a hint — it's one decision, not
one per bar.

`backtest` became an expression too, evaluating to a `BacktestResult`
whose fields (`r.sharpe`, `r.returns`, ...) feed ordinary code, so a
parameter sweep, a train/test split (its bounds are `Int` expressions
now) and a multiple-comparisons correction are all written in Allie.

### Two jump-boundary bugs found by checking a cross-reference claim

Phase 7.1 asked to "confirm both flags are still distinct and neither
accidentally subsumes the other". Confirming revealed `inside_when` was
**set but never read** — so `break`/`continue` inside a `when` action
could bind to a loop enclosing the *strategy declaration*, which the
per-bar callback can never actually reach at runtime. The same class of
bug existed for function bodies (`loop_depth` wasn't reset on entry).
Both now reset `loop_depth` at the boundary; `inside_when` only selects
the more specific error wording, and deliberately does not subsume
`loop_depth` — a loop written *inside* a `when` action is still a valid
jump target.

### The Phase 5.2 IR lowering for these constructs was rewritten

It was written while `strategy`/`when`/`backtest` were always-raising
stubs, and was structurally unable to express them: a nameless
`__strategy_declare`, each `when` body lowered *inline and
unconditionally* (running in sequence rather than when its signal
fired), and raw strategy/symbol strings passed in an argument list the
interpreter reads as register names. Nothing caught it because the
constructs never executed.

Each `when` body is now its own zero-argument IRProgram — the same
mechanism `FuncDecl` uses — registered in `functions`/`function_info`,
which is what lets the LLVM backend declare and compile them with no
special-casing. `allie.quant.backtest_engine` holds a dispatch table
mapping body names to zero-argument callables; the IR interpreter fills
it with closures over `run_ir_program`, and the JIT fills it with ctypes
wrappers around compiled function addresses (obtainable only after
`finalize_object()`). The engine just calls them.

Each clause's condition is evaluated where the strategy is declared, on
every backend, and the backtest receives that `Series<Bool>`. (v1's
tree walker deferred it to the `backtest` statement while the IR and
LLVM paths didn't, so reassigning an indicator between the two gave
different results; `tests/unit/test_strategy_dsl.py` now pins the
shared behavior.)

## Phase 8 — Testing & benchmarking infrastructure

### Test suite shape

- **`tests/unit/`** — one file per compiler stage.
- **`tests/integration/fixtures/*.qs`** paired with `.expected` files, run
  by one parametrized test. Three families are represented and a test
  guards their presence: pure-CS, quant-only, and combined. Every fixture
  runs on **all three backends** against the same `.expected` file, so
  this doubles as a continuous three-way agreement check rather than
  diffing one backend against itself.
- **`tests/golden/`** — IR and LLVM IR snapshots. These catch lowering
  regressions the behavioral tests miss: a lowering can emit different
  instructions and still compute the right answer, right up until it
  doesn't.

`@FIXTURE@` in a fixture source is substituted with an absolute CSV path
at run time. Fixtures can't hardcode a relative path because pytest's
working directory isn't guaranteed to be the repo root.

### Golden snapshots need normalization to exist at all

Verified by diffing two runs, not assumed: **LLVM IR dumps are not
deterministic.** Non-scalar constants (strings, and the packed specs the
backtest lowering emits) cross into JIT code as boxed handles — `id()`
of a live Python object — so their numeric value is a memory address
that changes every run. A naive golden file would fail immediately and
always.

`_normalize` therefore folds two things out of every snapshot:

1. Integer literals at or above `2**32` become `<handle>`. That
   threshold is a documented heuristic — object addresses sit far above
   it, program literals far below — not a guarantee. It can never
   produce a false pass, since surrounding instruction structure is
   still compared exactly.
2. The absolute fixture path becomes `@FIXTURE@`, which otherwise bakes
   the generating machine's home directory into a committed file.

A test asserts the normalization actually fires, so it can't silently
become a no-op.

### Coverage

Floor of **85%** (current ~92%) in `pyproject.toml`, enforced in CI.
Phase 8.1 specified four `--cov` roots from an earlier layout; this repo
flattened them under `allie/`, so one source root covers the same code.

### Benchmarks

See `benchmarks/README.md` for methodology and the reading of the
numbers. Structurally: three workload families (quant, pure-CS, blended),
each defined twice — Allie and hand-written Python — and checked to
compute the same answer before any timing counts; five implementations
timed over interleaved trials, phase by phase; a scaling study that
separates fixed cost from cost per iteration; and bootstrap confidence
intervals on every ratio.

**Revised after v1.** The first harness timed with `tracemalloc` running,
which slowed allocation-heavy Python several-fold, and counted LLVM's
code generation as run time. Its headline — the JIT ~127× faster than
the interpreter and ~11× faster than Python on a loop — was mostly those
two errors. Measured properly, a JIT-compiled loop iteration is roughly
75× faster than CPython's, but compilation costs more than the 50k-iteration
workload takes to run, so its total time is ~3× CPython's.
