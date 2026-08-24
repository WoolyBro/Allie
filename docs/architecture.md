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
     │                                             gradual: optional annotations, enforced where present
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

**Does:** characters → tokens, each carrying line and column. Resolves
keyword spellings through one `KEYWORD_ALIASES` dict, so `ts` and `let`
both emit `TokenType.LET`.

**Why:** the alias table is the entire implementation of the slang
keywords feature. Putting it here — a single dict lookup with no
branching — means every later stage is automatically spelling-agnostic,
and adding an alias is one line.

**Does not:** branch on which spelling it saw, or expose that
information downstream. It also does not lowercase identifiers: lookup
is case-sensitive on purpose, so `Ahhh` is `while` and `ahhh` is an
identifier.

An invalid character is recorded and scanning continues (so one bad
character doesn't hide the rest of the file); an unterminated string
raises immediately, since there's nothing left to resynchronize on.

## Parser (`allie/parser/parser.py`)

**Does:** recursive descent, one method per grammar production,
precedence climbing for expressions. Produces the dataclass AST in
`allie/ast/nodes.py`.

**Why:** hand-written rather than generated — the grammar is small
enough that a parser generator would add a build step and a debugging
layer for no benefit.

**Does not:** type-check, resolve names, or evaluate anything. On error
it records a `Diagnostic` and resynchronizes to the next statement
boundary rather than raising, so one syntax error doesn't mask the next
five.

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

**Does:** gradual typing — annotations optional but enforced where
present. Annotates `inferred_type` on every expression node in place,
which is what makes LLVM codegen possible later (it needs to know
whether `+` is `add` or `fadd`).

**Why:** fail-fast (raises on first error) rather than collect-and-report
like the parser. A cascade of type errors from one root cause is rarely
more useful than the first one.

**Does not:** infer function signatures. Every parameter and return type
must be annotated, including `-> Void`. Partial inference that works
sometimes is harder to use than none.

**Does not:** allow truthiness. Conditions must be exactly `Bool` —
`if 1 { }` is an error, not a shorthand.

## IR lowering (`allie/ir/`)

**Does:** flattens the AST into a linear instruction list over virtual
registers, with `Label`/`Jump`/`JumpIfFalse` for control flow.

**Design decision — `for` gets no instructions of its own.** It
desugars into exactly the shape `while` produces. The payoff is that
LLVM lowering needed *zero* `for`-specific code. One subtlety: `for`'s
`continue` targets a dedicated increment label, not the condition check
— jump to the condition and the counter never advances. A golden
snapshot pins this exact shape.

**Design decision — flat list, no control-flow graph.** Simple, and
sufficient for lowering to LLVM (which builds real basic blocks from the
labels). The cost is that optimizer passes have to reason about
reachability without a CFG, which is where both bugs in
[`how-it-works.md`](how-it-works.md) §3.4 came from. A real CFG is the
main v2 change here.

**Does not:** support first-class function values. Functions are static
and name-registered, so a function returned from another function and
called later has no way to work through this path. Documented, and the
cross-validation suite is scoped around it.

## Optimizer (`allie/optimizer/`)

**Does:** constant folding, dead-code elimination, common subexpression
elimination. Each is a pure `IRProgram -> IRProgram` function; the pass
manager is a single on/off switch so benchmarks can measure the
difference.

**Does not:** remove any `Call`, ever (DCE). It has no way to know
whether a call is side-effecting without consulting purity, so it
assumes every call matters. CSE *does* consult `PURE_FUNCTIONS` —
defined once, in one module, so the two passes cannot disagree. A test
asserts every registered builtin is classified pure or impure, so a new
builtin can't silently default to "never optimized".

**Does not:** do proper dataflow analysis. DCE's liveness asks "is this
read anywhere in the program", not "read after this point" — strictly
more conservative, but immune to the loop back-edge trap that a naive
forward scan falls into. CSE invalidates its caches at every label, for
the same reason from the other direction.

## Backends

### Tree-walking interpreter (`allie/runtime/interpreter/`)

Runs straight off the AST, no IR. The default for `allie run`, and the
reference implementation the other two are checked against.

### IR interpreter (`allie/ir/interpreter.py`)

Executes the flat instruction list directly. Operator logic is
deliberately **re-implemented** rather than imported from the
tree-walker: sharing the helpers would let a bug common to both hide
from cross-validation, defeating the point of having two.

### LLVM + MCJIT (`allie/codegen/`)

Labels become basic blocks, jumps become `br`/`cbranch`. Scalar
arithmetic (`Int`/`Float`/`Bool`) compiles to native instructions.

**Design decision — Series/Matrix codegen is out of scope for v1.**
Everything touching a `Series`, `Matrix`, or `String` crosses into
Python as an opaque `i64` handle, and every stdlib call routes through
`runtime_bridge`. Reimplementing vectorized rolling windows in LLVM IR
to *maybe* match numpy is a large amount of work for a likely-zero gain
— the benchmarks confirm quant workloads are already ~99% numpy time.
The cost is real and bounded: a user-defined function taking a `Series`
parameter is rejected with a clear error rather than compiled.

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

**Errors.** `LexError` and `QSTypeError` carry line/column and format
identically: `{Kind}: {message}. Line {L}, Column {C}`. `AllieRuntimeError`
uses the same shape but usually has no location — most runtime errors are
raised inside stdlib functions operating on data ("mean: series must not
be empty"), where no source position exists. Threading spans through the
runtime is v2 work. `tests/unit/test_error_formats.py` pins the shared
format so it can't drift again.

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
instructions (`add`/`fadd`/`sdiv`/`srem`/`icmp_signed`/`fcmp_ordered`,
including `%`). Anything else routes through the bridge, which calls the
*same* `_apply_binop`/`_apply_unary` the IR interpreter uses — so there
is no third reimplementation of operator semantics to drift out of sync.

A call to a user-defined function with a fully-scalar signature becomes a
direct native LLVM `call` (this is also how recursion works). A
user-defined function with a `Series`/`Matrix`/... parameter or return
type raises `LLVMLoweringError` rather than attempting boxed
function-boundary marshalling — a deliberate, documented scope limit,
since every real program calls *stdlib* functions for non-scalar work,
never a user-defined one.

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

Two entry points share one pipeline (`allie.cli.run_pipeline`):

```
python main.py <file> [--debug-tokens] [--debug-ast] [--debug-ir] [--no-optimize] [--backend ...]
allie run <file> [--backend {tree,ir,llvm}] [--debug-tokens] [--debug-ast] [--debug-ir] [--no-optimize]
```

`main.py` implements the Phase 6.4 interface and defaults to the
LLVM/JIT backend. `allie run` is the pre-existing installed console
script and keeps its tree-walking default so nothing already written
against it breaks. `--debug-ir` prints the Allie IR and the final LLVM IR
as two clearly labeled sections. All flags are independent and
combinable.

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

### Position and accounting model (v1)

Flat-or-long, all-in, no shorting or leverage. `buy()` while already
long and `sell()` while flat are no-ops, not errors — a crossing firing
twice before the opposite signal is normal. Equity marks to market every
bar, not just on trade bars. **Risk-free rate is 0** for the Sharpe
ratio, stated explicitly since "Sharpe" alone doesn't say which
convention is in use. An entry still open at the final bar is marked to
market but not counted as a trade — it has no realized result yet.

### `Series<Bool>` is a distinct runtime type

`crosses_above`/`crosses_below` return `BoolSeriesValue`, not a
float64 `SeriesValue` holding 1.0/0.0. `SeriesValue` enforces float64 by
construction, so a boolean series stored there would either break that
invariant or silently become numeric data that `mean`/`SMA` would
happily — and meaninglessly — accept.

### `when` does not accept arbitrary boolean expressions

Decided explicitly rather than left open. The committed grammar
(`whenClause → "when" expression (crosses_above|crosses_below)
expression ":" statement`) *requires* a crossing operator, so supporting
a bare `Bool` would mean a real grammar/parser/lowering change — and an
arbitrary `Bool` has no natural long/flat interpretation without extra
syntax distinguishing entry from exit. Not in v1.

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

**One documented semantic difference**: the IR/LLVM paths evaluate
clause operands at the *declaration* site, while the tree walker defers
them to the `backtest` statement. For any program that computes its
indicators before declaring the strategy — every realistic one, and
every one in the test suite — both give identical results.

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

See `benchmarks/README.md` for methodology and the honest reading of the
numbers. Structurally: three workload families (quant, pure-CS, blended),
each defined twice — Allie and hand-written Python — timed over N trials
with the first discarded, compile and run measured separately, and the
optimizer toggled on/off as the only difference between two columns.

The pure-CS workload is what makes the headline result visible: the JIT
is ~142x faster than the tree-walking interpreter and ~12x faster than
hand-written Python on loop-heavy code. On quant workloads Allie is
*slower* than Python, because both bottom out in the same numpy/pandas
calls — reported plainly rather than omitted.
