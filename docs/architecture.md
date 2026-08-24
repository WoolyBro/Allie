# Architecture

Updated as each phase lands — one paragraph per stage, written when the
stage is actually implemented, not speculatively ahead of it.

## Pipeline

```
source (.ql)
    │
    ▼
  Lexer            allie/lexer         — not yet implemented (Phase 2)
    │
    ▼
  Parser           allie/parser        — not yet implemented (Phase 2)
    │
    ▼
  AST              allie/ast           — not yet implemented (Phase 2)
    │
    ▼
  Semantic/        allie/semantic,     — not yet implemented (Phase 3+)
  gradual typing   allie/typechecker
    │
    ▼
  Interpreter      allie/runtime/interpreter — Phase 1: hand-built trees
                                                only, no lexer/parser
                                                feeding it yet
    │
    ▼
  (values)         allie/runtime/values, allie/runtime/environment
```

`allie/ir`, `allie/optimizer`, and `allie/codegen/{llvm_lowering,jit}` are
aspirational placeholders for a possible future execution backend — see
`docs/spec.md` "Considered alternatives". Nothing executes through them yet;
the interpreter runs directly off the AST (and, for now, off hand-built
Python tuples standing in for the AST).

## Phase 1 — hand-built evaluator

No lexer or parser exists yet. `allie.runtime.interpreter` evaluates trees
built by hand in Python as tuples (`('binop', '+', ('num', 1), ('num', 2))`),
so the evaluation semantics can be gotten right and tested in isolation
before a real parser has to produce trees in this same shape. The parser
(Phase 2) targets this exact node shape — this evaluator does not change
when it lands.

**Values.** Represented as raw Python `int` / `float` / `bool` / `str` — no
wrapper class yet (`allie/runtime/values/__init__.py`). A `Closure` value
(params, body, defining environment) is the only structured value so far,
for functions.

**Numeric promotion.** `int op float -> float`. Booleans and strings are
left alone by promotion (`+` on two strings concatenates instead).

**Scoping.** `allie/runtime/environment.py`'s `Environment` is a
parent-linked chain of `dict`s. `define()` introduces a binding in the
*current* frame (enabling shadowing); `assign()` walks up to find an
*existing* binding and mutates it in place, raising rather than silently
creating a new global. Every `{ ... }` block gets its own child
`Environment`.

**Control flow.** `break`/`continue`/`return` are implemented as internal
exceptions (`BreakSignal`/`ContinueSignal`/`ReturnSignal`) raised by their
respective statement nodes and caught by the nearest enclosing loop or call
frame. This is the same mechanism Phase 2's real interpreter reuses
unchanged — proven here, on hand-built trees, before a parser adds more
moving parts.

**Short-circuit `and`/`or`.** Special-cased in `eval_expr` *before*
evaluating the right operand — not implemented as an eager `binop`. Returns
one of the two operand values (Python/JS-style), not a coerced `bool`.

**Closures.** Capture their defining `Environment` *by reference*, not a
snapshot of its values at definition time. A closure's captured variables
reflect later mutations to that environment (proven in
`tests/unit/test_phase1_eval.py`).

**Runtime errors.** Division/modulo by zero raise
`allie.runtime.errors.AllieValueError`, not a bare Python
`ZeroDivisionError` — user-facing errors go through the
`allie.runtime.errors` hierarchy, never a leaked builtin exception (see
`docs/spec.md` "Runtime error philosophy").

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
