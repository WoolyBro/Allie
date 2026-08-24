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

`allie/typechecker/checker.py`. Gradual typing: annotations are
optional, but where present they're enforced. `Int` widens to `Float`;
nothing narrows; a scalar is never assignable to a `Series`. Every
expression node gets its resolved type written back onto it — which is
what makes LLVM codegen possible later, since it needs to know whether
`+` is an `add` or an `fadd`.

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

`allie/optimizer/`. Constant folding, dead-code elimination, common
subexpression elimination. Each is a pure `IRProgram -> IRProgram`
function; `--no-optimize` skips them. §3.4 is the interesting one.

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
  %".10" = mul i64 %"r9", 2
  %".11" = add i64 %"r8", %".10"
  store i64 %".11", i64* %"total_ptr"
  br label %"for_increment_1"
for_increment_1:
  %"r13" = load i64, i64* %"i_ptr"
  %".14" = add i64 %"r13", 1
  store i64 %".14", i64* %"i_ptr"
  br label %"for_loop_0"
for_end_2:
  %"r16" = load i64, i64* %"total_ptr"
  %".17" = call i64 @"box_int"(i64 %"r16")
  %".18" = call i64 @"call_builtin_bridge"(i64 2757324001792, ...)
  ret void
cont:
  %"r5" = load i64, i64* %"i_ptr"
  %".7" = icmp eq i64 %"r5", 2
  br i1 %".7", label %"cont.1", label %"if_end_4"
cont.1:
  br label %"for_increment_1"
}
```

The arithmetic is native `mul`/`add`/`icmp` on `i64`. The `print` call
isn't — it goes through `call_builtin_bridge` back into Python (§3.3).
That `2757324001792` is a pointer to a Python object, which is why LLVM
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

Is that a cop-out? Genuinely no, and the benchmarks say why: on quant
workloads, ~99% of runtime is already inside numpy's C code. Compiling
the thin Allie wrapper around a `numpy.mean` call can't speed up
`numpy.mean`. Reimplementing vectorized rolling windows in LLVM IR to
maybe match numpy is a large amount of work for a likely-zero gain.

Meanwhile the part where compilation *does* pay — scalar arithmetic and
control flow — is fully native, and that's exactly where the 127×
speedup shows up (§4).

The honest cost: a user-defined function taking a `Series` parameter
can't compile, and raises a clear error instead of silently misbehaving.
Every real program calls *stdlib* functions for series work, so this
hasn't bitten in practice — but it's a real limit, not a rounding error.

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

The fix here is deliberately blunt: liveness asks "is this read
*anywhere* in the program," not "read after this point." Position stops
mattering, so back-edges stop being a special case. That is strictly
more conservative than a proper dataflow analysis — it won't spot that
the first write in `x = 1; x = 2; print(x);` is dead — and given the IR
is a flat list with no control-flow graph, that's the honest trade.
There's a regression test with that exact loop shape.

A related one turned up in CSE, which had the same blind spot from the
other direction: it was reusing a register computed *before* a loop for
a read *inside* it, which is only valid if nothing in the loop
reassigns it. Fixed by invalidating the caches at every label —
i.e. at every place control flow can merge.

Both bugs share a root cause: a flat instruction list makes it easy to
write a pass that reasons about text order when it should be reasoning
about control flow.

---

## 4. Benchmarks

Full table and methodology in
[`benchmarks/README.md`](../benchmarks/README.md); regenerate with
`make bench`.

| Workload | Family | Python | Allie (interp) | Allie (JIT, unopt) | Allie (JIT, opt) | LOC (Py) | LOC (Allie) |
|---|---|---|---|---|---|---|---|
| `sma_10k` | quant | 104.40 | 199.71 | 227.00 | 221.21 | 6 | 6 |
| `rolling_std` | quant | 21.79 | 197.62 | 194.80 | 186.80 | 6 | 6 |
| `backtest_ma` | quant | 2.70 | 3.02 | 12.93 | 12.89 | 20 | 8 |
| `matmul_200` | quant | 1.51 | n/a | n/a | n/a | 3 | n/a |
| `loop_sum` | pure-cs | 72.19 | 834.57 | 6.65 | 6.57 | 6 | 7 |
| `blended_sweep` | blended | 7.04 | 4.85 | 14.95 | 14.76 | 8 | 7 |

Three things this shows.

**Compilation is worth a lot for control flow.** `loop_sum` — a plain
50k-iteration loop, no quant content — goes from 835 ms interpreted to
6.6 ms JIT-compiled. That's ~127×, and ~11× faster than the equivalent
hand-written Python. This is the number the project actually earns.

**Compilation is worth nothing for library-bound work.** On the quant
rows Allie is *slower* than Python: both bottom out in the same numpy
calls, and Allie pays front-end and bridge overhead on top. Expected,
given §3.3. The interesting column there is LOC — the backtest is 8
lines of Allie against 20 of Python.

**The optimizer barely moves the needle: low single-digit percent, near
noise.** The passes act on scalar IR, and these workloads spend their
time inside bridged calls the passes deliberately don't touch. The unit
tests prove the transformations fire on hand-built IR; this table shows
what that's worth on this mix of workloads. Reporting ~2% honestly beats
constructing a workload that flatters it.

(`rolling_std` used to be 112× slower here — a real finding the
benchmark surfaced, and a separate bug from anything compiler-related:
the implementation was a per-window Python loop where pandas is
vectorized. Rewriting it with `sliding_window_view` cut the gap to ~9×
with bit-identical results. Worth noting the *tempting* fix — cumulative
sums of `x` and `x²` — was rejected: it's faster still, but loses ~1e-08
accuracy to catastrophic cancellation on exactly the large-mean,
small-variance data this function exists to handle.)

---

## 5. What's not done

Stated as v2 work, not apologies.

**Flat IR, no control-flow graph.** Instructions are one list; basic
blocks only appear at LLVM lowering. This is what made both bugs in §3.4
possible, and it's why DCE has to be conservative. A real CFG with
proper dataflow analysis would make the passes both safer and stronger.

**LLVM codegen is scalar-only.** Series/Matrix operations bridge back to
Python (§3.3). A user-defined function with a `Series` parameter is
rejected at compile time rather than compiled.

**`import` parses but resolves nothing.** `import "foo";` type-checks and
is then ignored — there's no module loader, no path resolution, no
cross-file name binding. It's a parsed placeholder, deliberately inert
rather than half-working.

**`break`/`continue` can't escape a `when` action.** A `when` body is a
callback the backtest engine invokes per bar, so there's no enclosing
loop to jump out of. Rejected at resolve time with a specific message. A
loop written *inside* a `when` action is fine.

**Runtime errors mostly lack line numbers.** `LexError` and `QSTypeError`
carry line/column; runtime errors usually don't, because they're raised
inside stdlib functions operating on data ("mean: series must not be
empty") where no source position is available. Threading spans through
the runtime is a v2 change.

**No `Matrix` literal syntax.** `Matrix[Float]` exists as a type and the
matrix stdlib works, but nothing in the language can *construct* one —
array literals produce `Series`. The matrix benchmark has no Allie
implementation for exactly this reason, and says so.

**Assignment targets only bare names.** No `arr[i] = x`, no
`obj.field = x`.
