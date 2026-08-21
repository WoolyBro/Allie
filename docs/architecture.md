# Architecture

Updated as each phase lands — one paragraph per stage, written when the
stage is actually implemented, not speculatively ahead of it.

## Pipeline

```
source (.ql)
    │
    ▼
  Lexer            qubit/lexer         — not yet implemented (Phase 2)
    │
    ▼
  Parser           qubit/parser        — not yet implemented (Phase 2)
    │
    ▼
  AST              qubit/ast           — not yet implemented (Phase 2)
    │
    ▼
  Semantic/        qubit/semantic,     — not yet implemented (Phase 3+)
  gradual typing   qubit/typechecker
    │
    ▼
  Interpreter      qubit/runtime/interpreter — Phase 1: hand-built trees
                                                only, no lexer/parser
                                                feeding it yet
    │
    ▼
  (values)         qubit/runtime/values, qubit/runtime/environment
```

`qubit/ir`, `qubit/optimizer`, and `qubit/codegen/{llvm_lowering,jit}` are
aspirational placeholders for a possible future execution backend — see
`docs/spec.md` "Considered alternatives". Nothing executes through them yet;
the interpreter runs directly off the AST (and, for now, off hand-built
Python tuples standing in for the AST).

## Phase 1 — hand-built evaluator

No lexer or parser exists yet. `qubit.runtime.interpreter` evaluates trees
built by hand in Python as tuples (`('binop', '+', ('num', 1), ('num', 2))`),
so the evaluation semantics can be gotten right and tested in isolation
before a real parser has to produce trees in this same shape. The parser
(Phase 2) targets this exact node shape — this evaluator does not change
when it lands.

**Values.** Represented as raw Python `int` / `float` / `bool` / `str` — no
wrapper class yet (`qubit/runtime/values/__init__.py`). A `Closure` value
(params, body, defining environment) is the only structured value so far,
for functions.

**Numeric promotion.** `int op float -> float`. Booleans and strings are
left alone by promotion (`+` on two strings concatenates instead).

**Scoping.** `qubit/runtime/environment.py`'s `Environment` is a
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
`qubit.runtime.errors.QubitValueError`, not a bare Python
`ZeroDivisionError` — user-facing errors go through the
`qubit.runtime.errors` hierarchy, never a leaked builtin exception (see
`docs/spec.md` "Runtime error philosophy").
