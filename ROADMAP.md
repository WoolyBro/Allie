# Roadmap

## Phase 0 — Foundations (done)

- [x] Directory skeleton (`qubit/{lexer,parser,ast,semantic,typechecker,ir,optimizer,codegen,runtime,stdlib,quant}`, `tests/`, `examples/`, `docs/`, `benchmarks/`)
- [x] `.gitignore`, `LICENSE` (MIT), initial commit
- [x] CI (`.github/workflows/ci.yml`): ruff + mypy + pytest on every push, matrix over Python 3.11/3.12
- [x] Diagnostics infrastructure (`qubit/errors.py`): `SourceSpan`, `Diagnostic`, collect-and-report-all `DiagnosticBag`
- [x] Runtime error hierarchy (`qubit/runtime/errors/__init__.py`), distinct from compile-time diagnostics
- [x] CLI skeleton (`qubit/cli.py`): `qubit run FILE --dump-tokens --dump-ast --trace`
- [x] Canonical north-star example (`examples/hello_world.ql`)
- [x] Dependency management: `uv` + `uv.lock`, `pre-commit` wired to ruff/mypy
- [x] Design decisions locked in (`docs/spec.md`): execution model, typing, precision, syntax style, backtest loop, slippage model, calendar handling, determinism, concurrency, module system, statistical rigor, scope cut line

## Phase 1 — Tiny hand-built evaluator (done)

No lexer/parser yet — evaluation semantics validated on trees built by hand
in Python, so Phase 2's real parser output has a trusted evaluator waiting
for it.

- [x] Values (`qubit/runtime/values/__init__.py`): raw `int`/`float`/`bool`/`str`, no wrapper class; `int op float -> float` promotion
- [x] `Environment` (`qubit/runtime/environment.py`): parent chain, `define`/`get`/`assign`, shadowing, assign-to-outer-scope, assign-to-undefined raises rather than creating a global
- [x] Hand-built tree evaluator (`qubit/runtime/interpreter/__init__.py`): `eval_expr`/`exec_stmt` over plain-tuple nodes
- [x] Short-circuit `and`/`or`, proven before Phase 2 has more moving parts
- [x] `for` loop, `break`/`continue` via internal `BreakSignal`/`ContinueSignal` exceptions
- [x] Functions as closures (params, body, defining env), `return` via `ReturnSignal`, closures capture by reference (proven in `test_closures_capture_environment_by_reference_not_by_value`)
- [x] Division/modulo by zero raise `QubitValueError`, not a bare Python exception
- [x] Tests (`tests/unit/test_phase1_eval.py`, `tests/unit/test_environment.py`): precedence, if/else, while, for, break/continue at known iterations, short-circuit, recursive factorial(5)/fibonacci(10), closures — 25 tests passing, ruff + mypy clean

## Project rename: QuantLang → Qubit (done)

The language is called **Qubit**. Flattened the package layout to match:
`compiler/*` and the `quantlang/` CLI wrapper merged into a single `qubit/`
package (`qubit.lexer`, `qubit.runtime.interpreter`, `qubit.cli`, etc.) —
no more `compiler/` nesting. `QuantLang*Error` classes renamed to
`Qubit*Error`. CLI command is now `qubit`. `pyproject.toml` project name,
entry point, and package list all updated; `uv.lock` regenerated.

## Phase 2 — Lexer & Parser (in progress)

Spec received for **2.1 Lexer** (`qubit/lexer/`): token types, a
`KEYWORD_ALIASES` table mapping both canonical and slang spellings to the
same `TokenType` (single lookup table, no branching on which spelling was
used), `Lexer`/`scan_tokens`/`scan_token`, numbers/strings/comments,
case-sensitive keyword lookup, multi-error collection on invalid
characters, and a matching `tests/unit/test_lexer.py`. Not yet
implemented — building next.

**2.2 Grammar (`docs/grammar.md`)** was flagged as "write before coding the
parser" but the actual merged EBNF content was cut off mid-paste — still
waiting on that before finalizing `docs/grammar.md`. `docs/grammar.ebnf`
(Phase 0 placeholder) stays in place until it's superseded.

- [ ] `qubit/lexer/tokens.py`, `qubit/lexer/keywords.py`, `qubit/lexer/lexer.py`, `qubit/lexer/errors.py`
- [ ] `docs/keywords.md` generated from `KEYWORD_ALIASES`
- [ ] `docs/grammar.md` — blocked on the full EBNF from the user
- [ ] `tests/unit/test_lexer.py`
- [ ] Parser producing nodes in the exact tuple shape Phase 1's evaluator already consumes, recovering into `DiagnosticBag` instead of raising
- [ ] `--dump-tokens` / `--dump-ast` wired to real output
- Goal: `qubit run examples/hello_world.ql --dump-ast` prints a full AST, and the Phase 1 evaluator can run it end to end

## Phases 3–13

Fill in from prior planning notes (semantic analysis / gradual typing,
the 5 indicators, event-driven backtest engine, walk-forward +
multiple-comparisons stats tooling, benchmarking, paper writeup, etc.) —
not reconstructed here since these phases weren't pasted into this
session. Whatever list existed before this scaffold should slot in
against the directories already created.
