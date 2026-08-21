# Roadmap

## Phase 0 — Foundations (done)

- [x] Directory skeleton (`allie/{lexer,parser,ast,semantic,typechecker,ir,optimizer,codegen,runtime,stdlib,quant}`, `tests/`, `examples/`, `docs/`, `benchmarks/`)
- [x] `.gitignore`, `LICENSE` (MIT), initial commit
- [x] CI (`.github/workflows/ci.yml`): ruff + mypy + pytest on every push, matrix over Python 3.11/3.12
- [x] Diagnostics infrastructure (`allie/errors.py`): `SourceSpan`, `Diagnostic`, collect-and-report-all `DiagnosticBag`
- [x] Runtime error hierarchy (`allie/runtime/errors/__init__.py`), distinct from compile-time diagnostics
- [x] CLI skeleton (`allie/cli.py`): `allie run FILE --dump-tokens --dump-ast --trace`
- [x] Canonical north-star example (`examples/hello_world.ql`)
- [x] Dependency management: `uv` + `uv.lock`, `pre-commit` wired to ruff/mypy
- [x] Design decisions locked in (`docs/spec.md`): execution model, typing, precision, syntax style, backtest loop, slippage model, calendar handling, determinism, concurrency, module system, statistical rigor, scope cut line

## Phase 1 — Tiny hand-built evaluator (done)

No lexer/parser yet — evaluation semantics validated on trees built by hand
in Python, so Phase 2's real parser output has a trusted evaluator waiting
for it.

- [x] Values (`allie/runtime/values/__init__.py`): raw `int`/`float`/`bool`/`str`, no wrapper class; `int op float -> float` promotion
- [x] `Environment` (`allie/runtime/environment.py`): parent chain, `define`/`get`/`assign`, shadowing, assign-to-outer-scope, assign-to-undefined raises rather than creating a global
- [x] Hand-built tree evaluator (`allie/runtime/interpreter/__init__.py`): `eval_expr`/`exec_stmt` over plain-tuple nodes
- [x] Short-circuit `and`/`or`, proven before Phase 2 has more moving parts
- [x] `for` loop, `break`/`continue` via internal `BreakSignal`/`ContinueSignal` exceptions
- [x] Functions as closures (params, body, defining env), `return` via `ReturnSignal`, closures capture by reference (proven in `test_closures_capture_environment_by_reference_not_by_value`)
- [x] Division/modulo by zero raise `AllieValueError`, not a bare Python exception
- [x] Tests (`tests/unit/test_phase1_eval.py`, `tests/unit/test_environment.py`): precedence, if/else, while, for, break/continue at known iterations, short-circuit, recursive factorial(5)/fibonacci(10), closures — 25 tests passing, ruff + mypy clean

## Project rename: QuantLang → Allie (done)

The language is called **Allie**. Flattened the package layout to match:
`compiler/*` and the `quantlang/` CLI wrapper merged into a single `allie/`
package (`allie.lexer`, `allie.runtime.interpreter`, `allie.cli`, etc.) —
no more `compiler/` nesting. `QuantLang*Error` classes renamed to
`Allie*Error`. CLI command is now `allie`. `pyproject.toml` project name,
entry point, and package list all updated; `uv.lock` regenerated.

## Phase 2 — Lexer & Parser (2.1 done, 2.2 grammar written, parser next)

### 2.1 Lexer (done)

- [x] `allie/lexer/tokens.py`: `TokenType` (50 members), `Token` dataclass
- [x] `allie/lexer/keywords.py`: `KEYWORD_ALIASES` (single lookup table, canonical + slang spellings -> identical `TokenType`; quant keywords canonical-only in v1), `render_keywords_markdown()`
- [x] `allie/lexer/lexer.py`: `Lexer.scan_tokens`/`scan_token` — numbers (digit required on both sides of `.`), strings with escapes, `//` comments, case-sensitive keyword lookup, invalid characters collected into `.errors` (scanning continues), unterminated string raises immediately
- [x] `allie/lexer/errors.py`: `LexError`
- [x] `scripts/generate_keywords_doc.py` + `docs/keywords.md` (generated, not hand-edited)
- [x] `tests/unit/test_lexer.py`: 104 tests — every `TokenType` in isolation, exact token sequence, number/string/comment edge cases, multi-char operators, invalid-character collection, line/column tracking, every `KEYWORD_ALIASES` entry parametrized, mixed canonical/goofy/quant keyword file, and a drift check against `docs/keywords.md`

Reconciled while building: `and`/`or` were in `TokenType` but missing from
the pasted `KEYWORD_ALIASES` example — added as canonical-only entries
(`logic_and`/`logic_or` need them as keyword tokens).

### 2.2 Grammar (done)

- [x] `docs/grammar.md` — full merged EBNF, written before `parser.py` exists. Reconciled two conflicts against already-committed decisions rather than silently picking a side: type annotations kept **optional** (pasted grammar had no `?`, which would contradict the locked gradual-typing decision and break `hello_world.ql`); generics kept as **`Series[float]`** (pasted grammar used `<float>`, but everything already committed uses brackets, which also avoid the `<`/`>`-vs-comparison ambiguity). Also fixed `factor`'s multiplication alternative, dropped in the paste, and added the `arguments` production referenced but never defined.
- `docs/grammar.ebnf` (Phase 0 placeholder) is now superseded by `docs/grammar.md`.

### Next: Parser

- [ ] Recursive-descent parser producing nodes in the exact tuple shape Phase 1's evaluator already consumes, recovering into `DiagnosticBag` instead of raising
- [ ] `--dump-tokens` / `--dump-ast` wired to real output
- Goal: `allie run examples/hello_world.ql --dump-ast` prints a full AST, and the Phase 1 evaluator can run it end to end

## Phases 3–13

Fill in from prior planning notes (semantic analysis / gradual typing,
the 5 indicators, event-driven backtest engine, walk-forward +
multiple-comparisons stats tooling, benchmarking, paper writeup, etc.) —
not reconstructed here since these phases weren't pasted into this
session. Whatever list existed before this scaffold should slot in
against the directories already created.
