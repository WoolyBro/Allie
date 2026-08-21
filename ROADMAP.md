# Roadmap

## Phase 0 — Foundations (done)

- [x] Directory skeleton (`compiler/`, `runtime/`, `stdlib/`, `quant/`, `tests/`, `examples/`, `docs/`, `benchmarks/`)
- [x] `.gitignore`, `LICENSE` (MIT), initial commit
- [x] CI (`.github/workflows/ci.yml`): ruff + mypy + pytest on every push, matrix over Python 3.11/3.12
- [x] Diagnostics infrastructure (`compiler/errors.py`): `SourceSpan`, `Diagnostic`, collect-and-report-all `DiagnosticBag`
- [x] Runtime error hierarchy (`runtime/errors/__init__.py`), distinct from compile-time diagnostics
- [x] CLI skeleton (`quantlang/cli.py`): `quantlang run FILE --dump-tokens --dump-ast --trace`
- [x] Canonical north-star example (`examples/hello_world.ql`)
- [x] Dependency management: `uv` + `uv.lock`, `pre-commit` wired to ruff/mypy
- [x] Design decisions locked in (`docs/spec.md`): execution model, typing, precision, syntax style, backtest loop, slippage model, calendar handling, determinism, concurrency, module system, statistical rigor, scope cut line

## Phase 1 — Lexer & Parser (next)

- Tokenizer for the brace-based grammar in `docs/grammar.ebnf`, emitting tokens with `SourceSpan`s
- Recursive-descent parser producing `compiler/ast` nodes, recovering from errors into the `DiagnosticBag` instead of raising
- `--dump-tokens` / `--dump-ast` wired to real output
- Goal: `quantlang run examples/hello_world.ql --dump-ast` prints a full AST (execution still stubbed)

## Phases 2–13

Fill in from prior planning notes (semantic analysis / gradual typing,
tree-walking interpreter, the 5 indicators, event-driven backtest engine,
walk-forward + multiple-comparisons stats tooling, benchmarking, paper
writeup, etc.) — not reconstructed here since Phase 0 is scoped to the
foundational decisions above. Whatever list existed before this scaffold
should slot in against the directories already created.
