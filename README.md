# Allie

A gradually-typed DSL for quantitative trading research, with an
event-driven backtest engine that structurally prevents look-ahead bias.

Design decisions live in [`docs/spec.md`](docs/spec.md) — read that before
touching the lexer/parser/interpreter. The grammar target is
[`docs/grammar.ebnf`](docs/grammar.ebnf). [`examples/hello_world.ql`](examples/hello_world.ql)
is the north star: every phase should get incrementally closer to running it.

## Status

Phase 1 (this commit): a hand-built tree-walking evaluator
(`allie.runtime.interpreter`) validated against trees built by hand in
Python — see `docs/architecture.md`. No lexer/parser yet; that's Phase 2.

## Setup

```bash
uv sync --group dev
uv run pytest
```

## CLI

```bash
uv run allie run path/to/file.ql [--dump-tokens] [--dump-ast] [--trace]
```
