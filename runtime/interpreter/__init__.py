"""Phase 1+: tree-walking interpreter. Each `quantlang run` invocation constructs
a fresh Interpreter instance so concurrent/repeated runs have isolated state
(docs/spec.md "Concurrency model")."""
