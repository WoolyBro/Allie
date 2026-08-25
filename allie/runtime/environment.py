"""Lexical scoping, shared by the Phase 1 evaluator and every phase after it.

`assign()` walks the parent chain looking for an EXISTING binding and
mutates it in place -- it never creates a new global as a side effect of a
typo. Use `define()` to introduce a new binding (which shadows any binding
of the same name in an ancestor frame).
"""

from __future__ import annotations

from allie.runtime.errors import AllieNameError
from allie.runtime.values import Value


class Environment:
    def __init__(self, parent: Environment | None = None) -> None:
        self.parent = parent
        self.vars: dict[str, Value] = {}

    def define(self, name: str, value: Value) -> None:
        self.vars[name] = value

    def get(self, name: str) -> Value:
        env: Environment | None = self
        while env is not None:
            if name in env.vars:
                return env.vars[name]
            env = env.parent
        raise AllieNameError(f"undefined name: {name!r}")

    def assign(self, name: str, value: Value) -> None:
        env: Environment | None = self
        while env is not None:
            if name in env.vars:
                env.vars[name] = value
                return
            env = env.parent
        raise AllieNameError(f"cannot assign to undefined name: {name!r}")
