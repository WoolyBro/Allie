"""Single source of truth for builtin functions -- BUILTINS maps a name
to (FunctionType, Callable), read by both the type checker (call-site
type-checking) and the interpreter (execution), so registering a new
builtin is one entry here, not edits across multiple files.

Importing allie.stdlib (not just this module) triggers every submodule's
registration as a side effect -- see allie/stdlib/__init__.py.
"""

from __future__ import annotations

from collections.abc import Callable

from allie.runtime.environment import Environment
from allie.runtime.values import BuiltinFunction
from allie.typechecker.types import FunctionType

BUILTINS: dict[str, tuple[FunctionType, Callable[..., object]]] = {}


def register(name: str, signature: FunctionType, fn: Callable[..., object]) -> None:
    """Registers a builtin. Raises if `name` is already registered --
    catches an accidental duplicate at import time rather than silently
    letting the second registration win."""
    if name in BUILTINS:
        raise ValueError(f"builtin '{name}' is already registered")
    BUILTINS[name] = (signature, fn)


def create_global_environment() -> Environment:
    """A fresh Environment with every builtin pre-bound as a
    BuiltinFunction value. Called once per `allie run` invocation, never
    shared/mutated across runs (docs/spec.md "Concurrency model")."""
    env = Environment()
    for name, (_, fn) in BUILTINS.items():
        env.define(name, BuiltinFunction(name=name, fn=fn))
    return env


def builtin_function_types() -> dict[str, FunctionType]:
    """The subset the type checker needs: name -> FunctionType, without
    the Callable implementations."""
    return {name: signature for name, (signature, _) in BUILTINS.items()}
