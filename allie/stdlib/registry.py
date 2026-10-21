"""Single source of truth for builtin functions -- BUILTINS maps a name
to (FunctionType, Callable), read by both the type checker (call-site
type-checking) and the interpreter (execution), so registering a new
builtin is one entry here, not edits across multiple files.

Importing allie.stdlib (not just this module) triggers every submodule's
registration as a side effect -- see allie/stdlib/__init__.py.

Every registered function is wrapped so that it fails the way Allie
code fails, on every backend: a numpy/Python `ValueError` or
`ArithmeticError` escaping a builtin (a negative std handed to the RNG,
a singular matrix) becomes an `AllieValueError` naming the function,
instead of a Python traceback. And floating-point events inside a
builtin follow IEEE 754 silently -- `pct_change` across a zero is `inf`
for that bar, as `/` on series is -- rather than printing numpy's
RuntimeWarnings to stderr in the middle of a program's output.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np

from allie.runtime.environment import Environment
from allie.runtime.errors import AllieRuntimeError, AllieValueError
from allie.runtime.values import BuiltinFunction
from allie.typechecker.types import FunctionType

BUILTINS: dict[str, tuple[FunctionType, Callable[..., object]]] = {}


def register(name: str, signature: FunctionType, fn: Callable[..., object]) -> None:
    """Registers a builtin. Raises if `name` is already registered --
    catches an accidental duplicate at import time rather than silently
    letting the second registration win."""
    if name in BUILTINS:
        raise ValueError(f"builtin '{name}' is already registered")
    BUILTINS[name] = (signature, _as_allie_function(name, fn))


def _as_allie_function(name: str, fn: Callable[..., object]) -> Callable[..., object]:
    @functools.wraps(fn)
    def call(*args: object) -> object:
        try:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                return fn(*args)
        except AllieRuntimeError:
            raise
        except (ValueError, ArithmeticError) as error:
            raise AllieValueError(f"{name}: {error}") from error

    return call


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
    the Callable implementations. Names starting with `__` are internal
    lowering targets (e.g. the `__index_*` functions behind `s[i]`) and
    are left out, so source code can't call them by name."""
    return {
        name: signature for name, (signature, _) in BUILTINS.items() if not name.startswith("__")
    }
