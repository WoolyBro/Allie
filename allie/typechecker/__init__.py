"""Static type checker. Every expression gets a type: a `let` without
an annotation takes its initializer's, one with an annotation must match
it, and functions declare their signatures. See
docs/language-reference.md "Types"."""

from allie.typechecker.checker import TypeChecker, TypeEnvironment
from allie.typechecker.errors import QSTypeError
from allie.typechecker.types import (
    BOOL,
    FLOAT,
    INT,
    STRING,
    VOID,
    FunctionType,
    Type,
    is_assignable,
    resolve_base_type,
)

__all__ = [
    "BOOL",
    "FLOAT",
    "INT",
    "STRING",
    "VOID",
    "FunctionType",
    "QSTypeError",
    "Type",
    "TypeChecker",
    "TypeEnvironment",
    "is_assignable",
    "resolve_base_type",
]
