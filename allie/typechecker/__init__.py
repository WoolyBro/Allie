"""Phase 3: static type checker. Validates optional annotations
(`Series[Float]`-style); unannotated `let` bindings stay dynamic
(inferred from the initializer). See docs/spec.md "Typing"."""

from allie.typechecker.checker import TypeChecker, TypeEnvironment
from allie.typechecker.errors import QSTypeError
from allie.typechecker.types import (
    BOOL,
    FLOAT,
    INT,
    STRING,
    VOID,
    ArrayType,
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
    "ArrayType",
    "FunctionType",
    "QSTypeError",
    "Type",
    "TypeChecker",
    "TypeEnvironment",
    "is_assignable",
    "resolve_base_type",
]
