"""Static types -- Phase 3.1.

Designed so SeriesType/MatrixType/VectorType/DateType/DateRangeType
(Phase 4's quant types) can be added later without reworking
is_assignable or __eq__/__hash__: every Type subclass's equality and hash
are derived generically from its own __str__ (see the base Type class),
and is_assignable's _WIDENING table is keyed by (str, str) pairs rather
than a hardcoded enumeration of "the known types". Adding a new type
later means adding a subclass with its own __str__ and, if it widens to
or from something, a row in _WIDENING -- nothing existing changes shape.
"""

from __future__ import annotations


class Type:
    """Base class for every static type. __str__ (overridden by every
    subclass to match Allie's surface syntax, e.g. `Int`, `Array<Int>`)
    IS the type's identity for __eq__/__hash__, which gives structural
    equality "for free" on every subclass, including compound ones like
    ArrayType/FunctionType, without each needing its own __eq__."""

    def __str__(self) -> str:  # pragma: no cover - overridden everywhere
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Type) and str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


class _SingletonType(Type):
    """Base for the primitive types -- one shared instance each (INT,
    FLOAT, ...) below, so callers can compare by identity or by the
    inherited structural __eq__ interchangeably."""

    _name: str = ""

    def __str__(self) -> str:
        return self._name


class IntType(_SingletonType):
    _name = "Int"


class FloatType(_SingletonType):
    _name = "Float"


class BoolType(_SingletonType):
    _name = "Bool"


class StringType(_SingletonType):
    _name = "String"


class VoidType(_SingletonType):
    """A function that returns nothing. Needed because Allie's functions
    can omit '-> type' entirely for side-effecting functions, and the
    "does every path return" check (checker.py) only applies to
    non-Void functions -- distinguishing Void from every other type
    matters, unlike an earlier draft where every function implicitly
    returned something."""

    _name = "Void"


INT = IntType()
FLOAT = FloatType()
BOOL = BoolType()
STRING = StringType()
VOID = VoidType()


class ArrayType(Type):
    def __init__(self, elem: Type) -> None:
        self.elem = elem

    def __str__(self) -> str:
        return f"Array<{self.elem}>"


class FunctionType(Type):
    def __init__(self, params: tuple[Type, ...], ret: Type) -> None:
        self.params = params
        self.ret = ret

    def __str__(self) -> str:
        param_str = ", ".join(str(p) for p in self.params)
        return f"({param_str}) -> {self.ret}"


# ---- Assignability ----
#
# Keyed by (source_str, target_str) so a Phase 4 addition (e.g. an Int ->
# a hypothetical wider numeric quant type) is a new row here, not a
# change to is_assignable's logic. See docs/language-reference.md
# "Assignability" for the table rendered for humans.
_WIDENING: dict[tuple[str, str], bool] = {
    (str(INT), str(FLOAT)): True,
}


def is_assignable(source: Type, target: Type) -> bool:
    """Exact match is always OK. Int -> Float widens. Everything else --
    including Float -> Int narrowing -- is rejected."""
    if source == target:
        return True
    return _WIDENING.get((str(source), str(target)), False)


_BASE_TYPES_BY_NAME: dict[str, Type] = {
    "Int": INT,
    "Float": FLOAT,
    "Bool": BOOL,
    "String": STRING,
    "Void": VOID,
}


def resolve_base_type(name: str) -> Type | None:
    """Looks up a base type by its surface-syntax name (from a
    TypeAnnotation). Returns None for names not yet known -- Phase 4's
    Series/Matrix/Vector/Date/DateRange aren't registered here yet."""
    return _BASE_TYPES_BY_NAME.get(name)
