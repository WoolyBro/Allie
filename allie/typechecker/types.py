"""Static types -- Phase 3.1, extended with the quant types in Phase 4.1.

Every Type subclass's equality/hash are derived generically from its own
__str__ (see the base Type class), and is_assignable's _WIDENING table is
keyed by (str, str) pairs rather than a hardcoded enumeration of "the
known types" -- exactly what let SeriesType/MatrixType/VectorType/
DateType/DateRangeType get added below (Phase 4) without reworking either
mechanism, as planned back in Phase 3.
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


class DateType(_SingletonType):
    _name = "Date"


class DateRangeType(_SingletonType):
    _name = "DateRange"


class PriceSeriesType(_SingletonType):
    """The type of stdlib `load`'s return value: an OHLCV price history
    (allie.runtime.values.PriceSeriesValue). Not in Phase 4.1's original
    type list, but needed to give `load` a checkable FunctionType
    signature -- added here rather than leaving `load` untyped. Not
    parameterized: always exactly open/high/low/close/volume/dates, so
    no elem type is needed, unlike Series/Matrix/Vector."""

    _name = "PriceSeries"


class StrategyType(_SingletonType):
    """The type of a declared `strategy` (Phase 7.2), so a `backtest`
    statement can verify its target through the ordinary
    TypeEnvironment lookup rather than a separate strategy registry.
    Deliberately NOT registered in _BASE_TYPES_BY_NAME: a strategy is
    introduced by the `strategy` construct, never written as a type
    annotation."""

    _name = "Strategy"


class AnyType(_SingletonType):
    """Top type: is_assignable(source, ANY) is always True (see
    is_assignable below). Exists only so `print`'s parameter can accept
    anything -- deliberately NOT registered in _BASE_TYPES_BY_NAME, so a
    user can't write `let x: Any` themselves. Exposing that would
    undermine gradual typing's whole point; this is a narrow, justified
    exception for one builtin, not a general escape hatch."""

    _name = "Any"


INT = IntType()
FLOAT = FloatType()
BOOL = BoolType()
STRING = StringType()
VOID = VoidType()
DATE = DateType()
DATE_RANGE = DateRangeType()
PRICE_SERIES = PriceSeriesType()
STRATEGY = StrategyType()
ANY = AnyType()


class ArrayType(Type):
    """No longer produced by array-literal expressions as of Phase 4 --
    those now default to SeriesType (see checker.py's
    `_check_array_literal` and docs/language-reference.md "Array
    literals default to Series"). Kept for a hypothetical future
    user-writable `Array[T]` annotation, which the grammar doesn't
    expose yet (there's no `Array` keyword in `baseType`)."""

    def __init__(self, elem: Type) -> None:
        self.elem = elem

    def __str__(self) -> str:
        return f"Array<{self.elem}>"


class SeriesType(Type):
    def __init__(self, elem: Type) -> None:
        self.elem = elem

    def __str__(self) -> str:
        return f"Series<{self.elem}>"


class MatrixType(Type):
    def __init__(self, elem: Type) -> None:
        self.elem = elem

    def __str__(self) -> str:
        return f"Matrix<{self.elem}>"


class VectorType(Type):
    def __init__(self, elem: Type) -> None:
        self.elem = elem

    def __str__(self) -> str:
        return f"Vector<{self.elem}>"


class FunctionType(Type):
    def __init__(self, params: tuple[Type, ...], ret: Type) -> None:
        self.params = params
        self.ret = ret

    def __str__(self) -> str:
        param_str = ", ".join(str(p) for p in self.params)
        return f"({param_str}) -> {self.ret}"


# ---- Assignability ----
#
# Keyed by (source_str, target_str) so adding a widening rule is a new
# row here, not a change to is_assignable's logic. See
# docs/language-reference.md "Assignability" for the table rendered for
# humans.
#
# Note there is deliberately NO row for (Float, Series<Float>) or
# similar: a bare scalar is never assignable to a Series, even though
# both carry Float as their element type. Series operations (SMA, mean,
# ...) are defined over an actual sequence of observations; silently
# accepting a scalar there would be exactly the class of bug this type
# distinction exists to catch (e.g. passing today's single closing price
# where a whole price history was expected). This falls out of
# is_assignable's existing logic with no extra code -- see
# test_typechecker_quant.py's
# test_scalar_not_assignable_to_series for the explicit regression test
# guarding against a future change accidentally adding such a rule.
_WIDENING: dict[tuple[str, str], bool] = {
    (str(INT), str(FLOAT)): True,
}


def is_assignable(source: Type, target: Type) -> bool:
    """ANY accepts anything (see AnyType). Otherwise: exact match is
    always OK; Int -> Float widens; everything else -- including
    Float -> Int narrowing, and any scalar -> Series/Matrix/Vector -- is
    rejected."""
    if target == ANY:
        return True
    if source == target:
        return True
    return _WIDENING.get((str(source), str(target)), False)


_BASE_TYPES_BY_NAME: dict[str, Type] = {
    "Int": INT,
    "Float": FLOAT,
    "Bool": BOOL,
    "String": STRING,
    "Void": VOID,
    "Date": DATE,
    "DateRange": DATE_RANGE,
    "PriceSeries": PRICE_SERIES,
}

# Generic quant types -- annotation syntax is `Name[ElemType]` (brackets,
# not angle brackets -- see docs/grammar.md's reconciliation notes).
_GENERIC_TYPES_BY_NAME: dict[str, type[SeriesType | MatrixType | VectorType]] = {
    "Series": SeriesType,
    "Matrix": MatrixType,
    "Vector": VectorType,
}


def resolve_base_type(name: str) -> Type | None:
    """Looks up a non-generic base type by its surface-syntax name (from
    a TypeAnnotation with no bracketed argument). Returns None for names
    not yet known."""
    return _BASE_TYPES_BY_NAME.get(name)


def resolve_generic_type(name: str, elem: Type) -> Type | None:
    """Looks up a generic quant type (Series/Matrix/Vector) by name,
    applied to an already-resolved element type. Returns None if `name`
    isn't one of the known generic type constructors."""
    constructor = _GENERIC_TYPES_BY_NAME.get(name)
    if constructor is None:
        return None
    return constructor(elem)
