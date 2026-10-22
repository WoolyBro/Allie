"""Static types.

Every Type subclass's equality/hash are derived generically from its own
__str__ (see the base Type class), and is_assignable's _WIDENING table is
keyed by (str, str) pairs rather than a hardcoded enumeration of "the
known types" -- which is what lets a new type (the series and matrix
types, the quant types below) be added without reworking either
mechanism.
"""

from __future__ import annotations


class Type:
    """Base class for every static type. __str__ (overridden by every
    subclass to match Allie's surface syntax, e.g. `Int`, `Series<Int>`)
    IS the type's identity for __eq__/__hash__, which gives structural
    equality "for free" on every subclass, including compound ones like
    SeriesType/FunctionType, without each needing its own __eq__."""

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
    (allie.runtime.values.PriceSeriesValue), which gives `load` a
    checkable signature and `AAPL.close` a checkable field. Not
    parameterized: always exactly open/high/low/close/volume/dates, so
    no elem type is needed, unlike Series/Matrix/Vector."""

    _name = "PriceSeries"


class BacktestResultType(_SingletonType):
    """What a `backtest` expression evaluates to: its metrics (`sharpe`,
    `cagr`, `max_drawdown`, ...), trade count, and per-bar equity and
    returns -- so results can be compared, corrected for multiple
    testing, and fed to the significance functions in source."""

    _name = "BacktestResult"


class StrategyType(_SingletonType):
    """The type of a declared `strategy`, so a `backtest`
    statement can verify its target through the ordinary
    TypeEnvironment lookup rather than a separate strategy registry.
    Deliberately NOT registered in _BASE_TYPES_BY_NAME: a strategy is
    introduced by the `strategy` construct, never written as a type
    annotation."""

    _name = "Strategy"


class AnyType(_SingletonType):
    """Top type: is_assignable(source, ANY) is always True (see
    is_assignable below). Exists only for builtin signatures: `print`
    accepts anything, and `len`'s `Series<Any>` accepts a series of any
    element type. Deliberately NOT registered in _BASE_TYPES_BY_NAME, so
    a user can't write `let x: Any` themselves. Exposing that would
    undermine the point of static typing; this is a narrow, justified
    exception for builtins, not a general escape hatch."""

    _name = "Any"


class ElementwiseType(_SingletonType):
    """A builtin parameter that takes a number or a numeric series, for
    the math functions that apply to either (`log`, `sqrt`, `exp`). An
    Int argument is widened to Float; the result, the type checker's
    rule, is a Float for a number and a Series<Float> for a series. Like
    ANY, never written in source."""

    _name = "Float|Series<Float>"


class ElementwiseIntType(_SingletonType):
    """ElementwiseType for a function that keeps an Int an Int (`abs`)."""

    _name = "Int|Float|Series<Float>"


class ErrorType(_SingletonType):
    """The type of an expression that failed to type-check. Assignable
    to and from everything, so every check involving it passes silently:
    one mistake produces one error instead of a cascade of consequences.
    Never appears in a program that type-checks."""

    _name = "<error>"


INT = IntType()
FLOAT = FloatType()
BOOL = BoolType()
STRING = StringType()
VOID = VoidType()
DATE = DateType()
DATE_RANGE = DateRangeType()
PRICE_SERIES = PriceSeriesType()
BACKTEST_RESULT = BacktestResultType()
STRATEGY = StrategyType()
ANY = AnyType()
ELEMENTWISE = ElementwiseType()
ELEMENTWISE_INT = ElementwiseIntType()
ERROR = ErrorType()


class _GenericType(Type):
    """Base for the parameterized quant types. Mirrors `_SingletonType`
    above: one `_name` per subclass, everything else shared. The three
    subclasses were previously identical copies differing only in a
    literal inside `__str__`."""

    _name: str = ""

    def __init__(self, elem: Type) -> None:
        self.elem = elem

    def __str__(self) -> str:
        return f"{self._name}<{self.elem}>"


class SeriesType(_GenericType):
    _name = "Series"


class MatrixType(_GenericType):
    _name = "Matrix"


class VectorType(_GenericType):
    _name = "Vector"


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
# Series widen element-wise (Series<Int> -> Series<Float>), handled in
# is_assignable below rather than as a row here. That is sound only
# because series are immutable values: a mutable container must be
# invariant, or code holding it as Series<Float> could write a Float into
# what another reference still believes is a Series<Int>.
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
    """ANY accepts anything (see AnyType) and ERROR is compatible with
    everything (see ErrorType). Otherwise: exact match is always OK;
    Int -> Float widens, and so does Series<Int> -> Series<Float>;
    everything else -- including Float -> Int narrowing, and any
    scalar -> Series/Matrix/Vector -- is rejected."""
    if target == ANY or source == ERROR or target == ERROR:
        return True
    if source == target:
        return True
    if target in (ELEMENTWISE, ELEMENTWISE_INT):
        numeric = (INT, FLOAT)
        return source in numeric or (isinstance(source, SeriesType) and source.elem in numeric)
    if isinstance(source, SeriesType) and isinstance(target, SeriesType):
        return is_assignable(source.elem, target.elem)
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
    "BacktestResult": BACKTEST_RESULT,
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
