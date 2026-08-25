"""Tests for Phase 4.1: the quant type extensions to allie/typechecker/."""

from __future__ import annotations

import pytest

from allie.ast.nodes import TypeAnnotation
from allie.errors import SourceSpan
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.typechecker.checker import TypeChecker
from allie.typechecker.errors import QSTypeError
from allie.typechecker.types import (
    BOOL,
    FLOAT,
    INT,
    DateRangeType,
    DateType,
    MatrixType,
    SeriesType,
    VectorType,
    is_assignable,
)


def _check(source: str) -> None:
    tokens = Lexer(source).scan_tokens()
    program = Parser(tokens).parse()
    TypeChecker().check(program)


def _initializer_type(source: str) -> object:
    tokens = Lexer(f"let __t = {source};").scan_tokens()
    program = Parser(tokens).parse()
    TypeChecker().check(program)
    return program.statements[0].initializer.inferred_type  # type: ignore[attr-defined]


# ---- generic annotation resolution ----


def test_series_float_annotation_resolves() -> None:
    tokens = Lexer("let x: Series[Float] = [1.0, 2.0];").scan_tokens()
    program = Parser(tokens).parse()
    TypeChecker().check(program)
    annotation_type = program.statements[0].type_annotation  # type: ignore[attr-defined]
    assert annotation_type.name == "Series"


def test_matrix_and_vector_annotations_resolve() -> None:
    # Array literals always infer as Series (see below), so there's no
    # expression that currently produces a Matrix/Vector value to
    # assign -- test annotation resolution directly instead of going
    # through a full let-statement.
    span = SourceSpan(line=1, column=1, offset=0)
    checker = TypeChecker()
    matrix_annotation = TypeAnnotation(
        name="Matrix", span=span, type_arg=TypeAnnotation(name="Float", span=span)
    )
    vector_annotation = TypeAnnotation(
        name="Vector", span=span, type_arg=TypeAnnotation(name="Int", span=span)
    )
    assert str(checker._resolve_type_annotation(matrix_annotation)) == "Matrix<Float>"
    assert str(checker._resolve_type_annotation(vector_annotation)) == "Vector<Int>"


def test_date_and_daterange_are_plain_base_types() -> None:
    assert isinstance(DateType(), DateType)
    assert isinstance(DateRangeType(), DateRangeType)


def test_unknown_generic_name_fails() -> None:
    with pytest.raises(QSTypeError):
        _check("let x: Foo[Float] = [1.0];")


# ---- assignability ----


def test_scalar_not_assignable_to_series() -> None:
    # Deliberately no widening rule for scalar -> Series -- see
    # types.py's _WIDENING comment. Regression guard: this must stay
    # rejected even if someone "helpfully" adds a row later.
    assert not is_assignable(FLOAT, SeriesType(FLOAT))
    assert not is_assignable(INT, SeriesType(INT))


def test_let_series_from_bare_float_fails() -> None:
    with pytest.raises(QSTypeError):
        _check("let x: Series[Float] = 1.0;")


def test_series_int_not_assignable_to_series_float() -> None:
    # Series<Int> -> Series<Float> does NOT widen even though Int -> Float
    # does for bare scalars -- is_assignable only widens exact primitive
    # pairs in _WIDENING, and Series<Int> != Series<Float> structurally.
    assert not is_assignable(SeriesType(INT), SeriesType(FLOAT))


# ---- array literals default to Series ----


def test_array_literal_infers_series_of_element_type() -> None:
    assert str(_initializer_type("[1, 2, 3]")) == "Series<Int>"
    assert str(_initializer_type("[1.0, 2.0]")) == "Series<Float>"


def test_array_literal_matches_series_annotation() -> None:
    _check("let x: Series[Int] = [1, 2, 3];")


def test_array_literal_mismatched_elements_fails() -> None:
    with pytest.raises(QSTypeError):
        _check('let x = [1, "two"];')


def test_matrix_and_vector_types_are_distinct_from_series() -> None:
    assert MatrixType(FLOAT) != SeriesType(FLOAT)
    assert VectorType(FLOAT) != SeriesType(FLOAT)
    assert str(MatrixType(FLOAT)) == "Matrix<Float>"
    assert str(VectorType(FLOAT)) == "Vector<Float>"


# ---- when-clause requires Series<Float> ----


def test_when_clause_with_series_float_operands_passes() -> None:
    # Body is a plain expression statement, not a builtin call -- the
    # stdlib registry (Phase 4.3) doesn't exist yet, so `buy()` would
    # fail as an undefined name for reasons unrelated to what this test
    # is checking (the operand types, not call resolution).
    _check(
        """
        strategy S {
            when [1.0, 2.0] crosses_above [1.0, 2.0]: 1;
        }
        """
    )


def test_when_clause_with_scalar_operand_fails() -> None:
    with pytest.raises(QSTypeError):
        _check(
            """
            strategy S {
                when 1.0 crosses_above [1.0, 2.0]: buy();
            }
            """
        )


def test_when_clause_with_series_int_operand_fails() -> None:
    # Series<Int> is not Series<Float> -- no implicit per-element widening.
    with pytest.raises(QSTypeError):
        _check(
            """
            strategy S {
                when [1, 2] crosses_above [1.0, 2.0]: buy();
            }
            """
        )


def test_bool_still_rejected_where_series_float_expected() -> None:
    with pytest.raises(QSTypeError):
        _check(
            """
            strategy S {
                when true crosses_above [1.0, 2.0]: buy();
            }
            """
        )
    assert BOOL != SeriesType(FLOAT)
