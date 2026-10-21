"""The type system as a whole: that every error in a program is reported
(and only once), the soundness rules that keep the three backends in
agreement, the hints, and Int/Float arithmetic giving identical results
on the tree-walking interpreter, the IR interpreter and the LLVM JIT.
Rule-by-rule basics are in test_typechecker.py / test_typechecker_quant.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import allie.stdlib as stdlib
from allie.cli import BACKENDS, run_pipeline
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.typechecker.checker import TypeChecker
from allie.typechecker.errors import QSTypeError
from allie.typechecker.types import DATE, ERROR, FLOAT, INT, SeriesType, is_assignable

_REPO = Path(__file__).resolve().parents[2]
_LOAD = 'let AAPL = load("tests/fixtures/AAPL_sample.csv");\n'


def _errors(source: str) -> list[QSTypeError]:
    """Every type error in `source`, checked with the real builtins."""
    program = Parser(Lexer(source).scan_tokens()).parse()
    checker = TypeChecker()
    for name, signature in stdlib.builtin_function_types().items():
        checker.env.define(name, signature)
    try:
        checker.check(program)
    except QSTypeError as first:
        assert first is checker.errors[0]
    return checker.errors


def _only(source: str) -> QSTypeError:
    errors = _errors(source)
    assert len(errors) == 1, [str(e) for e in errors]
    return errors[0]


def _clean(source: str) -> None:
    errors = _errors(source)
    assert errors == [], [str(e) for e in errors]


@pytest.fixture
def repo_cwd() -> object:
    previous = Path.cwd()
    os.chdir(_REPO)
    yield
    os.chdir(previous)


def _run(tmp_path: Path, source: str, backend: str, *, optimize: bool = True) -> int:
    path = tmp_path / "p.qs"
    path.write_text(source, encoding="utf-8")
    return run_pipeline(path, backend=backend, optimize=optimize)


# ---- every error, once ----


def test_independent_errors_are_all_reported_in_source_order() -> None:
    source = (
        _LOAD
        + "let n: int = 3;\n"
        + "let total: Int = 7 / 2;\n"
        + "let s = SMA(AAPL.close);\n"
        + "let same = AAPL.close == AAPL.open;\n"
    )
    errors = _errors(source)
    assert [e.line for e in errors] == [2, 3, 4, 5]
    assert [e.message for e in errors] == [
        "unknown type 'int'",
        "cannot assign Float to Int",
        "'SMA' takes 2 arguments, got 1",
        "'==' can't compare whole series",
    ]


def test_check_raises_the_first_error_after_checking_everything() -> None:
    program = Parser(Lexer('let a: Int = "x";\nlet b: Bool = 1;').scan_tokens()).parse()
    checker = TypeChecker()
    with pytest.raises(QSTypeError, match="cannot assign String to Int"):
        checker.check(program)
    assert len(checker.errors) == 2


def test_a_failed_expression_does_not_cascade() -> None:
    # `c` gets the poison type, so every later use of it is silent.
    source = (
        _LOAD
        + "let c = AAPL.clsoe;\n"
        + "let fast = SMA(c, 5);\n"
        + "print(mean(fast) + 1.0);\n"
        + "if c == c { print(1); }\n"
    )
    error = _only(source)
    assert error.message == "PriceSeries has no field 'clsoe'"
    assert error.hint == "did you mean 'close'?"


def test_an_undefined_name_is_reported_once() -> None:
    error = _only("print(nope);\nprint(nope + 1);\nlet x = nope;")
    assert error.message == "undefined name 'nope'"


def test_type_errors_render_with_caret_and_hint() -> None:
    source = "let n: Int = 3;\nif n { print(n); }"
    error = _only(source)
    assert error.render(source).split("\n") == [
        "TypeError: condition must be Bool, got Int. Line 2, Column 4",
        "  if n { print(n); }",
        "     ^",
        "  help: Allie has no truthiness: compare explicitly, e.g. 'x != 0'",
    ]


def test_cli_prints_every_type_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = "".join(f'let x{i}: Int = "s";\n' for i in range(25))
    assert _run(tmp_path, source, "tree") == 1
    err = capsys.readouterr().err
    assert err.count("TypeError: cannot assign String to Int") == 20
    assert "... and 5 more type error(s)" in err


# ---- arithmetic typing ----


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("7 / 2", FLOAT),
        ("6 / 3", FLOAT),
        ("7 % 2", INT),
        ("7 * 2", INT),
        ("7 + 2.0", FLOAT),
        ("7.0 % 2", FLOAT),
        ("-7", INT),
    ],
)
def test_arithmetic_result_types(expression: str, expected: object) -> None:
    program = Parser(Lexer(f"let r = {expression};").scan_tokens()).parse()
    TypeChecker().check(program)
    assert program.statements[0].initializer.inferred_type == expected  # type: ignore[attr-defined]


def test_dividing_ints_into_an_int_is_rejected_with_a_hint() -> None:
    error = _only("let half: Int = 7 / 2;")
    assert error.message == "cannot assign Float to Int"
    assert error.hint == "Allie never narrows a Float to an Int implicitly"


@pytest.mark.parametrize(
    "source",
    ["print(2.5 == 2);", "print(1 != 1.0);", 'print("a" == "a");', "print(true != false);"],
)
def test_scalar_equality(source: str) -> None:
    _clean(source)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("let s: Series[Float] = [1.0];\nprint(s == s);", "'==' can't compare whole series"),
        ('print(1 == "1");', "'==' requires operands of the same type, got Int and String"),
        ("print(true == 1);", "'==' requires operands of the same type, got Bool and Int"),
    ],
)
def test_equality_that_has_no_meaning_is_rejected(source: str, message: str) -> None:
    assert _only(source).message == message


# ---- series ----


def test_series_widen_element_wise() -> None:
    assert is_assignable(SeriesType(INT), SeriesType(FLOAT))
    assert not is_assignable(SeriesType(FLOAT), SeriesType(INT))
    _clean("print(SMA([1, 2, 3, 4], 2));\nprint(len([1, 2, 3]));")


def test_mixed_numeric_literal_is_a_float_series() -> None:
    program = Parser(Lexer("let s = [1, 2.5, 3];").scan_tokens()).parse()
    TypeChecker().check(program)
    assert program.statements[0].initializer.inferred_type == SeriesType(FLOAT)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ('let s = ["a", "b"];', "a series can only hold numbers or Bools, got String"),
        ("let m = [[1.0], [2.0]];", "a series can only hold numbers or Bools, got Series<Float>"),
        ("let s = [1, true];", "array elements must all be the same type: Int vs Bool"),
    ],
)
def test_series_literals_hold_only_what_the_runtime_can_store(source: str, message: str) -> None:
    assert _only(source).message == message


def test_price_series_fields() -> None:
    program = Parser(Lexer(_LOAD + "let d = AAPL.dates;").scan_tokens()).parse()
    checker = TypeChecker()
    for name, signature in stdlib.builtin_function_types().items():
        checker.env.define(name, signature)
    checker.check(program)
    assert program.statements[1].initializer.inferred_type == SeriesType(DATE)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("source", "message", "hint"),
    [
        (_LOAD + "print(AAPL.clsoe);", "PriceSeries has no field 'clsoe'", "did you mean 'close'?"),
        (_LOAD + "print(AAPL.price);", "PriceSeries has no field 'price'", "its fields are"),
        ("let x: Int = 1;\nprint(x.close);", "Int has no field 'close'", "only a PriceSeries"),
    ],
)
def test_unknown_fields_are_errors(source: str, message: str, hint: str) -> None:
    error = _only(source)
    assert error.message == message
    assert error.hint is not None and error.hint.startswith(hint)


# ---- what only some backends could run ----


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "func f(a: Int) -> Int { return a; }\nlet g = f;",
            "'f' is a function; functions can't be used as values",
        ),
        ("let p = print;", "'print' is a function; functions can't be used as values"),
        (
            "strategy S { when [1.0] crosses_above [1.0]: 1; }\nlet t = S;",
            "'S' is a strategy; it can only be named in a 'backtest' statement",
        ),
        (
            "func outer(n: Int) -> Int {\n"
            "    func inner(k: Int) -> Int { return k + n; }\n"
            "    return inner(1);\n"
            "}",
            "'n' is a local variable of the enclosing function 'outer'",
        ),
        (
            "func outer(n: Int) -> Int {\n"
            "    func inner(k: Int) -> Int { n = k; return k; }\n"
            "    return inner(1);\n"
            "}",
            "'n' is a local variable of the enclosing function 'outer'",
        ),
        (
            "func a() -> Int { func h() -> Int { return 1; } return h(); }\n"
            "func b() -> Int { func h() -> Int { return 2; } return h(); }",
            "a function named 'h' is already declared at 1:19",
        ),
        (
            "func f() -> Int { return 1; }\nfunc f() -> Int { return 2; }",
            "a function named 'f' is already declared at 1:1",
        ),
        (
            "func f() -> Float { func mean(x: Float) -> Float { return x; } return mean(1.0); }",
            "'mean' is a built-in function; a nested function can't reuse its name",
        ),
        ("let x = print(1);", "'print' returns Void, so this call has no value"),
        ("print(print(1));", "'print' returns Void, so this call has no value"),
    ],
    ids=[
        "function-value",
        "builtin-value",
        "strategy-value",
        "capture-read",
        "capture-write",
        "nested-name-clash",
        "top-level-name-clash",
        "nested-builtin-name",
        "void-let",
        "void-argument",
    ],
)
def test_rules_that_keep_the_backends_in_agreement(source: str, message: str) -> None:
    """Each of these used to type-check, then run on the tree-walker
    and crash -- or silently compute something else -- on the IR and
    LLVM backends."""
    assert _only(source).message == message


def test_globals_and_parameters_are_not_captures() -> None:
    _clean(
        "let rate: Float = 0.05;\n"
        "func grow(p: Float) -> Float {\n"
        "    func step(q: Float) -> Float { return q * (1.0 + rate); }\n"
        "    return step(p);\n"
        "}\n"
    )


def test_a_top_level_function_may_override_a_builtin() -> None:
    # Consistent on every backend: calls everywhere resolve to it.
    _clean("func sharpe(x: Float) -> Float { return x; }\nprint(sharpe(1.0));")


# ---- hints and suggestions ----


@pytest.mark.parametrize(
    ("annotation", "hint"),
    [
        ("int", "did you mean 'Int'?"),
        ("float", "did you mean 'Float'?"),
        ("str", "did you mean 'String'?"),
        ("Strng", "did you mean 'String'?"),
        ("Series", "e.g. Series[Float]"),
    ],
)
def test_unknown_types_get_suggestions(annotation: str, hint: str) -> None:
    assert _only(f"let x: {annotation} = 1;").hint == hint


def test_undefined_name_suggestion() -> None:
    error = _only("let total: Int = 1;\nprint(totl);")
    assert error.hint == "did you mean 'total'?"


@pytest.mark.parametrize(
    ("source", "hint_start"),
    [
        ("let x: Int = 1;\nif x { }", "Allie has no truthiness"),
        ("let s: Series[Float] = 1.0;", "a single value isn't a series"),
        (_LOAD + "let c: Float = AAPL.close;", "this is a whole series"),
        ('print("n=" + 1);', "Allie never converts between numbers and Strings"),
        ("print(SMA(1.0, 5));", "a single value isn't a series"),
    ],
)
def test_hints_for_common_mismatches(source: str, hint_start: str) -> None:
    hint = _only(source).hint
    assert hint is not None and hint.startswith(hint_start)


def test_wrong_argument_names_the_function_and_position() -> None:
    error = _only(_LOAD + 'print(SMA(AAPL.close, "5"));')
    assert error.message == "argument 2 of 'SMA' must be Int, got String"
    assert error.hint == "signature: SMA(Series<Float>, Int) -> Series<Float>"


def test_poison_type_is_compatible_with_everything() -> None:
    assert is_assignable(ERROR, INT) and is_assignable(SeriesType(FLOAT), ERROR)


# ---- runs identically everywhere ----

_ARITHMETIC = """
let a: Int = 7;
let b: Int = 2;
print(a / b);
print(-7 % 2);
print(7 % -2);
print(-7 % -2);
print(-7.5 % 2.0);
print(6.0 % -2.0);
print(-0.0 % 2.0);
print(-(0.0));
let m: Int = 0 - 9223372036854775807 - 1;
print(m % -1);
print(m / -1);
let big: Float = 1e308;
print(big * 10.0);
print(a == 7.0);
let nan: Float = SMA([1.0, 2.0], 2)[0];  // the warm-up bar of an SMA is NaN
print(nan != nan);
print(nan == nan);
print(SMA([1, 2, 3, 4], 2));
print([1, 2.5, 3]);
print(even(10));
print(twice(21));
func even(n: Int) -> Bool { if n == 0 { return true; } return odd(n - 1); }
func odd(n: Int) -> Bool { if n == 0 { return false; } return even(n - 1); }
func twice(k: Int) -> Int { return k * 2; }
"""

_ARITHMETIC_EXPECTED = [
    "3.5",
    "1",
    "-1",
    "-1",
    "0.5",
    "-0.0",
    "0.0",
    "-0.0",
    "0",
    "9.223372036854776e+18",
    "inf",
    "True",
    "True",
    "False",
    "Series[nan, 1.5, 2.5, 3.5]",
    "Series[1, 2.5, 3]",
    "True",
    "42",
]


@pytest.mark.parametrize("optimize", [True, False], ids=["opt", "no-opt"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_arithmetic_agrees_on_every_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], backend: str, optimize: bool
) -> None:
    """Python-style floored `%` (including -0.0 and INT_MIN % -1), true
    division, NaN comparisons, Series<Int> widening and function hoisting
    -- each of which at least one backend used to get differently."""
    assert _run(tmp_path, _ARITHMETIC, backend, optimize=optimize) == 0
    assert capsys.readouterr().out.strip().split("\n") == _ARITHMETIC_EXPECTED


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "source",
    [
        "print(9223372036854775807 + 1);",
        "let m: Int = 0 - 9223372036854775807 - 1;\nprint(-m);",
        "let m: Int = 4000000000;\nprint(m * m);",
        "let m: Int = 4000000000;\nm * m;\nprint(1);",
    ],
    ids=["add-constants", "negate-min", "multiply", "unused-multiply"],
)
def test_int_overflow_is_an_error_on_every_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], backend: str, source: str
) -> None:
    """Int is 64-bit everywhere. The interpreters used to grow it without
    bound while LLVM wrapped around; constant folding must not fold an
    overflowing result, and DCE must not delete an unused one."""
    assert _run(tmp_path, source, backend, optimize=True) == 1
    captured = capsys.readouterr()
    assert "integer overflow" in captured.err
    assert captured.out == ""


@pytest.mark.usefixtures("repo_cwd")
@pytest.mark.parametrize("backend", BACKENDS)
def test_dates_field_runs_on_every_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], backend: str
) -> None:
    # `.dates` used to be typed Void, which crashed LLVM codegen.
    assert _run(tmp_path, _LOAD + "let d = AAPL.dates;\nprint(1);\n", backend) == 0
    assert capsys.readouterr().out.strip() == "1"
