"""The parser's error side: what each syntax error says and where it
points, how far recovery gets after one, the nesting/depth limits, and a
seeded fuzz test that it never raises. Structural parsing of valid code
is in test_parser.py."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from allie.ast.nodes import (
    ArrayLiteral,
    Binary,
    Block,
    Call,
    ExprStmt,
    FieldAccess,
    FuncDecl,
    Identifier,
    IfStmt,
    Index,
    LetStmt,
    Program,
    ReturnStmt,
    StrategyDecl,
    WhenClause,
)
from allie.cli import run_pipeline
from allie.errors import Diagnostic
from allie.lexer.lexer import Lexer
from allie.parser.parser import MAX_AST_DEPTH, MAX_NESTING, Parser

_REPO = Path(__file__).resolve().parents[2]


def _parse(source: str) -> tuple[Program, list[Diagnostic]]:
    parser = Parser(Lexer(source).scan_tokens())
    program = parser.parse()
    return program, parser.diagnostics.diagnostics


def _clean(source: str) -> Program:
    program, diagnostics = _parse(source)
    assert diagnostics == [], [d.render(source) for d in diagnostics]
    return program


def _only(source: str) -> Diagnostic:
    _, diagnostics = _parse(source)
    assert len(diagnostics) == 1, [d.render(source) for d in diagnostics]
    return diagnostics[0]


def _where(diagnostic: Diagnostic) -> tuple[int, int]:
    return diagnostic.span.line, diagnostic.span.column


# ---- what errors say, and where ----


def test_missing_semicolon_at_end_of_line_points_after_the_last_token() -> None:
    program, diagnostics = _parse("let x = 1\nlet y = 2;")
    assert len(diagnostics) == 1
    assert diagnostics[0].message == "expected ';' after the let statement"
    assert _where(diagnostics[0]) == (1, 10)
    # ...and the next statement is still parsed.
    assert [s.name for s in program.statements if isinstance(s, LetStmt)] == ["y"]


def test_missing_semicolon_on_the_same_line_names_what_was_found() -> None:
    program, diagnostics = _parse("let x = 1 let y = 2;")
    assert len(diagnostics) == 1
    assert diagnostics[0].message == "expected ';' after the let statement, found keyword 'let'"
    assert _where(diagnostics[0]) == (1, 11)
    assert [s.name for s in program.statements if isinstance(s, LetStmt)] == ["y"]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("print(1 + );", "expected an expression, found ')'"),
        ("let x =", "expected an expression, found end of file"),
        ('let x = 1 "two";', 'expected \';\' after the let statement, found string "two"'),
        ("print(1];", "expected ')' to close the call opened at 1:6, found ']'"),
        ("let x: Series[Float = 1;", "expected ']' to close the type argument opened at 1:14"),
    ],
)
def test_messages_name_the_token_that_was_found(source: str, message: str) -> None:
    assert _only(source).message.startswith(message)


def test_unclosed_paren_at_end_of_file_points_at_the_paren() -> None:
    error = _only("print(1 + 2")
    assert error.message == "this '(' is never closed"
    assert _where(error) == (1, 6)


def test_unclosed_brace_names_the_inner_block_whose_brace_is_misindented() -> None:
    source = (
        "func f(a: Int) -> Int {\n"
        "    if a > 1 {\n"
        "        return a;\n"
        "\n"
        "    return 0;\n"
        "}\n"
        "print(f(2));\n"
    )
    error = _only(source)
    assert error.message == "this '{' is never closed"
    assert _where(error) == (1, 23)
    assert error.hint is not None
    assert "the '{' at 2:14 may be the one missing its '}'" in error.hint
    assert "the '}' at 6:1" in error.hint


@pytest.mark.parametrize(
    ("source", "keyword", "hint_fragment"),
    [
        ("let ts = 1;", "ts", "another spelling of 'let'"),
        ("let fr = 1;", "fr", "another spelling of 'true'"),
        ("let to = 1;", "to", "'to' is a keyword"),
        ("func while(a: Int) -> Int { return a; }", "while", "'while' is a keyword"),
    ],
)
def test_keyword_used_as_a_name_is_explained_and_recovered(
    source: str, keyword: str, hint_fragment: str
) -> None:
    program, diagnostics = _parse(source)
    assert len(diagnostics) == 1
    assert diagnostics[0].message.endswith(f"found keyword '{keyword}'")
    assert diagnostics[0].hint is not None and hint_fragment in diagnostics[0].hint
    # Recovery uses the keyword as the name, so the statement still parses.
    assert len(program.statements) == 1


@pytest.mark.parametrize(
    "source",
    [
        "print(0 < a < 10);",
        "print(a <= b > c);",
        "print(a == b == c);",
        "print(a != b == c);",
        "print(a crosses_above b crosses_below c);",
        "print(a > b crosses_above c);",
    ],
)
def test_chained_comparisons_are_rejected_with_a_hint(source: str) -> None:
    error = _only(source)
    assert "can't be chained" in error.message
    assert error.hint is not None and "and" in error.hint


def test_parenthesized_comparison_is_not_a_chain() -> None:
    _clean("print((a < b) == c);")


@pytest.mark.parametrize(
    "source",
    [
        "if x = 1 { }",
        "while x = 1 { }",
        "for let i = 0; i = 1; i = i + 1 { }",
    ],
)
def test_assignment_used_as_a_condition_is_reported(source: str) -> None:
    error = _only(source)
    assert "condition is an assignment" in error.message
    assert error.hint is not None and "'=='" in error.hint


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("s[0] = 1.0;", "can't assign to a series element"),
        ("p.close = 1.0;", "can't assign to the field 'close'"),
        ("1 = 2;", "invalid assignment target"),
    ],
)
def test_invalid_assignment_targets(source: str, message: str) -> None:
    error = _only(source)
    assert error.message == message
    assert _where(error)[1] == source.index("=") + 1


def test_duplicate_parameter_names() -> None:
    error = _only("func f(a: Int, a: Int) -> Int { return a; }")
    assert error.message == "duplicate parameter name 'a'"
    assert _where(error) == (1, 16)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("else { print(1); }\nlet z = 1;", "'else' without a matching 'if'"),
        ("else if x { }\nlet z = 1;", "'else' without a matching 'if'"),
        ("when a crosses_above b: buy();\nlet z = 1;", "'when' outside a strategy"),
        ("}\nlet z = 1;", "unmatched '}'"),
    ],
)
def test_stray_tokens_cost_one_error_and_parsing_resumes(source: str, message: str) -> None:
    program, diagnostics = _parse(source)
    assert [d.message for d in diagnostics] == [message]
    assert isinstance(program.statements[-1], LetStmt)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("backtest S on A from 1.5 to 10;", "a bar index must be a whole number, found 1.5"),
        ("backtest S on A from 0 to 2.5;", "a bar index must be a whole number, found 2.5"),
        ("backtest S on A from -1 to 10;", "a bar index can't be negative"),
        ("backtest S on A from 20 to 10;", "empty bar range: 'from 20 to 10' selects no bars"),
        ("backtest S on A from 5 to 5;", "empty bar range: 'from 5 to 5' selects no bars"),
    ],
)
def test_backtest_bar_ranges_are_validated_at_compile_time(source: str, message: str) -> None:
    assert _only(source).message == message


def test_valid_backtest_range() -> None:
    program = _clean("backtest S on A from 0 to 100;")
    stmt = program.statements[0]
    assert (stmt.start.value, stmt.end.value) == (0, 100)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "source",
    [
        "backtest S on A from split to len(c);",
        "backtest S on A from 0 to n - 1;",
        "backtest S on A from k * 10 to (k + 1) * 10;",
    ],
)
def test_computed_backtest_bounds_are_left_to_the_type_checker(source: str) -> None:
    _clean(source)


def test_a_missing_bar_index_is_reported_as_one() -> None:
    assert _only("backtest S on A from 0 to;").message == (
        "expected a bar index after 'to', found ';'"
    )


def test_one_error_per_mistake_in_a_file_with_many() -> None:
    source = (
        "}\n"
        "else { print(1); }\n"
        "when x crosses_above y: buy();\n"
        "let n: Int = 3;\n"
        "if n = 4 { print(n); }\n"
        "func f(a: Int, a: Int) -> Int { return a; }\n"
        "print(f(1,\n"
        "  2\n"
        "let z = 1;\n"
    )
    _, diagnostics = _parse(source)
    assert [d.span.line for d in diagnostics] == [1, 2, 3, 5, 6, 8]


# ---- recovery ----


def test_bad_when_clause_does_not_cascade_through_the_strategy() -> None:
    source = (
        "strategy S {\n"
        "    when f crosses s: buy();\n"
        "    when f crosses_below s: { sell(); }\n"
        "}\n"
        "backtest S on A from 0 to 10;\n"
    )
    program, diagnostics = _parse(source)
    assert len(diagnostics) == 1
    assert diagnostics[0].message == "expected ':' after the when condition, found 'crosses'"
    assert diagnostics[0].hint == (
        "'crosses' isn't an operator; did you mean 'crosses_above' or 'crosses_below'?"
    )
    strategy = program.statements[0]
    assert isinstance(strategy, StrategyDecl)
    assert [_operator(c) for c in strategy.when_clauses] == ["crosses_below"]
    assert len(program.statements) == 2


def _operator(clause: WhenClause) -> str:
    assert isinstance(clause.condition, Binary)
    return clause.condition.operator


def test_a_when_clause_missing_its_colon_explains_what_a_condition_is() -> None:
    _, diagnostics = _parse("strategy S {\n    when fast > slow buy();\n}\n")
    assert len(diagnostics) == 1
    assert diagnostics[0].message == "expected ':' after the when condition, found 'buy'"
    assert diagnostics[0].hint is not None and "Series<Bool>" in diagnostics[0].hint


def test_recovery_skips_over_a_broken_clauses_action_block() -> None:
    source = (
        "strategy S {\n"
        "    when f crosses s: { buy(); }\n"
        "    when f crosses_above s: sell();\n"
        "}\n"
    )
    program, diagnostics = _parse(source)
    assert len(diagnostics) == 1
    strategy = program.statements[0]
    assert isinstance(strategy, StrategyDecl)
    assert [_operator(c) for c in strategy.when_clauses] == ["crosses_above"]


def test_error_inside_a_block_does_not_swallow_its_closing_brace() -> None:
    source = "func g(x: Int) -> Int {\n    let y = foo(;\n    return x;\n}\nlet z = 1;\n"
    program, diagnostics = _parse(source)
    assert len(diagnostics) == 1
    func = program.statements[0]
    assert isinstance(func, FuncDecl)
    assert isinstance(func.body.statements[-1], ReturnStmt)
    assert isinstance(program.statements[1], LetStmt)


def test_error_right_before_a_closing_brace_keeps_the_block_intact() -> None:
    program, diagnostics = _parse("if a { foo( }\nlet z = 1;")
    assert len(diagnostics) == 1
    assert isinstance(program.statements[0], IfStmt)
    assert isinstance(program.statements[1], LetStmt)


# ---- syntax the parser accepts ----


def test_trailing_commas_are_allowed() -> None:
    program = _clean(
        "func f(a: Int, b: Int,) -> Int { return a; }\n"
        "print(f(1, 2,));\n"
        "let s = [1, 2, 3,];\n"
    )
    func, call_stmt, let_stmt = program.statements
    assert isinstance(func, FuncDecl) and len(func.params) == 2
    assert isinstance(call_stmt, ExprStmt) and isinstance(call_stmt.expression, Call)
    inner = call_stmt.expression.arguments[0]
    assert isinstance(inner, Call) and len(inner.arguments) == 2
    assert isinstance(let_stmt, LetStmt) and isinstance(let_stmt.initializer, ArrayLiteral)
    assert len(let_stmt.initializer.elements) == 3


def test_a_lone_or_doubled_comma_is_still_an_error() -> None:
    assert "found ','" in _only("print(,);").message
    assert "found ','" in _only("print(1,,2);").message


def test_index_expressions() -> None:
    program = _clean("let x = s[i + 1];\nlet y = m[0][1];\nlet z = f(1)[0].close;")
    x, y, z = (stmt.initializer for stmt in program.statements)  # type: ignore[attr-defined]
    assert isinstance(x, Index) and isinstance(x.obj, Identifier) and isinstance(x.index, Binary)
    assert isinstance(y, Index) and isinstance(y.obj, Index)
    assert isinstance(z, FieldAccess) and isinstance(z.obj, Index) and isinstance(z.obj.obj, Call)


def test_else_if_chain_folds_into_nested_ifs() -> None:
    program = _clean("if a { } else if b { } else if c { } else { }")
    first = program.statements[0]
    assert isinstance(first, IfStmt)
    second = first.else_branch
    assert isinstance(second, IfStmt)
    third = second.else_branch
    assert isinstance(third, IfStmt)
    assert isinstance(third.else_branch, Block)
    assert [b.condition.name for b in (first, second, third)] == ["a", "b", "c"]  # type: ignore[attr-defined]


def test_long_else_if_chain_needs_no_recursion() -> None:
    _clean("if a { }" + " else if a { }" * 150 + " else { }")


# ---- limits ----


def test_nesting_up_to_the_limit_is_fine() -> None:
    depth = MAX_NESTING - 1  # the initializer itself is one level
    _clean("let x = " + "(" * depth + "1" + ")" * depth + ";")


@pytest.mark.parametrize(
    "source",
    [
        "let x = " + "(" * 1000 + "1" + ")" * 1000 + ";",
        "let x = " + "[" * 1000 + "1" + "]" * 1000 + ";",
        "let x = " + "f(" * 1000 + "1" + ")" * 1000 + ";",
        "let x = " + "-" * 1000 + "1;",
        "let x = " + "y = " * 1000 + "1;",
        "if a {" * 1000 + "}" * 1000,
        "let x: " + "Series[" * 1000 + "Float" + "]" * 1000 + " = 1;",
        "strategy S { when " + "(" * 1000 + "a" + ")" * 1000 + " crosses_above b: buy(); }",
    ],
    ids=["parens", "brackets", "calls", "unary", "assignments", "blocks", "types", "when"],
)
def test_pathological_nesting_is_one_error_not_a_crash(source: str) -> None:
    error = _only(source)
    assert error.message.startswith(f"nesting is too deep: more than {MAX_NESTING} levels")


def test_ast_depth_limit_is_exact() -> None:
    # `let x = 1+1+...` is LetStmt -> n Binary nodes -> a literal: depth n + 2.
    allowed = MAX_AST_DEPTH - 2
    _clean("let x = " + "+".join(["1"] * (allowed + 1)) + ";")
    error = _only("let x = " + "+".join(["1"] * (allowed + 2)) + ";")
    assert error.message.startswith("this statement is nested too deeply")
    assert _where(error) == (1, 1)


def test_deepest_allowed_program_runs_on_every_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The depth cap only means something if every later stage -- type
    checker, look-ahead pass, both interpreters, lowering -- survives
    the deepest tree it lets through. They recurse; this pins that."""
    terms = MAX_AST_DEPTH - 1
    path = tmp_path / "deep.qs"
    path.write_text(f"let x: Int = {'+'.join(['1'] * terms)};\nprint(x);\n", encoding="utf-8")
    for backend in ("tree", "ir", "llvm"):
        assert run_pipeline(path, backend=backend) == 0, backend
        assert capsys.readouterr().out.strip().endswith(str(terms)), backend


# ---- rendering ----


def test_parse_errors_render_with_source_line_caret_and_hint() -> None:
    source = "let n = 3;\nprint(0 < n < 10);"
    rendered = _only(source).render(source).split("\n")
    assert rendered == [
        "error: comparison operators can't be chained (2:13)",
        "  print(0 < n < 10);",
        "              ^",
        "  help: write 'a < x and x < b'; unlike Python, Allie doesn't chain comparisons",
    ]


def test_cli_prints_parse_errors_with_context(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "p.qs"
    path.write_text("let ts = 1;\n", encoding="utf-8")
    assert run_pipeline(path) == 1
    captured = capsys.readouterr()
    assert "  let ts = 1;\n      ^^\n  help: 'ts' is reserved" in captured.err
    assert captured.out == ""  # diagnostics never mix with program output


# ---- robustness ----


def _programs() -> list[str]:
    paths = sorted([*_REPO.glob("examples/*.qs"), *_REPO.glob("tests/**/*.qs")])
    return [p.read_text(encoding="utf-8") for p in paths]


def _check_never_raises(source: str) -> None:
    program, diagnostics = _parse(source)  # must not raise
    assert isinstance(program, Program)
    line_count = source.count("\n") + 1
    for diagnostic in diagnostics:
        assert 1 <= diagnostic.span.line <= line_count, (source, diagnostic)
        diagnostic.render(source)


def test_mutated_real_programs_never_crash_the_parser() -> None:
    """Delete, duplicate or swap random tokens of every real program --
    the shapes typos actually take -- and check the parser always
    returns, with every diagnostic pointing inside the file."""
    rng = random.Random(20261020)
    for source in _programs():
        tokens = [t for t in Lexer(source).scan_tokens() if t.lexeme]
        lexemes = [t.lexeme for t in tokens]
        for _ in range(60):
            mutated = list(lexemes)
            for _ in range(rng.randint(1, 4)):
                i = rng.randrange(len(mutated))
                operation = rng.choice(("delete", "duplicate", "swap", "insert"))
                if operation == "delete":
                    del mutated[i]
                elif operation == "duplicate":
                    mutated.insert(i, mutated[i])
                elif operation == "swap" and i + 1 < len(mutated):
                    mutated[i], mutated[i + 1] = mutated[i + 1], mutated[i]
                else:
                    mutated.insert(i, rng.choice(lexemes))
            _check_never_raises(" ".join(mutated))


def test_random_token_soup_never_crashes_the_parser() -> None:
    vocabulary = sorted({t.lexeme for s in _programs() for t in Lexer(s).scan_tokens() if t.lexeme})
    vocabulary += ["else", "when", "}", "{", "(", ")", "[", "]", ",", ";", "=", "<", "=="]
    rng = random.Random(7)
    for _ in range(1500):
        _check_never_raises(" ".join(rng.choice(vocabulary) for _ in range(rng.randint(0, 30))))
