"""Tests for allie/parser/ (Phase 2.3, designed alongside the AST since no
separate spec was provided). Targets docs/grammar.md."""

from __future__ import annotations

from allie.ast.nodes import (
    ArrayLiteral,
    Assignment,
    Backtest,
    BacktestStmt,
    Binary,
    Block,
    BoolLiteral,
    BreakStmt,
    Call,
    ContinueStmt,
    ExprStmt,
    FieldAccess,
    ForStmt,
    FuncDecl,
    Identifier,
    IfStmt,
    ImportStmt,
    LetStmt,
    Logical,
    NumberLiteral,
    ReturnStmt,
    StrategyDecl,
    StringLiteral,
    Unary,
    WhileStmt,
)
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser


def _parse(source: str):
    tokens = Lexer(source).scan_tokens()
    parser = Parser(tokens)
    program = parser.parse()
    assert not parser.diagnostics.has_errors(), parser.diagnostics.diagnostics
    return program


def test_let_without_annotation() -> None:
    program = _parse("let x = 1;")
    stmt = program.statements[0]
    assert isinstance(stmt, LetStmt)
    assert stmt.name == "x"
    assert stmt.type_annotation is None
    assert isinstance(stmt.initializer, NumberLiteral)
    assert stmt.initializer.value == 1


def test_let_with_annotation() -> None:
    program = _parse("let x: Int = 1;")
    stmt = program.statements[0]
    assert isinstance(stmt, LetStmt)
    assert stmt.type_annotation is not None
    assert stmt.type_annotation.name == "Int"


def test_let_with_bracket_generic_annotation() -> None:
    program = _parse("let prices: Series[Float] = x;")
    stmt = program.statements[0]
    assert isinstance(stmt, LetStmt)
    annotation = stmt.type_annotation
    assert annotation is not None
    assert annotation.name == "Series"
    assert annotation.type_arg is not None
    assert annotation.type_arg.name == "Float"


def test_arithmetic_precedence() -> None:
    program = _parse("let x = 1 + 2 * 3;")
    stmt = program.statements[0]
    assert isinstance(stmt, LetStmt)
    expr = stmt.initializer
    assert isinstance(expr, Binary)
    assert expr.operator == "+"
    assert isinstance(expr.left, NumberLiteral)
    assert isinstance(expr.right, Binary)
    assert expr.right.operator == "*"


def test_assignment_is_right_associative_and_only_targets_identifiers() -> None:
    program = _parse("let x = 1; { x = 2; }")
    block = program.statements[1]
    assert isinstance(block, Block)
    expr_stmt = block.statements[0]
    assignment = expr_stmt.expression  # type: ignore[attr-defined]
    assert isinstance(assignment, Assignment)
    assert assignment.name == "x"
    assert isinstance(assignment.value, NumberLiteral)


def test_logical_and_or_are_distinct_from_binary() -> None:
    program = _parse("let x = true and false or true;")
    stmt = program.statements[0]
    assert isinstance(stmt, LetStmt)
    expr = stmt.initializer
    # logic_or binds loosest: (true and false) or true
    assert isinstance(expr, Logical)
    assert expr.operator == "or"
    assert isinstance(expr.left, Logical)
    assert expr.left.operator == "and"


def test_unary_minus_and_bang() -> None:
    program = _parse("let x = -1; let y = !true;")
    neg = program.statements[0].initializer  # type: ignore[attr-defined]
    bang = program.statements[1].initializer  # type: ignore[attr-defined]
    assert isinstance(neg, Unary) and neg.operator == "-"
    assert isinstance(bang, Unary) and bang.operator == "!"


def test_call_and_field_access_chain() -> None:
    program = _parse("let x = SMA(prices.close, 20);")
    stmt = program.statements[0]
    assert isinstance(stmt, LetStmt)
    call = stmt.initializer
    assert isinstance(call, Call)
    assert isinstance(call.callee, Identifier)
    assert call.callee.name == "SMA"
    assert len(call.arguments) == 2
    field_access = call.arguments[0]
    assert isinstance(field_access, FieldAccess)
    assert field_access.name == "close"
    assert isinstance(field_access.obj, Identifier)
    assert field_access.obj.name == "prices"


def test_bool_literals() -> None:
    program = _parse("let a = true; let b = false;")
    assert isinstance(program.statements[0].initializer, BoolLiteral)  # type: ignore[attr-defined]
    assert program.statements[0].initializer.value is True  # type: ignore[attr-defined]
    assert program.statements[1].initializer.value is False  # type: ignore[attr-defined]


def test_if_else_if_chain() -> None:
    program = _parse(
        """
        if x == 1 {
            return 1;
        } else if x == 2 {
            return 2;
        } else {
            return 3;
        }
        """
    )
    stmt = program.statements[0]
    assert isinstance(stmt, IfStmt)
    assert isinstance(stmt.then_branch, Block)
    assert isinstance(stmt.else_branch, IfStmt)
    assert isinstance(stmt.else_branch.else_branch, Block)


def test_while_loop() -> None:
    program = _parse("while x < 10 { x = x + 1; }")
    stmt = program.statements[0]
    assert isinstance(stmt, WhileStmt)
    assert isinstance(stmt.condition, Binary)
    assert isinstance(stmt.body, Block)


def test_for_loop_matches_phase1_evaluator_shape() -> None:
    program = _parse("for let i = 0; i < 5; i = i + 1 { }")
    stmt = program.statements[0]
    assert isinstance(stmt, ForStmt)
    assert isinstance(stmt.initializer, LetStmt)
    assert stmt.initializer.name == "i"
    assert isinstance(stmt.condition, Binary)
    assert isinstance(stmt.increment, Assignment)
    assert isinstance(stmt.body, Block)


def test_break_and_continue() -> None:
    program = _parse("while true { break; continue; }")
    body = program.statements[0].body  # type: ignore[attr-defined]
    assert isinstance(body.statements[0], BreakStmt)
    assert isinstance(body.statements[1], ContinueStmt)


def test_func_decl_with_params_and_return_type() -> None:
    program = _parse("func add(a: Int, b: Int) -> Int { return a + b; }")
    stmt = program.statements[0]
    assert isinstance(stmt, FuncDecl)
    assert stmt.name == "add"
    assert [p.name for p in stmt.params] == ["a", "b"]
    assert [p.type_annotation.name for p in stmt.params] == ["Int", "Int"]
    assert stmt.return_type is not None
    assert stmt.return_type.name == "Int"
    assert isinstance(stmt.body.statements[0], ReturnStmt)


def test_func_decl_without_return_type() -> None:
    program = _parse("func log(msg: String) { return; }")
    stmt = program.statements[0]
    assert isinstance(stmt, FuncDecl)
    assert stmt.return_type is None


def test_import_statement() -> None:
    program = _parse('import "some/path.ql";')
    stmt = program.statements[0]
    assert isinstance(stmt, ImportStmt)
    assert stmt.path == "some/path.ql"


def test_strategy_and_when_clauses() -> None:
    program = _parse(
        """
        strategy Cross {
            when fast crosses_above slow: buy();
            when fast crosses_below slow: sell();
        }
        """
    )
    stmt = program.statements[0]
    assert isinstance(stmt, StrategyDecl)
    assert stmt.name == "Cross"
    assert len(stmt.when_clauses) == 2
    first, second = (clause.condition for clause in stmt.when_clauses)
    assert isinstance(first, Binary) and first.operator == "crosses_above"
    assert isinstance(second, Binary) and second.operator == "crosses_below"
    assert isinstance(stmt.when_clauses[0].body, ExprStmt)


def test_a_when_condition_is_any_expression() -> None:
    program = _parse(
        """
        strategy S {
            when fast crosses_above slow and rsi < 70.0: buy();
            when !(close > upper) or rsi crosses_below 30.0: sell();
        }
        """
    )
    stmt = program.statements[0]
    assert isinstance(stmt, StrategyDecl)
    first, second = (clause.condition for clause in stmt.when_clauses)
    # `and` binds looser than a crossing, which binds like a comparison.
    assert isinstance(first, Logical) and first.operator == "and"
    assert isinstance(first.left, Binary) and first.left.operator == "crosses_above"
    assert isinstance(first.right, Binary) and first.right.operator == "<"
    assert isinstance(second, Logical) and second.operator == "or"
    assert isinstance(second.left, Unary)


def test_crossings_are_ordinary_operators_outside_strategies_too() -> None:
    program = _parse("let up = fast crosses_above slow + 1.0;")
    stmt = program.statements[0]
    assert isinstance(stmt, LetStmt)
    crossing = stmt.initializer
    assert isinstance(crossing, Binary) and crossing.operator == "crosses_above"
    assert isinstance(crossing.right, Binary) and crossing.right.operator == "+"


def test_backtest_expression() -> None:
    program = _parse("let r = backtest Cross on AAPL from 0 to 100;")
    stmt = program.statements[0]
    assert isinstance(stmt, LetStmt)
    expr = stmt.initializer
    assert isinstance(expr, Backtest)
    assert (expr.strategy_name, expr.symbol) == ("Cross", "AAPL")


def test_backtest_statement() -> None:
    program = _parse("backtest Cross on AAPL from 0 to 100;")
    stmt = program.statements[0]
    assert isinstance(stmt, BacktestStmt)
    assert stmt.strategy_name == "Cross"
    assert stmt.symbol == "AAPL"
    assert isinstance(stmt.start, NumberLiteral) and stmt.start.value == 0
    assert isinstance(stmt.end, NumberLiteral) and stmt.end.value == 100


def test_backtest_bounds_are_int_expressions() -> None:
    program = _parse("backtest Cross on AAPL from split to len(AAPL.close) - 1;")
    stmt = program.statements[0]
    assert isinstance(stmt, BacktestStmt)
    assert isinstance(stmt.start, Identifier) and stmt.start.name == "split"
    assert isinstance(stmt.end, Binary) and stmt.end.operator == "-"


def test_array_literal() -> None:
    program = _parse("let xs = [1, 2, 3];")
    stmt = program.statements[0]
    assert isinstance(stmt, LetStmt)
    array = stmt.initializer
    assert isinstance(array, ArrayLiteral)
    assert [e.value for e in array.elements] == [1, 2, 3]  # type: ignore[attr-defined]


def test_parenthesized_expression() -> None:
    program = _parse("let x = (1 + 2) * 3;")
    stmt = program.statements[0]
    assert isinstance(stmt, LetStmt)
    expr = stmt.initializer
    assert isinstance(expr, Binary)
    assert expr.operator == "*"
    assert isinstance(expr.left, Binary)
    assert expr.left.operator == "+"


def test_string_literal_expression() -> None:
    program = _parse('let x = "hello";')
    stmt = program.statements[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.initializer, StringLiteral)
    assert stmt.initializer.value == "hello"


def test_error_recovery_collects_multiple_diagnostics_and_continues() -> None:
    # Two malformed lets (missing '=') followed by one valid statement.
    tokens = Lexer("let x 1; let y 2; let z = 3;").scan_tokens()
    parser = Parser(tokens)
    program = parser.parse()

    assert len(parser.diagnostics.diagnostics) == 2
    # Synchronization should have recovered enough to still parse the
    # trailing valid statement.
    assert isinstance(program.statements[-1], LetStmt)
    assert program.statements[-1].name == "z"


def test_invalid_assignment_target_is_diagnostic_not_crash() -> None:
    tokens = Lexer("1 = 2;").scan_tokens()
    parser = Parser(tokens)
    parser.parse()
    assert parser.diagnostics.has_errors()
