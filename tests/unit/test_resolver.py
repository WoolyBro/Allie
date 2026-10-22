"""Tests for allie/semantic/resolver.py: scopes, and where `break`,
`continue`, `return` and `when` are allowed."""

from __future__ import annotations

from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.semantic.resolver import Resolver


def _resolve(source: str) -> Resolver:
    tokens = Lexer(source).scan_tokens()
    program = Parser(tokens).parse()
    resolver = Resolver()
    resolver.resolve(program)
    return resolver


def _messages(resolver: Resolver) -> list[str]:
    return [d.message for d in resolver.diagnostics.diagnostics]


def test_undefined_variable_reference() -> None:
    resolver = _resolve("let x = y;")
    assert "undefined name 'y'" in _messages(resolver)


def test_defined_variable_reference_is_clean() -> None:
    resolver = _resolve("let x = 1; let y = x;")
    assert resolver.diagnostics.diagnostics == []


def test_shadowing_in_nested_block_is_allowed() -> None:
    resolver = _resolve("let x = 1; { let x = 2; }")
    assert resolver.diagnostics.diagnostics == []


def test_assign_to_undefined_name() -> None:
    resolver = _resolve("{ x = 1; }")
    assert "cannot assign to undefined name 'x'" in _messages(resolver)


def test_break_outside_loop() -> None:
    resolver = _resolve("break;")
    assert "'break' outside a loop" in _messages(resolver)


def test_continue_outside_loop() -> None:
    resolver = _resolve("continue;")
    assert "'continue' outside a loop" in _messages(resolver)


def test_break_inside_while_is_clean() -> None:
    resolver = _resolve("while true { break; }")
    assert resolver.diagnostics.diagnostics == []


def test_break_inside_for_is_clean() -> None:
    resolver = _resolve("for let i = 0; i < 5; i = i + 1 { break; continue; }")
    assert resolver.diagnostics.diagnostics == []


def test_break_inside_nested_function_body_but_outside_any_loop_still_errors() -> None:
    resolver = _resolve("func f() { break; }")
    assert "'break' outside a loop" in _messages(resolver)


def test_return_outside_function() -> None:
    resolver = _resolve("return 1;")
    assert "'return' outside a function" in _messages(resolver)


def test_return_inside_function_is_clean() -> None:
    resolver = _resolve("func f() { return 1; }")
    assert resolver.diagnostics.diagnostics == []


def test_when_clause_outside_strategy() -> None:
    # The parser only accepts 'when' inside a strategy body per the
    # grammar, so this exercises the resolver's own belt-and-suspenders
    # inside_strategy check directly rather than through parsed source.
    from allie.ast.nodes import BoolLiteral, ExprStmt, WhenClause
    from allie.errors import SourceSpan

    span = SourceSpan(line=1, column=1, offset=0)
    clause = WhenClause(
        span=span,
        condition=BoolLiteral(span=span, value=True),
        body=ExprStmt(span=span, expression=BoolLiteral(span=span, value=True)),
    )
    resolver = Resolver()
    resolver._resolve_when_clause(clause)
    assert "'when' clause outside a strategy" in _messages(resolver)


def test_when_clause_inside_strategy_is_clean() -> None:
    resolver = _resolve(
        """
        strategy S {
            when a crosses_above b: buy();
        }
        """
    )
    # 'a', 'b', 'buy' are all genuinely undefined here -- that's expected
    # (this test is about strategy/when structure, not name resolution) --
    # so only assert the structural error is absent.
    assert "'when' clause outside a strategy" not in _messages(resolver)


def test_call_target_identifier_not_checked_for_existence() -> None:
    # Builtins (SMA, print, buy, ...) aren't registered anywhere yet --
    # see the resolver's module docstring for why call *targets* are
    # deliberately not existence-checked.
    resolver = _resolve("SMA(1, 2);")
    assert resolver.diagnostics.diagnostics == []


def test_call_arguments_are_still_checked_for_existence() -> None:
    resolver = _resolve("f(undefined_arg);")
    assert "undefined name 'undefined_arg'" in _messages(resolver)


def test_backtest_undeclared_strategy() -> None:
    resolver = _resolve("backtest Ghost on AAPL from 0 to 1;")
    assert "backtest refers to undeclared strategy 'Ghost'" in _messages(resolver)


def test_backtest_declared_strategy_is_clean() -> None:
    resolver = _resolve(
        """
        strategy Cross {
            when a crosses_above b: buy();
        }
        backtest Cross on AAPL from 0 to 1;
        """
    )
    assert "backtest refers to undeclared strategy 'Cross'" not in _messages(resolver)


def test_recursive_function_resolves_its_own_name() -> None:
    resolver = _resolve("func fact(n) { return fact(n - 1); }")
    assert resolver.diagnostics.diagnostics == []
