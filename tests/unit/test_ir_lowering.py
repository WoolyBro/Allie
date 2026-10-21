"""Tests for allie/ir/lowering.py -- Phase 5.2. Structural assertions on
the emitted instruction shapes, not just "the value comes out right"
(that's what test_ir_cross_validation.py is for)."""

from __future__ import annotations

import pytest

from allie.ir.instructions import (
    BinOp,
    Call,
    Jump,
    Label,
    LoadConst,
    LoadVar,
    Return,
    StoreVar,
    UnaryOp,
)
from allie.ir.lowering import LoweringError, lower_program
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser


def _lower(source: str):
    # This file only cares about instruction shapes, not the Phase 6
    # register_types/function_info lower_program also returns -- drop them.
    tokens = Lexer(source).scan_tokens()
    program = Parser(tokens).parse()
    main_program, functions, _register_types, _function_info = lower_program(program)
    return main_program, functions


def test_arithmetic_lowers_to_loadconst_and_binop() -> None:
    main, _ = _lower("let x = 1 + 2;")
    kinds = [type(i).__name__ for i in main.instructions]
    assert kinds == ["LoadConst", "LoadConst", "BinOp", "StoreVar"]
    binop = main.instructions[2]
    assert isinstance(binop, BinOp)
    assert binop.op == "+"


def test_unary_lowers_to_unaryop() -> None:
    main, _ = _lower("let x = -1;")
    assert any(isinstance(i, UnaryOp) and i.op == "-" for i in main.instructions)


def test_registers_are_distinct_virtual_names() -> None:
    main, _ = _lower("let x = 1 + 2;")
    loadconsts = [i for i in main.instructions if isinstance(i, LoadConst)]
    assert loadconsts[0].dest != loadconsts[1].dest
    assert loadconsts[0].dest.startswith("r")


def test_if_else_uses_jumpiffalse_and_labels() -> None:
    main, _ = _lower("if true { let x = 1; } else { let y = 2; }")
    kinds = [type(i).__name__ for i in main.instructions]
    assert "JumpIfFalse" in kinds
    assert kinds.count("Label") == 2  # else_label, end_label
    assert kinds.count("Jump") == 1  # then-branch jumps over the else


def test_if_without_else_jumps_straight_to_end() -> None:
    main, _ = _lower("if true { let x = 1; }")
    kinds = [type(i).__name__ for i in main.instructions]
    assert kinds.count("Label") == 1
    assert kinds.count("Jump") == 0  # no else branch to jump over


def test_while_shape() -> None:
    main, _ = _lower("while x < 5 { let y = 1; }")
    kinds = [type(i).__name__ for i in main.instructions]
    # Label(loop), ... cond ..., JumpIfFalse, body, Jump(back), Label(end)
    assert kinds[0] == "Label"
    assert "JumpIfFalse" in kinds
    assert kinds[-2:] == ["Jump", "Label"]


def test_for_desugars_to_while_shaped_ir() -> None:
    main, _ = _lower("for let i = 0; i < 5; i = i + 1 { let y = 1; }")
    kinds = [type(i).__name__ for i in main.instructions]
    # init LoadConst+StoreVar, then Label/cond/JumpIfFalse/body/
    # Label(increment)/increment-instrs/Jump/Label(end) -- same
    # primitive shape as while, just with an extra Label marking where
    # continue lands.
    assert kinds[:2] == ["LoadConst", "StoreVar"]  # the init: `let i = 0;`
    assert "JumpIfFalse" in kinds
    assert kinds.count("Label") == 3  # loop, increment, end
    assert kinds.count("Jump") == 1  # the back-edge


def test_for_continue_targets_increment_not_condition_check() -> None:
    # The whole point of the increment_label: continue must still run
    # the increment before rechecking the condition.
    main, _ = _lower(
        """
        for let i = 0; i < 5; i = i + 1 {
            continue;
        }
        """
    )
    labels = [i for i in main.instructions if isinstance(i, Label)]
    loop_label, increment_label, _end_label = labels[0].name, labels[1].name, labels[2].name

    continue_jump = next(i for i in main.instructions if isinstance(i, Jump))
    assert continue_jump.label == increment_label
    assert continue_jump.label != loop_label


def test_while_continue_targets_condition_check() -> None:
    # Contrast with for: while has no increment step, so continue jumps
    # straight back to the condition check.
    main, _ = _lower(
        """
        while x < 5 {
            continue;
        }
        """
    )
    loop_label = next(i for i in main.instructions if isinstance(i, Label)).name
    continue_jump = next(i for i in main.instructions if isinstance(i, Jump))
    assert continue_jump.label == loop_label


def test_for_break_targets_end_skipping_increment() -> None:
    main, _ = _lower(
        """
        for let i = 0; i < 5; i = i + 1 {
            break;
        }
        """
    )
    labels = [i for i in main.instructions if isinstance(i, Label)]
    end_label = labels[-1].name
    break_jump = next(i for i in main.instructions if isinstance(i, Jump))
    assert break_jump.label == end_label


def test_break_outside_loop_raises() -> None:
    with pytest.raises(LoweringError):
        _lower("break;")


def test_continue_outside_loop_raises() -> None:
    with pytest.raises(LoweringError):
        _lower("continue;")


def test_logical_and_uses_move_and_jumpiffalse_no_unary_not() -> None:
    main, _ = _lower("let x = true and false;")
    kinds = [type(i).__name__ for i in main.instructions]
    assert kinds.count("Move") == 2  # tentative result, then overwritten
    assert "JumpIfFalse" in kinds
    assert "UnaryOp" not in kinds  # "and" doesn't need the "!" trick


def test_logical_or_uses_move_and_negates_for_jumpiffalse() -> None:
    main, _ = _lower("let x = true or false;")
    kinds = [type(i).__name__ for i in main.instructions]
    assert kinds.count("Move") == 2
    assert "UnaryOp" in kinds  # "or" needs "!" to emulate JumpIfTrue
    assert "JumpIfFalse" in kinds


def test_array_literal_lowers_to_make_series_call() -> None:
    main, _ = _lower("let x = [1, 2, 3];")
    call = next(i for i in main.instructions if isinstance(i, Call))
    assert call.fn_name == "__make_series"
    assert len(call.args) == 3


def test_field_access_lowers_to_memberget() -> None:
    main, _ = _lower("let x = prices.close;")
    member_get = next(i for i in main.instructions if hasattr(i, "field"))
    assert member_get.field == "close"


def test_call_by_name_lowers_directly() -> None:
    main, _ = _lower("let x = mean(prices);")
    call = next(i for i in main.instructions if isinstance(i, Call))
    assert call.fn_name == "mean"
    assert call.dest is not None


def test_return_with_and_without_value() -> None:
    main, functions = _lower("func f() { return 1; } func g() { return; }")
    f_program, _ = functions["f"]
    g_program, _ = functions["g"]
    f_return = next(i for i in f_program.instructions if isinstance(i, Return))
    g_return = next(i for i in g_program.instructions if isinstance(i, Return))
    assert f_return.value is not None
    assert g_return.value is None


def test_func_decl_registers_separate_irprogram_no_instruction_at_def_site() -> None:
    main, functions = _lower("func f(a, b) { return a + b; }")
    assert main.instructions == []  # nothing emitted at the definition site
    assert "f" in functions
    body_program, params = functions["f"]
    assert params == ["a", "b"]
    assert any(isinstance(i, BinOp) for i in body_program.instructions)


def test_recursive_function_calls_itself_by_name() -> None:
    _, functions = _lower(
        """
        func factorial(n) {
            if n <= 1 { return 1; } else { return n * factorial(n - 1); }
        }
        """
    )
    body_program, _ = functions["factorial"]
    call = next(i for i in body_program.instructions if isinstance(i, Call))
    assert call.fn_name == "factorial"


def test_call_through_non_identifier_callee_raises() -> None:
    # Documented limitation: only a plain name can be called -- no
    # first-class function values at the IR level. `(f)()` doesn't
    # exercise this -- parens are transparent in the parser, so `(f)`
    # collapses straight to the Identifier `f`. A call-that-returns-a-
    # callable, `f()()`, is what actually produces a non-Identifier
    # callee (the inner Call node) for the outer call.
    with pytest.raises(LoweringError):
        _lower("let x = f()();")


def test_strategy_and_backtest_lower_to_calls_not_inlined_control_flow() -> None:
    main, functions = _lower(
        """
        strategy S {
            when a crosses_above b: buy();
        }
        backtest S on AAPL from 0 to 1;
        """
    )
    call_names = [i.fn_name for i in main.instructions if isinstance(i, Call)]
    assert "__strategy_declare" in call_names
    assert "__strategy_clause" in call_names
    assert "__backtest_run" in call_names
    # No dedicated strategy/when/backtest instruction: the condition is an
    # ordinary expression (here one BinOp) evaluated where the strategy is
    # declared, and its register is what `__strategy_clause` receives.
    kinds = {type(i).__name__ for i in main.instructions}
    assert kinds <= {"Call", "LoadConst", "LoadVar", "BinOp"}
    (crossing,) = [i for i in main.instructions if isinstance(i, BinOp)]
    assert crossing.op == "crosses_above"
    (clause,) = [
        i for i in main.instructions if isinstance(i, Call) and i.fn_name == "__strategy_clause"
    ]
    assert clause.args[1] == crossing.dest


def test_when_body_lowers_to_a_separate_callable_program_not_inline() -> None:
    # Phase 7 rewrite: the Phase 5.2 lowering emitted the `when` body
    # inline, so it ran unconditionally in sequence instead of per-bar
    # when its signal fired. Each body is now its own zero-arg program,
    # registered like an ordinary function so the engine (and the LLVM
    # backend) can invoke it.
    main, functions = _lower(
        """
        strategy S {
            when a crosses_above b: buy();
            when a crosses_below b: sell();
        }
        """
    )
    assert "__when_body_S_0" in functions
    assert "__when_body_S_1" in functions

    # The bodies' calls live in the body programs, NOT in main.
    main_calls = {i.fn_name for i in main.instructions if isinstance(i, Call)}
    assert "buy" not in main_calls
    assert "sell" not in main_calls

    body0, params0 = functions["__when_body_S_0"]
    assert params0 == []
    assert any(isinstance(i, Call) and i.fn_name == "buy" for i in body0.instructions)


def test_backtest_passes_prices_as_a_value_not_a_name() -> None:
    # The LLVM bridge can't look a variable up by name, so the symbol is
    # loaded into a register and passed as a value. (The Phase 5.2
    # version put the raw strings in the arg list, which the IR
    # interpreter would have read as register names.)
    main, _ = _lower("backtest S on AAPL from 0 to 1;")
    load = next(i for i in main.instructions if isinstance(i, LoadVar) and i.name == "AAPL")
    call = next(i for i in main.instructions if isinstance(i, Call))
    assert call.fn_name == "__backtest_run"
    assert load.dest in call.args
    assert "AAPL" not in call.args  # a name, not a register, must not appear


def test_assignment_expression_returns_the_assigned_value_register() -> None:
    main, _ = _lower("let x = 0; x = 5;")
    store_instrs = [i for i in main.instructions if isinstance(i, StoreVar)]
    assert store_instrs[-1].name == "x"


def test_loadvar_used_for_identifier_reference() -> None:
    main, _ = _lower("let x = 1; let y = x;")
    assert any(isinstance(i, LoadVar) and i.name == "x" for i in main.instructions)
