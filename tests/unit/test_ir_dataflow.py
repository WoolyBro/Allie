"""The control-flow graph, the IR verifier, and the dataflow passes
built on them (constant propagation, liveness-based DCE), plus the CSE
fixes and regressions the verifier and the optimizer fuzz test found.
The older pass-by-pass tests are in test_optimizer.py."""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import allie.stdlib as stdlib
from allie.cli import BACKENDS, run_pipeline
from allie.ir.cfg import CFGError, build_cfg, flatten, without_unreachable_code
from allie.ir.instructions import (
    BinOp,
    Call,
    IRProgram,
    Jump,
    JumpIfFalse,
    Label,
    LoadConst,
    LoadVar,
    Move,
    Return,
    StoreVar,
)
from allie.ir.interpreter import run_ir_program
from allie.ir.lowering import lower_program
from allie.ir.verify import IRVerificationError, verify
from allie.lexer.lexer import Lexer
from allie.optimizer import DEFAULT_PASSES, constant_fold, cse, dce, optimize, run_passes
from allie.parser.parser import Parser
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker

_REPO = Path(__file__).resolve().parents[2]


def _lower(source: str) -> tuple[IRProgram, dict[str, tuple[IRProgram, list[str]]]]:
    program = Parser(Lexer(source).scan_tokens()).parse()
    Resolver().resolve(program)
    checker = TypeChecker()
    for name, signature in stdlib.builtin_function_types().items():
        checker.env.define(name, signature)
    checker.check(program)
    main, functions, _, _ = lower_program(program)
    return main, functions


def _optimized_main(source: str) -> IRProgram:
    main, functions = _lower(source)
    return optimize(main, functions, DEFAULT_PASSES, True, verify=True)[0]


def _printed_args(program: IRProgram) -> list[object]:
    """The constant value of each `print` argument, or None if it isn't a
    LoadConst -- what the optimizer managed to prove."""
    consts = {i.dest: i.value for i in program.instructions if isinstance(i, LoadConst)}
    return [
        consts.get(i.args[0])
        for i in program.instructions
        if isinstance(i, Call) and i.fn_name == "print"
    ]


# ---- control-flow graph ----


def test_straight_line_code_is_one_block() -> None:
    cfg = build_cfg(_lower("let a = 1; let b = a + 2; print(b);")[0])
    assert len(cfg.blocks) == 1
    assert cfg.blocks[0].successors == []


def test_if_else_has_two_branches_that_join() -> None:
    cfg = build_cfg(_lower("let c = true; if c { print(1); } else { print(2); } print(3);")[0])
    entry, then_block, else_block, join = cfg.blocks
    assert entry.successors == [then_block.index, else_block.index]
    assert sorted(join.predecessors) == [then_block.index, else_block.index]


def test_loop_header_has_a_back_edge() -> None:
    cfg = build_cfg(_lower("let i = 0; while i < 3 { i = i + 1; }")[0])
    header = next(b for b in cfg.blocks if b.label is not None and b.label.startswith("while_loop"))
    assert len(header.predecessors) == 2  # entry, and the loop body's jump back


def test_branch_to_the_next_block_has_one_successor() -> None:
    program = IRProgram([LoadConst("r0", True), JumpIfFalse("r0", "L"), Label("L")])
    assert build_cfg(program).blocks[0].successors == [1]


@pytest.mark.parametrize(
    ("instructions", "message"),
    [
        ([Jump("nowhere")], "jump to undefined label 'nowhere'"),
        ([Label("L"), Label("L")], "label 'L' is defined twice"),
    ],
)
def test_malformed_control_flow_is_rejected(instructions: list, message: str) -> None:
    with pytest.raises(CFGError, match=message):
        build_cfg(IRProgram(instructions))


def _programs() -> list[Path]:
    return sorted([*_REPO.glob("examples/*.qs"), *_REPO.glob("tests/**/*.qs")])


@pytest.mark.parametrize("path", _programs(), ids=lambda p: p.name)
def test_every_program_round_trips_and_verifies_after_every_pass(path: Path) -> None:
    try:
        main, functions = _lower(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- a negative example; nothing to lower
        pytest.skip("program does not type-check by design")
    for body in [main, *(f for f, _ in functions.values())]:
        assert flatten(build_cfg(body), is_function=body.is_function) == body
    optimize(main, functions, DEFAULT_PASSES, True, verify=True)  # raises on a bad pass


def test_lowering_drops_unreachable_code() -> None:
    main, functions = _lower(
        "func f(x: Int) -> Int {\n"
        "    for let i: Int = 0; i < 3; i = i + 1 { break; print(i); }\n"
        "    if x > 0 { return 1; } else { return 2; }\n"
        "    print(x);\n"
        "}"
    )
    body = functions["f"][0]
    assert without_unreachable_code(body) == body
    assert not any(isinstance(i, Call) and i.fn_name == "print" for i in body.instructions)


# ---- verifier ----


@pytest.mark.parametrize(
    ("instructions", "message"),
    [
        ([LoadConst("r0", 1), LoadConst("r0", 2)], "register r0 is written twice"),
        (
            [LoadConst("r0", 1), Move("r0", "r0")],
            "register r0 is written by both a Move and another instruction",
        ),
        ([Return("r9")], "reads r9 before it is written"),
        (
            # r1 is written only on the path through the `if`.
            [
                LoadConst("r0", True),
                JumpIfFalse("r0", "end"),
                LoadConst("r1", 5),
                Label("end"),
                Return("r1"),
            ],
            "Return(value='r1') reads r1 before it is written",
        ),
        ([Jump("missing")], "jump to undefined label 'missing'"),
    ],
    ids=["double-write", "move-and-other", "undefined", "one-path-only", "bad-label"],
)
def test_verifier_rejects_broken_ir(instructions: list, message: str) -> None:
    with pytest.raises(IRVerificationError, match=re.escape(message)):
        verify(IRProgram(instructions))


def test_verifier_checks_call_arity_against_the_function_table() -> None:
    functions = {"f": (IRProgram([Return(None)], is_function=True), ["a", "b"])}
    with pytest.raises(IRVerificationError, match="passes 1 argument"):
        verify(IRProgram([LoadConst("r0", 1), Call(None, "f", ["r0"])]), functions=functions)


def test_run_passes_names_the_pass_that_broke_the_ir() -> None:
    def drop_first(program: IRProgram) -> IRProgram:
        return IRProgram(program.instructions[1:])

    drop_first.__module__ = "allie.optimizer.broken_pass"
    program = IRProgram([LoadConst("r0", 1), Return("r0")])
    with pytest.raises(IRVerificationError, match="main, after broken_pass"):
        run_passes(program, [drop_first], True, verify=True)


def test_run_passes_reports_what_each_pass_removed() -> None:
    main, functions = _lower("let a = 2; let b = a * 3; print(b);")
    report: list[str] = []
    optimize(main, functions, DEFAULT_PASSES, True, report=report)
    assert [line.split(":")[1].split()[0] for line in report] == ["constant_fold", "dce", "cse"]
    assert report[0].startswith("main: constant_fold ")


# ---- constant propagation ----


def test_constants_propagate_through_variables() -> None:
    program = _optimized_main("let a: Int = 2;\nlet b: Int = a * 3;\nprint(b + 1);")
    assert _printed_args(program) == [7]
    assert not any(isinstance(i, BinOp) for i in program.instructions)


def test_a_constant_branch_is_folded_away() -> None:
    program = _optimized_main(
        'let debug = false;\nif debug { print("on"); } else { print("off"); }'
    )
    assert _printed_args(program) == ["off"]
    assert not any(isinstance(i, JumpIfFalse) for i in program.instructions)


def test_a_loop_carried_variable_is_not_a_constant() -> None:
    source = "let i: Int = 0;\nwhile i < 3 { i = i + 1; }\nprint(i);"
    assert _printed_args(_optimized_main(source)) == [None]


@pytest.mark.parametrize(
    "branch_values",
    [("0.0", "-0.0"), ("1", "1.0")],
    ids=["signed-zero", "int-vs-float"],
)
def test_join_distinguishes_values_python_considers_equal(branch_values: tuple[str, str]) -> None:
    first, second = branch_values
    program = IRProgram(
        [
            LoadVar("r0", "flag"),
            JumpIfFalse("r0", "else"),
            LoadConst("r1", eval(first)),
            StoreVar("x", "r1"),
            Jump("end"),
            Label("else"),
            LoadConst("r2", eval(second)),
            StoreVar("x", "r2"),
            Label("end"),
            LoadVar("r3", "x"),
            Call(None, "print", ["r3"]),
        ]
    )
    folded = constant_fold.run(program)
    assert LoadVar("r3", "x") in folded.instructions  # not folded to one constant


def test_a_user_function_call_may_change_any_variable() -> None:
    program = IRProgram(
        [
            LoadConst("r0", 1),
            StoreVar("x", "r0"),
            Call(None, "bump_x", []),
            LoadVar("r1", "x"),
            Call(None, "print", ["r1"]),
        ]
    )
    assert LoadVar("r1", "x") in constant_fold.run(program).instructions


def test_in_a_function_an_unassigned_variable_is_unknown() -> None:
    body = IRProgram([LoadVar("r0", "rate"), Return("r0")], is_function=True)
    assert constant_fold.run(body).instructions == body.instructions


@pytest.mark.parametrize(
    ("left", "op", "right"),
    [(1, "/", 0), (7, "%", 0), (2**62, "*", 4), (2**63 - 1, "+", 1)],
    ids=["div-zero", "mod-zero", "mul-overflow", "add-overflow"],
)
def test_operations_that_raise_at_runtime_are_not_folded(left: int, op: str, right: int) -> None:
    program = IRProgram(
        [LoadConst("r0", left), LoadConst("r1", right), BinOp("r2", op, "r0", "r1"), Return("r2")]
    )
    assert any(isinstance(i, BinOp) for i in constant_fold.run(program).instructions)


# ---- dead-code elimination ----


def test_an_overwritten_store_is_dead() -> None:
    program = _optimized_main("let x: Int = 1;\nx = 2;\nprint(x);")
    assert _printed_args(program) == [2]
    assert not any(isinstance(i, StoreVar) for i in program.instructions)


def test_a_store_overwritten_before_a_user_call_is_still_dead() -> None:
    program = IRProgram(
        [
            LoadConst("r0", 1),
            StoreVar("x", "r0"),
            LoadConst("r1", 2),
            StoreVar("x", "r1"),
            Call(None, "reads_x", []),
        ]
    )
    stores = [i for i in dce.run(program).instructions if isinstance(i, StoreVar)]
    assert stores == [StoreVar("x", "r1")]  # the call may read x; it can't see the first store


def test_at_a_function_exit_every_variable_is_live() -> None:
    body = IRProgram([LoadConst("r0", 1), StoreVar("total", "r0"), Return(None)], is_function=True)
    assert StoreVar("total", "r0") in dce.run(body).instructions


def test_code_after_return_and_redundant_jumps_are_removed() -> None:
    program = IRProgram(
        [
            LoadConst("r0", 1),
            Jump("next"),
            Label("next"),
            Return("r0"),
            Call(None, "print", ["r0"]),
        ]
    )
    assert dce.run(program).instructions == [LoadConst("r0", 1), Return("r0")]


def test_overflowing_arithmetic_is_kept_even_if_unused() -> None:
    program = IRProgram([LoadVar("r0", "a"), BinOp("r1", "*", "r0", "r0"), Return(None)])
    assert BinOp("r1", "*", "r0", "r0") in dce.run(program).instructions


# ---- CSE fixes ----


def test_cse_does_not_reuse_a_constant_written_only_inside_a_branch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Found by the IR verifier the first time it ran: the second `7`
    was aliased to a register written only inside the `if`, so with
    `flag` false the IR backend crashed reading an unwritten register."""
    source = "let x: Int = 0;\nlet flag: Bool = false;\nif flag { x = x + 7; }\nprint(x + 7);\n"
    path = tmp_path / "p.qs"
    path.write_text(source, encoding="utf-8")
    for backend in BACKENDS:
        assert run_pipeline(path, backend=backend) == 0, backend
        assert capsys.readouterr().out.strip() == "7", backend


def test_cse_keeps_signed_zeros_apart() -> None:
    program = IRProgram(
        [LoadConst("r0", 0.0), LoadConst("r1", -0.0), BinOp("r2", "+", "r0", "r1"), Return("r2")]
    )
    assert LoadConst("r1", -0.0) in cse.run(program).instructions


def test_cse_forgets_variables_after_a_user_function_call() -> None:
    program = IRProgram(
        [
            LoadVar("r0", "x"),
            Call(None, "bump_x", []),
            LoadVar("r1", "x"),
            BinOp("r2", "+", "r0", "r1"),
            Return("r2"),
        ]
    )
    assert LoadVar("r1", "x") in cse.run(program).instructions


def test_cse_forgets_a_cached_value_whose_register_a_move_rewrites() -> None:
    program = IRProgram(
        [
            LoadConst("r0", 1),
            Move("m", "r0"),
            StoreVar("x", "m"),
            LoadConst("r1", 2),
            Move("m", "r1"),
            LoadVar("r2", "x"),  # x is still 1, though `m` now holds 2
            Return("r2"),
        ]
    )
    optimized = cse.run(program)
    assert LoadVar("r2", "x") in optimized.instructions
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = run_ir_program(optimized, stdlib.create_global_environment(), {}, {})
    assert result == 1


# ---- widening made explicit ----


def test_widening_happens_identically_on_every_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Used to print `1` on the interpreters and crash LLVM, which passed
    an i64 to a double parameter."""
    source = (
        "let x: Float = 1;\n"
        "print(x);\n"
        "func half(a: Float) -> Float { return a / 2.0; }\n"
        "print(half(3));\n"
        "func one() -> Float { return 1; }\n"
        "print(one());\n"
    )
    path = tmp_path / "p.qs"
    path.write_text(source, encoding="utf-8")
    for backend in BACKENDS:
        assert run_pipeline(path, backend=backend) == 0, backend
        assert capsys.readouterr().out.split() == ["1.0", "1.5", "1.0"], backend
