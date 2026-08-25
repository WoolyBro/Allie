"""Tests for allie/optimizer/ -- Phase 5.4. Hand-built "before" IRPrograms
with an exact instruction-by-instruction "after" assertion, per the spec
-- these are meant to demonstrate the transformation itself, not just
that the output value comes out the same."""

from __future__ import annotations

from allie.ir.instructions import (
    BinOp,
    Call,
    IRProgram,
    Jump,
    JumpIfFalse,
    Label,
    LoadConst,
    LoadVar,
    Return,
    StoreVar,
)
from allie.optimizer import constant_fold, cse, dce

# ---- constant_fold ----


def test_constant_fold_collapses_arithmetic_into_one_loadconst() -> None:
    before = IRProgram(
        instructions=[
            LoadConst(dest="r0", value=10),
            LoadConst(dest="r1", value=2),
            BinOp(dest="r2", op="*", left="r0", right="r1"),
        ]
    )
    after = constant_fold.run(before)
    # r0/r1 are folded away entirely (unused elsewhere); only the
    # computed constant remains.
    assert after.instructions == [LoadConst(dest="r2", value=20)]


def test_constant_fold_keeps_operand_still_used_elsewhere() -> None:
    before = IRProgram(
        instructions=[
            LoadConst(dest="r0", value=10),
            LoadConst(dest="r1", value=2),
            BinOp(dest="r2", op="*", left="r0", right="r1"),
            StoreVar(name="x", src="r0"),  # r0 used again -- must survive
        ]
    )
    after = constant_fold.run(before)
    assert after.instructions == [
        LoadConst(dest="r0", value=10),
        LoadConst(dest="r2", value=20),
        StoreVar(name="x", src="r0"),
    ]


def test_constant_fold_leaves_non_constant_binop_alone() -> None:
    before = IRProgram(
        instructions=[
            LoadVar(dest="r0", name="x"),
            LoadConst(dest="r1", value=2),
            BinOp(dest="r2", op="*", left="r0", right="r1"),
        ]
    )
    after = constant_fold.run(before)
    assert after.instructions == before.instructions


def test_constant_fold_does_not_fold_division_by_zero_at_compile_time() -> None:
    # A folded 1/0 would raise at "compile" time instead of leaving the
    # existing runtime AllieValueError("division by zero") path intact.
    before = IRProgram(
        instructions=[
            LoadConst(dest="r0", value=1),
            LoadConst(dest="r1", value=0),
            BinOp(dest="r2", op="/", left="r0", right="r1"),
        ]
    )
    after = constant_fold.run(before)
    assert any(isinstance(i, BinOp) for i in after.instructions)


# ---- dce ----


def test_dce_removes_stored_but_never_read_variable() -> None:
    before = IRProgram(
        instructions=[
            LoadConst(dest="r0", value=1),
            StoreVar(name="unused", src="r0"),
        ]
    )
    after = dce.run(before)
    assert after.instructions == []


def test_dce_never_removes_a_call_even_if_result_unused() -> None:
    before = IRProgram(
        instructions=[
            LoadVar(dest="r0", name="x"),
            Call(dest="r1", fn_name="print", args=["r0"]),  # r1 never read
        ]
    )
    after = dce.run(before)
    assert after.instructions == before.instructions


def test_dce_does_not_remove_for_loop_increment_store() -> None:
    # The loop-carried-liveness trap: `i`'s condition-check LoadVar comes
    # *before* the increment's StoreVar in program order (the read is
    # only reached again via the back-edge Jump), so a naive
    # forward-only scan would call this StoreVar dead. It must survive.
    before = IRProgram(
        instructions=[
            LoadConst(dest="r0", value=0),
            StoreVar(name="i", src="r0"),  # let i = 0;
            Label(name="loop"),
            LoadVar(dest="r1", name="i"),
            LoadConst(dest="r2", value=5),
            BinOp(dest="r3", op="<", left="r1", right="r2"),
            JumpIfFalse(cond="r3", label="end"),
            Label(name="increment"),
            LoadVar(dest="r4", name="i"),
            LoadConst(dest="r5", value=1),
            BinOp(dest="r6", op="+", left="r4", right="r5"),
            StoreVar(name="i", src="r6"),  # the instruction under test
            Jump(label="loop"),
            Label(name="end"),
        ]
    )
    after = dce.run(before)
    store_i_instrs = [
        i for i in after.instructions if isinstance(i, StoreVar) and i.name == "i"
    ]
    assert len(store_i_instrs) == 2  # the init AND the increment both survive


def test_dce_removes_unread_loadvar() -> None:
    before = IRProgram(
        instructions=[
            LoadVar(dest="r0", name="x"),  # never used
            LoadConst(dest="r1", value=1),
            Return(value="r1"),
        ]
    )
    after = dce.run(before)
    assert after.instructions == [LoadConst(dest="r1", value=1), Return(value="r1")]


# ---- cse ----


def test_cse_collapses_identical_pure_call_with_no_intervening_reassignment() -> None:
    before = IRProgram(
        instructions=[
            LoadVar(dest="r0", name="prices"),
            LoadConst(dest="r1", value=20),
            Call(dest="r2", fn_name="SMA", args=["r0", "r1"]),
            LoadVar(dest="r3", name="prices"),
            LoadConst(dest="r4", value=20),
            Call(dest="r5", fn_name="SMA", args=["r3", "r4"]),
            Return(value="r5"),
        ]
    )
    after = cse.run(before)
    calls = [i for i in after.instructions if isinstance(i, Call)]
    assert len(calls) == 1
    # The Return, resolved through the alias, now points at the first
    # call's register.
    assert after.instructions[-1] == Return(value=calls[0].dest)


def test_cse_does_not_collapse_across_an_intervening_reassignment() -> None:
    before = IRProgram(
        instructions=[
            LoadVar(dest="r0", name="prices"),
            LoadConst(dest="r1", value=20),
            Call(dest="r2", fn_name="SMA", args=["r0", "r1"]),
            LoadVar(dest="r6", name="new_prices"),
            StoreVar(name="prices", src="r6"),  # reassignment in between
            LoadVar(dest="r3", name="prices"),
            LoadConst(dest="r4", value=20),
            Call(dest="r5", fn_name="SMA", args=["r3", "r4"]),
        ]
    )
    after = cse.run(before)
    calls = [i for i in after.instructions if isinstance(i, Call)]
    assert len(calls) == 2  # NOT collapsed -- prices was reassigned in between


def test_cse_never_collapses_impure_calls_even_with_identical_args() -> None:
    before = IRProgram(
        instructions=[
            LoadVar(dest="r0", name="x"),
            Call(dest=None, fn_name="print", args=["r0"]),
            LoadVar(dest="r1", name="x"),
            Call(dest=None, fn_name="print", args=["r1"]),
        ]
    )
    after = cse.run(before)
    calls = [i for i in after.instructions if isinstance(i, Call)]
    assert len(calls) == 2  # print is impure -- PURE_FUNCTIONS is consulted, not ignored


def test_cse_never_collapses_impure_call_with_no_arguments() -> None:
    before = IRProgram(
        instructions=[
            Call(dest=None, fn_name="buy", args=[]),
            Call(dest=None, fn_name="buy", args=[]),
        ]
    )
    after = cse.run(before)
    calls = [i for i in after.instructions if isinstance(i, Call)]
    assert len(calls) == 2


def test_cse_deduplicates_repeated_identical_constants() -> None:
    before = IRProgram(
        instructions=[
            LoadConst(dest="r0", value=20),
            LoadConst(dest="r1", value=20),
            BinOp(dest="r2", op="+", left="r0", right="r1"),
        ]
    )
    after = cse.run(before)
    loadconsts = [i for i in after.instructions if isinstance(i, LoadConst)]
    assert len(loadconsts) == 1


def test_cse_does_not_collide_int_and_float_constants_of_equal_value() -> None:
    before = IRProgram(
        instructions=[
            LoadConst(dest="r0", value=1),
            LoadConst(dest="r1", value=1.0),
            BinOp(dest="r2", op="+", left="r0", right="r1"),
        ]
    )
    after = cse.run(before)
    loadconsts = [i for i in after.instructions if isinstance(i, LoadConst)]
    assert len(loadconsts) == 2  # Int(1) and Float(1.0) are distinct types


def test_cse_does_not_alias_across_a_loop_back_edge() -> None:
    # Found by actually lowering+optimizing a real `for` loop and
    # watching it compute the wrong answer: `total` and `i` both start
    # at the literal 0 before the loop (legitimately sharing one
    # LoadConst), but reading them *inside* the loop body must not
    # reuse that shared pre-loop register -- each iteration's LoadVar
    # has to see whatever the loop body itself last wrote, not a
    # register whose "meaning" predates the loop entirely.
    before = IRProgram(
        instructions=[
            LoadConst(dest="r0", value=0),
            StoreVar(name="total", src="r0"),
            StoreVar(name="i", src="r0"),  # shares r0 with total -- fine so far
            Label(name="loop"),
            LoadVar(dest="r1", name="i"),
            LoadConst(dest="r2", value=5),
            BinOp(dest="r3", op="<", left="r1", right="r2"),
            JumpIfFalse(cond="r3", label="end"),
            LoadVar(dest="r4", name="total"),
            LoadVar(dest="r5", name="i"),
            BinOp(dest="r6", op="+", left="r4", right="r5"),
            StoreVar(name="total", src="r6"),
            LoadVar(dest="r7", name="i"),
            LoadConst(dest="r8", value=1),
            BinOp(dest="r9", op="+", left="r7", right="r8"),
            StoreVar(name="i", src="r9"),
            Jump(label="loop"),
            Label(name="end"),
        ]
    )
    after = cse.run(before)

    # The two LoadVars right after the loop label must be genuinely
    # distinct reads -- not both aliased to the pre-loop r0.
    loads_at_top_of_body = [
        i
        for i in after.instructions
        if isinstance(i, LoadVar) and i.name in ("total", "i")
    ]
    assert len(loads_at_top_of_body) >= 2
    total_load = next(i for i in loads_at_top_of_body if i.name == "total")
    i_load = next(i for i in loads_at_top_of_body if i.name == "i")
    assert total_load.dest != i_load.dest

    # And the BinOp combining them must reference those two distinct
    # registers, not the same one twice.
    expected_operands = {total_load.dest, i_load.dest}
    combine = next(
        i
        for i in after.instructions
        if isinstance(i, BinOp) and i.op == "+" and {i.left, i.right} == expected_operands
    )
    assert combine.left != combine.right


def test_cse_does_not_alias_across_an_if_else_join() -> None:
    # Same root cause as the loop case: the end_label is reached having
    # run only one of the two branches, so a register loaded before the
    # branch can't be assumed to still hold the right value for a read
    # after it if either branch reassigned that variable.
    before = IRProgram(
        instructions=[
            LoadConst(dest="r0", value=1),
            StoreVar(name="x", src="r0"),
            LoadConst(dest="r1", value=0),
            JumpIfFalse(cond="r1", label="else"),
            LoadConst(dest="r2", value=99),
            StoreVar(name="x", src="r2"),
            Jump(label="end"),
            Label(name="else"),
            Label(name="end"),
            LoadVar(dest="r3", name="x"),
            Return(value="r3"),
        ]
    )
    after = cse.run(before)
    final_load = next(
        i for i in reversed(after.instructions) if isinstance(i, LoadVar) and i.name == "x"
    )
    # Must NOT have been aliased back to r0 (the pre-branch value) --
    # it has to be a real load reflecting whichever branch actually ran.
    assert final_load.dest not in ("r0",)
    final_return = next(i for i in after.instructions if isinstance(i, Return))
    assert final_return.value == final_load.dest


# ---- purity classification discipline (Phase 9 cross-cutting check) ----


def test_every_registered_builtin_is_classified_pure_or_impure() -> None:
    """Adding a stdlib builtin without deciding whether it's pure must
    fail the build, not silently default to "never optimized".

    This test exists because the Phase 9 audit found `crosses_above` and
    `crosses_below` (added in Phase 7) in neither set -- so CSE had been
    quietly declining to optimize them, with nothing to flag it.
    """
    import allie.stdlib as stdlib
    from allie.optimizer.pure_functions import KNOWN_IMPURE_FUNCTIONS, PURE_FUNCTIONS

    registered = set(stdlib.BUILTINS)
    unclassified = sorted(registered - PURE_FUNCTIONS - KNOWN_IMPURE_FUNCTIONS)
    assert not unclassified, (
        f"builtins classified as neither pure nor impure: {unclassified}. "
        "Add each to PURE_FUNCTIONS or KNOWN_IMPURE_FUNCTIONS in "
        "allie/optimizer/pure_functions.py."
    )


def test_purity_sets_are_disjoint() -> None:
    from allie.optimizer.pure_functions import KNOWN_IMPURE_FUNCTIONS, PURE_FUNCTIONS

    assert not (PURE_FUNCTIONS & KNOWN_IMPURE_FUNCTIONS)


def test_dce_defines_no_local_notion_of_purity() -> None:
    """The checklist requires that neither pass redefine purity locally.
    `cse.py` imports the shared set; `dce.py` satisfies the same intent
    more strongly by never removing a Call at all, so it has no purity
    concept to disagree with. This pins that: if `dce.py` ever grows its
    own list, the two passes could drift.
    """
    from pathlib import Path

    source = Path("allie/optimizer/dce.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    body = code.split('"""')[-1]  # drop the module docstring
    assert "PURE" not in body, "dce.py appears to define or use a local purity set"


def test_cse_optimizes_a_pure_crossing_call() -> None:
    """Regression for the classification fix: two identical
    `crosses_above` calls with no intervening reassignment must now
    collapse. Before Phase 9 they did not."""
    before = IRProgram(
        instructions=[
            LoadVar(dest="r0", name="fast"),
            LoadVar(dest="r1", name="slow"),
            Call(dest="r2", fn_name="crosses_above", args=["r0", "r1"]),
            LoadVar(dest="r3", name="fast"),
            LoadVar(dest="r4", name="slow"),
            Call(dest="r5", fn_name="crosses_above", args=["r3", "r4"]),
            Return(value="r5"),
        ]
    )
    after = cse.run(before)
    calls = [i for i in after.instructions if isinstance(i, Call)]
    assert len(calls) == 1
