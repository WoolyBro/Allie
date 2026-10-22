"""Common subexpression elimination. `run(program) ->
IRProgram`, pure function, no shared state with the other passes.

A forward pass maintaining three caches, all keyed by *resolved*
register names (register aliases from an earlier elimination are
substituted in before any cache lookup, via `resolve()`):

- `const_cache`: `(type, repr) -> register` for `LoadConst` -- a second
  `LoadConst` of the identical literal aliases the first. Keyed by
  `(type(value), repr(value))`, not just `value`, so `LoadConst(1)`
  (Int) and `LoadConst(1.0)` (Float) don't collide just because
  `1 == 1.0` in Python -- and neither do `0.0` and `-0.0`, which are
  equal (and hash equal) but print differently.
- `var_reg_cache`: `variable name -> register currently holding its
  up-to-date value` -- updated on every fresh `LoadVar` *and* every
  `StoreVar`. This is what makes reassignment correctly invalidate CSE:
  after `StoreVar prices, r6`, `var_reg_cache["prices"]` becomes `r6`, so
  the *next* `LoadVar` of `prices` resolves to `r6` (not whatever
  register held the old value) -- which means a `BinOp`/`Call` cache key
  built from that later `LoadVar` naturally differs from one built
  before the reassignment, with no separate "clear the whole cache on
  any StoreVar" step needed.
- `expr_cache` / `call_cache`: `(op, left, right) -> register` and
  `(fn_name, args) -> register`. Calls
  are only cached when `fn_name` is in
  `allie.optimizer.pure_functions.PURE_FUNCTIONS` -- `print`/`load`/
  `normal` and the order builtins (impure) are never collapsed, imported from that
  single shared location rather than redefined here (see that module's
  docstring for why `dce.py` and this file must not be able to disagree
  about what's pure).

**Every `Label` invalidates every cache**, `const_cache` included.
Without this, a naive linear scan happily "reused" a pre-loop `LoadVar`'s
register for a read *inside* the loop body, because nothing in a single
forward pass distinguishes "this is the same static instruction running
again on iteration 2" from "this is genuinely still the same value" --
verified the hard way, by lowering+optimizing an actual `for` loop
(`total = total + i` inside a loop) and watching the optimized IR compute
the wrong answer because `total` and `i`'s reads got aliased to each
other's since-stale pre-loop constant. The same reasoning applies to an
`if`/`else` join: the `end_label` is reached having executed only one
branch, so a cache entry from either branch is not trustworthy there
either. This is the CSE-side analogue of the loop-carried-liveness trap
DCE's module docstring documents -- same root cause (reasoning about
text order instead of control flow), different pass.

`const_cache` used to survive labels on the theory that a literal's
value is path-independent. The *value* is; the *register* holding it is
not -- it is only written on the path through the instruction that
defined it. `if flag { x = x + 7; } let y = x + 7;` aliased the second
`7` to a register written only inside the `if`, so with `flag` false
the IR interpreter crashed reading an unwritten register. The IR
verifier (allie/ir/verify.py) caught it the first time it ran.

Two more invalidations keep the caches honest:

- A `Call` to anything that isn't a known builtin -- a user function, a
  `when` body run by `__backtest_run` -- may assign a module-level
  variable, so `var_reg_cache` is cleared after it.
- A `Move` re-writes its destination register, so any cache entry
  whose value lives in that register is dropped.
"""

from __future__ import annotations

from typing import TypeVar

from allie.ir.instructions import (
    BinOp,
    Call,
    IRInstr,
    IRProgram,
    JumpIfFalse,
    Label,
    LoadConst,
    LoadVar,
    MemberGet,
    Move,
    Return,
    StoreVar,
    UnaryOp,
    located,
)
from allie.optimizer.pure_functions import PURE_FUNCTIONS, VARIABLE_SAFE_CALLS

_K = TypeVar("_K")


def _forget(cache: dict[_K, str], register: str) -> None:
    """Drops every entry whose value lives in `register`."""
    for key in [k for k, value in cache.items() if value == register]:
        del cache[key]


def run(program: IRProgram) -> IRProgram:
    reg_alias: dict[str, str] = {}
    const_cache: dict[tuple[type, str], str] = {}
    var_reg_cache: dict[str, str] = {}
    expr_cache: dict[tuple[str, str, str], str] = {}
    call_cache: dict[tuple[str, tuple[str, ...]], str] = {}

    def resolve(reg: str) -> str:
        return reg_alias.get(reg, reg)

    kept: list[IRInstr] = []

    for instr in program.instructions:
        if isinstance(instr, Label):
            # A Label is a control-flow join point: a loop's back-edge
            # lands here on iteration 2+ with variables the loop body
            # already reassigned, and an if/else's end_label is reached
            # having run only ONE of the two branches. Either way, a
            # cache entry established before this point can't be trusted
            # after it -- not even a constant's, whose register may only
            # have been written on the other path (see the module
            # docstring). So every cache is cleared at every Label.
            const_cache.clear()
            var_reg_cache.clear()
            expr_cache.clear()
            call_cache.clear()
            kept.append(instr)
            continue

        if isinstance(instr, LoadConst):
            const_key = (type(instr.value), repr(instr.value))
            if const_key in const_cache:
                reg_alias[instr.dest] = const_cache[const_key]
                continue
            const_cache[const_key] = instr.dest
            kept.append(instr)

        elif isinstance(instr, LoadVar):
            if instr.name in var_reg_cache:
                reg_alias[instr.dest] = var_reg_cache[instr.name]
                continue
            var_reg_cache[instr.name] = instr.dest
            kept.append(instr)

        elif isinstance(instr, StoreVar):
            src = resolve(instr.src)
            var_reg_cache[instr.name] = src
            kept.append(StoreVar(name=instr.name, src=src))

        elif isinstance(instr, BinOp):
            left = resolve(instr.left)
            right = resolve(instr.right)
            expr_key = (instr.op, left, right)
            if expr_key in expr_cache:
                reg_alias[instr.dest] = expr_cache[expr_key]
                continue
            expr_cache[expr_key] = instr.dest
            kept.append(located(BinOp(dest=instr.dest, op=instr.op, left=left, right=right), instr))

        elif isinstance(instr, Call) and instr.fn_name in PURE_FUNCTIONS:
            args = tuple(resolve(a) for a in instr.args)
            call_key = (instr.fn_name, args)
            if instr.dest is not None and call_key in call_cache:
                reg_alias[instr.dest] = call_cache[call_key]
                continue
            if instr.dest is not None:
                call_cache[call_key] = instr.dest
            kept.append(
                located(Call(dest=instr.dest, fn_name=instr.fn_name, args=list(args)), instr)
            )

        elif isinstance(instr, UnaryOp):
            kept.append(
                located(
                    UnaryOp(dest=instr.dest, op=instr.op, operand=resolve(instr.operand)), instr
                )
            )

        elif isinstance(instr, Call):
            kept.append(
                located(
                    Call(
                        dest=instr.dest,
                        fn_name=instr.fn_name,
                        args=[resolve(a) for a in instr.args],
                    ),
                    instr,
                )
            )
            if instr.fn_name not in VARIABLE_SAFE_CALLS:
                var_reg_cache.clear()

        elif isinstance(instr, MemberGet):
            kept.append(
                located(
                    MemberGet(dest=instr.dest, obj=resolve(instr.obj), field=instr.field), instr
                )
            )

        elif isinstance(instr, Move):
            kept.append(Move(dest=instr.dest, src=resolve(instr.src)))
            _forget(const_cache, instr.dest)
            _forget(var_reg_cache, instr.dest)
            _forget(expr_cache, instr.dest)
            _forget(call_cache, instr.dest)

        elif isinstance(instr, JumpIfFalse):
            kept.append(JumpIfFalse(cond=resolve(instr.cond), label=instr.label))

        elif isinstance(instr, Return):
            kept.append(Return(value=resolve(instr.value) if instr.value is not None else None))

        else:
            # Label, Jump have no register operands to resolve.
            kept.append(instr)

    return IRProgram(instructions=kept, is_function=program.is_function)
