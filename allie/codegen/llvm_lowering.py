"""Custom IR -> LLVM IR lowering -- Phase 6.3.

## Type mapping (Phase 6.2)

| Allie type | LLVM type |
|---|---|
| `Int` | `i64` |
| `Float` | `double` |
| `Bool` | `i1` |
| `Void` | `void` |
| everything else (`String`, quant types, `Any`, `FunctionType`) | `i64` -- opaque boxed handle |

**Deviation from the literal spec, documented**: `String -> i8*` (a
pointer to a global constant) was the literal ask, scoped to "no dynamic
string ops needed yet". Allie's `+` supports runtime string
concatenation (already proven in the tree-walker/IR-interpreter tests),
so a string's value isn't always a compile-time constant a fixed global
could hold. Every non-native-scalar type uses the SAME boxed-`i64`-handle
scheme uniformly (see `allie.codegen.runtime_bridge`) instead of a
special case just for strings -- one mechanism, not two.

**Series/Matrix/Vector native codegen is explicitly out of scope for v1**
(also per the spec): calls to `SMA`, `mean`, etc. are lowered as external
calls back into the Python/numpy runtime via `runtime_bridge.
call_builtin_bridge`, never reimplemented in LLVM IR.

## Self-check confirming the Phase 5.2 desugar (Phase 6.3's own ask)

Traced a `for` loop (`sum_to_n`, see test_llvm_lowering.py) through this
lowering by hand: zero `for`-specific code exists here. `Label`/`Jump`/
`JumpIfFalse` become basic blocks and `br`/`cbranch` uniformly for
`if`/`while`/`for` alike, exactly because Phase 5.2 desugared `for` onto
`while`'s shape already. If a `for`-specific branch had crept in here,
that would mean the Phase 5.2 desugar wasn't applied consistently --
the fix would belong there, not here. None did.

## Native vs. bridged dispatch

A `BinOp`/`UnaryOp` compiles to a direct LLVM instruction (`fadd`/`add`/
`icmp_signed`/`fcmp_ordered`/...) only when every operand's recorded
type (`allie.ir.lowering`'s Phase 6 `register_types` retrofit) is a
native scalar (`Int`/`Float`/`Bool`); otherwise it's bridged through
`runtime_bridge.binop_bridge`/`unaryop_bridge`, which call the exact
same `_apply_binop`/`_apply_unary` the IR interpreter uses -- no third
reimplementation of operator semantics.

A `Call` to a name in `function_info` (a user-defined function with a
known, fully-scalar signature) compiles to a direct native LLVM `call`
between the two corresponding LLVM functions (this is also how
recursion works -- `factorial`/`fib` call themselves exactly like any
other native call). A `Call` to anything else (a stdlib builtin,
`__make_series`, or a strategy/backtest/when stub) goes through
`runtime_bridge.call_builtin_bridge`, boxing every native-scalar
argument first and unboxing the result if its type is a native scalar.

**Narrower than the tree-walker/IR interpreter, on purpose**: a
user-defined function with a non-scalar (Series/Matrix/...) parameter or
return type raises `LLVMLoweringError` rather than attempting boxed
function-boundary marshalling nothing in this project's test suite
actually needs (every real program -- `factorial`, `fib`, the quant
integration tests -- calls stdlib functions for anything non-scalar,
never a *user*-defined one).
"""

from __future__ import annotations

import llvmlite.ir as ir

from allie.codegen.runtime_bridge import box
from allie.ir.instructions import (
    BinOp,
    Call,
    IRProgram,
    Jump,
    JumpIfFalse,
    Label,
    LoadConst,
    LoadVar,
    MemberGet,
    Move,
    Return,
    StoreVar,
    UnaryOp,
)
from allie.ir.lowering import MAKE_SERIES_FN, FunctionInfo
from allie.typechecker.types import BOOL, FLOAT, INT, VOID, Type

I64 = ir.IntType(64)
DOUBLE = ir.DoubleType()
BOOL_T = ir.IntType(1)

_MAX_BRIDGED_ARITY = 4

_NATIVE_SCALARS = (INT, FLOAT, BOOL)


def _is_native_scalar(t: Type | None) -> bool:
    return t in _NATIVE_SCALARS


def llvm_type_for(t: Type) -> ir.Type:
    if t == INT:
        return I64
    if t == FLOAT:
        return DOUBLE
    if t == BOOL:
        return BOOL_T
    if t == VOID:
        return ir.VoidType()
    return I64  # boxed handle -- see this module's docstring


class LLVMLoweringError(Exception):
    """Raised for IR shapes this lowering can't (or deliberately won't)
    compile -- see this module's docstring for the documented scope
    limits (non-scalar user-function boundaries, more than
    `_MAX_BRIDGED_ARITY` arguments to a bridged call)."""


class LLVMLowering:
    def __init__(
        self,
        functions: dict[str, tuple[IRProgram, list[str]]],
        function_info: dict[str, FunctionInfo],
    ) -> None:
        self.module = ir.Module(name="allie_module")
        self.functions = functions
        self.function_info = function_info
        self.builder: ir.IRBuilder
        self.register_types: dict[str, Type] = {}
        self.reg_values: dict[str, ir.Value] = {}
        self.var_ptrs: dict[str, ir.Value] = {}
        self.var_types: dict[str, Type] = {}
        self._bridge_fns: dict[str, ir.Function] = {}
        self._reached_blocks: set[int] = set()
        self._move_slots: dict[str, ir.Value] = {}
        self._current_return_type: Type = VOID

    def _read_reg(self, reg: str) -> ir.Value:
        """Reads a virtual register's current LLVM value. A register
        written by `Move` lives in a stack slot rather than as a direct
        SSA value -- see `_lower_move` -- so it's loaded on each read."""
        if reg in self._move_slots:
            return self.builder.load(self._move_slots[reg], name=f"{reg}_val")
        return self.reg_values[reg]

    # ---- top-level entry points ----

    def declare_user_functions(self) -> None:
        """Pre-declares every scalar-signature user function's LLVM
        signature, so forward references and recursive calls can be
        emitted before (or while) their bodies are lowered."""
        for name, info in self.function_info.items():
            if name in self.module.globals:
                continue
            if not all(_is_native_scalar(t) for t in info.param_types):
                continue  # non-scalar param -- lower_function will raise if actually called
            if not (_is_native_scalar(info.return_type) or info.return_type == VOID):
                continue
            param_types = [llvm_type_for(t) for t in info.param_types]
            ret_type = llvm_type_for(info.return_type)
            ir.Function(self.module, ir.FunctionType(ret_type, param_types), name=name)

    def lower_function(self, name: str) -> None:
        info = self.function_info[name]
        if not all(_is_native_scalar(t) for t in info.param_types) or not (
            _is_native_scalar(info.return_type) or info.return_type == VOID
        ):
            raise LLVMLoweringError(
                f"function '{name}' has a non-scalar parameter or return type -- "
                "not supported by the LLVM backend (see this module's docstring)"
            )
        program, param_names = self.functions[name]
        func = self.module.get_global(name)
        assert isinstance(func, ir.Function)

        self.register_types = info.register_types
        self.reg_values = {}
        self.var_ptrs = {}
        self.var_types = {}
        self._move_slots = {}
        self._current_return_type = info.return_type

        entry_block = func.append_basic_block("entry")
        self._reached_blocks = {id(entry_block)}
        self.builder = ir.IRBuilder(entry_block)
        for pname, ptype, arg in zip(param_names, info.param_types, func.args, strict=True):
            ptr = self.builder.alloca(llvm_type_for(ptype), name=f"{pname}_ptr")
            self.builder.store(arg, ptr)
            self.var_ptrs[pname] = ptr
            self.var_types[pname] = ptype

        self._lower_instructions(program.instructions, func)
        self._finalize_function(func, info.return_type, name)

    def lower_main(self, program: IRProgram, register_types: dict[str, Type]) -> ir.Function:
        """Wraps the top-level program statements in an implicit
        void-returning `__allie_main` function."""
        func = ir.Function(self.module, ir.FunctionType(ir.VoidType(), []), name="__allie_main")
        entry_block = func.append_basic_block("entry")
        self._reached_blocks = {id(entry_block)}
        self.builder = ir.IRBuilder(entry_block)
        self.register_types = register_types
        self.reg_values = {}
        self.var_ptrs = {}
        self.var_types = {}
        self._move_slots = {}
        self._current_return_type = VOID

        self._lower_instructions(program.instructions, func)
        self._finalize_function(func, VOID, "__allie_main")
        return func

    def _finalize_function(self, func: ir.Function, return_type: Type, name: str) -> None:
        """LLVM requires every basic block to end in a terminator, even
        one nothing branches to: a `for`/`if` whose every live path
        already returned still leaves the `end_label` block our own
        lowering (Phase 5.2) always emits sitting empty and un-branched-
        to. Any block never recorded in `_reached_blocks` is exactly
        that -- genuinely unreachable, not a missing `return` -- so it
        gets an explicit `unreachable` terminator. The one block that
        *is* live and still lacks a terminator is the function falling
        off its natural end: `ret_void` for a Void function (the normal,
        intentional case of never writing an explicit trailing
        `return;`), or an error for anything else -- which Phase 3's
        "does every path return" check should already have caught, so
        reaching it here would mean these two checks disagree, not that
        the user's program is actually wrong.
        """
        for block in func.basic_blocks:
            if block.is_terminated:
                continue
            if id(block) in self._reached_blocks:
                if return_type == VOID:
                    ir.IRBuilder(block).ret_void()
                else:
                    raise LLVMLoweringError(f"function '{name}' does not return on every path")
            else:
                ir.IRBuilder(block).unreachable()

    # ---- instruction-list -> basic blocks ----

    def _alloc_move_slots(self, instructions: list) -> None:  # noqa: ANN001
        """Allocates one entry-block stack slot per `Move` destination,
        before any instruction is lowered -- see `_lower_move` for why
        this can't happen lazily at the Move itself."""
        for instr in instructions:
            if not isinstance(instr, Move) or instr.dest in self._move_slots:
                continue
            slot_type = self.register_types.get(instr.dest) or self.register_types.get(instr.src)
            self._move_slots[instr.dest] = self.builder.alloca(
                llvm_type_for(slot_type) if slot_type is not None else I64,
                name=f"{instr.dest}_slot",
            )

    def _lower_instructions(self, instructions: list, func: ir.Function) -> None:  # noqa: ANN001
        self._alloc_move_slots(instructions)

        blocks: dict[str, ir.Block] = {}
        for instr in instructions:
            if isinstance(instr, Label):
                blocks[instr.name] = func.append_basic_block(instr.name)

        for instr in instructions:
            if isinstance(instr, Label):
                if not self.builder.block.is_terminated:
                    self.builder.branch(blocks[instr.name])
                    self._reached_blocks.add(id(blocks[instr.name]))
                self.builder.position_at_end(blocks[instr.name])
                continue
            if self.builder.block.is_terminated:
                # Unreachable: e.g. both branches of an `if` end in
                # `return`, so the unconditional `Jump(end_label)` our
                # own lowering (Phase 5.2) always emits after the
                # then-branch never actually runs. LLVM requires a
                # terminator to be a basic block's *last* instruction,
                # so nothing here can be emitted into this block -- skip
                # straight to the next Label, which starts a fresh block.
                continue
            self._lower_one(instr, blocks, func)

    def _lower_one(self, instr: object, blocks: dict[str, ir.Block], func: ir.Function) -> None:
        if isinstance(instr, LoadConst):
            self._lower_load_const(instr)
        elif isinstance(instr, LoadVar):
            self._lower_load_var(instr)
        elif isinstance(instr, StoreVar):
            self._lower_store_var(instr)
        elif isinstance(instr, Move):
            self._lower_move(instr)
        elif isinstance(instr, BinOp):
            self._lower_binop(instr)
        elif isinstance(instr, UnaryOp):
            self._lower_unaryop(instr)
        elif isinstance(instr, MemberGet):
            self._lower_member_get(instr)
        elif isinstance(instr, Call):
            self._lower_call(instr)
        elif isinstance(instr, Jump):
            self.builder.branch(blocks[instr.label])
            self._reached_blocks.add(id(blocks[instr.label]))
        elif isinstance(instr, JumpIfFalse):
            cond_type = self.register_types.get(instr.cond)
            cond_val = self._as_bool(self._read_reg(instr.cond), cond_type)
            continue_block = func.append_basic_block("cont")
            self.builder.cbranch(cond_val, continue_block, blocks[instr.label])
            self._reached_blocks.add(id(continue_block))
            self._reached_blocks.add(id(blocks[instr.label]))
            self.builder.position_at_end(continue_block)
        elif isinstance(instr, Return):
            if instr.value is None:
                self.builder.ret_void()
            else:
                self.builder.ret(self._read_reg(instr.value))
        else:
            raise LLVMLoweringError(f"unknown IR instruction: {type(instr).__name__}")

    # ---- values ----

    def _lower_load_const(self, instr: LoadConst) -> None:
        reg_type = self.register_types.get(instr.dest)
        if reg_type == INT:
            self.reg_values[instr.dest] = ir.Constant(I64, instr.value)
        elif reg_type == FLOAT:
            self.reg_values[instr.dest] = ir.Constant(DOUBLE, instr.value)
        elif reg_type == BOOL:
            self.reg_values[instr.dest] = ir.Constant(BOOL_T, int(instr.value))
        else:
            # Non-scalar constant (currently only string literals) --
            # box it once, now, at compile time; the handle is a fixed
            # value for the rest of this program's lifetime.
            self.reg_values[instr.dest] = ir.Constant(I64, box(instr.value))

    def _lower_move(self, instr: Move) -> None:
        """`Move` is only ever emitted by short-circuit `and`/`or`
        lowering (Phase 5.2), where TWO Moves write the same destination
        register from different branches -- one before the
        short-circuit `JumpIfFalse`, one after evaluating the right
        operand. Aliasing `reg_values[dest] = reg_values[src]` breaks
        LLVM's SSA dominance rule there: the value computed on the
        not-taken path doesn't dominate its use at the join point
        ("Instruction does not dominate all uses"). So a Move's
        destination gets a real stack slot instead -- store on each
        write, load on first read -- which is exactly what an `alloca`
        is for, and what LLVM's own mem2reg pass would turn back into a
        phi node if it ran.

        The slot itself is allocated up front by `_alloc_move_slots`,
        not here: an `alloca` must sit in the entry block *before* its
        terminator, and by the time a Move is reached mid-lowering the
        entry block is usually already closed -- appending then puts the
        `alloca` after the `br`, which is invalid IR ("expected
        instruction opcode" at parse time).
        """
        self.builder.store(self._read_reg(instr.src), self._move_slots[instr.dest])

    def _lower_load_var(self, instr: LoadVar) -> None:
        ptr = self.var_ptrs[instr.name]
        self.reg_values[instr.dest] = self.builder.load(ptr, name=instr.dest)

    def _lower_store_var(self, instr: StoreVar) -> None:
        src_val = self._read_reg(instr.src)
        src_type = self.register_types.get(instr.src)

        if instr.name not in self.var_ptrs:
            var_type = src_type if src_type is not None else INT
            ptr = self.builder.alloca(llvm_type_for(var_type), name=f"{instr.name}_ptr")
            self.var_ptrs[instr.name] = ptr
            self.var_types[instr.name] = var_type

        declared_type = self.var_types[instr.name]
        if declared_type == FLOAT and src_type == INT:
            # Int -> Float widening on reassignment (is_assignable allows
            # it; the LLVM value itself still needs an explicit convert).
            src_val = self.builder.sitofp(src_val, DOUBLE)
        self.builder.store(src_val, self.var_ptrs[instr.name])

    def _as_bool(self, value: ir.Value, value_type: Type | None) -> ir.Value:
        if value_type == BOOL:
            return value
        raise LLVMLoweringError(f"condition must be a native Bool, got {value_type}")

    # ---- operators ----

    def _lower_binop(self, instr: BinOp) -> None:
        left_type = self.register_types.get(instr.left)
        right_type = self.register_types.get(instr.right)

        if _is_native_scalar(left_type) and _is_native_scalar(right_type):
            left_val = self._read_reg(instr.left)
            right_val = self._read_reg(instr.right)
            self.reg_values[instr.dest] = self._native_binop(
                instr.op, left_val, left_type, right_val, right_type
            )
            return

        op_handle = ir.Constant(I64, box(instr.op))
        left_handle = self._box_value(self._read_reg(instr.left), left_type)
        right_handle = self._box_value(self._read_reg(instr.right), right_type)
        fn = self._declare_bridge("binop_bridge", ir.FunctionType(I64, [I64, I64, I64]))
        result_handle = self.builder.call(fn, [op_handle, left_handle, right_handle])
        self.reg_values[instr.dest] = self._unbox_value(
            result_handle, self.register_types.get(instr.dest)
        )

    def _native_binop(
        self,
        op: str,
        left: ir.Value,
        left_type: Type | None,
        right: ir.Value,
        right_type: Type | None,
    ) -> ir.Value:
        is_float = FLOAT in (left_type, right_type)
        if is_float:
            if left_type == INT:
                left = self.builder.sitofp(left, DOUBLE)
            if right_type == INT:
                right = self.builder.sitofp(right, DOUBLE)

        if op == "+":
            return self.builder.fadd(left, right) if is_float else self.builder.add(left, right)
        if op == "-":
            return self.builder.fsub(left, right) if is_float else self.builder.sub(left, right)
        if op == "*":
            return self.builder.fmul(left, right) if is_float else self.builder.mul(left, right)
        if op == "/":
            self._guard_nonzero_divisor(right, is_float, "division by zero")
            return self.builder.fdiv(left, right) if is_float else self.builder.sdiv(left, right)
        if op == "%":
            self._guard_nonzero_divisor(right, is_float, "modulo by zero")
            return self.builder.frem(left, right) if is_float else self.builder.srem(left, right)
        if op in ("==", "!=", "<", ">", "<=", ">="):
            return (
                self.builder.fcmp_ordered(op, left, right)
                if is_float
                else self.builder.icmp_signed(op, left, right)
            )
        raise LLVMLoweringError(f"unknown native binary operator: {op!r}")

    def _guard_nonzero_divisor(self, divisor: ir.Value, is_float: bool, message: str) -> None:
        """Emits a branch that calls `raise_runtime_error_bridge` when
        `divisor` is zero, instead of letting a native `sdiv`/`srem` by
        zero execute. Integer division by zero is *undefined behavior*
        in LLVM (in practice, a hardware trap that kills the process),
        not a catchable error -- so the check has to happen before the
        instruction, not around it. Matches the tree-walking and IR
        interpreters, which both raise AllieValueError here.
        """
        zero = ir.Constant(DOUBLE, 0.0) if is_float else ir.Constant(I64, 0)
        is_zero = (
            self.builder.fcmp_ordered("==", divisor, zero)
            if is_float
            else self.builder.icmp_signed("==", divisor, zero)
        )
        raise_block = self.builder.append_basic_block("div_by_zero")
        continue_block = self.builder.append_basic_block("div_ok")
        self.builder.cbranch(is_zero, raise_block, continue_block)
        self._reached_blocks.add(id(raise_block))
        self._reached_blocks.add(id(continue_block))

        self.builder.position_at_end(raise_block)
        fn = self._declare_bridge("raise_runtime_error_bridge", ir.FunctionType(I64, [I64]))
        self.builder.call(fn, [ir.Constant(I64, box(message))])
        # The bridge records the error and RETURNS (it can't raise across
        # the ctypes boundary -- see runtime_bridge's docstring), so this
        # block must not fall through to the division: executing the
        # `sdiv`/`srem` anyway is the exact undefined behavior the guard
        # exists to prevent. Bail out of the whole function instead;
        # jit.compile_and_run re-raises the recorded error as soon as
        # the JIT call returns, so the abandoned return value is never
        # observed.
        self._emit_early_return()

        self.builder.position_at_end(continue_block)

    def _emit_early_return(self) -> None:
        """Returns from the function currently being lowered, with a
        dummy value of the right type when it isn't void."""
        if self._current_return_type == VOID:
            self.builder.ret_void()
            return
        llvm_ret = llvm_type_for(self._current_return_type)
        dummy = (
            ir.Constant(llvm_ret, 0.0)
            if self._current_return_type == FLOAT
            else ir.Constant(llvm_ret, 0)
        )
        self.builder.ret(dummy)

    def _lower_unaryop(self, instr: UnaryOp) -> None:
        operand_type = self.register_types.get(instr.operand)
        if _is_native_scalar(operand_type):
            operand_val = self._read_reg(instr.operand)
            if instr.op == "-":
                if operand_type == FLOAT:
                    self.reg_values[instr.dest] = self.builder.fsub(
                        ir.Constant(DOUBLE, 0.0), operand_val
                    )
                else:
                    self.reg_values[instr.dest] = self.builder.sub(
                        ir.Constant(I64, 0), operand_val
                    )
                return
            if instr.op == "!":
                self.reg_values[instr.dest] = self.builder.not_(operand_val)
                return
            raise LLVMLoweringError(f"unknown native unary operator: {instr.op!r}")

        op_handle = ir.Constant(I64, box(instr.op))
        operand_handle = self._box_value(self._read_reg(instr.operand), operand_type)
        fn = self._declare_bridge("unaryop_bridge", ir.FunctionType(I64, [I64, I64]))
        result_handle = self.builder.call(fn, [op_handle, operand_handle])
        self.reg_values[instr.dest] = self._unbox_value(
            result_handle, self.register_types.get(instr.dest)
        )

    def _lower_member_get(self, instr: MemberGet) -> None:
        # PriceSeriesValue field access always crosses the boundary --
        # bridged, same generic call mechanism as everything else.
        obj_handle = self._read_reg(instr.obj)  # already boxed (PriceSeries is never scalar)
        field_handle = ir.Constant(I64, box(instr.field))
        fn = self._declare_bridge(
            "member_get_bridge", ir.FunctionType(I64, [I64, I64])
        )
        result_handle = self.builder.call(fn, [obj_handle, field_handle])
        self.reg_values[instr.dest] = self._unbox_value(
            result_handle, self.register_types.get(instr.dest)
        )

    def _lower_call(self, instr: Call) -> None:
        if instr.fn_name in self.function_info and instr.fn_name in self.module.globals:
            self._lower_native_call(instr)
        else:
            self._lower_bridged_call(instr)

    def _lower_native_call(self, instr: Call) -> None:
        info = self.function_info[instr.fn_name]
        target = self.module.get_global(instr.fn_name)
        assert isinstance(target, ir.Function)
        args = [self._read_reg(a) for a in instr.args]
        result = self.builder.call(target, args)
        if instr.dest is not None and info.return_type != VOID:
            self.reg_values[instr.dest] = result

    def _lower_bridged_call(self, instr: Call) -> None:
        if instr.fn_name == MAKE_SERIES_FN:
            self._lower_make_series(instr)
            return
        if len(instr.args) > _MAX_BRIDGED_ARITY:
            raise LLVMLoweringError(
                f"bridged call to '{instr.fn_name}' has more than "
                f"{_MAX_BRIDGED_ARITY} arguments"
            )
        name_handle = ir.Constant(I64, box(instr.fn_name))
        arg_handles = [
            self._box_value(self._read_reg(reg), self.register_types.get(reg))
            for reg in instr.args
        ]
        while len(arg_handles) < _MAX_BRIDGED_ARITY:
            arg_handles.append(ir.Constant(I64, 0))
        argc = ir.Constant(I64, len(instr.args))

        fn = self._declare_bridge("call_builtin_bridge", ir.FunctionType(I64, [I64] * 6))
        result_handle = self.builder.call(fn, [name_handle, *arg_handles, argc])

        if instr.dest is not None:
            self.reg_values[instr.dest] = self._unbox_value(
                result_handle, self.register_types.get(instr.dest)
            )

    def _lower_make_series(self, instr: Call) -> None:
        """Array literals have arbitrarily many elements, so they can't
        go through `call_builtin_bridge`'s fixed 4-argument signature.
        Instead: stack-allocate an array of boxed handles, store each
        element into it, and hand the bridge a pointer plus a count."""
        count = len(instr.args)
        array_type = ir.ArrayType(I64, max(count, 1))
        array_ptr = self.builder.alloca(array_type, name="series_elems")
        for index, reg in enumerate(instr.args):
            handle = self._box_value(self._read_reg(reg), self.register_types.get(reg))
            elem_ptr = self.builder.gep(
                array_ptr, [ir.Constant(I64, 0), ir.Constant(I64, index)], inbounds=True
            )
            self.builder.store(handle, elem_ptr)

        base_ptr = self.builder.gep(
            array_ptr, [ir.Constant(I64, 0), ir.Constant(I64, 0)], inbounds=True
        )
        base_as_int = self.builder.ptrtoint(base_ptr, I64)
        fn = self._declare_bridge("make_series_bridge", ir.FunctionType(I64, [I64, I64]))
        result_handle = self.builder.call(fn, [base_as_int, ir.Constant(I64, count)])

        if instr.dest is not None:
            self.reg_values[instr.dest] = self._unbox_value(
                result_handle, self.register_types.get(instr.dest)
            )

    # ---- boxing helpers ----

    def _box_value(self, value: ir.Value, value_type: Type | None) -> ir.Value:
        if value_type == INT:
            fn = self._declare_bridge("box_int", ir.FunctionType(I64, [I64]))
            return self.builder.call(fn, [value])
        if value_type == FLOAT:
            fn = self._declare_bridge("box_float", ir.FunctionType(I64, [DOUBLE]))
            return self.builder.call(fn, [value])
        if value_type == BOOL:
            # i64, not i1: a Bool is native `i1` inside LLVM, but the
            # bridge is a C-ABI ctypes callback taking c_int64 (see
            # jit._BRIDGES), so it must be zero-extended on the way out
            # and truncated back on the way in (_unbox_value).
            fn = self._declare_bridge("box_bool", ir.FunctionType(I64, [I64]))
            zext = self.builder.zext(value, I64)
            return self.builder.call(fn, [zext])
        return value  # already a boxed handle

    def _unbox_value(self, handle: ir.Value, value_type: Type | None) -> ir.Value:
        if value_type == INT:
            fn = self._declare_bridge("unbox_int", ir.FunctionType(I64, [I64]))
            return self.builder.call(fn, [handle])
        if value_type == FLOAT:
            fn = self._declare_bridge("unbox_float", ir.FunctionType(DOUBLE, [I64]))
            return self.builder.call(fn, [handle])
        if value_type == BOOL:
            fn = self._declare_bridge("unbox_bool", ir.FunctionType(I64, [I64]))
            result = self.builder.call(fn, [handle])
            return self.builder.trunc(result, BOOL_T)
        return handle  # stays boxed

    def _declare_bridge(self, name: str, func_type: ir.FunctionType) -> ir.Function:
        if name not in self._bridge_fns:
            self._bridge_fns[name] = ir.Function(self.module, func_type, name=name)
        return self._bridge_fns[name]
