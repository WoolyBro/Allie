"""Custom IR -> LLVM IR lowering.

## Type mapping

| Allie type | LLVM type |
|---|---|
| `Int` | `i64` |
| `Float` | `double` |
| `Bool` | `i1` |
| `Void` | `void` |
| everything else (`String`, quant types, `Any`, `FunctionType`) | `i64` -- opaque boxed handle |

**Strings are handles too**, not `i8*` pointers to global constants:
`+` concatenates strings at run time, so a string isn't always a
compile-time constant a fixed global could hold. Every non-native-scalar
type uses the same boxed-`i64`-handle scheme (see
`allie.codegen.runtime_bridge`) -- one mechanism, not a special case
per type.

**Series and Matrix work isn't compiled**: calls to `SMA`, `mean`, etc.
are lowered as external calls back into the Python/numpy runtime via
`runtime_bridge.call_builtin_bridge`, never reimplemented in LLVM IR.
numpy already runs them as compiled loops (docs/how-it-works.md §3.3).

## No `for`-specific code

There is none here. `Label`/`Jump`/`JumpIfFalse` become basic blocks and
`br`/`cbranch` uniformly for `if`/`while`/`for` alike, because IR
lowering already desugars `for` into `while`'s shape. A `for`-specific
branch here would mean that desugaring hadn't been applied consistently
-- and the fix would belong there, not here.

## Native vs. bridged dispatch

A `BinOp`/`UnaryOp` compiles to a direct LLVM instruction (`fadd`/`add`/
`icmp_signed`/`fcmp_ordered`/...) only when every operand's recorded
type (`allie.ir.lowering`'s `register_types`) is a
native scalar (`Int`/`Float`/`Bool`); otherwise it's bridged through
`runtime_bridge.binop_bridge`/`unaryop_bridge`, which call the exact
same `_apply_binop`/`_apply_unary` the IR interpreter uses -- no third
reimplementation of operator semantics.

A `Call` to a user-defined function compiles to a direct native LLVM
`call` (this is also how recursion works). Parameters and results of any
type cross it: native scalars as themselves, everything else -- a
Series, a PriceSeries, a String -- as its boxed `i64` handle, exactly as
values are carried everywhere else. A `Call` to anything else (a stdlib
builtin, `__make_series`, or a strategy/backtest/when stub) goes through
`runtime_bridge.call_builtin_bridge`, boxing every native-scalar argument
first and unboxing the result if its type is a native scalar.

## Variables

A function's locals and parameters, and the main program's variables no
function uses, live in `alloca`s allocated up front in the entry block
(LLVM's `mem2reg` turns them into registers at `-O2`). A main-program
variable some function or `when` body uses (`@name` in its IR) lives in
an LLVM global instead, with a companion "has been assigned" byte: a
function called before the variable's `let` ran raises "undefined name"
like the interpreters, instead of reading a silent zero.

## How a run ends early

Native code can't raise. When something fails -- a zero divisor or an
Int overflow caught by a guard, an error inside a bridged Python call, a
use of an unassigned shared variable -- the bridge records the error and
sets the module's `__allie_error` byte, and native code checks that byte
after every call, returning at once. Without it, code after a failed
call ran on with a dummy result (and a loop could spin until its counter
overflowed). A `__allie_depth` counter caps nested user-function calls at
MAX_CALL_DEPTH, so deep recursion raises "recursion too deep" instead of
overflowing the native stack and killing the process.
"""

from __future__ import annotations

import llvmlite.ir as ir

from allie.codegen.runtime_bridge import box
from allie.errors import SourceSpan
from allie.ir.instructions import (
    GLOBAL_PREFIX,
    TO_FLOAT,
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
I8 = ir.IntType(8)
DOUBLE = ir.DoubleType()
BOOL_T = ir.IntType(1)

# Set (from Python, through its address) when a bridge records an error;
# checked after every call so native code stops at the next one.
ERROR_FLAG = "__allie_error"
# Nested user-function calls allowed before "recursion too deep". Native
# frames are small, but the JIT runs on the caller's thread stack.
MAX_CALL_DEPTH = 10_000

_MAX_BRIDGED_ARITY = 4

# Must match the interpreters' message exactly: --verify compares output.
_INT_OVERFLOW = "integer overflow: the result doesn't fit in a 64-bit Int"

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
    """Raised for IR shapes this lowering can't compile: more than
    `_MAX_BRIDGED_ARITY` arguments to a bridged call, or IR that breaks an
    invariant lowering relies on (a variable read but never stored)."""


class LLVMLowering:
    def __init__(
        self,
        functions: dict[str, tuple[IRProgram, list[str]]],
        function_info: dict[str, FunctionInfo],
    ) -> None:
        self.module = ir.Module(name="allie_module")
        self.functions = functions
        # The source span of the IR instruction being lowered, so a
        # runtime error raised by its code names the line and column the
        # interpreters name (allie.ir.instructions.IRInstr.span).
        self._span: SourceSpan | None = None
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
        self._in_user_function = False
        # Main-program variables some function or `when` body uses (as
        # `@name`). Those live in LLVM globals rather than `__allie_main`'s
        # stack frame, so every function sees the same storage.
        self._escaping = {
            instr.name[len(GLOBAL_PREFIX) :]
            for body, _ in functions.values()
            for instr in body.instructions
            if isinstance(instr, (LoadVar, StoreVar)) and instr.name.startswith(GLOBAL_PREFIX)
        }
        self._globals: dict[str, tuple[ir.GlobalVariable, ir.GlobalVariable, Type]] = {}
        self._runtime_globals: dict[str, ir.GlobalVariable] = {}

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
            param_types = [llvm_type_for(t) for t in info.param_types]
            ret_type = llvm_type_for(info.return_type)
            ir.Function(self.module, ir.FunctionType(ret_type, param_types), name=name)

    def lower_function(self, name: str) -> None:
        info = self.function_info[name]
        program, param_names = self.functions[name]
        func = self.module.get_global(name)
        assert isinstance(func, ir.Function)

        self.register_types = info.register_types
        self.reg_values = {}
        self.var_ptrs = {}
        self.var_types = {}
        self._move_slots = {}
        self._current_return_type = info.return_type
        self._in_user_function = True
        self._span = None

        entry_block = func.append_basic_block("entry")
        self._reached_blocks = {id(entry_block)}
        self.builder = ir.IRBuilder(entry_block)
        for pname, ptype, arg in zip(param_names, info.param_types, func.args, strict=True):
            ptr = self.builder.alloca(llvm_type_for(ptype), name=f"{pname}_ptr")
            self.builder.store(arg, ptr)
            self.var_ptrs[pname] = ptr
            self.var_types[pname] = ptype
        self._enter_call_depth(name)

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
        self._in_user_function = False
        self._span = None

        self._lower_instructions(program.instructions, func)
        self._finalize_function(func, VOID, "__allie_main")
        return func

    def _finalize_function(self, func: ir.Function, return_type: Type, name: str) -> None:
        """LLVM requires every basic block to end in a terminator, even
        one nothing branches to: a `for`/`if` whose every live path
        already returned still leaves the `end_label` block IR
        lowering always emits sitting empty and un-branched-to. Any
        block never recorded in `_reached_blocks` is exactly that --
        genuinely unreachable, not a missing `return` -- so it
        gets an explicit `unreachable` terminator. The one block that
        *is* live and still lacks a terminator is the function falling
        off its natural end: `ret_void` for a Void function (the normal,
        intentional case of never writing an explicit trailing
        `return;`), or an error for anything else -- which the type
        checker's "does every path return" check should already have caught, so
        reaching it here would mean these two checks disagree, not that
        the user's program is actually wrong.
        """
        for block in func.basic_blocks:
            if block.is_terminated:
                continue
            if id(block) in self._reached_blocks:
                if return_type == VOID:
                    self.builder.position_at_end(block)
                    self._ret(None)
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

    def _alloc_var_slots(self, instructions: list) -> None:  # noqa: ANN001
        """Allocates every variable's stack slot in the entry block, before
        any instruction is lowered, typed by the first value stored to it.

        Allocating lazily at the first `StoreVar` *in text order* put the
        `alloca` wherever that store happened to be -- often inside a
        branch or a loop body that doesn't dominate the variable's other
        uses, which LLVM rejects ("Instruction does not dominate all
        uses"). The optimizer fuzz test found it: once a pass removed a
        dead initializing store, the first remaining store was inside an
        `if`. Same reason `_alloc_move_slots` exists.
        """
        for instr in instructions:
            if not isinstance(instr, StoreVar) or instr.name in self.var_ptrs:
                continue
            if self._is_global(instr.name):
                continue
            var_type = self.register_types.get(instr.src) or INT
            self.var_ptrs[instr.name] = self.builder.alloca(
                llvm_type_for(var_type), name=f"{instr.name}_ptr"
            )
            self.var_types[instr.name] = var_type

    def _lower_instructions(self, instructions: list, func: ir.Function) -> None:  # noqa: ANN001
        self._alloc_move_slots(instructions)
        self._alloc_var_slots(instructions)

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
                # `return`, so the unconditional `Jump(end_label)` IR
                # lowering always emits after the then-branch never
                # actually runs. LLVM requires a
                # terminator to be a basic block's *last* instruction,
                # so nothing here can be emitted into this block -- skip
                # straight to the next Label, which starts a fresh block.
                continue
            self._lower_one(instr, blocks, func)

    def _lower_one(self, instr: object, blocks: dict[str, ir.Block], func: ir.Function) -> None:
        self._span = getattr(instr, "span", None)
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
            self._ret(None if instr.value is None else self._read_reg(instr.value))
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
        lowering, where TWO Moves write the same destination
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
        if self._is_global(instr.name):
            value, is_set, _ = self._global(instr.name, self.register_types.get(instr.dest))
            if instr.name.startswith(GLOBAL_PREFIX):
                self._guard_defined(is_set, f"undefined name: {self._source_name(instr.name)!r}")
            self.reg_values[instr.dest] = self.builder.load(value, name=instr.dest)
            return
        if instr.name not in self.var_ptrs:
            raise LLVMLoweringError(f"internal: variable '{instr.name}' is read but never stored")
        ptr = self.var_ptrs[instr.name]
        self.reg_values[instr.dest] = self.builder.load(ptr, name=instr.dest)

    # ---- module-level variables shared with functions ----

    def _is_global(self, name: str) -> bool:
        """`@name` in a function body, or -- in the main program -- a
        variable some function body uses."""
        if name.startswith(GLOBAL_PREFIX):
            return True
        return not self._in_user_function and name in self._escaping

    @staticmethod
    def _source_name(name: str) -> str:
        return name[len(GLOBAL_PREFIX) :] if name.startswith(GLOBAL_PREFIX) else name

    def _global(
        self, name: str, var_type: Type | None
    ) -> tuple[ir.GlobalVariable, ir.GlobalVariable, Type]:
        """The LLVM global holding module-level variable `name`, and its
        "has been assigned" flag -- created on first use. The flag is what
        lets a function called before the variable's `let` ran raise the
        same "undefined name" error as the interpreters, instead of
        reading a silent zero."""
        key = self._source_name(name)
        if key not in self._globals:
            resolved = var_type if var_type is not None else INT
            llvm_type = llvm_type_for(resolved)
            value = ir.GlobalVariable(self.module, llvm_type, name=f"global.{key}")
            value.initializer = ir.Constant(llvm_type, 0.0 if resolved == FLOAT else 0)
            is_set = ir.GlobalVariable(self.module, I8, name=f"global.{key}.set")
            is_set.initializer = ir.Constant(I8, 0)
            self._globals[key] = (value, is_set, resolved)
        return self._globals[key]

    def _guard_defined(self, is_set: ir.GlobalVariable, message: str) -> None:
        unset = self.builder.icmp_unsigned("==", self.builder.load(is_set), ir.Constant(I8, 0))
        self._guard(unset, ("AllieNameError", message))

    def _lower_store_var(self, instr: StoreVar) -> None:
        src_val = self._read_reg(instr.src)
        src_type = self.register_types.get(instr.src)

        if self._is_global(instr.name):
            value, is_set, declared = self._global(instr.name, src_type)
            if instr.name.startswith(GLOBAL_PREFIX):
                # A function assigning a module-level variable: like
                # Environment.assign, it must already exist.
                source = self._source_name(instr.name)
                self._guard_defined(is_set, f"cannot assign to undefined name: {source!r}")
            if declared == FLOAT and src_type == INT:
                src_val = self.builder.sitofp(src_val, DOUBLE)
            self.builder.store(src_val, value)
            self.builder.store(ir.Constant(I8, 1), is_set)
            return

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
        self._stop_if_failed()

    def _native_binop(
        self,
        op: str,
        left: ir.Value,
        left_type: Type | None,
        right: ir.Value,
        right_type: Type | None,
    ) -> ir.Value:
        # `/` is true division even for two Ints (the type checker types
        # it Float), so it always works in doubles.
        is_float = op == "/" or FLOAT in (left_type, right_type)
        if is_float:
            if left_type == INT:
                left = self.builder.sitofp(left, DOUBLE)
            if right_type == INT:
                right = self.builder.sitofp(right, DOUBLE)

        if op in ("+", "-", "*"):
            if is_float:
                float_op = {"+": self.builder.fadd, "-": self.builder.fsub, "*": self.builder.fmul}
                return float_op[op](left, right)
            return self._checked_int_op(op, left, right)
        if op == "/":
            self._guard_nonzero_divisor(right, True, "division by zero")
            return self.builder.fdiv(left, right)
        if op == "%":
            self._guard_nonzero_divisor(right, is_float, "modulo by zero")
            return self._floored_frem(left, right) if is_float else self._floored_srem(left, right)
        if op in ("==", "!=", "<", ">", "<=", ">="):
            if not is_float:
                return self.builder.icmp_signed(op, left, right)
            # Python semantics with NaN: every comparison is False except
            # `!=`, which is True -- that one needs the *unordered* form.
            if op == "!=":
                return self.builder.fcmp_unordered(op, left, right)
            return self.builder.fcmp_ordered(op, left, right)
        raise LLVMLoweringError(f"unknown native binary operator: {op!r}")

    def _checked_int_op(self, op: str, left: ir.Value, right: ir.Value) -> ir.Value:
        """i64 `+`/`-`/`*` that raises on overflow, like both interpreters,
        instead of silently wrapping around."""
        intrinsic = {
            "+": self.builder.sadd_with_overflow,
            "-": self.builder.ssub_with_overflow,
            "*": self.builder.smul_with_overflow,
        }[op]
        pair = intrinsic(left, right)
        self._guard(self.builder.extract_value(pair, 1), _INT_OVERFLOW)
        return self.builder.extract_value(pair, 0)

    def _floored_srem(self, left: ir.Value, right: ir.Value) -> ir.Value:
        """Allie's `%` takes the divisor's sign (-7 % 2 is 1), as Python's
        does; `srem` takes the dividend's, so a nonzero result whose sign
        differs from the divisor's is shifted by one divisor."""
        zero = ir.Constant(I64, 0)
        # srem(INT_MIN, -1) is undefined behavior (it traps on x86), and
        # x % -1 is 0 for every x -- exactly what x % 1 computes.
        is_minus_one = self.builder.icmp_signed("==", right, ir.Constant(I64, -1))
        safe_right = self.builder.select(is_minus_one, ir.Constant(I64, 1), right)
        remainder = self.builder.srem(left, safe_right)
        nonzero = self.builder.icmp_signed("!=", remainder, zero)
        signs_differ = self.builder.icmp_signed("<", self.builder.xor(remainder, safe_right), zero)
        needs_shift = self.builder.and_(nonzero, signs_differ)
        return self.builder.select(needs_shift, self.builder.add(remainder, safe_right), remainder)

    def _floored_frem(self, left: ir.Value, right: ir.Value) -> ir.Value:
        """Floating-point `%` with Python's semantics: `frem` is C's fmod
        (dividend's sign), shifted by one divisor when the signs differ;
        and a zero result takes the divisor's sign (6.0 % -2.0 is -0.0)."""
        zero = ir.Constant(DOUBLE, 0.0)
        remainder = self.builder.frem(left, right)
        right_negative = self.builder.fcmp_ordered("<", right, zero)
        nonzero = self.builder.fcmp_ordered("!=", remainder, zero)
        signs_differ = self.builder.xor(
            self.builder.fcmp_ordered("<", remainder, zero), right_negative
        )
        shifted = self.builder.select(
            self.builder.and_(nonzero, signs_differ),
            self.builder.fadd(remainder, right),
            remainder,
        )
        signed_zero = self.builder.select(right_negative, ir.Constant(DOUBLE, -0.0), zero)
        is_zero = self.builder.fcmp_ordered("==", remainder, zero)
        return self.builder.select(is_zero, signed_zero, shifted)

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
        self._guard(is_zero, message)

    def _guard(self, condition: ir.Value, message: str | tuple[str, str]) -> None:
        """Emits a branch that, when `condition` is true, records `message`
        as a runtime error and leaves the function; otherwise execution
        continues. Used before any instruction that would otherwise trap
        or silently produce a wrong value (a zero divisor, an overflow)."""
        raise_block = self.builder.append_basic_block("runtime_error")
        continue_block = self.builder.append_basic_block("guard_ok")
        self.builder.cbranch(condition, raise_block, continue_block)
        self._reached_blocks.add(id(raise_block))
        self._reached_blocks.add(id(continue_block))

        self.builder.position_at_end(raise_block)
        fn = self._declare_bridge("raise_runtime_error_bridge", ir.FunctionType(I64, [I64]))
        class_name, text = ("AllieValueError", message) if isinstance(message, str) else message
        payload = (class_name, text, self._span)
        self.builder.call(fn, [ir.Constant(I64, box(payload))])
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
        dummy value of the right type when it isn't void. Callers never
        use it: every call is followed by `_stop_if_failed`."""
        if self._current_return_type == VOID:
            self._ret(None)
            return
        llvm_ret = llvm_type_for(self._current_return_type)
        dummy = (
            ir.Constant(llvm_ret, 0.0)
            if self._current_return_type == FLOAT
            else ir.Constant(llvm_ret, 0)
        )
        self._ret(dummy)

    def _ret(self, value: ir.Value | None) -> None:
        """Every return goes through here, so a user function's call-depth
        counter is decremented on every path out of it."""
        if self._in_user_function:
            depth = self._runtime_global("__allie_depth", I64)
            self.builder.store(
                self.builder.sub(self.builder.load(depth), ir.Constant(I64, 1)), depth
            )
        if value is None:
            self.builder.ret_void()
        else:
            self.builder.ret(value)

    # ---- runtime state shared with the bridge ----

    def _runtime_global(self, name: str, llvm_type: ir.Type) -> ir.GlobalVariable:
        """A module-level counter or flag, zero-initialized, with external
        linkage -- so LLVM's optimizer can't assume nothing outside the
        module writes it (the bridge sets `__allie_error` from Python)."""
        if name not in self._runtime_globals:
            variable = ir.GlobalVariable(self.module, llvm_type, name=name)
            variable.initializer = ir.Constant(llvm_type, 0)
            self._runtime_globals[name] = variable
        return self._runtime_globals[name]

    def _stop_if_failed(self) -> None:
        """After a call that may have raised -- a bridge into Python, or a
        user function that bailed out -- leave this function too if an
        error is pending. Without it, code after a failed call ran on with
        a dummy result, so `while s[i] == 0 { i = i + 1; }` past the end of
        `s` looped until the counter overflowed instead of stopping."""
        flag = self._runtime_global(ERROR_FLAG, I8)
        failed = self.builder.icmp_unsigned("!=", self.builder.load(flag), ir.Constant(I8, 0))
        bail = self.builder.append_basic_block("error_exit")
        carry_on = self.builder.append_basic_block("no_error")
        self.builder.cbranch(failed, bail, carry_on)
        self._reached_blocks.add(id(bail))
        self._reached_blocks.add(id(carry_on))
        self.builder.position_at_end(bail)
        if self._span is not None:
            # Only on the way out after a failure, so the run pays nothing
            # for it; an error an inner call already located keeps its span.
            locate = self._declare_bridge("locate_error_bridge", ir.FunctionType(I64, [I64]))
            self.builder.call(locate, [ir.Constant(I64, box(self._span))])
        self._emit_early_return()
        self.builder.position_at_end(carry_on)

    def _enter_call_depth(self, name: str) -> None:
        """Counts nested user-function calls and raises past
        MAX_CALL_DEPTH, so deep recursion is an Allie error rather than a
        native stack overflow that kills the process."""
        depth = self._runtime_global("__allie_depth", I64)
        deeper = self.builder.add(self.builder.load(depth), ir.Constant(I64, 1))
        self.builder.store(deeper, depth)
        too_deep = self.builder.icmp_signed(">", deeper, ir.Constant(I64, MAX_CALL_DEPTH))
        self._guard(
            too_deep,
            (
                "AllieRuntimeError",
                f"recursion too deep -- more than {MAX_CALL_DEPTH:,} nested calls",
            ),
        )

    def _lower_unaryop(self, instr: UnaryOp) -> None:
        operand_type = self.register_types.get(instr.operand)
        if _is_native_scalar(operand_type):
            operand_val = self._read_reg(instr.operand)
            if instr.op == "-":
                if operand_type == FLOAT:
                    # fneg, not 0.0 - x: that would turn -(0.0) into 0.0
                    # where both interpreters give -0.0.
                    self.reg_values[instr.dest] = self.builder.fneg(operand_val)
                else:
                    self.reg_values[instr.dest] = self._checked_int_op(
                        "-", ir.Constant(I64, 0), operand_val
                    )
                return
            if instr.op == "!":
                self.reg_values[instr.dest] = self.builder.not_(operand_val)
                return
            if instr.op == TO_FLOAT:
                self.reg_values[instr.dest] = (
                    self.builder.sitofp(operand_val, DOUBLE) if operand_type == INT else operand_val
                )
                return
            raise LLVMLoweringError(f"unknown native unary operator: {instr.op!r}")

        op_handle = ir.Constant(I64, box(instr.op))
        operand_handle = self._box_value(self._read_reg(instr.operand), operand_type)
        fn = self._declare_bridge("unaryop_bridge", ir.FunctionType(I64, [I64, I64]))
        result_handle = self.builder.call(fn, [op_handle, operand_handle])
        self.reg_values[instr.dest] = self._unbox_value(
            result_handle, self.register_types.get(instr.dest)
        )
        self._stop_if_failed()

    def _lower_member_get(self, instr: MemberGet) -> None:
        # PriceSeriesValue field access always crosses the boundary --
        # bridged, same generic call mechanism as everything else.
        obj_handle = self._read_reg(instr.obj)  # already boxed (PriceSeries is never scalar)
        field_handle = ir.Constant(I64, box(instr.field))
        fn = self._declare_bridge("member_get_bridge", ir.FunctionType(I64, [I64, I64]))
        result_handle = self.builder.call(fn, [obj_handle, field_handle])
        self.reg_values[instr.dest] = self._unbox_value(
            result_handle, self.register_types.get(instr.dest)
        )
        self._stop_if_failed()

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
        self._stop_if_failed()

    def _lower_bridged_call(self, instr: Call) -> None:
        if instr.fn_name == MAKE_SERIES_FN:
            self._lower_make_series(instr)
            return
        if len(instr.args) > _MAX_BRIDGED_ARITY:
            raise LLVMLoweringError(
                f"bridged call to '{instr.fn_name}' has more than {_MAX_BRIDGED_ARITY} arguments"
            )
        name_handle = ir.Constant(I64, box(instr.fn_name))
        arg_handles = [
            self._box_value(self._read_reg(reg), self.register_types.get(reg)) for reg in instr.args
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
        self._stop_if_failed()

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
