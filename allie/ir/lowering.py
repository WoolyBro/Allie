"""AST -> IR lowering -- Phase 5.2.

Two decisions the spec left open, made and documented here:

**Function bodies: separate IRProgram per function**, not labeled
regions in one flat list. A flat-list-with-labels design would need
call-stack/return-address machinery (push a return address, jump to the
function's label, pop and jump back on Return) just to support a call
returning to its call site -- a separate IRProgram per function avoids
all of that: allie.ir.interpreter.call_function recursively runs the
callee's own IRProgram and simply returns its Return value, the same
shape as the tree-walking interpreter's Closure calls (new Environment,
run the body, catch the return).

**Functions are static, name-registered, and NOT first-class values.**
Every `FuncDecl` -- top-level or nested -- registers into one shared
`functions: dict[name -> (IRProgram, params)]` at lowering time; no
instruction is emitted at the *definition* site, and `Call` always
refers to a callee *by name* (this only works when the AST's `Call.
callee` is a plain `Identifier` -- anything else raises `LoweringError`).
This is a real, deliberate limitation, not an oversight: it makes
top-level functions and recursion (`factorial`, `fib` call themselves by
name -- no closure capture needed) work identically to the tree-walking
interpreter, but it does NOT support genuine lexical closures -- a
function returned from another function and called later, capturing
that specific call's local variable (`make_adder`/`adder`, or the
shared-scope get/set pattern) has no way to work when calls dispatch by
static name rather than through a captured-environment value. The
Phase 5.3 cross-validation suite is scoped around this: it covers every
Phase 1 for/break/continue/recursion program and the Phase 4 quant
integration programs, not the two closure-capture tests, which the IR
path cannot execute by construction.

Short-circuit `and`/`or` (`_lower_logical`) needs a `JumpIfFalse` plus
one `UnaryOp("!", ...)` to emulate a `JumpIfTrue` this instruction set
doesn't have, and a `Move` (see instructions.py's docstring) to converge
the left/right branches into one result register.

**Phase 6 retrofit: `register_types`.** LLVM codegen needs to know each
register's static type (to pick `fadd` vs `add`, decide native-scalar vs
boxed-pointer representation, ...), but this IR's registers don't carry
type information on their own. Rather than a separate re-inference pass
over the IR, this lowering records `register_types[dest] = expr.
inferred_type` as it goes -- the type checker (Phase 3) already
annotated every expression node with its resolved type, so this is
free to capture at the exact point each register is created. Function
signatures (param/return types) are re-resolved from each `FuncDecl`'s
`TypeAnnotation`s the same way the checker does (`resolve_base_type`/
`resolve_generic_type`) -- safe to assume present, since Phase 3
requires explicit types on every function. This is additive: `functions`
(program + param *names*) is unchanged from Phase 5, so the IR
interpreter and its existing tests don't need to know any of this
exists. New info lives in `FunctionInfo` (param/return *types* and the
function body's own `register_types`), collected in `function_info`,
alongside `functions` rather than folded into it.

`for`'s `continue` target is deliberately NOT the same label as
`while`'s: a `for` loop's `continue` must still run the increment before
re-checking the condition (matching the Phase 1 interpreter's
`_exec_for`), so it jumps to a dedicated `increment_label`, not straight
back to the condition-check `loop_label` the way `while`'s `continue`
does. Getting this wrong silently breaks `for` (either skipping the
increment forever, or -- if aliased wrong -- skipping the condition
recheck); see test_ir_lowering.py's continue-target tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from allie.ast.nodes import (
    ArrayLiteral,
    Assignment,
    BacktestStmt,
    Binary,
    Block,
    BoolLiteral,
    BreakStmt,
    ContinueStmt,
    Expr,
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
    Program,
    ReturnStmt,
    Stmt,
    StrategyDecl,
    StringLiteral,
    Unary,
    WhenClause,
    WhileStmt,
)
from allie.ast.nodes import Call as CallNode
from allie.ir.instructions import (
    BinOp,
    Call,
    IRInstr,
    IRProgram,
    Jump,
    JumpIfFalse,
    Label,
    LoadConst,
    LoadVar,
    MemberGet,
    Move,
    RegisterAllocator,
    Return,
    StoreVar,
    UnaryOp,
)
from allie.typechecker.types import BOOL, Type, resolve_base_type, resolve_generic_type

MAKE_SERIES_FN = "__make_series"
STRATEGY_DECLARE_FN = "__strategy_declare"
BACKTEST_RUN_FN = "__backtest_run"


@dataclass
class FunctionInfo:
    """LLVM-lowering-specific metadata (Phase 6) -- collected *alongside*
    `IRLowering.functions` (program + param names, unchanged from Phase
    5), not folded into it, so the IR interpreter and its existing
    tests don't need to know any of this exists."""

    param_types: list[Type]
    return_type: Type
    register_types: dict[str, Type]


class LoweringError(Exception):
    """Raised for AST shapes the IR lowering can't handle -- see this
    module's docstring for the specific, documented limitation (calling
    through anything other than a plain name)."""


class IRLowering:
    def __init__(
        self,
        functions: dict[str, tuple[IRProgram, list[str]]] | None = None,
        registers: RegisterAllocator | None = None,
        function_info: dict[str, FunctionInfo] | None = None,
    ) -> None:
        self.instructions: list[IRInstr] = []
        self.functions: dict[str, tuple[IRProgram, list[str]]] = (
            functions if functions is not None else {}
        )
        self.function_info: dict[str, FunctionInfo] = (
            function_info if function_info is not None else {}
        )
        self.registers = registers if registers is not None else RegisterAllocator()
        self.register_types: dict[str, Type] = {}
        self._loop_stack: list[tuple[str, str]] = []  # (continue_label, break_label)

    def lower_program(self, program: Program) -> IRProgram:
        for stmt in program.statements:
            self._lower_stmt(stmt)
        return IRProgram(instructions=self.instructions)

    def _emit(self, instr: IRInstr) -> None:
        self.instructions.append(instr)

    def _resolve_type_annotation(self, annotation: object) -> Type:
        # Mirrors allie.typechecker.checker._resolve_type_annotation --
        # safe to assume the annotation is well-formed and resolvable
        # here, since IR lowering only ever runs on a program that
        # already passed type-checking.
        name = annotation.name  # type: ignore[attr-defined]
        type_arg = annotation.type_arg  # type: ignore[attr-defined]
        if type_arg is not None:
            elem = self._resolve_type_annotation(type_arg)
            generic = resolve_generic_type(name, elem)
            if generic is None:
                raise LoweringError(f"'{name}' is not a generic type")
            return generic
        base = resolve_base_type(name)
        if base is None:
            raise LoweringError(f"unknown type '{name}'")
        return base

    # ---- statements ----

    def _lower_stmt(self, stmt: Stmt) -> None:
        if isinstance(stmt, LetStmt):
            src = self._lower_expr(stmt.initializer)
            self._emit(StoreVar(name=stmt.name, src=src))
        elif isinstance(stmt, Block):
            for s in stmt.statements:
                self._lower_stmt(s)
        elif isinstance(stmt, ExprStmt):
            self._lower_expr(stmt.expression)
        elif isinstance(stmt, IfStmt):
            self._lower_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self._lower_while(stmt)
        elif isinstance(stmt, ForStmt):
            self._lower_for(stmt)
        elif isinstance(stmt, BreakStmt):
            if not self._loop_stack:
                raise LoweringError("'break' outside a loop")
            _, break_label = self._loop_stack[-1]
            self._emit(Jump(label=break_label))
        elif isinstance(stmt, ContinueStmt):
            if not self._loop_stack:
                raise LoweringError("'continue' outside a loop")
            continue_label, _ = self._loop_stack[-1]
            self._emit(Jump(label=continue_label))
        elif isinstance(stmt, ReturnStmt):
            value = self._lower_expr(stmt.value) if stmt.value is not None else None
            self._emit(Return(value=value))
        elif isinstance(stmt, FuncDecl):
            self._lower_func_decl(stmt)
        elif isinstance(stmt, ImportStmt):
            pass  # no module system yet -- inert, matching every other pass
        elif isinstance(stmt, StrategyDecl):
            self._lower_strategy(stmt)
        elif isinstance(stmt, BacktestStmt):
            self._lower_backtest(stmt)
        else:
            raise LoweringError(f"unhandled statement node: {type(stmt).__name__}")

    def _lower_if(self, stmt: IfStmt) -> None:
        cond_reg = self._lower_expr(stmt.condition)
        else_label = self.registers.new_label("if_else")
        end_label = self.registers.new_label("if_end")
        self._emit(JumpIfFalse(cond=cond_reg, label=else_label if stmt.else_branch else end_label))
        self._lower_stmt(stmt.then_branch)
        if stmt.else_branch is not None:
            self._emit(Jump(label=end_label))
            self._emit(Label(name=else_label))
            self._lower_stmt(stmt.else_branch)
        self._emit(Label(name=end_label))

    def _lower_while(self, stmt: WhileStmt) -> None:
        loop_label = self.registers.new_label("while_loop")
        end_label = self.registers.new_label("while_end")
        self._emit(Label(name=loop_label))
        cond_reg = self._lower_expr(stmt.condition)
        self._emit(JumpIfFalse(cond=cond_reg, label=end_label))
        # while's continue target IS the condition check -- no separate
        # increment step exists to route through (contrast _lower_for).
        self._loop_stack.append((loop_label, end_label))
        self._lower_stmt(stmt.body)
        self._loop_stack.pop()
        self._emit(Jump(label=loop_label))
        self._emit(Label(name=end_label))

    def _lower_for(self, stmt: ForStmt) -> None:
        self._lower_stmt(stmt.initializer)

        loop_label = self.registers.new_label("for_loop")
        increment_label = self.registers.new_label("for_increment")
        end_label = self.registers.new_label("for_end")

        self._emit(Label(name=loop_label))
        cond_reg = self._lower_expr(stmt.condition)
        self._emit(JumpIfFalse(cond=cond_reg, label=end_label))

        # continue -> increment_label (run the increment, then recheck
        # the condition); break -> end_label (skip the increment
        # entirely). See this module's docstring.
        self._loop_stack.append((increment_label, end_label))
        self._lower_stmt(stmt.body)
        self._loop_stack.pop()

        self._emit(Label(name=increment_label))
        self._lower_expr(stmt.increment)
        self._emit(Jump(label=loop_label))
        self._emit(Label(name=end_label))

    def _lower_func_decl(self, stmt: FuncDecl) -> None:
        sub_lowering = IRLowering(
            functions=self.functions,
            registers=RegisterAllocator(),
            function_info=self.function_info,
        )
        for s in stmt.body.statements:
            sub_lowering._lower_stmt(s)
        body_program = IRProgram(instructions=sub_lowering.instructions)
        self.functions[stmt.name] = (body_program, [param.name for param in stmt.params])

        # Phase 3 requires explicit types on every function, so this
        # normally succeeds -- but lowering also runs directly on
        # untyped programs in tests (Phase 5's cross-validation suite
        # deliberately skips type-checking, reusing the tree-walker's
        # original untyped test programs). Rather than crash on those,
        # just skip building FunctionInfo: the IR interpreter doesn't
        # need it (only `functions`, unaffected above), and nothing
        # untyped will be run through the LLVM path anyway, since that
        # needs FunctionInfo to exist.
        if stmt.return_type is None or any(p.type_annotation is None for p in stmt.params):
            return
        param_types = [self._resolve_type_annotation(p.type_annotation) for p in stmt.params]
        return_type = self._resolve_type_annotation(stmt.return_type)
        self.function_info[stmt.name] = FunctionInfo(
            param_types=param_types,
            return_type=return_type,
            register_types=sub_lowering.register_types,
        )

    def _lower_strategy(self, stmt: StrategyDecl) -> None:
        # Lowered as Calls into the runtime backtest engine (Phase 7) --
        # the engine's bar-by-bar iteration stays entirely outside the
        # IR's own control-flow machinery; it's invoked, not inlined.
        self._emit(Call(dest=None, fn_name=STRATEGY_DECLARE_FN, args=[]))
        for clause in stmt.when_clauses:
            self._lower_when_clause(clause)

    def _lower_when_clause(self, clause: WhenClause) -> None:
        left = self._lower_expr(clause.left)
        right = self._lower_expr(clause.right)
        self._emit(Call(dest=None, fn_name=f"__when_{clause.operator}", args=[left, right]))
        self._lower_stmt(clause.body)

    def _lower_backtest(self, stmt: BacktestStmt) -> None:
        start_reg = self.registers.new_reg()
        self._emit(LoadConst(dest=start_reg, value=stmt.start))
        end_reg = self.registers.new_reg()
        self._emit(LoadConst(dest=end_reg, value=stmt.end))
        self._emit(
            Call(
                dest=None,
                fn_name=BACKTEST_RUN_FN,
                args=[stmt.strategy_name, stmt.symbol, start_reg, end_reg],
            )
        )

    # ---- expressions (each returns the register holding its result) ----

    def _lower_expr(self, expr: Expr) -> str:
        dest = self._lower_expr_impl(expr)
        # Phase 6 retrofit: record every register's static type from the
        # type checker's own annotation (see this module's docstring).
        # `inferred_type` is only None if type-checking never ran --
        # lowering isn't meaningful in that case anyway.
        if expr.inferred_type is not None:
            self.register_types[dest] = expr.inferred_type
        return dest

    def _lower_expr_impl(self, expr: Expr) -> str:
        if isinstance(expr, NumberLiteral):
            dest = self.registers.new_reg()
            self._emit(LoadConst(dest=dest, value=expr.value))
            return dest
        if isinstance(expr, StringLiteral):
            dest = self.registers.new_reg()
            self._emit(LoadConst(dest=dest, value=expr.value))
            return dest
        if isinstance(expr, BoolLiteral):
            dest = self.registers.new_reg()
            self._emit(LoadConst(dest=dest, value=expr.value))
            return dest
        if isinstance(expr, Identifier):
            dest = self.registers.new_reg()
            self._emit(LoadVar(dest=dest, name=expr.name))
            return dest
        if isinstance(expr, Assignment):
            src = self._lower_expr(expr.value)
            self._emit(StoreVar(name=expr.name, src=src))
            return src
        if isinstance(expr, Unary):
            operand = self._lower_expr(expr.operand)
            dest = self.registers.new_reg()
            self._emit(UnaryOp(dest=dest, op=expr.operator, operand=operand))
            return dest
        if isinstance(expr, Binary):
            left = self._lower_expr(expr.left)
            right = self._lower_expr(expr.right)
            dest = self.registers.new_reg()
            self._emit(BinOp(dest=dest, op=expr.operator, left=left, right=right))
            return dest
        if isinstance(expr, Logical):
            return self._lower_logical(expr)
        if isinstance(expr, CallNode):
            return self._lower_call(expr)
        if isinstance(expr, FieldAccess):
            obj = self._lower_expr(expr.obj)
            dest = self.registers.new_reg()
            self._emit(MemberGet(dest=dest, obj=obj, field=expr.name))
            return dest
        if isinstance(expr, ArrayLiteral):
            arg_regs = [self._lower_expr(element) for element in expr.elements]
            dest = self.registers.new_reg()
            self._emit(Call(dest=dest, fn_name=MAKE_SERIES_FN, args=arg_regs))
            return dest
        raise LoweringError(f"unhandled expression node: {type(expr).__name__}")

    def _lower_logical(self, expr: Logical) -> str:
        left = self._lower_expr(expr.left)
        result = self.registers.new_reg()
        self._emit(Move(dest=result, src=left))
        end_label = self.registers.new_label("logical_end")

        if expr.operator == "and":
            # Falsy left -> short-circuit, keep result = left.
            self._emit(JumpIfFalse(cond=left, label=end_label))
        elif expr.operator == "or":
            # Truthy left -> short-circuit, keep result = left. No
            # JumpIfTrue exists, so negate and reuse JumpIfFalse.
            not_left = self.registers.new_reg()
            self._emit(UnaryOp(dest=not_left, op="!", operand=left))
            self.register_types[not_left] = BOOL  # synthetic register, not a real AST node
            self._emit(JumpIfFalse(cond=not_left, label=end_label))
        else:
            raise LoweringError(f"unknown logical operator: {expr.operator!r}")

        right = self._lower_expr(expr.right)
        self._emit(Move(dest=result, src=right))
        self._emit(Label(name=end_label))
        return result

    def _lower_call(self, expr: CallNode) -> str:
        if not isinstance(expr.callee, Identifier):
            raise LoweringError(
                "the IR lowering only supports calling a plain name (no first-class "
                "function values) -- see this module's docstring"
            )
        arg_regs = [self._lower_expr(arg) for arg in expr.arguments]
        dest = self.registers.new_reg()
        self._emit(Call(dest=dest, fn_name=expr.callee.name, args=arg_regs))
        return dest


def lower_program(
    program: Program,
) -> tuple[
    IRProgram,
    dict[str, tuple[IRProgram, list[str]]],
    dict[str, Type],
    dict[str, FunctionInfo],
]:
    """Convenience entry point: returns (main_program, functions,
    register_types, function_info). The last two are additive (Phase 6)
    -- callers that only need the first two (the IR interpreter, its
    existing tests) can keep unpacking just those and ignore the rest."""
    lowering = IRLowering()
    main_program = lowering.lower_program(program)
    return main_program, lowering.functions, lowering.register_types, lowering.function_info
