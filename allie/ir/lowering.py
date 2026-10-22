"""AST -> IR lowering.

Two decisions, made and documented here:

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
It makes top-level functions and recursion (`factorial`, `fib` call
themselves by name) work identically to the tree-walking interpreter.
There are no closures to support: the type checker rejects a nested
function that reads its enclosing function's locals, on every backend
alike, so a call by static name is all a call ever needs.

Short-circuit `and`/`or` (`_lower_logical`) needs a `JumpIfFalse` plus
one `UnaryOp("!", ...)` to emulate a `JumpIfTrue` this instruction set
doesn't have, and a `Move` (see instructions.py's docstring) to converge
the left/right branches into one result register.

**`register_types`: every register's static type.** LLVM codegen needs
each register's type (to pick `fadd` vs `add`, to decide native-scalar
vs boxed-handle representation, ...), but this IR's registers don't
carry types on their own. Rather than a separate re-inference pass over
the IR, this lowering records `register_types[dest] = expr.
inferred_type` as it goes -- the type checker already annotated every
expression node, so the type is free to capture at the exact point each
register is created. Function signatures (param/return types) are
re-resolved from each `FuncDecl`'s `TypeAnnotation`s the same way the
checker does (`resolve_base_type`/`resolve_generic_type`) -- safe, since
the checker requires explicit types on every function. The IR
interpreter needs none of this, so it lives in `FunctionInfo` (param and
return types, and the body's own `register_types`), collected in
`function_info` beside `functions` (program + param names) rather than
folded into it.

`for`'s `continue` target is deliberately NOT the same label as
`while`'s: a `for` loop's `continue` must still run the increment before
re-checking the condition (as the tree-walking interpreter does), so it
jumps to a dedicated `increment_label`, not straight
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
    Backtest,
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
    Index,
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
    Widen,
)
from allie.ast.nodes import Call as CallNode
from allie.errors import SourceSpan
from allie.ir.cfg import without_unreachable_code
from allie.ir.instructions import (
    GLOBAL_PREFIX,
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
from allie.ir.instructions import TO_FLOAT as TO_FLOAT
from allie.stdlib.indexing import index_function_for
from allie.typechecker.types import (
    BACKTEST_RESULT,
    BOOL,
    FLOAT,
    INT,
    PRICE_SERIES,
    STRING,
    VOID,
    SeriesType,
    Type,
    resolve_base_type,
    resolve_generic_type,
)

MAKE_SERIES_FN = "__make_series"
STRATEGY_DECLARE_FN = "__strategy_declare"
STRATEGY_CLAUSE_FN = "__strategy_clause"
BACKTEST_OPTION_FN = "__backtest_option"
BACKTEST_RUN_FN = "__backtest_run"


# The instructions that can fail at run time, and so carry a source span.
_FALLIBLE = (LoadVar, BinOp, UnaryOp, MemberGet, Call)


@dataclass
class FunctionInfo:
    """A function's types, for the LLVM backend -- kept *alongside*
    `IRLowering.functions` (program + param names), not folded into
    it, because the IR interpreter doesn't need them."""

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
        parent: IRLowering | None = None,
    ) -> None:
        self.instructions: list[IRInstr] = []
        # The lowering of the enclosing code, for a function or `when`
        # body -- how a module-level name used inside one is resolved.
        self._parent = parent
        self.functions: dict[str, tuple[IRProgram, list[str]]] = (
            functions if functions is not None else {}
        )
        self.function_info: dict[str, FunctionInfo] = (
            function_info if function_info is not None else {}
        )
        self.registers = registers if registers is not None else RegisterAllocator()
        self.register_types: dict[str, Type] = {}
        self._loop_stack: list[tuple[str, str]] = []  # (continue_label, break_label)
        # Scope stack of source-name -> IR-name, for alpha-renaming
        # shadowed variables (see _declare_var).
        self._scopes: list[dict[str, str]] = [{}]
        self._taken_names: set[str] = set()
        # Source position of the expression or statement being lowered,
        # stamped on the instructions that can fail (see IRInstr.span).
        self._span: SourceSpan | None = None

    def lower_program(self, program: Program) -> IRProgram:
        for stmt in program.statements:
            self._lower_stmt(stmt)
        return IRProgram(instructions=self.instructions)

    def _emit(self, instr: IRInstr) -> None:
        if isinstance(instr, _FALLIBLE) and instr.span is None:
            instr.span = self._span
        self.instructions.append(instr)

    # ---- variable scoping (alpha-renaming) ----
    #
    # Lowering flattens blocks away, but both the IR interpreter and the
    # LLVM backend keep ONE name -> value map. Without renaming, a
    # shadowing `let` inside a block would clobber the outer binding and
    # never restore it -- so `let x=1; { let x=2; } print(x);` printed 2
    # on those backends and 1 on the tree-walker.
    #
    # Blocks are still visible *here*, so the fix belongs here: a
    # shadowing declaration gets a fresh, unique IR name, leaving the
    # flat namespace genuinely unambiguous. Neither backend needed a
    # change. Names only get suffixed on an actual collision, so ordinary
    # programs lower to exactly the IR they did before.

    def _push_scope(self) -> None:
        self._scopes.append({})

    def _pop_scope(self) -> None:
        self._scopes.pop()

    def _declare_var(self, name: str) -> str:
        ir_name = name
        if name in self._taken_names:
            suffix = 1
            while f"{name}${suffix}" in self._taken_names:
                suffix += 1
            ir_name = f"{name}${suffix}"
        self._taken_names.add(ir_name)
        self._scopes[-1][name] = ir_name
        return ir_name

    def _lookup(self, name: str) -> str | None:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def _resolve_var(self, name: str) -> str:
        """The IR name for a variable reference.

        Inside a function or `when` body, a name the body doesn't declare
        is a module-level variable (the type checker rejects references
        to an enclosing *function's* locals). It is written `@name`,
        where `name` is the main program's IR name for the binding the
        function sees at its definition -- so `@rate` means "the
        module-level `rate`", distinct from any local `rate`, and every
        pass and backend can tell the two apart."""
        local = self._lookup(name)
        if local is not None:
            return local
        if self._parent is None:
            return name
        root = self._parent
        while root._parent is not None:
            root = root._parent
        return GLOBAL_PREFIX + (root._lookup(name) or name)

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
        outer, self._span = self._span, stmt.span
        try:
            self._lower_stmt_impl(stmt)
        finally:
            self._span = outer

    def _lower_stmt_impl(self, stmt: Stmt) -> None:
        if isinstance(stmt, LetStmt):
            # Initializer is lowered BEFORE declaring, so
            # `let x = x + 1;` still reads the outer `x`.
            src = self._lower_expr(stmt.initializer)
            self._emit(StoreVar(name=self._declare_var(stmt.name), src=src))
        elif isinstance(stmt, Block):
            self._push_scope()
            for s in stmt.statements:
                self._lower_stmt(s)
            self._pop_scope()
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
            # The statement form prints the results table.
            result = self._lower_backtest(stmt)
            self._emit(Call(dest=None, fn_name="print", args=[result]))
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
        # The loop variable lives in the loop's own scope -- matching
        # the tree-walker, which gives _exec_for a child Environment.
        self._push_scope()
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
        self._pop_scope()

    def _lower_func_decl(self, stmt: FuncDecl) -> None:
        sub_lowering = IRLowering(
            functions=self.functions,
            registers=RegisterAllocator(),
            function_info=self.function_info,
            parent=self,
        )
        for param in stmt.params:
            sub_lowering._declare_var(param.name)
        for s in stmt.body.statements:
            sub_lowering._lower_stmt(s)
        body_program = IRProgram(instructions=sub_lowering.instructions, is_function=True)
        self.functions[stmt.name] = (body_program, [param.name for param in stmt.params])

        # The type checker requires explicit types on every function,
        # so this normally succeeds -- but some tests lower programs
        # without type-checking them first. Rather than crash on those,
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
        """Lowered as `Call`s into the runtime backtest engine -- its
        bar-by-bar iteration stays entirely outside the IR's own
        control-flow machinery; it's invoked, not inlined.

        A `when` body can't be lowered inline: it must run only on the
        bars its signal fires on, which only the engine knows. So each
        body becomes its own zero-argument IRProgram -- the same
        mechanism `FuncDecl` uses -- and the engine invokes it per
        bar. Body programs are named `__when_body_<n>` and
        registered in `functions`/`function_info` alongside ordinary
        user functions, which is what lets the LLVM backend declare and
        compile them with no special-casing.

        Clause conditions are evaluated here, at the declaration, as
        on every backend: reassigning `fast` afterwards doesn't change
        a strategy already declared.
        """
        name_reg = self._const(stmt.name)
        self._emit(Call(dest=None, fn_name=STRATEGY_DECLARE_FN, args=[name_reg]))
        for index, clause in enumerate(stmt.when_clauses):
            self._lower_when_clause(stmt.name, index, clause)

    def _lower_when_clause(self, strategy_name: str, index: int, clause: WhenClause) -> None:
        body_fn = f"__when_body_{strategy_name}_{index}"

        sub_lowering = IRLowering(
            functions=self.functions,
            registers=RegisterAllocator(),
            function_info=self.function_info,
            parent=self,
        )
        sub_lowering._lower_stmt(clause.body)
        self.functions[body_fn] = (
            IRProgram(instructions=sub_lowering.instructions, is_function=True),
            [],
        )
        self.function_info[body_fn] = FunctionInfo(
            param_types=[],
            return_type=VOID,
            register_types=sub_lowering.register_types,
        )

        # The condition is evaluated here, at the declaration, into one
        # Series<Bool>; strategy and body names travel packed in one spec
        # constant to fit the LLVM bridge's fixed arity.
        spec_reg = self._const(f"{strategy_name}|{body_fn}")
        condition = self._lower_expr(clause.condition)
        self._emit(Call(dest=None, fn_name=STRATEGY_CLAUSE_FN, args=[spec_reg, condition]))

    def _lower_backtest(self, stmt: BacktestStmt | Backtest) -> str:
        """Runs the backtest; returns the register holding its result."""
        # `on SYM` names an ordinary variable holding a PriceSeries, so
        # it's loaded as a value rather than passed as a name -- the
        # LLVM bridge has no way to look a variable up by name.
        spec_reg = self._const(f"{stmt.strategy_name}|{stmt.symbol}")
        prices_reg = self.registers.new_reg()
        self._emit(LoadVar(dest=prices_reg, name=self._resolve_var(stmt.symbol)))
        self.register_types[prices_reg] = PRICE_SERIES
        start_reg = self._lower_expr(stmt.start)
        end_reg = self._lower_expr(stmt.end)
        # One call per `with` option, each setting it for the run below --
        # a single call carrying them all wouldn't fit the LLVM bridge's
        # fixed arity.
        for option in stmt.options:
            name_reg = self._const(option.name)
            value_reg = self._lower_expr(option.value)
            self._emit(Call(dest=None, fn_name=BACKTEST_OPTION_FN, args=[name_reg, value_reg]))
        result = self.registers.new_reg()
        self.register_types[result] = BACKTEST_RESULT
        self._emit(
            Call(
                dest=result,
                fn_name=BACKTEST_RUN_FN,
                args=[spec_reg, prices_reg, start_reg, end_reg],
            )
        )
        return result

    def _const(self, value: object) -> str:
        """Emits a LoadConst into a fresh register, recording its type
        so the LLVM backend knows how to represent it."""
        reg = self.registers.new_reg()
        self._emit(LoadConst(dest=reg, value=value))
        if isinstance(value, bool):
            self.register_types[reg] = BOOL
        elif isinstance(value, int):
            self.register_types[reg] = INT
        elif isinstance(value, float):
            self.register_types[reg] = FLOAT
        elif isinstance(value, str):
            self.register_types[reg] = STRING
        return reg

    # ---- expressions (each returns the register holding its result) ----

    def _lower_expr(self, expr: Expr) -> str:
        outer, self._span = self._span, expr.span
        try:
            dest = self._lower_expr_impl(expr)
        finally:
            self._span = outer
        # Record every register's static type from the type checker's
        # own annotation (see this module's docstring).
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
            self._emit(LoadVar(dest=dest, name=self._resolve_var(expr.name)))
            return dest
        if isinstance(expr, Assignment):
            src = self._lower_expr(expr.value)
            self._emit(StoreVar(name=self._resolve_var(expr.name), src=src))
            return src
        if isinstance(expr, Unary):
            operand = self._lower_expr(expr.operand)
            dest = self.registers.new_reg()
            self._emit(UnaryOp(dest=dest, op=expr.operator, operand=operand))
            return dest
        if isinstance(expr, Backtest):
            return self._lower_backtest(expr)
        if isinstance(expr, Widen):
            operand = self._lower_expr(expr.operand)
            dest = self.registers.new_reg()
            self._emit(UnaryOp(dest=dest, op=TO_FLOAT, operand=operand))
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
        if isinstance(expr, Index):
            # An ordinary builtin call, so the IR interpreter and the LLVM
            # bridge need nothing index-specific (see allie.stdlib.indexing).
            obj = self._lower_expr(expr.obj)
            index = self._lower_expr(expr.index)
            dest = self.registers.new_reg()
            fn_name = index_function_for(expr.inferred_type)
            self._emit(Call(dest=dest, fn_name=fn_name, args=[obj, index]))
            return dest
        if isinstance(expr, ArrayLiteral):
            arg_regs = [self._lower_expr(element) for element in expr.elements]
            dest = self.registers.new_reg()
            self._emit(Call(dest=dest, fn_name=MAKE_SERIES_FN, args=arg_regs))
            return dest
        raise LoweringError(f"unhandled expression node: {type(expr).__name__}")

    def _lower_logical(self, expr: Logical) -> str:
        if isinstance(expr.inferred_type, SeriesType):
            # Series `and`/`or` combine bar by bar -- both sides evaluated,
            # nothing to short-circuit -- so it's an ordinary BinOp.
            left = self._lower_expr(expr.left)
            right = self._lower_expr(expr.right)
            dest = self.registers.new_reg()
            self._emit(BinOp(dest=dest, op=expr.operator, left=left, right=right))
            return dest
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
    register_types, function_info). The IR interpreter needs only the
    first two; the LLVM backend needs all four.

    Lowered IR never contains unreachable code -- the statements after a
    `break`/`continue`/`return`, or after an `if` whose branches both
    return. It is dropped here, whether or not the optimizer runs: the
    LLVM backend creates a variable's stack slot at its first store, so a
    loop written after a `break` (its counter's initializing store
    skipped as unreachable, its header still reached by its own
    back-edge) crashed codegen with an unoptimized build. The optimizer
    fuzz test found it."""
    lowering = IRLowering()
    main_program = without_unreachable_code(lowering.lower_program(program))
    functions = {
        name: (without_unreachable_code(body), params)
        for name, (body, params) in lowering.functions.items()
    }
    return main_program, functions, lowering.register_types, lowering.function_info
