"""Type checker -- Phase 3.2. Walks the AST (allie.ast.nodes) produced by
allie.parser.parser, after allie.semantic.resolver.Resolver has already
run. Annotates `expr.inferred_type` on every expression node in place and
raises QSTypeError (fail-fast) on the first mismatch -- see
allie.typechecker.errors for why this phase doesn't collect-and-report
like the lexer/parser do.

TypeEnvironment below is a *separate* scope-stack from the resolver's
Scope (allie.semantic.resolver): same push/pop discipline, but this one
maps names to *types*; the resolver's only tracks that a name exists.
Don't conflate the two.

v1 requires explicit return types on every function (including
`-> Void` for one with no meaningful return value) and explicit
parameter types -- simpler and more honest to document as a known
limitation than silently doing full type inference. See
docs/language-reference.md "Known limitations".
"""

from __future__ import annotations

from allie.ast.nodes import (
    ArrayLiteral,
    Assignment,
    BacktestStmt,
    Binary,
    Block,
    BoolLiteral,
    BreakStmt,
    Call,
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
    TypeAnnotation,
    Unary,
    WhenClause,
    WhileStmt,
)
from allie.typechecker.errors import QSTypeError
from allie.typechecker.types import (
    BOOL,
    FLOAT,
    INT,
    PRICE_SERIES,
    STRATEGY,
    STRING,
    VOID,
    FunctionType,
    SeriesType,
    Type,
    is_assignable,
    resolve_base_type,
    resolve_generic_type,
)

_ARITHMETIC_OPS = frozenset({"+", "-", "*", "/", "%"})
_COMPARISON_OPS = frozenset({"<", ">", "<=", ">="})
_EQUALITY_OPS = frozenset({"==", "!="})


class TypeEnvironment:
    """Scope stack of name -> Type. Separate from the resolver's Scope --
    see this module's docstring."""

    def __init__(self, parent: TypeEnvironment | None = None) -> None:
        self.parent = parent
        self.types: dict[str, Type] = {}

    def define(self, name: str, type_: Type) -> None:
        self.types[name] = type_

    def get(self, name: str) -> Type | None:
        env: TypeEnvironment | None = self
        while env is not None:
            if name in env.types:
                return env.types[name]
            env = env.parent
        return None


class TypeChecker:
    def __init__(self) -> None:
        self.env = TypeEnvironment()
        self._function_return_stack: list[Type] = []

    def check(self, program: Program) -> None:
        for stmt in program.statements:
            self._check_stmt(stmt)

    def _push_scope(self) -> None:
        self.env = TypeEnvironment(parent=self.env)

    def _pop_scope(self) -> None:
        assert self.env.parent is not None
        self.env = self.env.parent

    def _resolve_type_annotation(self, annotation: TypeAnnotation) -> Type:
        if annotation.type_arg is not None:
            elem = self._resolve_type_annotation(annotation.type_arg)
            generic = resolve_generic_type(annotation.name, elem)
            if generic is None:
                raise QSTypeError(
                    f"'{annotation.name}' is not a generic type (expected Series/Matrix/Vector)",
                    annotation.span.line,
                    annotation.span.column,
                )
            return generic
        base = resolve_base_type(annotation.name)
        if base is None:
            raise QSTypeError(
                f"unknown type '{annotation.name}'", annotation.span.line, annotation.span.column
            )
        return base

    # ---- statements ----

    def _check_stmt(self, stmt: Stmt) -> None:
        if isinstance(stmt, LetStmt):
            self._check_let(stmt)
        elif isinstance(stmt, FuncDecl):
            self._check_func_decl(stmt)
        elif isinstance(stmt, IfStmt):
            self._check_condition(stmt.condition)
            self._check_stmt(stmt.then_branch)
            if stmt.else_branch is not None:
                self._check_stmt(stmt.else_branch)
        elif isinstance(stmt, WhileStmt):
            self._check_condition(stmt.condition)
            self._check_stmt(stmt.body)
        elif isinstance(stmt, ForStmt):
            self._check_for(stmt)
        elif isinstance(stmt, (BreakStmt, ContinueStmt, ImportStmt)):
            pass
        elif isinstance(stmt, ReturnStmt):
            self._check_return(stmt)
        elif isinstance(stmt, StrategyDecl):
            self._check_strategy(stmt)
        elif isinstance(stmt, BacktestStmt):
            self._check_backtest(stmt)
        elif isinstance(stmt, ExprStmt):
            self._check_expr(stmt.expression)
        elif isinstance(stmt, Block):
            self._push_scope()
            for s in stmt.statements:
                self._check_stmt(s)
            self._pop_scope()
        else:
            raise QSTypeError(
                f"internal: unhandled statement {type(stmt).__name__}",
                stmt.span.line,
                stmt.span.column,
            )

    def _check_condition(self, condition: Expr) -> None:
        # No truthy-int coercion -- matches the Phase 1 interpreter's
        # explicit-bool decision. if/while/for all go through this one
        # helper so the rule can't drift between the three.
        cond_type = self._check_expr(condition)
        if cond_type != BOOL:
            raise QSTypeError(
                f"condition must be Bool, got {cond_type}",
                condition.span.line,
                condition.span.column,
            )

    def _check_for(self, stmt: ForStmt) -> None:
        self._push_scope()
        self._check_stmt(stmt.initializer)
        self._check_condition(stmt.condition)
        self._check_expr(stmt.increment)
        self._check_stmt(stmt.body)
        self._pop_scope()

    def _check_let(self, stmt: LetStmt) -> None:
        init_type = self._check_expr(stmt.initializer)
        if stmt.type_annotation is not None:
            declared = self._resolve_type_annotation(stmt.type_annotation)
            if not is_assignable(init_type, declared):
                raise QSTypeError(
                    f"cannot assign {init_type} to {declared}",
                    stmt.initializer.span.line,
                    stmt.initializer.span.column,
                )
            self.env.define(stmt.name, declared)
        else:
            self.env.define(stmt.name, init_type)

    def _check_func_decl(self, stmt: FuncDecl) -> None:
        if stmt.return_type is None:
            raise QSTypeError(
                f"function '{stmt.name}' has no declared return type "
                "(v1 requires an explicit '-> Type', even '-> Void')",
                stmt.span.line,
                stmt.span.column,
            )
        param_types: list[Type] = []
        for param in stmt.params:
            if param.type_annotation is None:
                raise QSTypeError(
                    f"parameter '{param.name}' of '{stmt.name}' has no type annotation "
                    "(v1 requires explicit parameter types)",
                    param.span.line,
                    param.span.column,
                )
            param_types.append(self._resolve_type_annotation(param.type_annotation))
        return_type = self._resolve_type_annotation(stmt.return_type)

        function_type = FunctionType(params=tuple(param_types), ret=return_type)
        # Defined in the *enclosing* scope before the body is checked, so
        # a recursive call inside the body type-checks against it.
        self.env.define(stmt.name, function_type)

        self._push_scope()
        for param, param_type in zip(stmt.params, param_types, strict=True):
            self.env.define(param.name, param_type)
        self._function_return_stack.append(return_type)
        for s in stmt.body.statements:
            self._check_stmt(s)
        self._function_return_stack.pop()
        self._pop_scope()

        if return_type != VOID and not _all_paths_return(stmt.body):
            raise QSTypeError(
                f"function '{stmt.name}' does not return on every path",
                stmt.span.line,
                stmt.span.column,
            )

    def _check_return(self, stmt: ReturnStmt) -> None:
        if not self._function_return_stack:
            raise QSTypeError("'return' outside a function", stmt.span.line, stmt.span.column)
        expected = self._function_return_stack[-1]
        actual = VOID if stmt.value is None else self._check_expr(stmt.value)
        if not is_assignable(actual, expected):
            raise QSTypeError(
                f"return type mismatch: expected {expected}, got {actual}",
                stmt.span.line,
                stmt.span.column,
            )

    def _check_strategy(self, stmt: StrategyDecl) -> None:
        # Bind the name before checking the clauses so a `backtest` can
        # resolve it (and so a strategy could reference itself without
        # tripping the undefined-name check).
        self.env.define(stmt.name, STRATEGY)
        for clause in stmt.when_clauses:
            self._check_when_clause(clause)

    def _check_when_clause(self, clause: WhenClause) -> None:
        # Wired for real now that SeriesType exists (was a Phase 3 TODO
        # stub). Both operands must be Series<Float> -- a strategy
        # crossover check operates on a price/indicator history, not a
        # single scalar.
        for operand, side in ((clause.left, "left"), (clause.right, "right")):
            operand_type = self._check_expr(operand)
            if operand_type != SeriesType(FLOAT):
                raise QSTypeError(
                    f"'when' clause {side} operand must be Series<Float>, got {operand_type}",
                    operand.span.line,
                    operand.span.column,
                )
        self._check_stmt(clause.body)

    def _check_backtest(self, stmt: BacktestStmt) -> None:
        # Phase 3.2 left this half-stubbed ("must resolve to a declared
        # StrategyDecl once strategy type-checking exists") -- wired for
        # real now, through the same TypeEnvironment lookup any other
        # identifier goes through, not a separate mechanism.
        strategy_type = self.env.get(stmt.strategy_name)
        if strategy_type is None:
            raise QSTypeError(
                f"undefined strategy '{stmt.strategy_name}'", stmt.span.line, stmt.span.column
            )
        if strategy_type != STRATEGY:
            raise QSTypeError(
                f"'{stmt.strategy_name}' is not a strategy (it is {strategy_type})",
                stmt.span.line,
                stmt.span.column,
            )

        # `on SYM` names a variable holding a PriceSeries -- the grammar
        # makes it an IDENTIFIER, so it resolves like any other name.
        symbol_type = self.env.get(stmt.symbol)
        if symbol_type is None:
            raise QSTypeError(
                f"undefined name '{stmt.symbol}'", stmt.span.line, stmt.span.column
            )
        if symbol_type != PRICE_SERIES:
            raise QSTypeError(
                f"backtest symbol '{stmt.symbol}' must be a PriceSeries, got {symbol_type}",
                stmt.span.line,
                stmt.span.column,
            )

    # ---- expressions ----

    def _check_expr(self, expr: Expr) -> Type:
        result: Type
        if isinstance(expr, NumberLiteral):
            result = FLOAT if isinstance(expr.value, float) else INT
        elif isinstance(expr, StringLiteral):
            result = STRING
        elif isinstance(expr, BoolLiteral):
            result = BOOL
        elif isinstance(expr, Identifier):
            result = self._check_identifier(expr)
        elif isinstance(expr, Assignment):
            result = self._check_assignment(expr)
        elif isinstance(expr, Unary):
            result = self._check_unary(expr)
        elif isinstance(expr, Binary):
            result = self._check_binary(expr)
        elif isinstance(expr, Logical):
            result = self._check_logical(expr)
        elif isinstance(expr, Call):
            result = self._check_call(expr)
        elif isinstance(expr, FieldAccess):
            result = self._check_field_access(expr)
        elif isinstance(expr, ArrayLiteral):
            result = self._check_array_literal(expr)
        else:
            raise QSTypeError(
                f"internal: unhandled expression {type(expr).__name__}",
                expr.span.line,
                expr.span.column,
            )

        expr.inferred_type = result
        return result

    def _check_identifier(self, expr: Identifier) -> Type:
        found = self.env.get(expr.name)
        if found is None:
            raise QSTypeError(f"undefined name '{expr.name}'", expr.span.line, expr.span.column)
        return found

    def _check_field_access(self, expr: FieldAccess) -> Type:
        obj_type = self._check_expr(expr.obj)
        if obj_type == PRICE_SERIES and expr.name in (
            "open",
            "high",
            "low",
            "close",
            "volume",
        ):
            return SeriesType(FLOAT)
        # No general record/module type system exists yet (docs/
        # language-reference.md "Known limitations") -- anything else,
        # including PriceSeries's own `.dates` field (there's no
        # Series<Date> type to give it), is accepted as Void rather than
        # claiming a type that can't be verified.
        return VOID

    def _check_assignment(self, expr: Assignment) -> Type:
        declared = self.env.get(expr.name)
        if declared is None:
            raise QSTypeError(
                f"cannot assign to undefined name '{expr.name}'", expr.span.line, expr.span.column
            )
        value_type = self._check_expr(expr.value)
        if not is_assignable(value_type, declared):
            raise QSTypeError(
                f"cannot assign {value_type} to '{expr.name}' ({declared})",
                expr.span.line,
                expr.span.column,
            )
        return declared

    def _check_unary(self, expr: Unary) -> Type:
        operand_type = self._check_expr(expr.operand)
        if expr.operator == "-":
            if operand_type not in (INT, FLOAT):
                raise QSTypeError(
                    f"unary '-' requires a numeric operand, got {operand_type}",
                    expr.span.line,
                    expr.span.column,
                )
            return operand_type
        if expr.operator == "!":
            if operand_type != BOOL:
                raise QSTypeError(
                    f"unary '!' requires a Bool operand, got {operand_type}",
                    expr.span.line,
                    expr.span.column,
                )
            return BOOL
        raise QSTypeError(
            f"unknown unary operator '{expr.operator}'", expr.span.line, expr.span.column
        )

    def _check_binary(self, expr: Binary) -> Type:
        left_type = self._check_expr(expr.left)
        right_type = self._check_expr(expr.right)

        if expr.operator in _ARITHMETIC_OPS:
            # Found while wiring up Phase 6's LLVM backend: `+` on two
            # Strings type-checks fine dynamically in both the tree-
            # walking and IR interpreters (string concatenation), but
            # this check required numeric operands unconditionally --
            # a real, pre-existing gap between the checker and what the
            # interpreters actually do, invisible until something tried
            # to run typed string concatenation through the full
            # pipeline for the first time. Only the clean String+String
            # case is legalized here, not the interpreters' looser
            # "either side is a string" coercion -- that's a dynamic-
            # typing leniency, not a rule this static checker should
            # necessarily also encode.
            if expr.operator == "+" and left_type == STRING and right_type == STRING:
                return STRING
            if left_type not in (INT, FLOAT) or right_type not in (INT, FLOAT):
                raise QSTypeError(
                    f"'{expr.operator}' requires numeric operands, "
                    f"got {left_type} and {right_type}",
                    expr.span.line,
                    expr.span.column,
                )
            return FLOAT if FLOAT in (left_type, right_type) else INT

        if expr.operator in _COMPARISON_OPS:
            if left_type not in (INT, FLOAT) or right_type not in (INT, FLOAT):
                raise QSTypeError(
                    f"'{expr.operator}' requires numeric operands, "
                    f"got {left_type} and {right_type}",
                    expr.span.line,
                    expr.span.column,
                )
            return BOOL

        if expr.operator in _EQUALITY_OPS:
            if left_type != right_type:
                raise QSTypeError(
                    f"'{expr.operator}' requires operands of the same type, "
                    f"got {left_type} and {right_type}",
                    expr.span.line,
                    expr.span.column,
                )
            return BOOL

        raise QSTypeError(
            f"unknown binary operator '{expr.operator}'", expr.span.line, expr.span.column
        )

    def _check_logical(self, expr: Logical) -> Type:
        left_type = self._check_expr(expr.left)
        right_type = self._check_expr(expr.right)
        if left_type != BOOL or right_type != BOOL:
            raise QSTypeError(
                f"'{expr.operator}' requires Bool operands, got {left_type} and {right_type}",
                expr.span.line,
                expr.span.column,
            )
        return BOOL

    def _check_call(self, expr: Call) -> Type:
        callee_type = self._check_expr(expr.callee)
        if not isinstance(callee_type, FunctionType):
            raise QSTypeError(
                f"cannot call a value of type {callee_type}", expr.span.line, expr.span.column
            )
        if len(expr.arguments) != len(callee_type.params):
            raise QSTypeError(
                f"expected {len(callee_type.params)} argument(s), got {len(expr.arguments)}",
                expr.span.line,
                expr.span.column,
            )
        for arg, param_type in zip(expr.arguments, callee_type.params, strict=True):
            arg_type = self._check_expr(arg)
            if not is_assignable(arg_type, param_type):
                raise QSTypeError(
                    f"argument type mismatch: expected {param_type}, got {arg_type}",
                    arg.span.line,
                    arg.span.column,
                )
        return callee_type.ret

    def _check_array_literal(self, expr: ArrayLiteral) -> Type:
        # Array literals default to Series, not a bare Array: this
        # language operates on series by default (docs/language-
        # reference.md "Array literals default to Series") -- `[1, 2, 3]`
        # is a Series<Int> a strategy could feed straight into SMA/mean/
        # etc., not an opaque collection type nothing else understands.
        if not expr.elements:
            raise QSTypeError(
                "cannot infer the type of an empty array literal (v1 has no "
                "annotation syntax for array literals yet)",
                expr.span.line,
                expr.span.column,
            )
        element_types = [self._check_expr(e) for e in expr.elements]
        first = element_types[0]
        for element, element_type in zip(expr.elements[1:], element_types[1:], strict=False):
            if element_type != first:
                raise QSTypeError(
                    f"array elements must all be the same type: {first} vs {element_type}",
                    element.span.line,
                    element.span.column,
                )
        return SeriesType(first)


def _all_paths_return(block: Block) -> bool:
    """Flat "does every branch end in return" check -- not full CFG
    reachability. Matches the general v1-vs-later posture in
    docs/spec.md "Scope cut line": good enough to catch the common case,
    documented as a known limitation rather than silently pretending to
    be exhaustive."""
    for stmt in block.statements:
        if isinstance(stmt, ReturnStmt):
            return True
        if isinstance(stmt, IfStmt) and stmt.else_branch is not None:
            if _stmt_returns(stmt.then_branch) and _stmt_returns(stmt.else_branch):
                return True
        if isinstance(stmt, Block) and _all_paths_return(stmt):
            return True
    return False


def _stmt_returns(stmt: Stmt) -> bool:
    if isinstance(stmt, ReturnStmt):
        return True
    if isinstance(stmt, Block):
        return _all_paths_return(stmt)
    if isinstance(stmt, IfStmt) and stmt.else_branch is not None:
        return _stmt_returns(stmt.then_branch) and _stmt_returns(stmt.else_branch)
    return False
