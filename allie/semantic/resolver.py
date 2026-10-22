"""Resolver: scopes and placement. Runs after parsing (allie.parser),
before type-checking (allie.typechecker.checker), and tracks
loop_depth/inside_strategy/inside_when so that a jump or a `when` is
only accepted where it can mean something.

Catches structural errors syntax alone can't: `break`/`continue` outside
a loop, `return` outside a function, a `backtest` referring to an
undeclared strategy. Tracks a scope stack (`Scope`) purely for
name-existence checks -- has this identifier been declared anywhere
visible. This is a *separate* structure from the type checker's own
TypeEnvironment (allie.typechecker.checker), which maps names to types;
Scope only tracks that a name exists. Don't conflate the two.

Design choice: a bare Identifier used as a Call's callee is NOT checked
for existence here. Builtins (SMA, print, ...) live in the stdlib
registry, not in any scope this pass keeps, so treating every call
target as "must be locally declared" would misfire on every builtin
call. The type checker validates that a call target resolves to a
FunctionType; this resolver only validates plain value references and
assignments.
"""

from __future__ import annotations

import difflib

from allie.ast.nodes import (
    ArrayLiteral,
    Assignment,
    Backtest,
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
)
from allie.errors import DiagnosticBag, SourceSpan


class Scope:
    def __init__(self, parent: Scope | None = None) -> None:
        self.parent = parent
        self.names: set[str] = set()

    def declare(self, name: str) -> None:
        self.names.add(name)

    def visible_names(self) -> set[str]:
        names: set[str] = set()
        scope: Scope | None = self
        while scope is not None:
            names.update(scope.names)
            scope = scope.parent
        return names

    def is_declared(self, name: str) -> bool:
        scope: Scope | None = self
        while scope is not None:
            if name in scope.names:
                return True
            scope = scope.parent
        return False


class Resolver:
    def __init__(self, diagnostics: DiagnosticBag | None = None) -> None:
        self.diagnostics = diagnostics if diagnostics is not None else DiagnosticBag()
        self.scope = Scope()
        self.loop_depth = 0
        self.inside_strategy = False
        self.inside_when = False
        self.inside_function = False

    def resolve(self, program: Program) -> None:
        for stmt in program.statements:
            self._resolve_stmt(stmt)

    def _push_scope(self) -> None:
        self.scope = Scope(parent=self.scope)

    def _pop_scope(self) -> None:
        assert self.scope.parent is not None
        self.scope = self.scope.parent

    # ---- statements ----

    def _resolve_stmt(self, stmt: Stmt) -> None:
        if isinstance(stmt, LetStmt):
            self._resolve_expr(stmt.initializer)
            self.scope.declare(stmt.name)
        elif isinstance(stmt, FuncDecl):
            self.scope.declare(stmt.name)
            self._resolve_function(stmt)
        elif isinstance(stmt, IfStmt):
            self._resolve_expr(stmt.condition)
            self._resolve_stmt(stmt.then_branch)
            if stmt.else_branch is not None:
                self._resolve_stmt(stmt.else_branch)
        elif isinstance(stmt, WhileStmt):
            self._resolve_expr(stmt.condition)
            self.loop_depth += 1
            self._resolve_stmt(stmt.body)
            self.loop_depth -= 1
        elif isinstance(stmt, ForStmt):
            self._resolve_for(stmt)
        elif isinstance(stmt, BreakStmt):
            self._check_jump_allowed("break", stmt.span)
        elif isinstance(stmt, ContinueStmt):
            self._check_jump_allowed("continue", stmt.span)
        elif isinstance(stmt, ReturnStmt):
            self._resolve_return(stmt)
        elif isinstance(stmt, ImportStmt):
            pass
        elif isinstance(stmt, StrategyDecl):
            self._resolve_strategy(stmt)
        elif isinstance(stmt, BacktestStmt):
            self._resolve_backtest(stmt.strategy_name, stmt.span)
            for part in (stmt.start, stmt.end, *(option.value for option in stmt.options)):
                self._resolve_expr(part)
        elif isinstance(stmt, ExprStmt):
            self._resolve_expr(stmt.expression)
        elif isinstance(stmt, Block):
            self._push_scope()
            for s in stmt.statements:
                self._resolve_stmt(s)
            self._pop_scope()
        else:
            raise TypeError(f"resolver: unhandled statement node {type(stmt).__name__}")

    def _check_jump_allowed(self, keyword: str, span: SourceSpan) -> None:
        """`break`/`continue` need an enclosing loop *in the same
        execution context*. `loop_depth` is reset to 0 when entering a
        function body or a `when` action (see `_resolve_function` /
        `_resolve_when_clause`), so a jump that would have to cross
        either boundary reports here rather than silently binding to a
        lexically-enclosing loop it could never actually reach at
        runtime. `inside_when` only selects the more specific wording --
        it is NOT what does the rejecting, and it deliberately does not
        subsume `loop_depth`: a loop written *inside* a `when` action is
        a perfectly valid jump target."""
        if self.loop_depth > 0:
            return
        if self.inside_when:
            self.diagnostics.error(
                f"'{keyword}' inside a 'when' action must target a loop within that action "
                "(a 'when' action runs per-bar; it cannot jump out to an enclosing loop)",
                span,
            )
        else:
            self.diagnostics.error(f"'{keyword}' outside a loop", span)

    def _resolve_for(self, stmt: ForStmt) -> None:
        self._push_scope()
        self._resolve_stmt(stmt.initializer)
        self._resolve_expr(stmt.condition)
        self.loop_depth += 1
        self._resolve_stmt(stmt.body)
        self.loop_depth -= 1
        self._resolve_expr(stmt.increment)
        self._pop_scope()

    def _resolve_function(self, decl: FuncDecl) -> None:
        was_inside_function = self.inside_function
        # A function body is its own execution context: a `break` here
        # cannot target a loop that lexically encloses the *declaration*.
        saved_loop_depth = self.loop_depth
        self.inside_function = True
        self.loop_depth = 0
        self._push_scope()
        for param in decl.params:
            self.scope.declare(param.name)
        self._resolve_stmt(decl.body)
        self._pop_scope()
        self.loop_depth = saved_loop_depth
        self.inside_function = was_inside_function

    def _resolve_return(self, stmt: ReturnStmt) -> None:
        if not self.inside_function:
            self.diagnostics.error("'return' outside a function", stmt.span)
        if stmt.value is not None:
            self._resolve_expr(stmt.value)

    def _resolve_strategy(self, stmt: StrategyDecl) -> None:
        self.scope.declare(stmt.name)
        was_inside_strategy = self.inside_strategy
        self.inside_strategy = True
        for clause in stmt.when_clauses:
            self._resolve_when_clause(clause)
        self.inside_strategy = was_inside_strategy

    def _resolve_backtest(self, strategy_name: str, span: SourceSpan) -> None:
        if not self.scope.is_declared(strategy_name):
            self.diagnostics.error(
                f"backtest refers to undeclared strategy '{strategy_name}'",
                span,
                hint=self._suggest(strategy_name),
            )

    def _resolve_when_clause(self, clause: WhenClause) -> None:
        if not self.inside_strategy:
            self.diagnostics.error("'when' clause outside a strategy", clause.span)
        self._resolve_expr(clause.condition)
        was_inside_when = self.inside_when
        # Same reasoning as _resolve_function: a `when` action is a
        # per-bar callback the backtest engine invokes, so a loop
        # enclosing the *strategy declaration* is not a reachable jump
        # target from inside the action.
        saved_loop_depth = self.loop_depth
        self.inside_when = True
        self.loop_depth = 0
        self._resolve_stmt(clause.body)
        self.loop_depth = saved_loop_depth
        self.inside_when = was_inside_when

    # ---- expressions ----

    def _suggest(self, name: str) -> str | None:
        """A "did you mean" hint when a declared name is a near miss."""
        names = sorted(self.scope.visible_names())
        for candidate in names:
            if candidate.lower() == name.lower():
                return f"did you mean '{candidate}'?"
        matches = difflib.get_close_matches(name, names, n=1, cutoff=0.75)
        return f"did you mean '{matches[0]}'?" if matches else None

    def _resolve_expr(self, expr: Expr) -> None:
        if isinstance(expr, (NumberLiteral, StringLiteral, BoolLiteral)):
            return
        if isinstance(expr, Identifier):
            if not self.scope.is_declared(expr.name):
                self.diagnostics.error(
                    f"undefined name '{expr.name}'", expr.span, hint=self._suggest(expr.name)
                )
            return
        if isinstance(expr, Assignment):
            if not self.scope.is_declared(expr.name):
                self.diagnostics.error(
                    f"cannot assign to undefined name '{expr.name}'",
                    expr.span,
                    hint=self._suggest(expr.name),
                )
            self._resolve_expr(expr.value)
            return
        if isinstance(expr, Unary):
            self._resolve_expr(expr.operand)
            return
        if isinstance(expr, (Binary, Logical)):
            self._resolve_expr(expr.left)
            self._resolve_expr(expr.right)
            return
        if isinstance(expr, Call):
            # See module docstring: call targets aren't existence-checked
            # here, since builtins have no registry yet.
            if not isinstance(expr.callee, Identifier):
                self._resolve_expr(expr.callee)
            for arg in expr.arguments:
                self._resolve_expr(arg)
            return
        if isinstance(expr, FieldAccess):
            self._resolve_expr(expr.obj)
            return
        if isinstance(expr, Index):
            self._resolve_expr(expr.obj)
            self._resolve_expr(expr.index)
            return
        if isinstance(expr, Backtest):
            self._resolve_backtest(expr.strategy_name, expr.span)
            for part in (expr.start, expr.end, *(option.value for option in expr.options)):
                self._resolve_expr(part)
            return
        if isinstance(expr, ArrayLiteral):
            for element in expr.elements:
                self._resolve_expr(element)
            return
        raise TypeError(f"resolver: unhandled expression node {type(expr).__name__}")
