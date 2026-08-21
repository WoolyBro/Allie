"""AST node definitions -- Phase 2.3, designed here since no separate AST
spec was provided (Phase 3's typechecker spec assumes one exists).

Nodes are plain dataclasses, not the hand-built tuples Phase 1's evaluator
consumed. That's a deliberate departure from the earlier "the parser
targets this exact tuple shape" plan: Phase 3 needs a mutable
`inferred_type` field on every expression node, and tuples can't hold
that. allie.runtime.interpreter has NOT been adapted to walk these nodes
yet -- see ROADMAP.md for that follow-up.

Every node carries a SourceSpan for diagnostics. `Expr`/`Stmt` are bare
marker base classes (not dataclasses themselves) so subclasses can freely
mix required and defaulted fields without dataclass field-ordering
constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from allie.errors import SourceSpan

# inferred_type is typed Any (not allie.typechecker.types.Type) to avoid
# a circular import -- the typechecker imports these node classes, so
# these node classes can't import the typechecker back.


class Expr:
    """Marker base for expression nodes -- see docs/grammar.md. The bare
    `span` annotation (no default, not a dataclass field here) exists
    only so mypy knows every subclass has one; each concrete subclass
    still declares its own `span` field for real."""

    span: SourceSpan


class Stmt:
    """Marker base for statement nodes -- see docs/grammar.md. See Expr's
    docstring for why `span` is declared here without making Stmt itself
    a dataclass."""

    span: SourceSpan


@dataclass(kw_only=True)
class TypeAnnotation:
    """Surface-syntax type, e.g. `Int` or `Series[Float]` (grammar's
    `type` production). Kept separate from the resolved
    allie.typechecker.types.Type so the parser doesn't need to import the
    typechecker."""

    name: str
    span: SourceSpan
    type_arg: TypeAnnotation | None = None


# ---- Expressions ----


@dataclass(kw_only=True)
class NumberLiteral(Expr):
    span: SourceSpan
    value: int | float
    inferred_type: Any = field(default=None, repr=False, compare=False)


@dataclass(kw_only=True)
class StringLiteral(Expr):
    span: SourceSpan
    value: str
    inferred_type: Any = field(default=None, repr=False, compare=False)


@dataclass(kw_only=True)
class BoolLiteral(Expr):
    span: SourceSpan
    value: bool
    inferred_type: Any = field(default=None, repr=False, compare=False)


@dataclass(kw_only=True)
class Identifier(Expr):
    span: SourceSpan
    name: str
    inferred_type: Any = field(default=None, repr=False, compare=False)


@dataclass(kw_only=True)
class Assignment(Expr):
    span: SourceSpan
    name: str
    value: Expr
    inferred_type: Any = field(default=None, repr=False, compare=False)


@dataclass(kw_only=True)
class Unary(Expr):
    span: SourceSpan
    operator: str
    operand: Expr
    inferred_type: Any = field(default=None, repr=False, compare=False)


@dataclass(kw_only=True)
class Binary(Expr):
    span: SourceSpan
    operator: str
    left: Expr
    right: Expr
    inferred_type: Any = field(default=None, repr=False, compare=False)


@dataclass(kw_only=True)
class Logical(Expr):
    """`and`/`or` -- kept distinct from Binary so the interpreter/checker
    can special-case short-circuiting without inspecting `operator`."""

    span: SourceSpan
    operator: str
    left: Expr
    right: Expr
    inferred_type: Any = field(default=None, repr=False, compare=False)


@dataclass(kw_only=True)
class Call(Expr):
    span: SourceSpan
    callee: Expr
    arguments: list[Expr]
    inferred_type: Any = field(default=None, repr=False, compare=False)


@dataclass(kw_only=True)
class FieldAccess(Expr):
    """`a.b` -- the grammar's `call → primary (... | "." IDENTIFIER)*`
    alternative that isn't a call. No dedicated grammar production name
    was given for this, so it gets its own node rather than overloading
    Call."""

    span: SourceSpan
    obj: Expr
    name: str
    inferred_type: Any = field(default=None, repr=False, compare=False)


@dataclass(kw_only=True)
class ArrayLiteral(Expr):
    span: SourceSpan
    elements: list[Expr]
    inferred_type: Any = field(default=None, repr=False, compare=False)


# ---- Statements ----


@dataclass(kw_only=True)
class Param:
    name: str
    type_annotation: TypeAnnotation | None
    span: SourceSpan


@dataclass(kw_only=True)
class LetStmt(Stmt):
    span: SourceSpan
    name: str
    type_annotation: TypeAnnotation | None
    initializer: Expr


@dataclass(kw_only=True)
class Block(Stmt):
    span: SourceSpan
    statements: list[Stmt]


@dataclass(kw_only=True)
class FuncDecl(Stmt):
    span: SourceSpan
    name: str
    params: list[Param]
    return_type: TypeAnnotation | None
    body: Block


@dataclass(kw_only=True)
class IfStmt(Stmt):
    span: SourceSpan
    condition: Expr
    then_branch: Stmt
    else_branch: Stmt | None


@dataclass(kw_only=True)
class WhileStmt(Stmt):
    span: SourceSpan
    condition: Expr
    body: Stmt


@dataclass(kw_only=True)
class ForStmt(Stmt):
    span: SourceSpan
    initializer: LetStmt
    condition: Expr
    increment: Expr
    body: Stmt


@dataclass(kw_only=True)
class BreakStmt(Stmt):
    span: SourceSpan


@dataclass(kw_only=True)
class ContinueStmt(Stmt):
    span: SourceSpan


@dataclass(kw_only=True)
class ReturnStmt(Stmt):
    span: SourceSpan
    value: Expr | None


@dataclass(kw_only=True)
class ImportStmt(Stmt):
    span: SourceSpan
    path: str


@dataclass(kw_only=True)
class WhenClause:
    span: SourceSpan
    left: Expr
    operator: str  # "crosses_above" | "crosses_below"
    right: Expr
    body: Stmt


@dataclass(kw_only=True)
class StrategyDecl(Stmt):
    span: SourceSpan
    name: str
    when_clauses: list[WhenClause]


@dataclass(kw_only=True)
class BacktestStmt(Stmt):
    span: SourceSpan
    strategy_name: str
    symbol: str
    start: int | float
    end: int | float


@dataclass(kw_only=True)
class ExprStmt(Stmt):
    span: SourceSpan
    expression: Expr


@dataclass(kw_only=True)
class Program:
    statements: list[Stmt]
