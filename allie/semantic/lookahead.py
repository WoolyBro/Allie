"""Look-ahead bias analysis -- a compile-time guarantee, not a runtime one.

`docs/spec.md` states the project's thesis: *"Look-ahead bias is the
single biggest source of silent backtest bugs. Making 'you cannot see
the future' a property of the execution model (not a discipline the
strategy author has to maintain) is the whole point."*

The backtest engine already honours that at runtime -- `buy()`/`sell()`
can only ever read `close[current_index]`, so an action physically
cannot reach a later bar. But that only constrains the *engine*. It says
nothing about the **data** a strategy hands the engine, and that is
where the bug actually lives:

    let tomorrow: Series[Float] = shift(AAPL.close, -1);
    strategy Clairvoyant {
        when tomorrow crosses_above slow: buy();
    }

`shift(s, -1)[i]` is `s[i+1]` -- tomorrow's price, available today.
Before this pass existed the language accepted that happily, and the
backtest reported Sharpe 3.34 / CAGR +41.56% against the honest
strategy's -0.16 / -2.89%. Nothing anywhere said the number was
meaningless.

This pass makes that program **fail to compile**.

## What it does

A taint analysis over `Series` values. Every series is either

- **point-in-time safe** -- every element is derived only from
  observations at or before its own index, or
- **future-tainted** -- at least one element may embed data from a
  later index.

Taint is introduced, propagated, and then rejected at a sink:

- **source** -- `shift(s, k)` with a negative `k`, or a `k` this pass
  cannot prove non-negative.
- **propagation** -- through `let`/assignment, through any call that
  receives a tainted argument, through field access, and through any
  user function whose body contains a source.
- **sink** -- a `when` clause operand: the one place a *trading
  decision* is made at a specific bar.

## Why the sink is only `when`

Look-ahead bias is not "touching future data" -- it is "making a
decision at time t informed by data from after t". Computing
`print(mean(shift(x, -1)))` for analysis is perfectly legitimate and
stays legal. Feeding that series into a crossing signal that drives
`buy()` is not, and that is exactly what gets rejected.

## Soundness

Deliberately biased toward false positives over false negatives, because
this is a safety property: a missed violation is a silently wrong
backtest, while a spurious one is a compile error the user can see and
fix. Concretely, a `shift` whose period is not a literal is treated as
tainted -- the pass cannot prove it non-negative, so it does not
pretend to.

**What it does not catch**, stated plainly rather than implied:

- Data that was already shifted *outside* the language (a CSV whose
  `close` column was pre-shifted). Nothing in the source text reveals
  that.
- A series indexed by hand, since the language has no indexing.
- Taint laundered through a user function's *parameter* rather than its
  body -- handled by the argument rule, but only because arguments are
  checked at the call site; a function that stores and later returns a
  tainted value would need real interprocedural dataflow.

Runs after type checking (so `inferred_type` is populated for
diagnostics) and reports through `DiagnosticBag`, collect-and-report-all
like the parser and resolver rather than fail-fast.
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
    Unary,
    WhenClause,
    WhileStmt,
)
from allie.errors import DiagnosticBag, SourceSpan

# Functions that can move data *backwards* in time -- i.e. place a later
# observation at an earlier index. Maps name -> index of the argument
# controlling the direction.
#
# `shift` is the only one today. Every other series function in the
# stdlib is backward-looking by construction (SMA/EMA/rolling_* average
# a trailing window; diff/pct_change compare against the *previous*
# element; crosses_* compare bar i against bar i-1). If a genuinely
# forward-looking function is ever added, it belongs here -- and
# `test_lookahead.py` asserts this table still covers every stdlib
# function whose name suggests it could shift data.
TIME_SHIFTING_FUNCTIONS: dict[str, int] = {"shift": 1}


class LookaheadAnalysis:
    def __init__(self, diagnostics: DiagnosticBag | None = None) -> None:
        self.diagnostics = diagnostics if diagnostics is not None else DiagnosticBag()
        # Scope stack of variable name -> is-tainted.
        self._scopes: list[dict[str, bool]] = [{}]
        # User functions whose body contains a taint source, so calling
        # them yields a tainted value regardless of arguments.
        self._tainting_functions: set[str] = set()

    # ---- entry point ----

    def analyze(self, program: Program) -> None:
        # Pre-scan: a function may be called before its declaration is
        # reached in a single forward pass, so taint-producing functions
        # are identified up front.
        for stmt in program.statements:
            if isinstance(stmt, FuncDecl) and self._contains_taint_source(stmt.body):
                self._tainting_functions.add(stmt.name)

        for stmt in program.statements:
            self._check_stmt(stmt)

    # ---- scopes ----

    def _push(self) -> None:
        self._scopes.append({})

    def _pop(self) -> None:
        self._scopes.pop()

    def _bind(self, name: str, tainted: bool) -> None:
        self._scopes[-1][name] = tainted

    def _assign(self, name: str, tainted: bool) -> None:
        """Assignment mutates the binding where it was declared, so taint
        introduced inside a block is visible to the enclosing scope --
        matching the runtime, where the value really does escape."""
        for scope in reversed(self._scopes):
            if name in scope:
                scope[name] = tainted
                return
        self._bind(name, tainted)

    def _is_tainted_name(self, name: str) -> bool:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return False

    # ---- statements ----

    def _check_stmt(self, stmt: Stmt) -> None:
        if isinstance(stmt, LetStmt):
            self._bind(stmt.name, self._is_tainted(stmt.initializer))
        elif isinstance(stmt, ExprStmt):
            self._is_tainted(stmt.expression)
        elif isinstance(stmt, Block):
            self._push()
            for inner in stmt.statements:
                self._check_stmt(inner)
            self._pop()
        elif isinstance(stmt, IfStmt):
            self._is_tainted(stmt.condition)
            self._check_stmt(stmt.then_branch)
            if stmt.else_branch is not None:
                self._check_stmt(stmt.else_branch)
        elif isinstance(stmt, WhileStmt):
            self._is_tainted(stmt.condition)
            self._check_stmt(stmt.body)
        elif isinstance(stmt, ForStmt):
            self._push()
            self._check_stmt(stmt.initializer)
            self._is_tainted(stmt.condition)
            self._is_tainted(stmt.increment)
            self._check_stmt(stmt.body)
            self._pop()
        elif isinstance(stmt, FuncDecl):
            self._push()
            for param in stmt.params:
                self._bind(param.name, False)
            self._check_stmt(stmt.body)
            self._pop()
        elif isinstance(stmt, ReturnStmt):
            if stmt.value is not None:
                self._is_tainted(stmt.value)
        elif isinstance(stmt, StrategyDecl):
            for clause in stmt.when_clauses:
                self._check_when_clause(stmt.name, clause)
        elif isinstance(stmt, (BreakStmt, ContinueStmt, ImportStmt, BacktestStmt)):
            pass

    def _check_when_clause(self, strategy_name: str, clause: WhenClause) -> None:
        """The sink. A `when` operand decides a trade at a specific bar,
        so it must be provably point-in-time safe."""
        for operand, side in ((clause.left, "left"), (clause.right, "right")):
            if self._is_tainted(operand):
                self.diagnostics.error(
                    f"look-ahead bias: the {side} operand of this 'when' clause in "
                    f"strategy '{strategy_name}' may contain data from a future bar. "
                    f"A trading decision cannot depend on data the strategy would not "
                    f"have had at that bar. {self._describe_source(operand)}",
                    operand.span,
                )
        self._check_stmt(clause.body)

    def _describe_source(self, expr: Expr) -> str:
        name = _root_identifier(expr)
        if name is not None:
            return (
                f"'{name}' is derived from a backwards time shift "
                "(shift(series, k) with a negative k places bar i+k at bar i)."
            )
        return "It is derived from a backwards time shift."

    # ---- expressions: returns True if the value may embed future data ----

    def _is_tainted(self, expr: Expr) -> bool:
        if isinstance(expr, (NumberLiteral, StringLiteral, BoolLiteral)):
            return False
        if isinstance(expr, Identifier):
            return self._is_tainted_name(expr.name)
        if isinstance(expr, Assignment):
            tainted = self._is_tainted(expr.value)
            self._assign(expr.name, tainted)
            return tainted
        if isinstance(expr, Unary):
            return self._is_tainted(expr.operand)
        if isinstance(expr, (Binary, Logical)):
            left = self._is_tainted(expr.left)
            right = self._is_tainted(expr.right)
            return left or right
        if isinstance(expr, FieldAccess):
            return self._is_tainted(expr.obj)
        if isinstance(expr, ArrayLiteral):
            return any(self._is_tainted(element) for element in expr.elements)
        if isinstance(expr, Call):
            return self._is_tainted_call(expr)
        return False

    def _is_tainted_call(self, call: Call) -> bool:
        # Evaluate every argument for its own side effects (nested
        # assignments) before deciding, so nothing is short-circuited away.
        argument_taint = [self._is_tainted(argument) for argument in call.arguments]

        callee = call.callee
        name = callee.name if isinstance(callee, Identifier) else None

        if name is not None and name in TIME_SHIFTING_FUNCTIONS:
            if self._shifts_backwards(call, TIME_SHIFTING_FUNCTIONS[name]):
                return True

        if name is not None and name in self._tainting_functions:
            return True

        return any(argument_taint)

    def _shifts_backwards(self, call: Call, period_index: int) -> bool:
        """True unless the shift period is provably non-negative.

        A non-literal period is treated as unsafe: this pass cannot prove
        `shift(s, n)` moves data backwards in time, and for a safety
        property "cannot prove safe" has to mean "reject". The fix is to
        pass a literal.
        """
        if period_index >= len(call.arguments):
            return False  # arity error -- the type checker reports it
        period = call.arguments[period_index]
        if isinstance(period, NumberLiteral):
            return bool(period.value < 0)
        if isinstance(period, Unary) and period.operator == "-":
            # `-1` parses as a unary negation of a positive literal.
            return isinstance(period.operand, NumberLiteral) and period.operand.value != 0
        return True

    # ---- pre-scan helper ----

    def _contains_taint_source(self, node: Stmt | Expr) -> bool:
        """Whether a function body introduces taint on its own, ignoring
        its arguments. Used only for the interprocedural approximation
        described in this module's docstring."""
        found = False

        def walk(current: object) -> None:
            nonlocal found
            if found or current is None:
                return
            if isinstance(current, Call):
                callee = current.callee
                if (
                    isinstance(callee, Identifier)
                    and callee.name in TIME_SHIFTING_FUNCTIONS
                    and self._shifts_backwards(current, TIME_SHIFTING_FUNCTIONS[callee.name])
                ):
                    found = True
                    return
            for child in _children(current):
                walk(child)

        walk(node)
        return found


def _children(node: object) -> list[object]:
    """Every AST child of `node`, for generic traversal."""
    fields: list[object] = []
    for attribute in vars(node).values() if hasattr(node, "__dict__") else []:
        if isinstance(attribute, list):
            fields.extend(attribute)
        else:
            fields.append(attribute)
    return [f for f in fields if isinstance(f, (Expr, Stmt))]


def _root_identifier(expr: Expr) -> str | None:
    """The variable name an expression is rooted at, for error messages."""
    if isinstance(expr, Identifier):
        return expr.name
    if isinstance(expr, FieldAccess):
        return _root_identifier(expr.obj)
    if isinstance(expr, Call) and expr.arguments:
        return _root_identifier(expr.arguments[0])
    return None


def check_lookahead(program: Program, diagnostics: DiagnosticBag | None = None) -> DiagnosticBag:
    """Convenience entry point: analyzes `program`, returns the bag."""
    analysis = LookaheadAnalysis(diagnostics)
    analysis.analyze(program)
    return analysis.diagnostics


__all__ = [
    "TIME_SHIFTING_FUNCTIONS",
    "LookaheadAnalysis",
    "SourceSpan",
    "check_lookahead",
]
