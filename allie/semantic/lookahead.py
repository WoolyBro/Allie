"""Look-ahead bias analysis -- a compile-time guarantee, not a runtime one.

`docs/spec.md` states the project's thesis: *"Look-ahead bias is the
single biggest source of silent backtest bugs. Making 'you cannot see
the future' a property of the execution model (not a discipline the
strategy author has to maintain) is the whole point."*

The backtest engine already honours that at runtime -- an order fills at
the next bar's open, and an action can't read a later bar. But that only
constrains the *engine*. It says nothing about the **data** a strategy
hands the engine, and that is where the bug actually lives:

    let tomorrow: Series[Float] = shift(AAPL.close, -1);
    strategy Clairvoyant {
        when tomorrow crosses_above slow: buy();
    }

`shift(s, -1)[i]` is `s[i+1]` -- tomorrow's price, available today.
Without this pass the language accepts that happily, and the backtest
reports Sharpe 3.31 / CAGR +41.78%, against 1.14 / +13.39% for the same
rule on today's close (examples/cheat_demo.qs; tests/integration/
test_demo.py measures both). Nothing anywhere says the number is
meaningless. This pass makes that program **fail to compile**.

## What it does

A taint analysis. Every value is either point-in-time safe -- at every
bar it depends only on that bar and earlier ones -- or may carry data
from a later bar, in which case it carries the *sources* that brought
the future in.

**Sources** -- where future data enters:

- `shift(s, k)` with a `k` that isn't provably 0 or more. A small sign
  analysis proves it for a literal, a variable holding one (`let lag: Int
  = 1`), sums and products of those, `x % 5`, and a loop counter that
  starts at 0 or more and only grows.
- A builtin that reads a whole series (WHOLE_HISTORY_FUNCTIONS: `mean(c)`,
  `zscore(c)`, `percentile(...)`, `volatility(...)`): its value at any
  bar reflects the later ones. Every builtin is classified, as that, as
  time-shifting, or as point-in-time (POINT_IN_TIME_FUNCTIONS).
- `len(s)`: the number of bars counts the ones still to come. (Fine as
  a loop bound or a backtest's range, which aren't decisions.)
- One bar of a series, `s[k]`, for any `k` but a literal 0: compared with
  every bar, `c[len(c) - 1]` is the future for all but the last.
- A statistic of a whole backtest, `r.sharpe`: it covers the bars it
  would be compared with. (`r.equity` and the other per-bar fields are
  point-in-time.)
- A variable a backtest's `when` actions assigned, after that backtest:
  it holds what they did on every bar.
- A series assembled by hand from indexed elements, `[s[3], s[2]]`:
  which bar each element came from isn't something the pass can see.

**Propagation** -- through `let` and assignment, operators, field access
and indexing, and every call. A user function is analyzed *at each call
site*, with its parameters bound to the arguments' taints and its
module-level variables read and written where it's called -- so a
function returning a tainted global taints its result, one assigning a
tainted value to a global taints that global from the call on, and a
function that is declared but never called changes nothing.

**Flow** -- the analysis follows control flow: a later `s = c` clears an
earlier `s = shift(c, -1)`; the two branches of an `if` are both
possible afterwards; and a loop is analyzed to a fixpoint, so a value
tainted late in one iteration is tainted early in the next. Control
dependence counts too: a value assigned under a condition that may see
the future -- `if mean(c) > 150.0 { s = fast; }` -- carries its taint,
as does what follows an `if` that may break, continue or return. A
`when` action is analyzed where it's declared and again at each
backtest of its strategy, where it really runs, bar by bar.

**Sinks** -- the places a trading decision is made, which must be
provably point-in-time:

- a `when` condition;
- a condition inside a `when` action (`if mean(c) > 150.0 { buy(); }`);
- the size of an order (`target(mean(c) / c[0])`);
- a backtest option (`with stop_loss: volatility(r, 252)` calibrates the
  stop on the whole period's volatility);
- the condition of an `if` or loop that decides whether a backtest runs.

Each error points at the sub-expression that brings the future in and
follows with notes tracing it back to the source.

## Why the sinks are only decisions

Look-ahead bias is not "touching future data" -- it is "making a
decision at time t informed by data from after t". `print(mean(shift(x,
-1)))` is legitimate post-hoc analysis and stays legal.

## Soundness

Deliberately biased toward false positives over false negatives, because
this is a safety property: a missed violation is a silently wrong
backtest, while a spurious one is a compile error the user can see and
fix. Unprovable means rejected. allie/semantic/audit.py checks the same
property by experiment -- a decision at bar t that changes when the bars
after t are removed -- and tests/unit/test_lookahead_audit.py holds the
analysis to it: every generated program the analysis accepts must pass.

**What it does not catch**, stated plainly:

- Data that was already shifted *outside* the language (a CSV whose
  `close` column was pre-shifted). Nothing in the source text reveals it.
- A function's (or a `when` action's) module-level variables are
  resolved where it is called (or backtested), not where it was
  declared; a local there with the same name as one of them would be
  read instead.

Runs after type checking (it reads `inferred_type`) and reports through
`DiagnosticBag`, collect-and-report-all like the parser and resolver.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
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
    Widen,
)
from allie.errors import DiagnosticBag, SourceSpan
from allie.typechecker.types import BACKTEST_RESULT, PRICE_SERIES, MatrixType, SeriesType

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

# Builtins that read a whole series -- every bar of it -- to produce their
# result. `zscore(c)[i]` is measured against the mean and spread of all of
# `c`, later bars included, and `c > mean(c)` compares each bar with an
# average of the future too: at bar i both already know what comes after
# i. Taking a series (or a matrix of series) makes a call to one of these
# a taint source; on plain numbers (`bonferroni(p, n)`) it isn't one.
WHOLE_HISTORY_FUNCTIONS: frozenset[str] = frozenset(
    {
        "mean",
        "median",
        "std",
        "variance",
        "sum",
        "min",
        "max",
        "count",
        "covariance",
        "correlation",
        "percentile",
        "zscore",
        "max_drawdown",
        "volatility",
        "sharpe",
        "sortino",
        "value_at_risk",
        "expected_shortfall",
        "beta",
        "permutation_pvalue",
        "backtest_pvalue",
        "probabilistic_sharpe",
        "deflated_sharpe",
        "benjamini_hochberg",
        "holm",
        # How many bars there are, the later ones included: fine as a
        # loop bound or a backtest's range, not inside a decision.
        "len",
        # Matrices mix bars: a row of `matrix(c, 6, 10)` holds ten of them.
        "matrix",
        "row",
        "transpose",
        "inverse",
        "determinant",
        "dot",
        "covariance_matrix",
        "ols",
    }
)

# Builtins whose value at bar i depends only on bar i and earlier ones
# (or on no series at all): a taint passes through them from their
# arguments, and they add none. `tests/unit/test_lookahead.py` asserts
# every builtin is in exactly one of these tables and
# TIME_SHIFTING_FUNCTIONS, so a new one has to be classified on purpose.
POINT_IN_TIME_FUNCTIONS: frozenset[str] = frozenset(
    {
        "SMA",
        "EMA",
        "rolling_mean",
        "rolling_sum",
        "rolling_std",
        "rolling_zscore",
        "rolling_corr",
        "rolling_max",
        "rolling_min",
        "RSI",
        "ATR",
        "cumsum",
        "diff",
        # One more element at the end; every earlier one is as it was.
        "append",
        "pct_change",
        "log_returns",
        "cumulative_returns",
        "drawdown",
        "abs",
        "sqrt",
        "log",
        "exp",
        "floor",
        "ceil",
        "round",
        "bonferroni",
        "normal",
        "load",
        "print",
        # Keep the bar-per-row layout of the series they take or give.
        "with_column",
        "column",
        "rows",
        "cols",
        "identity",
        "buy",
        "sell",
        "short",
        "cover",
        "target",
    }
)

# The per-bar fields of a BacktestResult; every other field summarizes the
# whole backtest.
_PER_BAR_RESULT_FIELDS = frozenset({"equity", "returns", "drawdown", "position"})

# Loops and recursion are analyzed to a fixpoint; taints only grow and are
# finite sets, so this is a safety net, not a limit real programs reach.
_MAX_ITERATIONS = 12
_MAX_CALL_DEPTH = 32


# ---- taint ----


@dataclass(frozen=True)
class Source:
    """Where future data enters a value."""

    reason: str  # completes "'x' ..." in the error
    note: str  # said at the source itself
    hint: str
    span: SourceSpan
    certain: bool = True  # False: the future *may* enter here; the pass can't tell


Taint = frozenset[Source]
_CLEAN: Taint = frozenset()

_SHIFT_REASON = (
    "is derived from a backwards time shift "
    "(shift(series, k) with a negative k places bar i+k at bar i)."
)
_UNPROVEN_SHIFT_REASON = (
    "is derived from a time shift whose period can't be proven 0 or more "
    "(shift(series, k) with a negative k places bar i+k at bar i)."
)
_SHIFT_HINT = "a positive shift looks back: shift(x, 1) is the previous bar's value"
_WHOLE_HISTORY = (
    "is computed from the whole series, every bar of it, so at bar i it already "
    "reflects the bars after i. In a condition, use a trailing version: "
    "rolling_mean, rolling_std, rolling_zscore, rolling_max/rolling_min."
)
_WHOLE_HINT = "trailing versions: SMA, rolling_std, rolling_zscore, rolling_max, rolling_min"
_INDEX_REASON = (
    "is a single bar of a series (s[k]): compared with every bar, it is the future "
    "for the bars before k."
)
_INDEX_HINT = "compare with a trailing value instead, such as shift(c, 1) or rolling_max(c, 20)"
_RESULT_REASON = (
    "is a statistic of a whole backtest, which covers bars after the ones it would "
    "be compared with."
)
_RESULT_HINT = "decide from the bars so far, not from a backtest's results over the same bars"
_HAND_BUILT = (
    "is assembled from individually indexed elements (s[i]), and the compiler "
    "can't check which bars those come from. Build signals from series "
    "functions such as SMA or shift instead."
)
_HAND_BUILT_HINT = "build signals from series functions such as SMA or shift"
_LENGTH_REASON = (
    "is the number of bars in the whole series, so at bar i it counts the bars after i."
)
_LENGTH_HINT = "a fixed window, such as SMA(c, 20), doesn't depend on how many bars follow"
_ACTION_STATE_REASON = (
    "is changed by the 'when' actions of a strategy, so after a backtest of it, it holds "
    "what they did on every bar of that backtest."
)
_ACTION_STATE_HINT = "read it inside the action, bar by bar, or reset it before using it again"


@dataclass(frozen=True)
class _Binding:
    """What the analysis knows about a variable."""

    taint: Taint = _CLEAN
    nonneg: bool = False  # proven to be 0 or more
    span: SourceSpan | None = None  # where it was last given future data
    derived_from: tuple[str, ...] = ()  # the tainted variables that data came through
    chosen: bool = False  # the value is clean; the condition that picked it isn't


def _join(a: _Binding, b: _Binding) -> _Binding:
    if a == b:
        return a
    return _Binding(
        taint=a.taint | b.taint,
        nonneg=a.nonneg and b.nonneg,
        span=a.span if a.taint else b.span,
        derived_from=tuple(dict.fromkeys(a.derived_from + b.derived_from)),
        chosen=(a.chosen or not a.taint) and (b.chosen or not b.taint),
    )


_State = list[dict[str, _Binding]]


def _join_states(a: _State, b: _State) -> _State:
    joined: _State = []
    for left, right in zip(a, b, strict=False):
        scope = dict(left)
        for name, binding in right.items():
            scope[name] = _join(scope[name], binding) if name in scope else binding
        joined.append(scope)
    return joined


@dataclass(frozen=True)
class Checkpoint:
    """The analysis's state between REPL submissions."""

    scopes: _State
    functions: dict[str, FuncDecl]


@dataclass
class _LoopExits:
    depth: int  # the scope depth at the loop, which a break or continue returns to
    breaks: list[_State]
    continues: list[_State]


class LookaheadAnalysis:
    def __init__(self, diagnostics: DiagnosticBag | None = None) -> None:
        self.diagnostics = diagnostics if diagnostics is not None else DiagnosticBag()
        self._scopes: _State = [{}]
        self._functions: dict[str, FuncDecl] = {}
        # Per expression node (by id): its taint from the latest
        # evaluation, and the source it introduces itself, if any.
        self._taints: dict[int, Taint] = {}
        self._sources: dict[int, Source] = {}
        self._quiet = 0  # inside a fixpoint iteration: findings aren't final yet
        self._reported: set[tuple[int, int, str]] = set()
        self._loops: list[_LoopExits] = []
        self._returns: list[list[tuple[Taint, SourceSpan]]] = []
        self._returned: dict[int, SourceSpan] = {}  # call -> a return that carried the future
        self._active: dict[str, Taint] = {}  # functions being analyzed -> result so far
        self._recursed: set[str] = set()
        self._action: str | None = None  # the strategy whose `when` action is being analyzed
        self._in_array = 0
        self._assigned: dict[str, _Binding] = {}  # module-level assignments since the checkpoint
        # The tainted conditions control currently depends on: an `if`
        # being analyzed, a loop's condition, or an earlier `if` whose
        # branch may break, continue or return.
        self._guards: list[tuple[Expr, Taint]] = []
        self._strategies: dict[str, StrategyDecl] = {}
        # Per `when` action being analyzed at a backtest: the scope depth
        # at the action, and the variables outside it the action assigns.
        self._action_writes: list[tuple[int, set[str]]] = []

    # ---- entry point ----

    def analyze(self, program: Program) -> None:
        self._reported = set()
        self._taints, self._sources, self._returned = {}, {}, {}
        for decl in _function_declarations(program.statements):
            self._functions[decl.name] = decl
        for stmt in program.statements:
            self._stmt(stmt)

    # ---- the REPL's undo ----

    def checkpoint(self) -> Checkpoint:
        """The state to go back to if the next submission is rejected."""
        self._assigned = {}
        return Checkpoint(self._state(), dict(self._functions))

    def rollback(self, checkpoint: Checkpoint, *, ran: bool) -> None:
        """Forgets a rejected submission. If it ran part of the way, a
        variable it assigned may now hold any value it was given, so each
        such variable keeps that possibility: forgetting `s = shift(c, -1)`
        when it did run would let a later condition read `s` unchecked."""
        state = [dict(scope) for scope in checkpoint.scopes]
        if ran:
            for name, binding in self._assigned.items():
                if name in state[0]:
                    state[0][name] = _join(state[0][name], binding)
        self._scopes = state
        self._functions = dict(checkpoint.functions)

    # ---- state ----

    def _push(self) -> None:
        self._scopes.append({})

    def _pop(self) -> None:
        self._scopes.pop()

    def _lookup(self, name: str) -> _Binding | None:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def _bind(self, name: str, binding: _Binding) -> None:
        self._scopes[-1][name] = binding

    def _assign(self, name: str, binding: _Binding) -> None:
        """Updates the binding where it was declared, so a value assigned
        inside a block or a function escapes to it, as at run time."""
        for scope in reversed(self._scopes):
            if name in scope:
                break
        else:
            scope = self._scopes[0]
        scope[name] = binding
        depth = next(i for i, inner in enumerate(self._scopes) if inner is scope)
        for action_depth, written in self._action_writes:
            if depth < action_depth:
                written.add(name)
        if scope is self._scopes[0]:
            earlier = self._assigned.get(name)
            self._assigned[name] = binding if earlier is None else _join(earlier, binding)

    def _state(self, depth: int | None = None) -> _State:
        return [dict(scope) for scope in self._scopes[: depth if depth is not None else None]]

    def _restore(self, state: _State) -> None:
        self._scopes[: len(state)] = [dict(scope) for scope in state]

    def _value(self, expr: Expr, span: SourceSpan) -> _Binding:
        """The binding a variable gets when assigned `expr`: its taint,
        and the taint of every condition that decides whether this
        assignment happens -- `if mean(c) > 150.0 { s = fast; } else { s
        = slow; }` picks `s` with the future."""
        taint = self._expr(expr) | self._pc()
        if not taint:
            return _Binding(nonneg=self._nonneg(expr))
        own = self._taints.get(id(expr), _CLEAN)
        names = self._tainted_names(expr)
        for condition, _ in self._guards:
            names += self._tainted_names(condition)
        return _Binding(
            taint, self._nonneg(expr), span, tuple(dict.fromkeys(names)), chosen=not own
        )

    # ---- control dependence ----

    def _pc(self) -> Taint:
        return _union(taint for _, taint in self._guards)

    @contextmanager
    def _guarded(self, condition: Expr, taint: Taint) -> Iterator[None]:
        """Analysis inside a branch or loop that `condition` decides."""
        if not taint:
            yield
            return
        self._guards.append((condition, taint))
        try:
            yield
        finally:
            self._guards.pop()

    @contextmanager
    def _frame(self) -> Iterator[None]:
        """A loop or a function body: an `if` inside it that may break,
        continue or return keeps deciding what runs until it ends."""
        depth = len(self._guards)
        try:
            yield
        finally:
            del self._guards[depth:]

    # ---- statements ----

    def _stmt(self, stmt: Stmt) -> None:
        if isinstance(stmt, LetStmt):
            self._bind(stmt.name, self._value(stmt.initializer, stmt.span))
        elif isinstance(stmt, ExprStmt):
            self._expr(stmt.expression)
        elif isinstance(stmt, Block):
            self._push()
            try:
                for inner in stmt.statements:
                    self._stmt(inner)
            finally:
                self._pop()
        elif isinstance(stmt, IfStmt):
            taint = self._branch_condition(stmt.condition)
            with self._guarded(stmt.condition, taint):
                before = self._state()
                self._stmt(stmt.then_branch)
                after_then = self._state()
                self._restore(before)
                if stmt.else_branch is not None:
                    self._stmt(stmt.else_branch)
                self._restore(_join_states(after_then, self._state()))
            if taint and _jumps(stmt):
                # What follows runs only if the branch didn't jump away.
                self._guards.append((stmt.condition, taint))
        elif isinstance(stmt, WhileStmt):
            self._loop(stmt.condition, stmt.body, None)
        elif isinstance(stmt, ForStmt):
            self._push()
            try:
                self._stmt(stmt.initializer)
                self._loop(stmt.condition, stmt.body, stmt.increment)
            finally:
                self._pop()
        elif isinstance(stmt, BreakStmt):
            if self._loops:
                self._loops[-1].breaks.append(self._state(self._loops[-1].depth))
        elif isinstance(stmt, ContinueStmt):
            if self._loops:
                self._loops[-1].continues.append(self._state(self._loops[-1].depth))
        elif isinstance(stmt, ReturnStmt):
            taint = self._expr(stmt.value) if stmt.value is not None else _CLEAN
            taint |= self._pc()
            if self._returns:
                self._returns[-1].append((taint, stmt.span))
        elif isinstance(stmt, StrategyDecl):
            self._strategies[stmt.name] = stmt
            for clause in stmt.when_clauses:
                self._when(stmt.name, clause)
        elif isinstance(stmt, BacktestStmt):
            self._backtest(stmt)
        elif isinstance(stmt, (FuncDecl, ImportStmt)):
            pass  # a function is analyzed where it's called
        else:
            raise TypeError(f"look-ahead analysis: unhandled statement {type(stmt).__name__}")

    def _loop(self, condition: Expr, body: Stmt, increment: Expr | None) -> None:
        """To a fixpoint first -- quietly, since findings from a state that
        isn't final yet could be wrong -- then once more for the record,
        from the loop's settled state."""
        with self._frame():
            self._loop_to_fixpoint(condition, body, increment)

    def _loop_to_fixpoint(self, condition: Expr, body: Stmt, increment: Expr | None) -> None:
        self._quiet += 1
        try:
            for _ in range(_MAX_ITERATIONS):
                header = self._state()
                self._iteration(condition, body, increment, _LoopExits(len(self._scopes), [], []))
                self._restore(_join_states(header, self._state()))
                if self._state() == header:
                    break
        finally:
            self._quiet -= 1
        header = self._state()
        exits = _LoopExits(len(self._scopes), [], [])
        self._iteration(condition, body, increment, exits)
        state = _join_states(header, self._state())
        for broke in exits.breaks:
            state = _join_states(state, broke)
        self._restore(state)

    def _iteration(
        self, condition: Expr, body: Stmt, increment: Expr | None, exits: _LoopExits
    ) -> None:
        taint = self._branch_condition(condition)
        with self._guarded(condition, taint):
            self._loops.append(exits)
            try:
                self._stmt(body)
            finally:
                self._loops.pop()
            for skipped in exits.continues:
                self._restore(_join_states(self._state(), skipped))
            if increment is not None:
                self._expr(increment)

    def _branch_condition(self, condition: Expr) -> Taint:
        """A branch inside a `when` action decides what the action does."""
        if self._action is None:
            return self._expr(condition)
        return self._sink(
            condition, f"this condition inside a 'when' action of strategy '{self._action}'"
        )

    def _when(self, strategy: str, clause: WhenClause) -> None:
        """A `when` condition decides a trade at a specific bar, so it
        must be provably point-in-time safe; so must every decision its
        action makes."""
        self._sink(clause.condition, f"this 'when' condition in strategy '{strategy}'")
        self._action_body(strategy, clause)

    def _action_body(self, strategy: str, clause: WhenClause) -> None:
        outer, self._action = self._action, strategy
        self._push()
        try:
            self._stmt(clause.body)
        finally:
            self._pop()
            self._action = outer

    def _backtest(self, node: BacktestStmt | Backtest) -> None:
        self._expr(node.start)
        self._expr(node.end)
        for option in node.options:
            self._sink(option.value, f"backtest option '{option.name}'")
        for condition, _ in self._guards:
            self._report(
                condition,
                f"this condition, which decides whether strategy '{node.strategy_name}' "
                "is backtested,",
            )
        strategy = self._strategies.get(node.strategy_name)
        if strategy is not None:
            self._run_actions(strategy, node)

    def _run_actions(self, strategy: StrategyDecl, node: BacktestStmt | Backtest) -> None:
        """A backtest runs the strategy's actions bar by bar, with the
        variables they read as they are *now* -- analyzed like a loop over
        the bars. Afterwards, what the actions assigned reflects every bar
        of the backtest."""
        written: set[str] = set()
        self._action_writes.append((len(self._scopes), written))
        try:
            with self._frame():
                self._quiet += 1
                try:
                    for _ in range(_MAX_ITERATIONS):
                        header = self._state()
                        for clause in strategy.when_clauses:
                            self._action_body(strategy.name, clause)
                        self._restore(_join_states(header, self._state()))
                        if self._state() == header:
                            break
                finally:
                    self._quiet -= 1
                header = self._state()
                for clause in strategy.when_clauses:
                    self._action_body(strategy.name, clause)
                self._restore(_join_states(header, self._state()))
        finally:
            self._action_writes.pop()
        source = Source(
            _ACTION_STATE_REASON,
            f"the future enters here: this backtest runs the actions of '{strategy.name}' "
            "on every bar",
            _ACTION_STATE_HINT,
            node.span,
        )
        for name in sorted(written):
            binding = self._lookup(name)
            if binding is not None:
                self._assign(
                    name,
                    _join(binding, _Binding(frozenset({source}), False, node.span)),
                )

    # ---- sinks ----

    def _sink(self, expr: Expr, what: str) -> Taint:
        """Analyzes `expr`, a decision, and reports it if it may see the
        future. Returns its taint."""
        taint = self._expr(expr)
        if taint:
            self._report(expr, what)
        return taint

    def _report(self, expr: Expr, what: str) -> None:
        """Reports an analyzed, tainted expression: one error per culprit."""
        for culprit, source in self._culprits(expr):
            self._error(
                f"look-ahead bias: {what} may contain data from a future bar. "
                "A trading decision cannot depend on data the strategy would not have "
                f"had at that bar. {_subject(culprit)} {source.reason}",
                culprit.span,
                source.hint,
                self._trail(culprit, source),
            )

    def _culprits(self, node: Expr) -> list[tuple[Expr, Source]]:
        """The smallest sub-expressions that bring the future in: a
        tainted variable, a source, or a call whose result is tainted
        though its arguments aren't. Operators and calls that only pass a
        taint along are looked through."""
        taint = self._taints.get(id(node), _CLEAN)
        if not taint:
            return []
        own = self._sources.get(id(node))
        if own is not None and own.certain:
            return [(node, own)]
        if isinstance(node, Identifier):
            return [(node, _first(taint))]
        found = [culprit for child in _children(node) for culprit in self._culprits(child)]
        if own is not None and not any(source.certain for _, source in found):
            return [(node, own)]  # a possible source, and nothing inside it is certain
        return found or [(node, _first(taint))]

    def _trail(self, culprit: Expr, source: Source) -> list[tuple[str, SourceSpan]]:
        """Notes from the culprit back to where the future entered: each
        variable the data passed through, then the source itself."""
        notes: list[tuple[str, SourceSpan]] = []
        returned = self._returned.get(id(culprit))
        if returned is not None and isinstance(culprit, Call):
            name = _subject(culprit).strip("'").removesuffix("(...)")
            notes.append((f"'{name}' returns data from a future bar here", returned))
        queue = [culprit.name] if isinstance(culprit, Identifier) else []
        seen: set[str] = set()
        while queue and len(notes) < 4:
            name = queue.pop(0)
            if name in seen:
                continue
            seen.add(name)
            binding = self._lookup(name)
            if binding is not None and binding.span is not None:
                text = (
                    f"'{name}' is set here by a branch that a condition on future data chose"
                    if binding.chosen
                    else f"'{name}' is given data from a future bar here"
                )
                notes.append((text, binding.span))
                queue.extend(binding.derived_from)
        if source.span != culprit.span and all(source.span != span for _, span in notes):
            notes.append((source.note, source.span))
        return notes

    def _error(
        self, message: str, span: SourceSpan, hint: str, notes: list[tuple[str, SourceSpan]]
    ) -> None:
        if self._quiet:
            return
        key = (span.line, span.column, message)
        if key in self._reported:
            return
        self._reported.add(key)
        self.diagnostics.error(message, span, hint=hint)
        for text, where in notes:
            self.diagnostics.note(text, where)

    # ---- expressions ----

    def _expr(self, node: Expr) -> Taint:
        self._sources.pop(id(node), None)
        taint = self._expr_impl(node)
        self._taints[id(node)] = taint
        return taint

    def _source(self, node: Expr, source: Source) -> Taint:
        self._sources[id(node)] = source
        return frozenset({source})

    def _expr_impl(self, node: Expr) -> Taint:
        if isinstance(node, (NumberLiteral, StringLiteral, BoolLiteral)):
            return _CLEAN
        if isinstance(node, Identifier):
            binding = self._lookup(node.name)
            return binding.taint if binding is not None else _CLEAN
        if isinstance(node, Assignment):
            binding = self._value(node.value, node.span)
            self._assign(node.name, binding)
            return binding.taint
        if isinstance(node, (Unary, Widen)):
            return self._expr(node.operand)
        if isinstance(node, Logical):
            # The right operand runs only if the left doesn't decide it.
            left = self._expr(node.left)
            with self._guarded(node.left, left):
                return left | self._expr(node.right)
        if isinstance(node, Binary):
            return self._expr(node.left) | self._expr(node.right)
        if isinstance(node, FieldAccess):
            taint = self._expr(node.obj)
            if (
                node.obj.inferred_type == BACKTEST_RESULT
                and node.name not in _PER_BAR_RESULT_FIELDS
            ):
                taint |= self._source(
                    node,
                    Source(
                        _RESULT_REASON,
                        f"the future enters here: .{node.name} summarizes the whole backtest",
                        _RESULT_HINT,
                        node.span,
                    ),
                )
            return taint
        if isinstance(node, Index):
            taint = self._expr(node.obj) | self._expr(node.index)
            if (
                not self._in_array
                and isinstance(node.obj.inferred_type, SeriesType)
                and not _is_literal(node.index, 0)
            ):
                taint |= self._source(
                    node,
                    Source(
                        _INDEX_REASON,
                        "the future enters here: one fixed bar of a series",
                        _INDEX_HINT,
                        node.span,
                    ),
                )
            return taint
        if isinstance(node, ArrayLiteral):
            # An element `s[i]` is covered by the hand-built rule below,
            # not reported on its own.
            self._in_array += 1
            try:
                taint = _union(self._expr(element) for element in node.elements)
            finally:
                self._in_array -= 1
            if any(_contains_index(element) for element in node.elements):
                taint |= self._source(
                    node,
                    Source(
                        _HAND_BUILT,
                        "the future may enter here: a series built from individual elements",
                        _HAND_BUILT_HINT,
                        node.span,
                        certain=False,
                    ),
                )
            return taint
        if isinstance(node, Call):
            return self._call(node)
        if isinstance(node, Backtest):
            # Results are computed bar by bar from the past; a strategy that
            # cheated was rejected at its own `when`. Its scalar fields are
            # sources where they're read (FieldAccess, above).
            self._backtest(node)
            return _CLEAN
        raise TypeError(f"look-ahead analysis: unhandled expression {type(node).__name__}")

    def _call(self, node: Call) -> Taint:
        arguments = [self._expr(argument) for argument in node.arguments]
        name = node.callee.name if isinstance(node.callee, Identifier) else None
        if name == "target" and self._action is not None and node.arguments:
            self._sink(node.arguments[0], f"the size of this order in strategy '{self._action}'")
        if name is not None and name in self._functions:
            return self._call_function(self._functions[name], node, arguments)
        taint = _union(arguments)
        if name in TIME_SHIFTING_FUNCTIONS:
            period_index = TIME_SHIFTING_FUNCTIONS[name]
            if period_index < len(node.arguments) and not self._nonneg(
                node.arguments[period_index]
            ):
                taint |= self._source(node, _shift_source(node, node.arguments[period_index]))
        elif name == "len" and _reads_whole_history(node):
            taint |= self._source(
                node,
                Source(
                    _LENGTH_REASON,
                    "the future enters here: len(...) counts every bar, later ones included",
                    _LENGTH_HINT,
                    node.span,
                ),
            )
        elif _reads_whole_history(node):
            taint |= self._source(
                node,
                Source(
                    _WHOLE_HISTORY,
                    f"the future enters here: {name}(...) reads every bar of its argument, "
                    "later ones included",
                    _WHOLE_HINT,
                    node.span,
                ),
            )
        return taint

    def _call_function(self, decl: FuncDecl, node: Call, arguments: list[Taint]) -> Taint:
        """Analyzes the function's body here, with its parameters bound to
        these arguments: its result is what its returns carry, and what it
        assigns to module-level variables lands on them. A recursive call
        is analyzed to a fixpoint."""
        name = decl.name
        if name in self._active:
            self._recursed.add(name)
            return self._active[name] | _union(arguments)
        if len(self._active) >= _MAX_CALL_DEPTH:
            return _union(arguments)
        nonneg = [self._nonneg(argument) for argument in node.arguments]
        self._active[name] = _CLEAN
        try:
            result: Taint = _CLEAN
            where: SourceSpan | None = None
            for _ in range(_MAX_ITERATIONS):
                self._recursed.discard(name)
                result, where = self._function_body(decl, arguments, nonneg)
                if name not in self._recursed or result == self._active[name]:
                    break
                self._active[name] = result
            if where is not None:
                self._returned[id(node)] = where
            return result
        finally:
            del self._active[name]

    def _function_body(
        self, decl: FuncDecl, arguments: list[Taint], nonneg: list[bool]
    ) -> tuple[Taint, SourceSpan | None]:
        """The function's result taint, and a return that carried it."""
        self._push()
        for param, taint, is_nonneg in zip(decl.params, arguments, nonneg, strict=False):
            self._bind(param.name, _Binding(taint, is_nonneg, param.span if taint else None))
        self._returns.append([])
        try:
            with self._frame():
                for stmt in decl.body.statements:
                    self._stmt(stmt)
        finally:
            returned = self._returns.pop()
            self._pop()
        where = next((span for taint, span in returned if taint), None)
        return _union(taint for taint, _ in returned), where

    # ---- the sign analysis ----

    def _nonneg(self, node: Expr) -> bool:
        """Whether `node` is provably 0 or more -- enough to accept
        `shift(c, lag)` for `let lag: Int = 1`, or a loop counter."""
        if isinstance(node, NumberLiteral):
            return node.value >= 0
        if isinstance(node, Widen):
            return self._nonneg(node.operand)
        if isinstance(node, Unary):
            return node.operator == "-" and _is_literal(node.operand, 0)
        if isinstance(node, Identifier):
            binding = self._lookup(node.name)
            return binding is not None and binding.nonneg
        if isinstance(node, Assignment):
            return self._nonneg(node.value)
        if isinstance(node, Binary):
            if node.operator in ("+", "*", "/"):
                return self._nonneg(node.left) and self._nonneg(node.right)
            if node.operator == "%":
                # Floored modulo takes the divisor's sign.
                return isinstance(node.right, NumberLiteral) and node.right.value > 0
            return False
        if isinstance(node, Call) and isinstance(node.callee, Identifier):
            name = node.callee.name
            if name in ("len", "count", "rows", "cols", "abs", "sqrt"):
                return True
            if name in ("floor", "ceil", "round") and node.arguments:
                return self._nonneg(node.arguments[0])
        return False

    def _tainted_names(self, expr: Expr) -> tuple[str, ...]:
        names: dict[str, None] = {}
        for node in _walk(expr):
            if isinstance(node, Identifier):
                binding = self._lookup(node.name)
                if binding is not None and binding.taint:
                    names[node.name] = None
        return tuple(names)


# ---- helpers ----


def _union(taints: Iterable[Taint]) -> Taint:
    result = _CLEAN
    for taint in taints:
        result |= taint
    return result


def _first(taint: Taint) -> Source:
    """The source an error names: a certain one over a possible one, then
    the earliest."""
    return min(
        taint,
        key=lambda s: (not s.certain, s.span.line, s.span.column, s.reason),
    )


def _is_literal(node: Expr, value: int) -> bool:
    return isinstance(node, NumberLiteral) and node.value == value


def _shift_source(call: Call, period: Expr) -> Source:
    literal = None
    if isinstance(period, NumberLiteral):
        literal = int(period.value)
    elif (
        isinstance(period, Unary)
        and period.operator == "-"
        and isinstance(period.operand, NumberLiteral)
    ):
        literal = -int(period.operand.value)
    if literal is not None and literal < 0:
        ahead = -literal
        note = (
            f"the future enters here: shift by {literal} gives each bar the value from "
            f"{ahead} bar{'s' if ahead != 1 else ''} later"
        )
        return Source(_SHIFT_REASON, note, _SHIFT_HINT, call.span)
    return Source(
        _UNPROVEN_SHIFT_REASON,
        "the future may enter here: this shift's period can't be proven 0 or more",
        _SHIFT_HINT,
        call.span,
        certain=False,
    )


def _reads_whole_history(call: Call) -> bool:
    """A whole-history builtin applied to something with bars in it."""
    callee = call.callee
    return (
        isinstance(callee, Identifier)
        and callee.name in WHOLE_HISTORY_FUNCTIONS
        and any(
            isinstance(argument.inferred_type, (SeriesType, MatrixType))
            or argument.inferred_type in (PRICE_SERIES, BACKTEST_RESULT)
            for argument in call.arguments
        )
    )


def _children(node: Expr) -> list[Expr]:
    """The sub-expressions whose values make up `node`'s value."""
    if isinstance(node, (Binary, Logical)):
        return [node.left, node.right]
    if isinstance(node, (Unary, Widen)):
        return [node.operand]
    if isinstance(node, Call):
        return list(node.arguments)
    if isinstance(node, FieldAccess):
        return [node.obj]
    if isinstance(node, Index):
        return [node.obj, node.index]
    if isinstance(node, ArrayLiteral):
        return list(node.elements)
    if isinstance(node, Assignment):
        return [node.value]
    return []


def _walk(node: Expr) -> Iterator[Expr]:
    yield node
    for child in _children(node):
        yield from _walk(child)


def _jumps(stmt: Stmt) -> bool:
    """Whether `stmt` may break, continue or return."""
    if isinstance(stmt, (BreakStmt, ContinueStmt, ReturnStmt)):
        return True
    if isinstance(stmt, Block):
        return any(_jumps(inner) for inner in stmt.statements)
    if isinstance(stmt, IfStmt):
        return _jumps(stmt.then_branch) or (
            stmt.else_branch is not None and _jumps(stmt.else_branch)
        )
    if isinstance(stmt, (WhileStmt, ForStmt)):
        return _jumps(stmt.body)
    return False


def _contains_index(node: Expr) -> bool:
    return any(isinstance(inner, Index) for inner in _walk(node))


def _function_declarations(statements: list[Stmt]) -> Iterator[FuncDecl]:
    """Every function declared anywhere in `statements` -- names are
    unique program-wide (the type checker's rule), so one table serves."""
    for stmt in statements:
        if isinstance(stmt, FuncDecl):
            yield stmt
            yield from _function_declarations(stmt.body.statements)
        elif isinstance(stmt, Block):
            yield from _function_declarations(stmt.statements)
        elif isinstance(stmt, IfStmt):
            yield from _function_declarations([stmt.then_branch])
            if stmt.else_branch is not None:
                yield from _function_declarations([stmt.else_branch])
        elif isinstance(stmt, (WhileStmt, ForStmt)):
            yield from _function_declarations([stmt.body])
        elif isinstance(stmt, StrategyDecl):
            yield from _function_declarations([clause.body for clause in stmt.when_clauses])


def _subject(culprit: Expr) -> str:
    """How an error names a culprit, as the subject of its reason."""
    name = _dotted(culprit)
    if name is not None:
        return f"'{name}'"
    if isinstance(culprit, Index):
        obj = _dotted(culprit.obj)
        if obj is not None:
            return f"'{obj}[...]'"
    if isinstance(culprit, Call) and isinstance(culprit.callee, Identifier):
        return f"'{culprit.callee.name}(...)'"
    if isinstance(culprit, ArrayLiteral):
        return "This series literal"
    return "It"


def _dotted(node: Expr) -> str | None:
    """`r`, `AAPL.close`, `r.sharpe` -- a name, or names joined by dots."""
    if isinstance(node, Identifier):
        return node.name
    if isinstance(node, FieldAccess):
        obj = _dotted(node.obj)
        return f"{obj}.{node.name}" if obj is not None else None
    return None


def check_lookahead(program: Program, diagnostics: DiagnosticBag | None = None) -> DiagnosticBag:
    """Convenience entry point: analyzes `program`, returns the bag."""
    analysis = LookaheadAnalysis(diagnostics)
    analysis.analyze(program)
    return analysis.diagnostics


__all__ = [
    "POINT_IN_TIME_FUNCTIONS",
    "TIME_SHIFTING_FUNCTIONS",
    "WHOLE_HISTORY_FUNCTIONS",
    "Checkpoint",
    "LookaheadAnalysis",
    "Source",
    "SourceSpan",
    "check_lookahead",
]
