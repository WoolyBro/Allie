"""Tree-walking interpreter: runs a type-checked AST directly.

The reference backend -- what `allie run` uses by default, and what the
IR interpreter and the LLVM JIT are cross-validated against
(tests/unit/test_ir_cross_validation.py, test_llvm_cross_validation.py,
the differential fuzzer).

## Compiled to closures, then run

The AST is walked once, up front, turning every node into a Python
closure that does that node's work: an `Identifier` becomes a function
that looks one name up, `a + b` on two Ints a function that adds and
range-checks. Running the program is then calling closures. Compared
with re-inspecting each node with an `isinstance` chain every time it
executes, this does the inspection once:

- **Dispatch** is decided at compile time, per node.
- **Arithmetic is specialized by type.** The type checker annotated
  every expression (`inferred_type`), so `+` on two Ints compiles to an
  add and a 64-bit range check, `+` on two Floats to a bare add, and a
  series operator to the element-wise implementation -- no per-execution
  type tests. A node without an annotation (an unchecked AST) falls back
  to the generic, dynamically-typed operator.
- **`break` / `continue` / `return` are return values**, not exceptions:
  a statement's closure returns None to carry on, or a signal that
  every enclosing block passes up until a loop or call consumes it.
- **Scopes cost only where they're needed.** A block that declares
  nothing (`{ total = total + i; }`) runs in its enclosing scope rather
  than allocating a new one on every iteration; one with a `let` gets a
  fresh scope each time it runs, as it must.

Each `allie run` constructs a fresh Environment, so runs are isolated
(docs/spec.md "Concurrency model").

## Runtime errors know where they happened

An AllieRuntimeError escaping a node is given that node's source span
if it has none yet, so the innermost failing expression is the one
reported: `AllieValueError: division by zero. Line 3, Column 11`. The IR
interpreter and the JIT attach the same spans (allie.ir.instructions.
IRInstr.span), so all three backends report a failure identically.

## Tracing

`run_program(..., trace=f)` calls `f(statement)` before each statement
runs -- what `allie run --trace` prints.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

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
    WhileStmt,
    Widen,
)
from allie.errors import SourceSpan
from allie.quant.backtest_engine import (
    BacktestResult,
    WhenSignal,
    config_from_options,
    run_backtest,
)
from allie.runtime import elementwise
from allie.runtime.elementwise import as_signal
from allie.runtime.environment import Environment
from allie.runtime.errors import AllieNameError, AllieRuntimeError, AllieValueError
from allie.runtime.values import (
    BoolSeriesValue,
    BuiltinFunction,
    Closure,
    PriceSeriesValue,
    SeriesValue,
    StrategyValue,
    Value,
    promote_numeric,
)
from allie.stdlib.indexing import index_bool, index_float, index_int
from allie.typechecker.types import BOOL, FLOAT, INT, STRING, SeriesType

ExprCode = Callable[[Environment], Value]
# A statement's closure returns None to continue, or a control signal.
StmtCode = Callable[[Environment], object]
Trace = Callable[[Stmt], None]


class _Signal:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


BREAK = _Signal("break")
CONTINUE = _Signal("continue")


class _Return:
    """`return v`, on its way up to the call that consumes it."""

    __slots__ = ("value",)

    def __init__(self, value: Value) -> None:
        self.value = value


_INT_MIN = -(2**63)
_INT_MAX = 2**63 - 1
_PRICE_SERIES_FIELDS = ("open", "high", "low", "close", "volume", "dates")


def run_program(program: Program, env: Environment, *, trace: Trace | None = None) -> None:
    _Compiler(trace).statements(program.statements, scoped=False)(env)


def evaluate(expression: Expr, env: Environment) -> Value:
    """The value of one type-checked expression in `env` -- what the REPL
    shows for an expression typed on its own."""
    return _Compiler(None).expr(expression)(env)


def _locate(error: AllieRuntimeError, span: SourceSpan) -> None:
    """Gives `error` this span unless an inner node already gave it one."""
    if error.span is None:
        error.span = span


def _overflow(span: SourceSpan) -> AllieValueError:
    return AllieValueError("integer overflow: the result doesn't fit in a 64-bit Int", span)


def _is_series(t: object) -> bool:
    return isinstance(t, SeriesType)


class _Compiler:
    def __init__(self, trace: Trace | None) -> None:
        self.trace = trace

    # ---- statements ----

    def statements(self, statements: list[Stmt], *, scoped: bool) -> StmtCode:
        """A statement list. Functions are hoisted, matching the type
        checker: every function declared directly in the list is bound
        before any of them runs, so a call can precede its declaration
        and two functions can call each other. `scoped` lists get their
        own Environment -- but only if they declare something."""
        hoisted = [s for s in statements if isinstance(s, FuncDecl)]
        rest = [s for s in statements if not isinstance(s, FuncDecl)]
        body = [self.stmt(s) for s in hoisted + rest]
        declares = any(isinstance(s, (LetStmt, FuncDecl, StrategyDecl)) for s in statements)

        if scoped and declares:

            def run_scoped(env: Environment) -> object:
                inner = Environment(env)
                for code in body:
                    signal = code(inner)
                    if signal is not None:
                        return signal
                return None

            return run_scoped

        if len(body) == 1:
            return body[0]

        def run(env: Environment) -> object:
            for code in body:
                signal = code(env)
                if signal is not None:
                    return signal
            return None

        return run

    def stmt(self, node: Stmt) -> StmtCode:
        compile_stmt = self._STATEMENTS.get(type(node))
        if compile_stmt is None:
            raise AllieValueError(f"unknown statement node: {type(node).__name__}")
        code = compile_stmt(self, node)
        if self.trace is None or isinstance(node, Block):
            return code
        trace = self.trace

        def traced(env: Environment) -> object:
            trace(node)
            return code(env)

        return traced

    def _let(self, node: LetStmt) -> StmtCode:
        name, initializer = node.name, self.expr(node.initializer)

        def run(env: Environment) -> None:
            env.vars[name] = initializer(env)

        return run

    def _expr_stmt(self, node: ExprStmt) -> StmtCode:
        expression = self.expr(node.expression)

        def run(env: Environment) -> None:
            expression(env)

        return run

    def _if(self, node: IfStmt) -> StmtCode:
        condition, then = self.expr(node.condition), self.stmt(node.then_branch)
        otherwise = self.stmt(node.else_branch) if node.else_branch is not None else None

        def run(env: Environment) -> object:
            if condition(env):
                return then(env)
            if otherwise is not None:
                return otherwise(env)
            return None

        return run

    def _while(self, node: WhileStmt) -> StmtCode:
        condition, body = self.expr(node.condition), self.stmt(node.body)

        def run(env: Environment) -> object:
            while condition(env):
                signal = body(env)
                if signal is not None:
                    if signal is BREAK:
                        break
                    if signal is not CONTINUE:
                        return signal
            return None

        return run

    def _for(self, node: ForStmt) -> StmtCode:
        initializer = self.stmt(node.initializer)
        condition, increment = self.expr(node.condition), self.expr(node.increment)
        body = self.stmt(node.body)

        def run(env: Environment) -> object:
            loop_env = Environment(env)
            initializer(loop_env)
            while condition(loop_env):
                signal = body(loop_env)
                if signal is not None:
                    if signal is BREAK:
                        break
                    if signal is not CONTINUE:
                        return signal
                increment(loop_env)
            return None

        return run

    def _block(self, node: Block) -> StmtCode:
        return self.statements(node.statements, scoped=True)

    def _break(self, node: BreakStmt) -> StmtCode:
        return lambda env: BREAK

    def _continue(self, node: ContinueStmt) -> StmtCode:
        return lambda env: CONTINUE

    def _return(self, node: ReturnStmt) -> StmtCode:
        if node.value is None:
            return lambda env: _Return(None)
        value = self.expr(node.value)
        return lambda env: _Return(value(env))

    def _func(self, node: FuncDecl) -> StmtCode:
        name, params = node.name, [param.name for param in node.params]
        # The body runs directly in the call's Environment, which already
        # holds the parameters -- one scope per call, not two.
        code = self.statements(node.body.statements, scoped=False)

        def run(env: Environment) -> None:
            env.vars[name] = Closure(params=params, body=node.body, defining_env=env, code=code)

        return run

    def _import(self, node: ImportStmt) -> StmtCode:
        # No module system yet -- inert, as in the resolver and checker.
        return lambda env: None

    def _strategy(self, node: StrategyDecl) -> StmtCode:
        name = node.name
        conditions = [self.expr(clause.condition) for clause in node.when_clauses]
        actions = [self.stmt(clause.body) for clause in node.when_clauses]

        def run(env: Environment) -> None:
            # Each condition is evaluated here, at the declaration -- the
            # same point the IR and LLVM backends evaluate it.
            clauses = [
                (as_signal(condition(env)), action)
                for condition, action in zip(conditions, actions, strict=True)
            ]
            env.vars[name] = StrategyValue(name=name, clauses=clauses, defining_env=env)

        return run

    def _backtest_stmt(self, node: BacktestStmt) -> StmtCode:
        backtest = self._backtest(node)

        def run(env: Environment) -> None:
            print(cast(BacktestResult, backtest(env)).render())

        return run

    _STATEMENTS: dict[type, Callable[[Any, Any], StmtCode]] = {
        LetStmt: _let,
        ExprStmt: _expr_stmt,
        IfStmt: _if,
        WhileStmt: _while,
        ForStmt: _for,
        Block: _block,
        BreakStmt: _break,
        ContinueStmt: _continue,
        ReturnStmt: _return,
        FuncDecl: _func,
        ImportStmt: _import,
        StrategyDecl: _strategy,
        BacktestStmt: _backtest_stmt,
    }

    # ---- expressions ----

    def expr(self, node: Expr) -> ExprCode:
        compile_expr = self._EXPRESSIONS.get(type(node))
        if compile_expr is None:
            raise AllieValueError(f"unknown expression node: {type(node).__name__}")
        return compile_expr(self, node)

    def _literal(self, node: NumberLiteral | StringLiteral | BoolLiteral) -> ExprCode:
        value = node.value
        return lambda env: value

    def _identifier(self, node: Identifier) -> ExprCode:
        name, span = node.name, node.span

        def load(env: Environment) -> Value:
            scope: Environment | None = env
            while scope is not None:
                variables = scope.vars
                if name in variables:
                    return variables[name]
                scope = scope.parent
            raise AllieNameError(f"undefined name: {name!r}", span)

        return load

    def _assignment(self, node: Assignment) -> ExprCode:
        name, span, value_code = node.name, node.span, self.expr(node.value)

        def assign(env: Environment) -> Value:
            value = value_code(env)
            scope: Environment | None = env
            while scope is not None:
                if name in scope.vars:
                    scope.vars[name] = value
                    return value
                scope = scope.parent
            raise AllieNameError(f"cannot assign to undefined name: {name!r}", span)

        return assign

    def _widen(self, node: Widen) -> ExprCode:
        operand = self.expr(node.operand)
        return lambda env: float(operand(env))  # type: ignore[arg-type]

    def _unary(self, node: Unary) -> ExprCode:
        operand, op, span = self.expr(node.operand), node.operator, node.span
        operand_type = node.operand.inferred_type
        if op == "!" and not _is_series(operand_type):
            return lambda env: not operand(env)
        if op == "-" and operand_type == FLOAT:
            return lambda env: -operand(env)  # type: ignore[operator]
        if op == "-" and operand_type == INT:

            def negate_int(env: Environment) -> Value:
                value = -operand(env)  # type: ignore[operator]
                if value > _INT_MAX:  # only -(-2**63) overflows
                    raise _overflow(span)
                return value

            return negate_int

        def generic(env: Environment) -> Value:
            value = operand(env)
            try:
                return _generic_unary(op, value, span)
            except AllieRuntimeError as error:
                _locate(error, span)
                raise

        return generic

    def _logical(self, node: Logical) -> ExprCode:
        left, right, op, span = (
            self.expr(node.left),
            self.expr(node.right),
            node.operator,
            node.span,
        )
        if _is_series(node.inferred_type):

            def bar_by_bar(env: Environment) -> Value:
                a, b = left(env), right(env)  # both sides, always
                try:
                    return elementwise.apply_binary(op, a, b)
                except AllieRuntimeError as error:
                    _locate(error, span)
                    raise

            return bar_by_bar
        if op == "and":

            def logical_and(env: Environment) -> Value:
                value = left(env)
                return right(env) if value else value

            return logical_and

        def logical_or(env: Environment) -> Value:
            value = left(env)
            return value if value else right(env)

        return logical_or

    def _binary(self, node: Binary) -> ExprCode:
        left, right, op, span = (
            self.expr(node.left),
            self.expr(node.right),
            node.operator,
            node.span,
        )
        types = (node.left.inferred_type, node.right.inferred_type)
        result_type = node.inferred_type

        if _is_series(result_type) or any(_is_series(t) for t in types):

            def series(env: Environment) -> Value:
                a, b = left(env), right(env)
                try:
                    return elementwise.apply_binary(op, a, b)
                except AllieRuntimeError as error:
                    _locate(error, span)
                    raise

            return series

        typed = None not in types and result_type is not None
        if typed and op in _COMPARE:
            compare = _COMPARE[op]
            return lambda env: compare(left(env), right(env))
        if typed and op in ("+", "-", "*") and result_type == INT:
            arithmetic = _ARITHMETIC[op]

            def int_arithmetic(env: Environment) -> Value:
                value = arithmetic(left(env), right(env))
                if _INT_MIN <= value <= _INT_MAX:
                    return value
                raise _overflow(span)

            return int_arithmetic
        if typed and op in ("+", "-", "*") and result_type == FLOAT:
            arithmetic = _ARITHMETIC[op]
            return lambda env: arithmetic(left(env), right(env))
        if typed and op == "+" and result_type == STRING:
            return lambda env: f"{left(env)}{right(env)}"
        if typed and op == "/":

            def divide(env: Environment) -> Value:
                a, b = left(env), right(env)
                if b == 0:
                    raise AllieValueError("division by zero", span)
                return float(a) / float(b)  # type: ignore[arg-type]

            return divide
        if typed and op == "%":

            def modulo(env: Environment) -> Value:
                a, b = left(env), right(env)
                if b == 0:
                    raise AllieValueError("modulo by zero", span)
                # Floored: the result takes the divisor's sign (-7 % 2 is 1).
                return a % b  # type: ignore[operator]

            return modulo

        def generic(env: Environment) -> Value:
            a, b = left(env), right(env)
            try:
                return _generic_binary(op, a, b, span)
            except AllieRuntimeError as error:
                _locate(error, span)
                raise

        return generic

    def _call(self, node: Call) -> ExprCode:
        callee = self.expr(node.callee)
        arguments = [self.expr(argument) for argument in node.arguments]
        span = node.span

        def call(env: Environment) -> Value:
            function = callee(env)
            args = [argument(env) for argument in arguments]
            kind = type(function)
            if kind is Closure:
                closure = cast(Closure, function)
                if len(args) != len(closure.params):
                    raise AllieValueError(
                        f"expected {len(closure.params)} argument(s), got {len(args)}", span
                    )
                call_env = Environment(closure.defining_env)
                call_env.vars.update(zip(closure.params, args, strict=True))
                code = closure.code or _Compiler(None).statements(
                    closure.body.statements, scoped=False
                )
                signal = code(call_env)
                return signal.value if isinstance(signal, _Return) else None
            if kind is BuiltinFunction:
                try:
                    return cast(BuiltinFunction, function).fn(*args)
                except AllieRuntimeError as error:
                    _locate(error, span)
                    raise
            raise AllieValueError(f"cannot call non-function value: {function!r}", span)

        return call

    def _field(self, node: FieldAccess) -> ExprCode:
        obj_code, name, span = self.expr(node.obj), node.name, node.span

        def field(env: Environment) -> Value:
            obj = obj_code(env)
            if isinstance(obj, PriceSeriesValue) and name in _PRICE_SERIES_FIELDS:
                return cast(Value, getattr(obj, name))
            try:
                if isinstance(obj, BacktestResult):
                    return cast(Value, obj.field(name))
                raise AllieValueError(f"cannot access field '{name}' on {type(obj).__name__}")
            except AllieRuntimeError as error:
                _locate(error, span)
                raise

        return field

    def _index(self, node: Index) -> ExprCode:
        """Same stdlib functions the IR and LLVM paths call, so bounds
        checking and the element conversion can't differ between backends."""
        series_code, index_code, span = self.expr(node.obj), self.expr(node.index), node.span
        static = node.obj.inferred_type
        by_type = (
            index_int
            if isinstance(static, SeriesType) and static.elem == INT
            else index_bool
            if isinstance(static, SeriesType) and static.elem == BOOL
            else index_float
        )

        def index(env: Environment) -> Value:
            series, i = series_code(env), index_code(env)
            try:
                if isinstance(i, bool) or not isinstance(i, int):
                    raise AllieValueError(f"a series index must be an Int, got {i!r}")
                if isinstance(series, BoolSeriesValue):
                    return index_bool(series, i)
                if not isinstance(series, SeriesValue):
                    raise AllieValueError(f"cannot index a {type(series).__name__}")
                return by_type(series, i)
            except AllieRuntimeError as error:
                _locate(error, span)
                raise

        return index

    def _array(self, node: ArrayLiteral) -> ExprCode:
        elements = [self.expr(element) for element in node.elements]
        return lambda env: elementwise.series_literal([element(env) for element in elements])

    def _backtest(self, node: BacktestStmt | Backtest) -> ExprCode:
        """`backtest S on SYM from N to M` -- hands the strategy's signals
        to the one engine entry point every backend calls
        (allie.quant.backtest_engine.run_backtest)."""
        strategy_name, symbol, span = node.strategy_name, node.symbol, node.span
        start_code, end_code = self.expr(node.start), self.expr(node.end)
        options = [(option.name, self.expr(option.value)) for option in node.options]

        def backtest(env: Environment) -> Value:
            try:
                strategy = env.get(strategy_name)
                if not isinstance(strategy, StrategyValue):
                    raise AllieValueError(f"'{strategy_name}' is not a strategy")
                # `on SYM` names an ordinary variable holding a PriceSeries.
                prices = env.get(symbol)
                if not isinstance(prices, PriceSeriesValue):
                    raise AllieValueError(
                        f"backtest: '{symbol}' must be a PriceSeries (from load(...)), "
                        f"got {type(prices).__name__}"
                    )
                clauses = [
                    WhenSignal(signal=signal, action=_when_action(action, strategy.defining_env))
                    for signal, action in strategy.clauses
                ]
                # The bounds, then the options, in source order -- the order
                # the compiled backends evaluate them in.
                start, end = start_code(env), end_code(env)
                config = config_from_options({name: code(env) for name, code in options})
                return cast(
                    Value,
                    run_backtest(
                        strategy_name=strategy_name,
                        symbol=symbol,
                        prices=prices,
                        clauses=clauses,
                        start=cast(int, start),
                        end=cast(int, end),
                        config=config,
                    ),
                )
            except AllieRuntimeError as error:
                _locate(error, span)
                raise

        return backtest

    _EXPRESSIONS: dict[type, Callable[[Any, Any], ExprCode]] = {
        NumberLiteral: _literal,
        StringLiteral: _literal,
        BoolLiteral: _literal,
        Identifier: _identifier,
        Assignment: _assignment,
        Widen: _widen,
        Unary: _unary,
        Logical: _logical,
        Binary: _binary,
        Call: _call,
        FieldAccess: _field,
        Index: _index,
        ArrayLiteral: _array,
        Backtest: _backtest,
    }


def _when_action(action: StmtCode, env: Environment) -> Callable[[], None]:
    def run() -> None:
        action(Environment(env))

    return run


_COMPARE: dict[str, Callable[[Any, Any], bool]] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}
_ARITHMETIC: dict[str, Callable[[Any, Any], Any]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}


# ---- the dynamically-typed operators, for nodes without a type ----


def _checked_int(value: Value, span: SourceSpan) -> Value:
    if isinstance(value, int) and not isinstance(value, bool):
        if not _INT_MIN <= value <= _INT_MAX:
            raise _overflow(span)
    return value


def _generic_unary(op: str, operand: Value, span: SourceSpan) -> Value:
    if elementwise.is_series(operand):
        return elementwise.apply_unary(op, operand)  # type: ignore[arg-type]
    if op == "-":
        if isinstance(operand, bool) or not isinstance(operand, (int, float)):
            raise AllieValueError(f"unary '-' requires a number, got {operand!r}")
        return _checked_int(-operand, span)
    if op == "!":
        return not operand
    raise AllieValueError(f"unknown unary operator: {op!r}")


def _generic_binary(op: str, left: Value, right: Value, span: SourceSpan) -> Value:
    if elementwise.is_series(left) or elementwise.is_series(right):
        return elementwise.apply_binary(op, left, right)
    if op in _COMPARE:
        return bool(_COMPARE[op](left, right))
    if op == "+":
        if isinstance(left, str) or isinstance(right, str):
            return f"{left}{right}"
        a, b = promote_numeric(left, right)
        return _checked_int(a + b, span)
    if op in ("-", "*"):
        a, b = promote_numeric(left, right)
        return _checked_int((a - b) if op == "-" else (a * b), span)
    if op == "/":
        a, b = promote_numeric(left, right)
        if b == 0:
            raise AllieValueError("division by zero")
        return float(a) / float(b)
    if op == "%":
        a, b = promote_numeric(left, right)
        if b == 0:
            raise AllieValueError("modulo by zero")
        return a % b
    raise AllieValueError(f"unknown binary operator: {op!r}")


__all__ = ["BREAK", "CONTINUE", "evaluate", "run_program"]
