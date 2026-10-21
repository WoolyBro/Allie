"""Static type checker.

Runs after the resolver. Writes the resolved type onto every expression
node (`inferred_type` -- LLVM codegen needs it to choose `add` or
`fadd`) and reports every type error in the program.

## The type system

Allie is statically typed with local type inference. A `let` without an
annotation takes its initializer's type; function parameters and return
types must be written, including `-> Void`. There is no dynamic type,
and the only implicit conversions are widenings: Int -> Float, and
Series<Int> -> Series<Float> (sound because series are immutable). There
is no truthiness: conditions must be exactly Bool.

## Only what every backend runs identically type-checks

A program that type-checks must behave the same on the tree-walking
interpreter, the IR interpreter and the LLVM JIT. Several rules exist
for exactly that reason -- each closes a hole where one backend accepted
a program another crashed on or ran differently:

- `/` always produces a Float (`7 / 2` is `3.5`); there is no silently
  truncating integer division.
- Functions are not values (`let g = f;` is rejected), and neither are
  strategies. A nested function can't use its enclosing function's local
  variables -- there are no closures -- and function names are unique
  across the program, because compiled code calls functions by name.
- A `Void` result can't be used as a value.
- `==`/`!=` compare scalars -- numbers (Int and Float mix freely), Bools,
  Strings -- never whole series.
- A series literal holds numbers or Bools; field access knows only a
  PriceSeries's six fields.

Operators work on whole series, bar by bar: `+ - * /` between two series
or a series and a number give a Series<Float>; `< > <= >=`,
`crosses_above` and `crosses_below` give a Series<Bool>; `and`/`or`/`!`
combine Series<Bool>. A `when` condition is any Series<Bool>.

Functions are hoisted within their scope, so a function can be called
above its declaration and two functions can call each other.

Implicit widening is made explicit: wherever an Int lands in a Float
slot -- an annotated `let`, an assignment, an argument, a `return` -- the
checker wraps it in a `Widen` node. Before that, each backend guessed:
the interpreters kept the Python int (so `let x: Float = 1; print(x);`
printed `1`), and LLVM crashed passing an i64 to a double parameter.

## Errors

Every type error is collected in `self.errors`, in source order, instead
of stopping at the first. An expression that fails to type-check gets
the poison type ERROR, which every later check accepts silently -- so one
mistake produces one error, not a cascade. An undefined name is reported
once, with a "did you mean" suggestion when a visible name is close.
`check()` raises the first error once the whole program has been
checked, so callers that want fail-fast behavior keep it.

TypeEnvironment is a *separate* scope stack from the resolver's Scope
(allie.semantic.resolver): same push/pop discipline, but this one maps
names to types; the resolver's only tracks that a name exists.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

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
    TypeAnnotation,
    Unary,
    WhenClause,
    WhileStmt,
    Widen,
)
from allie.errors import SourceSpan
from allie.quant.backtest_engine import OPTIONS, ORDER_FUNCTIONS, option_error
from allie.typechecker.errors import QSTypeError
from allie.typechecker.types import (
    BACKTEST_RESULT,
    BOOL,
    DATE,
    ELEMENTWISE,
    ELEMENTWISE_INT,
    ERROR,
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
_CROSSINGS = frozenset({"crosses_above", "crosses_below"})
_NUMERIC = (INT, FLOAT)
_SERIES_ELEMENTS = (INT, FLOAT, BOOL)

_PRICE_SERIES_FIELDS: dict[str, Type] = {
    "open": SeriesType(FLOAT),
    "high": SeriesType(FLOAT),
    "low": SeriesType(FLOAT),
    "close": SeriesType(FLOAT),
    "volume": SeriesType(FLOAT),
    "dates": SeriesType(DATE),
}

# A backtest's results, as fields of the value a `backtest` expression
# evaluates to (allie.quant.backtest_engine.BacktestResult.field).
_BACKTEST_RESULT_FIELDS: dict[str, Type] = {
    "sharpe": FLOAT,
    "sortino": FLOAT,
    "cagr": FLOAT,
    "total_return": FLOAT,
    "buy_hold_return": FLOAT,
    "volatility": FLOAT,
    "max_drawdown": FLOAT,
    "initial_equity": FLOAT,
    "final_equity": FLOAT,
    "exposure": FLOAT,
    "costs": FLOAT,
    "trades": INT,
    "win_rate": FLOAT,
    "profit_factor": FLOAT,
    "avg_trade_return": FLOAT,
    # One value per bar of the backtest window:
    "equity": SeriesType(FLOAT),
    "returns": SeriesType(FLOAT),
    "drawdown": SeriesType(FLOAT),
    "position": SeriesType(FLOAT),
    # One value per completed trade:
    "trade_returns": SeriesType(FLOAT),
}
_OPTION_TYPES: dict[str, Type] = {"Float": FLOAT, "Int": INT, "String": STRING}
_FIELDS: dict[Type, dict[str, Type]] = {
    PRICE_SERIES: _PRICE_SERIES_FIELDS,
    BACKTEST_RESULT: _BACKTEST_RESULT_FIELDS,
}

_TYPE_NAMES = (
    "BacktestResult",
    "Int",
    "Float",
    "Bool",
    "String",
    "Void",
    "Date",
    "DateRange",
    "PriceSeries",
    "Series",
    "Matrix",
    "Vector",
)
# What people coming from other languages reach for first.
_FOREIGN_TYPE_NAMES = {
    "str": "String",
    "double": "Float",
    "list": "Series[...]",
    "array": "Series[...]",
    "None": "Void",
}

_NO_TRUTHINESS = "Allie has no truthiness: compare explicitly, e.g. 'x != 0'"
_SERIES_OPERATORS = (
    "series support + - * / with each other or a number, < > <= >= and "
    "crosses_above/crosses_below (giving a Series<Bool>), and and/or/!"
)


class TypeEnvironment:
    """Scope stack of name -> Type. `function` names the function whose
    body this scope belongs to (None outside any function); that is how a
    nested function's use of its enclosing function's locals is caught."""

    def __init__(self, parent: TypeEnvironment | None = None, function: str | None = None) -> None:
        self.parent = parent
        self.function = function
        self.types: dict[str, Type] = {}

    def define(self, name: str, type_: Type) -> None:
        self.types[name] = type_

    def lookup(self, name: str) -> tuple[Type, TypeEnvironment] | None:
        env: TypeEnvironment | None = self
        while env is not None:
            if name in env.types:
                return env.types[name], env
            env = env.parent
        return None

    def get(self, name: str) -> Type | None:
        found = self.lookup(name)
        return None if found is None else found[0]

    def visible_names(self) -> set[str]:
        names: set[str] = set()
        env: TypeEnvironment | None = self
        while env is not None:
            names.update(name for name in env.types if not name.startswith("__"))
            env = env.parent
        return names


class TypeChecker:
    def __init__(self) -> None:
        self.env = TypeEnvironment()
        self.errors: list[QSTypeError] = []
        self._root = self.env
        self._function_return_stack: list[Type] = []
        self._signatures: dict[int, FunctionType] = {}
        self._function_spans: dict[str, SourceSpan] = {}
        self._builtin_names: frozenset[str] = frozenset()
        self._reported_undefined: set[str] = set()
        self._in_when_action = 0  # nesting depth; order builtins are allowed inside

    def check(self, program: Program) -> None:
        """Checks the whole program, collecting every error into
        `self.errors`; then raises the first one, if any."""
        self._builtin_names = frozenset(
            name for name, type_ in self._root.types.items() if isinstance(type_, FunctionType)
        )
        self._check_statements(program.statements)
        self.errors.sort(key=lambda error: (error.line, error.column))
        if self.errors:
            raise self.errors[0]

    # ---- helpers ----

    def _error(self, span: SourceSpan, message: str, *, hint: str | None = None) -> Type:
        """Records an error; returns ERROR for the caller to use as the
        failing expression's type."""
        self.errors.append(QSTypeError(message, span.line, span.column, span.length, hint))
        return ERROR

    def _push_scope(self, function: str | None = None) -> None:
        self.env = TypeEnvironment(
            parent=self.env, function=function if function is not None else self.env.function
        )

    def _pop_scope(self) -> None:
        assert self.env.parent is not None
        self.env = self.env.parent

    def _resolve_type_annotation(self, annotation: TypeAnnotation) -> Type:
        if annotation.type_arg is not None:
            elem = self._resolve_type_annotation(annotation.type_arg)
            generic = resolve_generic_type(annotation.name, elem)
            if generic is None:
                return self._error(
                    annotation.span,
                    f"'{annotation.name}' is not a generic type (expected Series/Matrix/Vector)",
                )
            return generic
        base = resolve_base_type(annotation.name)
        if base is not None:
            return base
        if resolve_generic_type(annotation.name, INT) is not None:
            return self._error(
                annotation.span,
                f"'{annotation.name}' needs an element type",
                hint=f"e.g. {annotation.name}[Float]",
            )
        suggestion = _FOREIGN_TYPE_NAMES.get(annotation.name) or _closest(
            annotation.name, _TYPE_NAMES
        )
        return self._error(
            annotation.span,
            f"unknown type '{annotation.name}'",
            hint=f"did you mean '{suggestion}'?" if suggestion else None,
        )

    # ---- statements ----

    def _check_statements(self, statements: list[Stmt]) -> None:
        """Hoists every function declared directly in `statements` before
        checking any of them, so calls may precede declarations."""
        for stmt in statements:
            if isinstance(stmt, FuncDecl):
                self._declare_function(stmt)
        for stmt in statements:
            self._check_stmt(stmt)

    def _check_stmt(self, stmt: Stmt) -> None:
        if isinstance(stmt, LetStmt):
            self._check_let(stmt)
        elif isinstance(stmt, FuncDecl):
            self._check_function_body(stmt)
        elif isinstance(stmt, IfStmt):
            self._check_condition(stmt.condition)
            self._check_stmt(stmt.then_branch)
            if stmt.else_branch is not None:
                self._check_stmt(stmt.else_branch)
        elif isinstance(stmt, WhileStmt):
            self._check_condition(stmt.condition)
            self._check_stmt(stmt.body)
        elif isinstance(stmt, ForStmt):
            self._push_scope()
            self._check_stmt(stmt.initializer)
            self._check_condition(stmt.condition)
            self._check_expr(stmt.increment)
            self._check_stmt(stmt.body)
            self._pop_scope()
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
            self._check_statements(stmt.statements)
            self._pop_scope()
        else:
            raise TypeError(f"type checker: unhandled statement {type(stmt).__name__}")

    def _check_condition(self, condition: Expr) -> None:
        # No truthiness: if/while/for all go through this one helper so
        # the rule can't drift between the three.
        cond_type = self._value(condition)
        if cond_type in (BOOL, ERROR):
            return
        self._error(
            condition.span,
            f"condition must be Bool, got {cond_type}",
            hint=_NO_TRUTHINESS if cond_type in _NUMERIC else None,
        )

    def _check_let(self, stmt: LetStmt) -> None:
        init_type = self._value(stmt.initializer)
        if stmt.type_annotation is None:
            self.env.define(stmt.name, init_type)
            return
        declared = self._resolve_type_annotation(stmt.type_annotation)
        if not is_assignable(init_type, declared):
            self._error(
                stmt.initializer.span,
                f"cannot assign {init_type} to {declared}",
                hint=_conversion_hint(init_type, declared),
            )
        stmt.initializer = _coerce(stmt.initializer, init_type, declared)
        self.env.define(stmt.name, declared)

    def _declare_function(self, stmt: FuncDecl) -> None:
        """Resolves a function's signature and binds its name, before any
        body in the same scope is checked."""
        previous = self._function_spans.get(stmt.name)
        if previous is not None:
            self._error(
                stmt.span,
                f"a function named '{stmt.name}' is already declared at {previous}",
                hint="function names must be unique across the program: "
                "compiled code calls functions by name",
            )
        elif self.env is not self._root and stmt.name in self._builtin_names:
            self._error(
                stmt.span,
                f"'{stmt.name}' is a built-in function; a nested function can't reuse its name",
                hint="pick a different name",
            )
        else:
            self._function_spans[stmt.name] = stmt.span

        param_types: list[Type] = []
        for param in stmt.params:
            if param.type_annotation is None:
                param_types.append(
                    self._error(
                        param.span,
                        f"parameter '{param.name}' of '{stmt.name}' has no type annotation "
                        "(v1 requires explicit parameter types)",
                    )
                )
            else:
                param_types.append(self._resolve_type_annotation(param.type_annotation))
        if stmt.return_type is None:
            return_type = self._error(
                stmt.span,
                f"function '{stmt.name}' has no declared return type "
                "(v1 requires an explicit '-> Type', even '-> Void')",
            )
        else:
            return_type = self._resolve_type_annotation(stmt.return_type)

        signature = FunctionType(params=tuple(param_types), ret=return_type)
        self._signatures[id(stmt)] = signature
        self.env.define(stmt.name, signature)

    def _check_function_body(self, stmt: FuncDecl) -> None:
        if id(stmt) not in self._signatures:
            self._declare_function(stmt)  # declared somewhere hoisting doesn't reach
        signature = self._signatures[id(stmt)]

        self._push_scope(function=stmt.name)
        for param, param_type in zip(stmt.params, signature.params, strict=True):
            self.env.define(param.name, param_type)
        self._function_return_stack.append(signature.ret)
        self._check_statements(stmt.body.statements)
        self._function_return_stack.pop()
        self._pop_scope()

        if signature.ret not in (VOID, ERROR) and not _all_paths_return(stmt.body):
            self._error(
                stmt.span,
                f"function '{stmt.name}' does not return on every path",
                hint=f"every path through '{stmt.name}' must end in 'return <{signature.ret}>'",
            )

    def _check_return(self, stmt: ReturnStmt) -> None:
        if not self._function_return_stack:
            self._error(stmt.span, "'return' outside a function")
            return
        expected = self._function_return_stack[-1]
        actual = VOID if stmt.value is None else self._check_expr(stmt.value)
        if not is_assignable(actual, expected):
            self._error(
                stmt.span,
                f"return type mismatch: expected {expected}, got {actual}",
                hint=_conversion_hint(actual, expected),
            )
        if stmt.value is not None:
            stmt.value = _coerce(stmt.value, actual, expected)

    def _check_strategy(self, stmt: StrategyDecl) -> None:
        # Bound before the clauses are checked so a `backtest` can resolve
        # it through the same TypeEnvironment as any other name.
        self.env.define(stmt.name, STRATEGY)
        for clause in stmt.when_clauses:
            self._check_when_clause(clause)

    def _check_when_clause(self, clause: WhenClause) -> None:
        condition = self._value(clause.condition)
        if condition not in (SeriesType(BOOL), ERROR):
            if condition == BOOL:
                hint = "this is one true/false value; compare whole series, e.g. 'rsi < 30.0'"
            elif isinstance(condition, SeriesType):
                hint = "compare it with something, e.g. 'close > SMA(close, 20)'"
            else:
                hint = None
            self._error(
                clause.condition.span,
                f"a 'when' condition must be a Series<Bool> (one value per bar), got {condition}",
                hint=hint,
            )
        self._in_when_action += 1
        self._check_stmt(clause.body)
        self._in_when_action -= 1

    def _check_backtest(self, stmt: BacktestStmt | Backtest) -> Type:
        strategy_type = self.env.get(stmt.strategy_name)
        if strategy_type is None:
            self._error(stmt.span, f"undefined strategy '{stmt.strategy_name}'")
        elif strategy_type not in (STRATEGY, ERROR):
            self._error(
                stmt.span, f"'{stmt.strategy_name}' is not a strategy (it is {strategy_type})"
            )

        # `on SYM` names a variable holding a PriceSeries.
        symbol_type = self.env.get(stmt.symbol)
        if symbol_type is None:
            self._error(stmt.span, f"undefined name '{stmt.symbol}'")
        elif symbol_type not in (PRICE_SERIES, ERROR):
            self._error(
                stmt.span,
                f"backtest symbol '{stmt.symbol}' must be a PriceSeries, got {symbol_type}",
                hint='load one with load("prices.csv")',
            )

        self._check_backtest_options(stmt)
        for bound in (stmt.start, stmt.end):
            bound_type = self._value(bound)
            if bound_type not in (INT, ERROR):
                self._error(
                    bound.span,
                    f"a bar index must be an Int, got {bound_type}",
                    hint="bars are counted 0, 1, 2, ...; len(AAPL.close) is one past the last",
                )
        return BACKTEST_RESULT

    def _check_backtest_options(self, stmt: BacktestStmt | Backtest) -> None:
        """The `with` clause: known names, each once, values of the
        option's type -- and a literal value in range, since that much
        is knowable now. (The engine checks every value when it runs.)"""
        seen: set[str] = set()
        for option in stmt.options:
            value_type = self._value(option.value)
            if option.name not in OPTIONS:
                suggestion = _closest(option.name, OPTIONS)
                self._error(
                    option.span,
                    f"unknown backtest option '{option.name}'",
                    hint=f"did you mean '{suggestion}'?"
                    if suggestion
                    else "the options are " + ", ".join(OPTIONS),
                )
                continue
            if option.name in seen:
                self._error(option.span, f"'{option.name}' is set twice", hint="set it once")
                continue
            seen.add(option.name)
            expected = _OPTION_TYPES[OPTIONS[option.name][0]]
            if not is_assignable(value_type, expected):
                if value_type != ERROR:
                    self._error(
                        option.value.span,
                        f"backtest option '{option.name}' must be {expected}, got {value_type}",
                        hint='fill is "open" or "close", in quotes'
                        if option.name == "fill"
                        else _conversion_hint(value_type, expected),
                    )
                continue
            option.value = _coerce(option.value, value_type, expected)
            literal = _literal_value(option.value)
            if literal is not None:
                problem = option_error(option.name, literal)
                if problem is not None:
                    self._error(option.value.span, problem)

    # ---- expressions ----

    def _value(self, expr: Expr) -> Type:
        """Checks an expression whose result is used as a value -- an
        initializer, an argument, an operand -- where Void is an error."""
        result = self._check_expr(expr)
        if result != VOID:
            return result
        if isinstance(expr, Call) and isinstance(expr.callee, Identifier):
            return self._error(
                expr.callee.span, f"'{expr.callee.name}' returns Void, so this call has no value"
            )
        return self._error(expr.span, "this expression has no value (its type is Void)")

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
        elif isinstance(expr, Index):
            result = self._check_index(expr)
        elif isinstance(expr, ArrayLiteral):
            result = self._check_array_literal(expr)
        elif isinstance(expr, Backtest):
            result = self._check_backtest(expr)
        elif isinstance(expr, Widen):
            self._check_expr(expr.operand)  # only met when an AST is checked twice
            result = FLOAT
        else:
            raise TypeError(f"type checker: unhandled expression {type(expr).__name__}")

        expr.inferred_type = result
        return result

    def _check_identifier(self, expr: Identifier, *, callee: bool = False) -> Type:
        found = self.env.lookup(expr.name)
        if found is None:
            return self._undefined(expr.name, expr.span)
        type_, scope = found
        if isinstance(type_, FunctionType):
            if callee:
                return type_
            return self._error(
                expr.span,
                f"'{expr.name}' is a function; functions can't be used as values",
                hint=f"call it: {expr.name}(...)",
            )
        if type_ == STRATEGY:
            return self._error(
                expr.span,
                f"'{expr.name}' is a strategy; it can only be named in a 'backtest' statement",
            )
        captured = self._captured_from(scope)
        if captured is not None:
            return self._error(
                expr.span,
                f"'{expr.name}' is a local variable of the enclosing function '{captured}'",
                hint="Allie has no closures, so a nested function can't use its "
                "enclosing function's variables; pass it in as a parameter",
            )
        return type_

    def _captured_from(self, scope: TypeEnvironment) -> str | None:
        """The enclosing function owning `scope`, if a name found there
        would be captured by the function currently being checked."""
        if scope.function is not None and scope.function != self.env.function:
            return scope.function
        return None

    def _undefined(self, name: str, span: SourceSpan) -> Type:
        if name in self._reported_undefined:
            return ERROR
        self._reported_undefined.add(name)
        suggestion = _closest(name, self.env.visible_names())
        return self._error(
            span,
            f"undefined name '{name}'",
            hint=f"did you mean '{suggestion}'?" if suggestion else None,
        )

    def _check_field_access(self, expr: FieldAccess) -> Type:
        obj_type = self._value(expr.obj)
        if obj_type == ERROR:
            return ERROR
        fields = _FIELDS.get(obj_type)
        if fields is not None:
            field_type = fields.get(expr.name)
            if field_type is not None:
                return field_type
            suggestion = _closest(expr.name, fields)
            return self._error(
                expr.span,
                f"{obj_type} has no field '{expr.name}'",
                hint=f"did you mean '{suggestion}'?"
                if suggestion
                else "its fields are " + ", ".join(fields),
            )
        return self._error(
            expr.span,
            f"{obj_type} has no field '{expr.name}'",
            hint="only a PriceSeries or a BacktestResult has fields, e.g. AAPL.close",
        )

    def _check_index(self, expr: Index) -> Type:
        """`s[i]`: `s` must be a Series of Int, Float or Bool and `i` an
        Int. The result is the element type."""
        obj_type = self._value(expr.obj)
        index_type = self._value(expr.index)
        if obj_type == ERROR:
            return ERROR
        if not isinstance(obj_type, SeriesType):
            return self._error(
                expr.span,
                f"cannot index a value of type {obj_type} (only a Series can be indexed)",
            )
        if obj_type.elem not in (*_SERIES_ELEMENTS, ERROR):
            return self._error(
                expr.span,
                f"cannot index {obj_type}: only series of Int, Float or Bool can be indexed",
            )
        if index_type not in (INT, ERROR):
            self._error(expr.index.span, f"a series index must be Int, got {index_type}")
        return obj_type.elem

    def _check_assignment(self, expr: Assignment) -> Type:
        value_type = self._value(expr.value)
        found = self.env.lookup(expr.name)
        if found is None:
            if expr.name not in self._reported_undefined:
                self._reported_undefined.add(expr.name)
                self._error(expr.span, f"cannot assign to undefined name '{expr.name}'")
            return ERROR
        declared, scope = found
        if isinstance(declared, FunctionType) or declared == STRATEGY:
            return self._error(expr.span, f"cannot assign to '{expr.name}': it is not a variable")
        captured = self._captured_from(scope)
        if captured is not None:
            return self._error(
                expr.span,
                f"'{expr.name}' is a local variable of the enclosing function '{captured}'",
                hint="Allie has no closures; return the new value instead",
            )
        if not is_assignable(value_type, declared):
            self._error(
                expr.span,
                f"cannot assign {value_type} to '{expr.name}' ({declared})",
                hint=_conversion_hint(value_type, declared),
            )
        expr.value = _coerce(expr.value, value_type, declared)
        return declared

    def _check_unary(self, expr: Unary) -> Type:
        operand_type = self._value(expr.operand)
        if operand_type == ERROR:
            return ERROR
        if expr.operator == "-":
            if operand_type in _NUMERIC:
                return operand_type
            if _numeric_series(operand_type):
                return SeriesType(FLOAT)
            return self._error(
                expr.span, f"unary '-' requires a number or a numeric series, got {operand_type}"
            )
        if expr.operator == "!":
            if operand_type in (BOOL, SeriesType(BOOL)):
                return operand_type
            return self._error(
                expr.span,
                f"unary '!' requires a Bool operand, got {operand_type}",
                hint=_NO_TRUTHINESS if operand_type in _NUMERIC else None,
            )
        raise TypeError(f"type checker: unknown unary operator {expr.operator!r}")

    def _check_binary(self, expr: Binary) -> Type:
        left = self._value(expr.left)
        right = self._value(expr.right)
        if ERROR in (left, right):
            return ERROR
        op = expr.operator

        if isinstance(left, SeriesType) or isinstance(right, SeriesType) or op in _CROSSINGS:
            return self._check_series_binary(expr, op, left, right)

        if op in _ARITHMETIC_OPS:
            if op == "+" and left == STRING and right == STRING:
                return STRING
            if left in _NUMERIC and right in _NUMERIC:
                # `/` is true division on every backend: 7 / 2 is 3.5.
                if op == "/" or FLOAT in (left, right):
                    return FLOAT
                return INT
            return self._error(
                expr.span,
                f"'{op}' requires numeric operands, got {left} and {right}",
                hint=_operand_hint(left, right),
            )

        if op in _COMPARISON_OPS:
            if left in _NUMERIC and right in _NUMERIC:
                return BOOL
            return self._error(
                expr.span,
                f"'{op}' requires numeric operands, got {left} and {right}",
                hint=_operand_hint(left, right),
            )

        if op in _EQUALITY_OPS:
            if left in _NUMERIC and right in _NUMERIC:
                return BOOL
            if left == right and left in (BOOL, STRING):
                return BOOL
            if isinstance(left, SeriesType) or isinstance(right, SeriesType):
                return self._error(
                    expr.span,
                    f"'{op}' can't compare whole series",
                    hint="compare elements instead, e.g. s[i] == t[i]",
                )
            return self._error(
                expr.span,
                f"'{op}' requires operands of the same type, got {left} and {right}",
            )

        raise TypeError(f"type checker: unknown binary operator {op!r}")

    def _check_series_binary(self, expr: Binary, op: str, left: Type, right: Type) -> Type:
        """An operator with a series operand, applied bar by bar; the other
        side may be a series or a single number (broadcast to every bar)."""

        def numeric(t: Type) -> bool:
            return t in _NUMERIC or _numeric_series(t)

        if op in ("+", "-", "*", "/") and numeric(left) and numeric(right):
            return SeriesType(FLOAT)
        if op in _COMPARISON_OPS and numeric(left) and numeric(right):
            return SeriesType(BOOL)
        if op in _CROSSINGS:
            if not (isinstance(left, SeriesType) or isinstance(right, SeriesType)):
                return self._error(
                    expr.span,
                    f"'{op}' compares series over time, got {left} and {right}",
                    hint="at least one side must be a series, e.g. 'rsi crosses_below 30.0'",
                )
            if numeric(left) and numeric(right):
                return SeriesType(BOOL)
        if op in _EQUALITY_OPS:
            return self._error(
                expr.span,
                f"'{op}' can't compare whole series",
                hint="use '<'/'>' for a per-bar condition, or compare elements, e.g. s[i] == t[i]",
            )
        return self._error(
            expr.span,
            f"'{op}' isn't defined for {left} and {right}",
            hint=_SERIES_OPERATORS,
        )

    def _check_logical(self, expr: Logical) -> Type:
        left = self._value(expr.left)
        right = self._value(expr.right)
        if ERROR in (left, right):
            return ERROR
        series_bool = SeriesType(BOOL)
        if series_bool in (left, right) and {left, right} <= {BOOL, series_bool}:
            return series_bool  # bar by bar; both sides always evaluated
        if left != BOOL or right != BOOL:
            return self._error(
                expr.span,
                f"'{expr.operator}' requires Bool operands, got {left} and {right}",
                hint=_NO_TRUTHINESS if left in _NUMERIC or right in _NUMERIC else None,
            )
        return BOOL

    def _check_call(self, expr: Call) -> Type:
        callee = expr.callee
        if isinstance(callee, Identifier):
            callee_type = self._check_identifier(callee, callee=True)
            callee.inferred_type = callee_type
            name = callee.name
        else:
            callee_type = self._check_expr(callee)
            name = "this function"
        argument_types = [self._value(argument) for argument in expr.arguments]
        if (
            name in ORDER_FUNCTIONS
            and name in self._builtin_names
            and name not in self._function_spans  # not a user function of the same name
            and not self._in_when_action
            and not self._function_return_stack  # a function may be called from an action
        ):
            self._error(
                callee.span,
                f"'{name}()' places an order, which only a strategy's 'when' action can do",
                hint=f"e.g. strategy S {{ when fast crosses_above slow: {name}("
                + ("1.0" if name == "target" else "")
                + "); }",
            )

        if callee_type == ERROR:
            return ERROR
        # Call-level errors point at the function's name, not the `(`.
        where = callee.span if isinstance(callee, Identifier) else expr.span
        if not isinstance(callee_type, FunctionType):
            return self._error(where, f"cannot call a value of type {callee_type}")

        signature = f"{name}({', '.join(map(str, callee_type.params))}) -> {callee_type.ret}"
        expected = len(callee_type.params)
        if len(expr.arguments) != expected:
            self._error(
                where,
                f"'{name}' takes {expected} argument{'s' if expected != 1 else ''}, "
                f"got {len(expr.arguments)}",
                hint=f"signature: {signature}",
            )
            elementwise = callee_type.ret in (ELEMENTWISE, ELEMENTWISE_INT)
            return ERROR if elementwise else callee_type.ret
        for position, (argument, argument_type, parameter_type) in enumerate(
            zip(expr.arguments, argument_types, callee_type.params, strict=True), start=1
        ):
            if not is_assignable(argument_type, parameter_type):
                self._error(
                    argument.span,
                    f"argument {position} of '{name}' must be {parameter_type}, "
                    f"got {argument_type}",
                    hint=_conversion_hint(argument_type, parameter_type)
                    or f"signature: {signature}",
                )
            target = FLOAT if parameter_type == ELEMENTWISE else parameter_type
            expr.arguments[position - 1] = _coerce(argument, argument_type, target)
        if callee_type.ret in (ELEMENTWISE, ELEMENTWISE_INT):
            return _elementwise_result(callee_type, argument_types)
        return callee_type.ret

    def _check_array_literal(self, expr: ArrayLiteral) -> Type:
        # Array literals are Series: this language operates on series by
        # default, so `[1, 2, 3]` can go straight into SMA/mean/etc.
        if not expr.elements:
            return self._error(
                expr.span,
                "cannot infer the type of an empty array literal",
                hint="a series literal needs at least one element",
            )
        element_types = [self._value(element) for element in expr.elements]
        if ERROR in element_types:
            return ERROR
        if all(t in _NUMERIC for t in element_types):
            return SeriesType(FLOAT if FLOAT in element_types else INT)
        first = element_types[0]
        for element, element_type in zip(expr.elements, element_types, strict=True):
            if element_type not in _SERIES_ELEMENTS:
                return self._error(
                    element.span,
                    f"a series can only hold numbers or Bools, got {element_type}",
                    hint="there is no Matrix literal yet"
                    if isinstance(element_type, SeriesType)
                    else None,
                )
            if element_type != first:
                return self._error(
                    element.span,
                    f"array elements must all be the same type: {first} vs {element_type}",
                )
        return SeriesType(first)


def _numeric_series(t: Type) -> bool:
    return isinstance(t, SeriesType) and t.elem in _NUMERIC


def _coerce(expr: Expr, source: Type, target: Type) -> Expr:
    """Makes an implicit Int -> Float widening explicit in the AST, so
    every backend converts at exactly this point (see Widen)."""
    if source == INT and target == FLOAT:
        return Widen(span=expr.span, operand=expr, inferred_type=FLOAT)
    return expr


def _conversion_hint(source: Type, target: Type) -> str | None:
    """Why a value of `source` isn't accepted where `target` is expected,
    for the mismatches people actually hit."""
    if source == FLOAT and target == INT:
        return "Allie never narrows a Float to an Int implicitly"
    if isinstance(target, SeriesType) and not isinstance(source, SeriesType):
        return "a single value isn't a series; pass a whole series, e.g. AAPL.close"
    if isinstance(source, SeriesType) and not isinstance(target, SeriesType):
        return "this is a whole series; reduce it (e.g. mean(s)) or pick one element with s[i]"
    return None


def _operand_hint(left: Type, right: Type) -> str | None:
    if isinstance(left, SeriesType) or isinstance(right, SeriesType):
        return _SERIES_OPERATORS
    if STRING in (left, right):
        return "Allie never converts between numbers and Strings implicitly"
    if BOOL in (left, right):
        return "Bools aren't numbers in Allie"
    return None


def _elementwise_result(signature: FunctionType, argument_types: list[Type]) -> Type:
    """`log(x)`'s type follows `x`: a series gives a Series<Float>, a
    number a Float -- or an Int, for a function that keeps one (`abs`)."""
    argument = next(
        argument_type
        for argument_type, parameter in zip(argument_types, signature.params, strict=True)
        if parameter in (ELEMENTWISE, ELEMENTWISE_INT)
    )
    if argument == ERROR:
        return ERROR
    if isinstance(argument, SeriesType):
        return SeriesType(FLOAT)
    return INT if argument == INT and signature.ret == ELEMENTWISE_INT else FLOAT


def _literal_value(expr: Expr) -> object | None:
    """The value of a literal -- `5`, `-0.5`, `"close"`, or one widened to
    Float -- else None."""
    if isinstance(expr, (NumberLiteral, StringLiteral)):
        return expr.value
    if isinstance(expr, Widen):
        inner = _literal_value(expr.operand)
        return float(inner) if isinstance(inner, int) else None
    if isinstance(expr, Unary) and expr.operator == "-":
        inner = _literal_value(expr.operand)
        return -inner if isinstance(inner, (int, float)) else None
    return None


def _closest(name: str, candidates: Iterable[str]) -> str | None:
    """The closest candidate to a misspelled `name`, if any is close: an
    exact case-insensitive match first (`int` -> `Int`), then difflib."""
    names = sorted(candidates)
    for candidate in names:
        if candidate.lower() == name.lower():
            return candidate
    matches = difflib.get_close_matches(name, names, n=1, cutoff=0.75)
    return matches[0] if matches else None


def _all_paths_return(block: Block) -> bool:
    """Flat "does every branch end in return" check -- not full CFG
    reachability: good enough for the common case, and documented as a
    known limitation rather than pretending to be exhaustive."""
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
