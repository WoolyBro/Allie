"""Recursive-descent parser: tokens -> the dataclass AST in allie.ast.nodes.

One method per production in docs/grammar.md, in the same order, so the
two can be read side by side. Expressions get one method per precedence
level, lowest first: assignment, or, and, equality, comparison, term,
factor, unary, postfix (call / field / index), primary.

## Errors and recovery

Problems are recorded in a DiagnosticBag and parsing continues, so one
run reports every syntax error. What keeps those reports useful rather
than noisy:

- **Messages say what was found, and point where the fix goes.** A
  missing `;` at the end of a line is reported just after the last token
  on that line, not at the first token of the next one. An unclosed `{`
  is reported at the brace itself, with a hint naming the inner block
  whose closing `}` is indented suspiciously (usually the real culprit).
- **Recovery is local.** After an error the parser skips to the next
  statement boundary (`_synchronize`) but never past a `}` that closes
  an enclosing block, and a bad `when` clause skips only to the next
  `when`. One mistake doesn't cascade through the rest of the file.
- **Predictable mistakes get their own message**: chained comparisons
  (`0 < x < 10`), `=` where `==` was meant in a condition, a keyword used
  as a name (including slang spellings such as `ts`), `else` without an
  `if`, `when` outside a strategy, assigning into a series element, a
  fractional or negative backtest bar index, an empty bar range.

`_ParseError` is an internal control-flow signal, never surfaced to the
caller: it unwinds to the nearest recovery point after a diagnostic has
already been recorded.

## Limits

Nesting -- parentheses, brackets, calls, blocks, unary operators,
chained assignments, type arguments -- is capped at MAX_NESTING. Each
level of expression nesting costs about a dozen Python stack frames
here, so without the cap a pathological input crashes the parser with a
RecursionError; past the cap the rest of the file is abandoned after one
error. Separately, the finished AST's depth is capped at MAX_AST_DEPTH:
a long `a + b + c + ...` chain is parsed iteratively but becomes a deep
left-leaning tree, and every later stage (type checker, interpreters,
lowering) walks trees recursively. Both caps are far above anything
written by hand.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import NamedTuple, TypeVar

from allie.ast.nodes import (
    ArrayLiteral,
    Assignment,
    Backtest,
    BacktestOption,
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
    Param,
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
from allie.errors import DiagnosticBag, SourceSpan
from allie.lexer.keywords import KEYWORD_ALIASES
from allie.lexer.tokens import Token, TokenType

MAX_NESTING = 64
MAX_AST_DEPTH = 200

_SYNCHRONIZE_STARTS = frozenset(
    {
        TokenType.LET,
        TokenType.FUNC,
        TokenType.IF,
        TokenType.WHILE,
        TokenType.FOR,
        TokenType.BREAK,
        TokenType.CONTINUE,
        TokenType.RETURN,
        TokenType.STRATEGY,
        TokenType.BACKTEST,
        TokenType.IMPORT,
    }
)

# `crosses_above`/`crosses_below` sit with the comparisons: like `<`, they
# compare two series bar by bar and produce a Series<Bool>.
_COMPARISON_OPERATORS = (
    TokenType.LESS,
    TokenType.GREATER,
    TokenType.LESS_EQUALS,
    TokenType.GREATER_EQUALS,
    TokenType.CROSSES_ABOVE,
    TokenType.CROSSES_BELOW,
)
_EQUALITY_OPERATORS = (TokenType.EQUALS_EQUALS, TokenType.BANG_EQUALS)

_CLOSERS: dict[TokenType, str] = {
    TokenType.RBRACE: "}",
    TokenType.RPAREN: ")",
    TokenType.RBRACKET: "]",
}

_WHEN_HINT = (
    "a 'when' condition is any Series<Bool>, e.g. "
    "'when fast crosses_above slow: buy();' or 'when rsi < 30.0: buy();'"
)

_T = TypeVar("_T")


class _BacktestParts(NamedTuple):
    """What the statement and expression forms of `backtest` share --
    named like their fields, so either node is `Node(**parts._asdict())`."""

    span: SourceSpan
    strategy_name: str
    symbol: str
    start: Expr
    end: Expr
    options: list[BacktestOption]


def _whole_number(expr: Expr) -> int | None:
    """The value of a whole-number literal, else None."""
    if isinstance(expr, NumberLiteral) and isinstance(expr.value, int):
        return expr.value
    return None


def _canonical_spellings() -> dict[TokenType, str]:
    """The first spelling listed for each keyword token is its canonical
    one (`let`, not `ts`) -- see allie/lexer/keywords.py."""
    canonical: dict[TokenType, str] = {}
    for spelling, token_type in KEYWORD_ALIASES.items():
        canonical.setdefault(token_type, spelling)
    return canonical


_CANONICAL = _canonical_spellings()


class _ParseError(Exception):
    """Internal signal to unwind to the nearest recovery point after a
    Diagnostic has already been recorded. Never surfaces to the caller."""


class Parser:
    def __init__(self, tokens: list[Token], diagnostics: DiagnosticBag | None = None) -> None:
        self.tokens = tokens
        self.current = 0
        self.diagnostics = diagnostics if diagnostics is not None else DiagnosticBag()
        self._nesting = 0
        self._block_depth = 0
        self._abandoned = False
        # Column of the first token on each line -- its indentation -- so
        # an unclosed-brace report can point at a suspiciously indented `}`.
        self._line_indent: dict[int, int] = {}
        for token in tokens:
            self._line_indent.setdefault(token.line, token.column)
        self._misaligned_close: tuple[Token, Token] | None = None

    def parse(self) -> Program:
        statements: list[Stmt] = []
        while not self._is_at_end():
            stmt = self._declaration()
            if stmt is not None:
                statements.append(stmt)
        program = Program(statements=statements)
        if not self._abandoned:
            self._check_depth(program)
        return program

    # ---- declarations ----

    def _declaration(self) -> Stmt | None:
        statement_start = self.current
        try:
            if self._block_depth == 0 and self._check(TokenType.RBRACE):
                self._error_at(
                    self._span(self._peek()),
                    "unmatched '}'",
                    hint="there are more '}' than '{' above this point",
                )
                self._advance()
                return None
            if self._check(TokenType.ELSE):
                self._stray_else()
                return None
            if self._check(TokenType.WHEN):
                self._stray_when()
                return None
            if self._match(TokenType.LET):
                return self._let_statement()
            if self._match(TokenType.FUNC):
                return self._func_declaration()
            if self._match(TokenType.STRATEGY):
                return self._strategy_declaration()
            return self._statement()
        except _ParseError:
            self._synchronize(statement_start)
            return None

    def _stray_else(self) -> None:
        """`else` with no `if` before it. Its block (or `else if`) is
        parsed and discarded, so recovery doesn't trip over its braces."""
        self._error_at(
            self._span(self._advance()),
            "'else' without a matching 'if'",
            hint="'else' must come right after the '}' that closes an if block",
        )
        if self._check(TokenType.LBRACE) or self._check(TokenType.IF):
            self._statement()

    def _stray_when(self) -> None:
        """A `when` clause outside any strategy, parsed and discarded."""
        self._error_at(
            self._span(self._advance()),
            "'when' outside a strategy",
            hint="'when' clauses go inside 'strategy Name { ... }'",
        )
        self._when_clause()

    def _let_statement(self) -> LetStmt:
        start_span = self._span(self._previous())
        name_token = self._consume(TokenType.IDENTIFIER, "expected a variable name after 'let'")
        type_annotation = None
        if self._match(TokenType.COLON):
            type_annotation = self._parse_type()
        self._consume(TokenType.EQUALS, "expected '=' after the variable name")
        initializer = self._expression()
        self._consume(TokenType.SEMICOLON, "expected ';' after the let statement")
        return LetStmt(
            span=start_span,
            name=name_token.lexeme,
            type_annotation=type_annotation,
            initializer=initializer,
        )

    def _func_declaration(self) -> FuncDecl:
        start_span = self._span(self._previous())
        name_token = self._consume(TokenType.IDENTIFIER, "expected a function name after 'func'")
        opening = self._consume(TokenType.LPAREN, "expected '(' after the function name")
        params = self._comma_separated(TokenType.RPAREN, self._param)
        self._expect_closing(TokenType.RPAREN, opening, "parameter list")

        seen: set[str] = set()
        for param in params:
            if param.name in seen:
                self._error_at(param.span, f"duplicate parameter name '{param.name}'")
            seen.add(param.name)

        return_type = self._parse_type() if self._match(TokenType.ARROW) else None
        body_opening = self._consume(TokenType.LBRACE, "expected '{' to begin the function body")
        body = self._block(body_opening)
        return FuncDecl(
            span=start_span,
            name=name_token.lexeme,
            params=params,
            return_type=return_type,
            body=body,
        )

    def _param(self) -> Param:
        name_token = self._consume(TokenType.IDENTIFIER, "expected a parameter name")
        type_annotation = self._parse_type() if self._match(TokenType.COLON) else None
        return Param(
            name=name_token.lexeme, type_annotation=type_annotation, span=self._span(name_token)
        )

    def _strategy_declaration(self) -> StrategyDecl:
        start_span = self._span(self._previous())
        name_token = self._consume(TokenType.IDENTIFIER, "expected a strategy name")
        opening = self._consume(TokenType.LBRACE, "expected '{' after the strategy name")

        when_clauses: list[WhenClause] = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            try:
                if not self._match(TokenType.WHEN):
                    self._error_expected("expected a 'when' clause inside the strategy body")
                    raise _ParseError
                when_clauses.append(self._when_clause())
            except _ParseError:
                if self._abandoned:
                    raise
                self._skip_to_next_when()
        self._expect_closing(TokenType.RBRACE, opening, "strategy body")

        return StrategyDecl(span=start_span, name=name_token.lexeme, when_clauses=when_clauses)

    def _when_clause(self) -> WhenClause:
        start_span = self._span(self._previous())
        condition = self._expression()
        if not self._check(TokenType.COLON):
            found = self._peek()
            # `when f crosses s:` / `when f cross_above s:` -- a misspelt
            # crossing operator lexes as a name right after the operand.
            misspelt_crossing = found.type == TokenType.IDENTIFIER and "cross" in found.lexeme
            hint = (
                f"'{found.lexeme}' isn't an operator; "
                "did you mean 'crosses_above' or 'crosses_below'?"
                if misspelt_crossing
                else _WHEN_HINT
            )
            self._error_expected("expected ':' after the when condition", hint=hint)
            raise _ParseError
        self._advance()
        body = self._statement()
        return WhenClause(span=start_span, condition=condition, body=body)

    def _skip_to_next_when(self) -> None:
        """Recovery inside a strategy body: skip to the next `when` or to
        the strategy's closing `}`, stepping over any braces a broken
        clause's action block opened."""
        depth = 0
        while not self._is_at_end():
            token = self._peek()
            if depth == 0 and token.type in (TokenType.WHEN, TokenType.RBRACE):
                return
            if token.type == TokenType.LBRACE:
                depth += 1
            elif token.type == TokenType.RBRACE:
                depth -= 1
            self._advance()

    # ---- statements ----

    def _statement(self) -> Stmt:
        if self._match(TokenType.IF):
            return self._if_statement()
        if self._match(TokenType.WHILE):
            return self._while_statement()
        if self._match(TokenType.FOR):
            return self._for_statement()
        if self._match(TokenType.BREAK):
            return self._break_statement()
        if self._match(TokenType.CONTINUE):
            return self._continue_statement()
        if self._match(TokenType.RETURN):
            return self._return_statement()
        if self._match(TokenType.IMPORT):
            return self._import_statement()
        if self._match(TokenType.BACKTEST):
            return self._backtest_statement()
        if self._match(TokenType.LBRACE):
            return self._block(self._previous())
        return self._expr_statement()

    def _block(self, opening: Token) -> Block:
        """`opening` is the `{`, already consumed."""
        statements: list[Stmt] = []
        with self._nested(opening):
            self._block_depth += 1
            try:
                while not self._check(TokenType.RBRACE) and not self._is_at_end():
                    stmt = self._declaration()
                    if stmt is not None:
                        statements.append(stmt)
            finally:
                self._block_depth -= 1
            closing = self._expect_closing(TokenType.RBRACE, opening, "block")
        self._note_alignment(opening, closing)
        return Block(span=self._span(opening), statements=statements)

    def _if_statement(self) -> IfStmt:
        """`if`/`else if`/`else` chains are collected in a loop and folded
        into nested IfStmts afterwards, so a long chain costs no
        recursion here."""
        branches: list[tuple[SourceSpan, Expr, Block]] = []
        else_block: Block | None = None
        span = self._span(self._previous())
        while True:
            condition = self._condition("if")
            opening = self._consume(TokenType.LBRACE, "expected '{' after the if condition")
            branches.append((span, condition, self._block(opening)))
            if not self._match(TokenType.ELSE):
                break
            if self._match(TokenType.IF):
                span = self._span(self._previous())
                continue
            opening = self._consume(TokenType.LBRACE, "expected '{' or 'if' after 'else'")
            else_block = self._block(opening)
            break

        result: Stmt | None = else_block
        for branch_span, condition, then_branch in reversed(branches):
            result = IfStmt(
                span=branch_span,
                condition=condition,
                then_branch=then_branch,
                else_branch=result,
            )
        assert isinstance(result, IfStmt)
        return result

    def _while_statement(self) -> WhileStmt:
        start_span = self._span(self._previous())
        condition = self._condition("while")
        opening = self._consume(TokenType.LBRACE, "expected '{' after the while condition")
        body = self._block(opening)
        return WhileStmt(span=start_span, condition=condition, body=body)

    def _for_statement(self) -> ForStmt:
        start_span = self._span(self._previous())
        self._consume(TokenType.LET, "expected 'let' to begin the for-loop initializer")
        initializer = self._let_statement()
        condition = self._condition("for")
        self._consume(TokenType.SEMICOLON, "expected ';' after the for-loop condition")
        increment = self._expression()
        opening = self._consume(TokenType.LBRACE, "expected '{' after the for-loop clauses")
        body = self._block(opening)
        return ForStmt(
            span=start_span,
            initializer=initializer,
            condition=condition,
            increment=increment,
            body=body,
        )

    def _condition(self, keyword: str) -> Expr:
        condition = self._expression()
        if isinstance(condition, Assignment):
            self._error_at(
                condition.span,
                f"this {keyword} condition is an assignment",
                hint="use '==' to compare; '=' assigns a value",
            )
        return condition

    def _break_statement(self) -> BreakStmt:
        span = self._span(self._previous())
        self._consume(TokenType.SEMICOLON, "expected ';' after 'break'")
        return BreakStmt(span=span)

    def _continue_statement(self) -> ContinueStmt:
        span = self._span(self._previous())
        self._consume(TokenType.SEMICOLON, "expected ';' after 'continue'")
        return ContinueStmt(span=span)

    def _return_statement(self) -> ReturnStmt:
        span = self._span(self._previous())
        value: Expr | None = None
        if not self._check(TokenType.SEMICOLON):
            value = self._expression()
        self._consume(TokenType.SEMICOLON, "expected ';' after the return statement")
        return ReturnStmt(span=span, value=value)

    def _import_statement(self) -> ImportStmt:
        span = self._span(self._previous())
        path_token = self._consume(TokenType.STRING, "expected a string path after 'import'")
        self._consume(TokenType.SEMICOLON, "expected ';' after the import statement")
        return ImportStmt(span=span, path=path_token.literal)

    def _backtest_statement(self) -> BacktestStmt:
        parts = self._backtest_parts()
        self._consume(TokenType.SEMICOLON, "expected ';' after the backtest statement")
        return BacktestStmt(**parts._asdict())

    def _backtest_parts(self) -> _BacktestParts:
        """`S on SYM from N to M (with name: value, ...)?`, after the
        `backtest` keyword -- shared by the statement form (which prints
        the results table) and the expression form (which evaluates to
        the results)."""
        start_span = self._span(self._previous())
        strategy_token = self._consume(
            TokenType.IDENTIFIER, "expected a strategy name after 'backtest'"
        )
        self._consume(TokenType.ON, "expected 'on' after the strategy name")
        symbol_token = self._consume(TokenType.IDENTIFIER, "expected a symbol name after 'on'")
        self._consume(TokenType.FROM, "expected 'from' after the symbol")
        start_token = self._peek()
        start = self._bar_bound("from")
        self._consume(TokenType.TO, "expected 'to' after the start bar")
        end = self._bar_bound("to")

        literal_start, literal_end = _whole_number(start), _whole_number(end)
        if literal_start is not None and literal_end is not None and literal_start >= literal_end:
            self._error_at(
                self._span(start_token),
                f"empty bar range: 'from {literal_start} to {literal_end}' selects no bars",
                hint="the range is [from, to), so 'to' must be greater than 'from'",
            )
        options = self._backtest_options() if self._match(TokenType.WITH) else []
        return _BacktestParts(
            span=start_span,
            strategy_name=strategy_token.lexeme,
            symbol=symbol_token.lexeme,
            start=start,
            end=end,
            options=options,
        )

    def _backtest_options(self) -> list[BacktestOption]:
        """`name: value, ...` after `with`. Values are at `+`/`-`
        precedence, like the bounds. Which names exist, and their types,
        is the type checker's business."""
        options: list[BacktestOption] = []
        while True:
            after = "'with'" if not options else "','"
            name = self._consume(TokenType.IDENTIFIER, f"expected an option name after {after}")
            self._consume(TokenType.COLON, f"expected ':' after the option name '{name.lexeme}'")
            value = self._term()
            if (
                name.lexeme == "fill"
                and isinstance(value, Identifier)
                and value.name in ("open", "close")
            ):
                # Otherwise reported as an undefined variable, which hides the fix.
                self._error_at(
                    value.span,
                    f"the fill point is a string, not a name: fill: \"{value.name}\"",
                    hint="'fill' is \"open\" (the next bar's open) or \"close\"",
                )
            options.append(BacktestOption(span=self._span(name), name=name.lexeme, value=value))
            if not self._match(TokenType.COMMA):
                return options

    def _bar_bound(self, keyword: str) -> Expr:
        """A backtest bound: an Int expression, at `+`/`-` precedence --
        `from 0 to 100`, `from split to len(c) - 1`. A literal's mistakes
        are reported here, where they're certain; anything else is the
        type checker's (it must be an Int) and the engine's (the bars
        must exist)."""
        first = self._peek()
        if first.type in (TokenType.SEMICOLON, TokenType.TO, TokenType.WITH, TokenType.EOF):
            self._error_expected(f"expected a bar index after '{keyword}'")
            raise _ParseError
        bound = self._term()
        if isinstance(bound, NumberLiteral) and not isinstance(bound.value, int):
            self._error_at(
                self._span(first),
                f"a bar index must be a whole number, found {first.lexeme}",
                hint="bars are counted 0, 1, 2, ...",
            )
        elif (
            isinstance(bound, Unary)
            and bound.operator == "-"
            and isinstance(bound.operand, NumberLiteral)
        ):
            self._error_at(
                self._span(first),
                "a bar index can't be negative",
                hint="bars are counted from 0",
            )
        return bound

    def _expr_statement(self) -> ExprStmt:
        start_span = self._span(self._peek())
        expr = self._expression()
        self._consume(TokenType.SEMICOLON, "expected ';' after the expression")
        return ExprStmt(span=start_span, expression=expr)

    # ---- types ----

    def _parse_type(self) -> TypeAnnotation:
        name_token = self._consume(TokenType.IDENTIFIER, "expected a type name")
        type_arg = None
        if self._match(TokenType.LBRACKET):
            opening = self._previous()
            with self._nested(opening):
                type_arg = self._parse_type()
            self._expect_closing(TokenType.RBRACKET, opening, "type argument")
        return TypeAnnotation(
            name=name_token.lexeme, span=self._span(name_token), type_arg=type_arg
        )

    # ---- expressions, lowest precedence first ----

    def _expression(self) -> Expr:
        with self._nested(self._peek()):
            return self._assignment()

    def _assignment(self) -> Expr:
        expr = self._logic_or()
        if not self._match(TokenType.EQUALS):
            return expr
        equals = self._previous()
        value = self._expression()
        if isinstance(expr, Identifier):
            return Assignment(span=expr.span, name=expr.name, value=value)
        if isinstance(expr, Index):
            message = "can't assign to a series element"
            hint = "series are immutable values; build a new series instead"
        elif isinstance(expr, FieldAccess):
            message = f"can't assign to the field '{expr.name}'"
            hint = "fields are read-only"
        else:
            message = "invalid assignment target"
            hint = "only a variable name can appear on the left of '='"
        self._error_at(self._span(equals), message, hint=hint)
        return value

    def _logic_or(self) -> Expr:
        expr = self._logic_and()
        while self._match(TokenType.OR):
            operator_span = self._span(self._previous())
            right = self._logic_and()
            expr = Logical(span=operator_span, operator="or", left=expr, right=right)
        return expr

    def _logic_and(self) -> Expr:
        expr = self._equality()
        while self._match(TokenType.AND):
            operator_span = self._span(self._previous())
            right = self._equality()
            expr = Logical(span=operator_span, operator="and", left=expr, right=right)
        return expr

    def _equality(self) -> Expr:
        return self._non_chaining(
            self._comparison,
            _EQUALITY_OPERATORS,
            "equality operators can't be chained",
            "compare one pair at a time and join the results with 'and'",
        )

    def _comparison(self) -> Expr:
        return self._non_chaining(
            self._term,
            _COMPARISON_OPERATORS,
            "comparison operators can't be chained",
            "write 'a < x and x < b'; unlike Python, Allie doesn't chain comparisons",
        )

    def _non_chaining(
        self,
        operand: Callable[[], Expr],
        operators: tuple[TokenType, ...],
        message: str,
        hint: str,
    ) -> Expr:
        """`0 < x < 10` would otherwise parse as `(0 < x) < 10` and fail
        later with a baffling "compare Bool to Int" type error. Reported
        once per chain; parsing continues normally."""
        expr = operand()
        count = 0
        while self._match(*operators):
            operator = self._previous()
            count += 1
            if count == 2:
                self._error_at(self._span(operator), message, hint=hint)
            right = operand()
            expr = Binary(
                span=self._span(operator), operator=operator.lexeme, left=expr, right=right
            )
        return expr

    def _term(self) -> Expr:
        expr = self._factor()
        while self._match(TokenType.PLUS, TokenType.MINUS):
            operator = self._previous()
            right = self._factor()
            expr = Binary(
                span=self._span(operator), operator=operator.lexeme, left=expr, right=right
            )
        return expr

    def _factor(self) -> Expr:
        expr = self._unary()
        while self._match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            operator = self._previous()
            right = self._unary()
            expr = Binary(
                span=self._span(operator), operator=operator.lexeme, left=expr, right=right
            )
        return expr

    def _unary(self) -> Expr:
        if self._match(TokenType.MINUS, TokenType.BANG):
            operator = self._previous()
            with self._nested(operator):
                operand = self._unary()
            return Unary(span=self._span(operator), operator=operator.lexeme, operand=operand)
        return self._call()

    def _call(self) -> Expr:
        """Postfix operators: call `f(...)`, field `a.b`, index `s[i]`."""
        expr = self._primary()
        while True:
            if self._match(TokenType.LPAREN):
                opening = self._previous()
                arguments = self._comma_separated(TokenType.RPAREN, self._expression)
                self._expect_closing(TokenType.RPAREN, opening, "call")
                # A call to a named function is located at the name, so
                # errors about the call underline `SMA`, not its `(`.
                span = expr.span if isinstance(expr, Identifier) else self._span(opening)
                expr = Call(span=span, callee=expr, arguments=arguments)
            elif self._match(TokenType.DOT):
                name = self._consume(TokenType.IDENTIFIER, "expected a field name after '.'")
                expr = FieldAccess(span=self._span(name), obj=expr, name=name.lexeme)
            elif self._match(TokenType.LBRACKET):
                opening = self._previous()
                index = self._expression()
                self._expect_closing(TokenType.RBRACKET, opening, "index")
                expr = Index(span=self._span(opening), obj=expr, index=index)
            else:
                break
        return expr

    def _primary(self) -> Expr:
        if self._match(TokenType.NUMBER):
            token = self._previous()
            return NumberLiteral(span=self._span(token), value=token.literal)
        if self._match(TokenType.STRING):
            token = self._previous()
            return StringLiteral(span=self._span(token), value=token.literal)
        if self._match(TokenType.TRUE):
            return BoolLiteral(span=self._span(self._previous()), value=True)
        if self._match(TokenType.FALSE):
            return BoolLiteral(span=self._span(self._previous()), value=False)
        if self._match(TokenType.IDENTIFIER):
            token = self._previous()
            return Identifier(span=self._span(token), name=token.lexeme)
        if self._match(TokenType.LPAREN):
            opening = self._previous()
            expr = self._expression()
            self._expect_closing(TokenType.RPAREN, opening, "parenthesis")
            return expr
        if self._match(TokenType.LBRACKET):
            opening = self._previous()
            elements = self._comma_separated(TokenType.RBRACKET, self._expression)
            self._expect_closing(TokenType.RBRACKET, opening, "array literal")
            return ArrayLiteral(span=self._span(opening), elements=elements)
        if self._match(TokenType.BACKTEST):
            return Backtest(**self._backtest_parts()._asdict())

        self._error_expected("expected an expression")
        raise _ParseError

    def _comma_separated(self, closer: TokenType, parse_item: Callable[[], _T]) -> list[_T]:
        """Items up to, not including, `closer`. A trailing comma is
        allowed, so multi-line lists can end every line with one."""
        items: list[_T] = []
        while not self._check(closer) and not self._is_at_end():
            items.append(parse_item())
            if not self._match(TokenType.COMMA):
                break
        return items

    # ---- token-stream helpers ----

    def _match(self, *types: TokenType) -> bool:
        for token_type in types:
            if self._check(token_type):
                self._advance()
                return True
        return False

    def _check(self, token_type: TokenType) -> bool:
        if self._is_at_end():
            return False
        return self._peek().type == token_type

    def _advance(self) -> Token:
        if not self._is_at_end():
            self.current += 1
        return self._previous()

    def _is_at_end(self) -> bool:
        return self._peek().type == TokenType.EOF

    def _peek(self) -> Token:
        return self.tokens[self.current]

    def _previous(self) -> Token:
        return self.tokens[self.current - 1]

    def _consume(self, token_type: TokenType, message: str) -> Token:
        if self._check(token_type):
            return self._advance()
        found = self._peek()
        same_line = self.current == 0 or found.line == self._previous().line
        if token_type == TokenType.IDENTIFIER and found.lexeme in KEYWORD_ALIASES and same_line:
            canonical = _CANONICAL[found.type]
            if canonical == found.lexeme:
                hint = f"'{found.lexeme}' is a keyword; pick a different name"
            else:
                hint = (
                    f"'{found.lexeme}' is reserved as another spelling of '{canonical}'; "
                    "pick a different name"
                )
            self._error_at(
                self._span(found), f"{message}, found keyword '{found.lexeme}'", hint=hint
            )
            # Recover by using the keyword as the name: the rest of the
            # statement then parses normally instead of the keyword being
            # re-read as the start of a new statement (`ts` *is* `let`).
            return self._advance()
        self._error_expected(message)
        raise _ParseError

    def _expect_closing(self, closer: TokenType, opening: Token, what: str) -> Token:
        if self._check(closer):
            return self._advance()
        if self._abandoned:
            raise _ParseError
        closing_text = _CLOSERS[closer]
        if self._is_at_end():
            hint = f"add a matching '{closing_text}'"
            if closer == TokenType.RBRACE and self._misaligned_close is not None:
                inner, early = self._misaligned_close
                self._misaligned_close = None
                hint = (
                    f"the '{{' at {inner.line}:{inner.column} may be the one missing its "
                    f"'}}': the '}}' at {early.line}:{early.column} that closed it is "
                    "indented less than the line that opened it"
                )
            self._error_at(
                self._span(opening), f"this '{opening.lexeme}' is never closed", hint=hint
            )
        else:
            self._error_expected(
                f"expected '{closing_text}' to close the {what} opened at "
                f"{opening.line}:{opening.column}"
            )
        raise _ParseError

    def _note_alignment(self, opening: Token, closing: Token) -> None:
        """Remembers the first block whose `}` sits left of the line that
        opened it -- the usual sign that a `}` is missing inside it and an
        outer block's brace closed it instead."""
        if (
            self._misaligned_close is None
            and closing.line != opening.line
            and closing.column < self._line_indent.get(opening.line, 1)
        ):
            self._misaligned_close = (opening, closing)

    @contextmanager
    def _nested(self, token: Token) -> Iterator[None]:
        self._nesting += 1
        try:
            if self._nesting > MAX_NESTING:
                self._error_at(
                    self._span(token),
                    f"nesting is too deep: more than {MAX_NESTING} levels of "
                    "parentheses, brackets, calls and blocks",
                    hint="split it into smaller pieces with intermediate 'let' bindings",
                )
                # Nothing after this point can be parsed reliably, and
                # resynchronizing at this depth would only add noise.
                self._abandoned = True
                self.current = len(self.tokens) - 1
                raise _ParseError
            yield
        finally:
            self._nesting -= 1

    def _check_depth(self, program: Program) -> None:
        """Reports the first statement whose AST nests deeper than
        MAX_AST_DEPTH, at the start of the innermost statement containing
        the too-deep part (the middle of a long expression would be a
        useless place for the caret)."""
        stack: list[tuple[object, int, Stmt]] = [(stmt, 1, stmt) for stmt in program.statements]
        while stack:
            node, depth, owner = stack.pop()
            if isinstance(node, Stmt):
                owner = node
            if depth > MAX_AST_DEPTH:
                self._error_at(
                    owner.span,
                    f"this statement is nested too deeply: more than {MAX_AST_DEPTH} "
                    "levels of expressions and blocks",
                    hint="break it up with intermediate 'let' bindings",
                )
                return
            stack.extend((child, depth + 1, owner) for child in _children(node))

    def _span(self, token: Token) -> SourceSpan:
        return SourceSpan(
            line=token.line,
            column=token.column,
            offset=token.offset,
            length=len(token.lexeme) or 1,
        )

    def _error_at(self, span: SourceSpan, message: str, *, hint: str | None = None) -> None:
        self.diagnostics.error(message, span, hint=hint)

    def _error_expected(self, message: str, *, hint: str | None = None) -> None:
        """Reports `message` against the next token. When that token is on
        a later line than the previous one, the error goes just after the
        previous token instead: that's where the missing `;` or `)` goes,
        and pointing at the next line's first token misleads."""
        found = self._peek()
        if self.current > 0 and found.line > self._previous().line:
            previous = self._previous()
            end = len(previous.lexeme)
            span = SourceSpan(
                line=previous.line,
                column=previous.column + end,
                offset=previous.offset + end,
                length=1,
            )
            self._error_at(span, message, hint=hint)
            return
        self._error_at(self._span(found), f"{message}, found {_describe(found)}", hint=hint)

    def _synchronize(self, statement_start: int) -> None:
        """Skips to a likely statement boundary: just past a `;`, or just
        before a statement keyword. Never consumes a `}` while inside a
        block -- that brace closes the block being parsed. Always makes
        progress past `statement_start`, so recovery can't loop."""
        while not self._is_at_end():
            token = self._peek()
            if self.current > statement_start:
                if token.type in _SYNCHRONIZE_STARTS:
                    return
                if token.type == TokenType.RBRACE and self._block_depth > 0:
                    return
            self._advance()
            if token.type == TokenType.SEMICOLON:
                return


def _describe(token: Token) -> str:
    """How a token is named in "found ..." messages."""
    if token.type == TokenType.EOF:
        return "end of file"
    if token.lexeme in KEYWORD_ALIASES:
        return f"keyword '{token.lexeme}'"
    lexeme = token.lexeme if len(token.lexeme) <= 24 else token.lexeme[:21] + "..."
    if token.type == TokenType.STRING:
        return f"string {lexeme}"
    return f"'{lexeme}'"


def _children(node: object) -> list[object]:
    """Direct AST children of `node`, for the depth check."""
    children: list[object] = []
    for value in vars(node).values():
        items = value if isinstance(value, list) else [value]
        children.extend(
            item for item in items if isinstance(item, (Expr, Stmt, WhenClause))
        )
    return children
