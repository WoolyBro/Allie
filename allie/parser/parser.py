"""Recursive-descent parser -- Phase 2.3, designed here since no separate
parser spec was provided (Phase 3's typechecker spec assumes a parser and
AST already exist).

Targets docs/grammar.md directly and reuses the project's established
patterns rather than inventing new ones:

- Errors go through allie.errors.DiagnosticBag (collect-and-report-all,
  docs/spec.md "Diagnostics strategy"), the same mechanism the lexer's
  invalid-character handling is meant to eventually feed into.
- `_ParseError` is an internal control-flow signal -- never surfaced to
  the caller -- used only to unwind to `_synchronize()`'s next recovery
  point after a Diagnostic has already been recorded. Same shape as the
  interpreter's BreakSignal/ContinueSignal (Phase 1).

Token has no `offset` field (Phase 2.1's Token dataclass: type, lexeme,
literal, line, column), so SourceSpans built from a Token here use
offset=0 -- nothing reads offset yet (Diagnostic.render only uses
line/column for its caret).
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
from allie.lexer.tokens import Token, TokenType

_SYNCHRONIZE_STARTS = frozenset(
    {
        TokenType.LET,
        TokenType.FUNC,
        TokenType.IF,
        TokenType.WHILE,
        TokenType.FOR,
        TokenType.RETURN,
        TokenType.STRATEGY,
        TokenType.BACKTEST,
        TokenType.IMPORT,
    }
)


class _ParseError(Exception):
    """Internal signal to unwind to the nearest recovery point after a
    Diagnostic has already been recorded. Never surfaces to the caller."""


class Parser:
    def __init__(self, tokens: list[Token], diagnostics: DiagnosticBag | None = None) -> None:
        self.tokens = tokens
        self.current = 0
        self.diagnostics = diagnostics if diagnostics is not None else DiagnosticBag()

    def parse(self) -> Program:
        statements: list[Stmt] = []
        while not self._is_at_end():
            stmt = self._declaration()
            if stmt is not None:
                statements.append(stmt)
        return Program(statements=statements)

    # ---- declarations ----

    def _declaration(self) -> Stmt | None:
        try:
            if self._match(TokenType.LET):
                return self._let_statement()
            if self._match(TokenType.FUNC):
                return self._func_declaration()
            if self._match(TokenType.STRATEGY):
                return self._strategy_declaration()
            return self._statement()
        except _ParseError:
            self._synchronize()
            return None

    def _let_statement(self) -> LetStmt:
        start_span = self._span(self._previous())
        name_token = self._consume(TokenType.IDENTIFIER, "expected variable name after 'let'")
        type_annotation = None
        if self._match(TokenType.COLON):
            type_annotation = self._parse_type()
        self._consume(TokenType.EQUALS, "expected '=' after let target")
        initializer = self._expression()
        self._consume(TokenType.SEMICOLON, "expected ';' after let statement")
        return LetStmt(
            span=start_span,
            name=name_token.lexeme,
            type_annotation=type_annotation,
            initializer=initializer,
        )

    def _func_declaration(self) -> FuncDecl:
        start_span = self._span(self._previous())
        name_token = self._consume(TokenType.IDENTIFIER, "expected function name")
        self._consume(TokenType.LPAREN, "expected '(' after function name")

        params: list[Param] = []
        if not self._check(TokenType.RPAREN):
            while True:
                param_name = self._consume(TokenType.IDENTIFIER, "expected parameter name")
                param_type = None
                if self._match(TokenType.COLON):
                    param_type = self._parse_type()
                params.append(
                    Param(
                        name=param_name.lexeme,
                        type_annotation=param_type,
                        span=self._span(param_name),
                    )
                )
                if not self._match(TokenType.COMMA):
                    break
        self._consume(TokenType.RPAREN, "expected ')' after parameters")

        return_type = None
        if self._match(TokenType.ARROW):
            return_type = self._parse_type()

        self._consume(TokenType.LBRACE, "expected '{' to begin function body")
        body = self._block()
        return FuncDecl(
            span=start_span,
            name=name_token.lexeme,
            params=params,
            return_type=return_type,
            body=body,
        )

    def _strategy_declaration(self) -> StrategyDecl:
        start_span = self._span(self._previous())
        name_token = self._consume(TokenType.IDENTIFIER, "expected strategy name")
        self._consume(TokenType.LBRACE, "expected '{' after strategy name")

        when_clauses: list[WhenClause] = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            self._consume(TokenType.WHEN, "expected 'when' clause inside strategy body")
            when_clauses.append(self._when_clause())
        self._consume(TokenType.RBRACE, "expected '}' to close strategy body")

        return StrategyDecl(span=start_span, name=name_token.lexeme, when_clauses=when_clauses)

    def _when_clause(self) -> WhenClause:
        start_span = self._span(self._previous())
        left = self._expression()
        if self._match(TokenType.CROSSES_ABOVE):
            operator = "crosses_above"
        elif self._match(TokenType.CROSSES_BELOW):
            operator = "crosses_below"
        else:
            self._error(self._peek(), "expected 'crosses_above' or 'crosses_below' in when clause")
            raise _ParseError
        right = self._expression()
        self._consume(TokenType.COLON, "expected ':' after when-clause condition")
        body = self._statement()
        return WhenClause(span=start_span, left=left, operator=operator, right=right, body=body)

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
            return self._block()
        return self._expr_statement()

    def _block(self) -> Block:
        """Assumes the opening '{' was already consumed by the caller."""
        start_span = self._span(self._previous())
        statements: list[Stmt] = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            stmt = self._declaration()
            if stmt is not None:
                statements.append(stmt)
        self._consume(TokenType.RBRACE, "expected '}' to close block")
        return Block(span=start_span, statements=statements)

    def _if_statement(self) -> IfStmt:
        start_span = self._span(self._previous())
        condition = self._expression()
        self._consume(TokenType.LBRACE, "expected '{' after if condition")
        then_branch = self._block()
        else_branch: Stmt | None = None
        if self._match(TokenType.ELSE):
            if self._match(TokenType.IF):
                else_branch = self._if_statement()
            else:
                self._consume(TokenType.LBRACE, "expected '{' after else")
                else_branch = self._block()
        return IfStmt(
            span=start_span, condition=condition, then_branch=then_branch, else_branch=else_branch
        )

    def _while_statement(self) -> WhileStmt:
        start_span = self._span(self._previous())
        condition = self._expression()
        self._consume(TokenType.LBRACE, "expected '{' after while condition")
        body = self._block()
        return WhileStmt(span=start_span, condition=condition, body=body)

    def _for_statement(self) -> ForStmt:
        start_span = self._span(self._previous())
        self._consume(TokenType.LET, "expected 'let' to begin for-loop initializer")
        initializer = self._let_statement()
        condition = self._expression()
        self._consume(TokenType.SEMICOLON, "expected ';' after for-loop condition")
        increment = self._expression()
        self._consume(TokenType.LBRACE, "expected '{' after for-loop clauses")
        body = self._block()
        return ForStmt(
            span=start_span,
            initializer=initializer,
            condition=condition,
            increment=increment,
            body=body,
        )

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
        self._consume(TokenType.SEMICOLON, "expected ';' after return statement")
        return ReturnStmt(span=span, value=value)

    def _import_statement(self) -> ImportStmt:
        span = self._span(self._previous())
        path_token = self._consume(TokenType.STRING, "expected a string path after 'import'")
        self._consume(TokenType.SEMICOLON, "expected ';' after import statement")
        return ImportStmt(span=span, path=path_token.literal)

    def _backtest_statement(self) -> BacktestStmt:
        start_span = self._span(self._previous())
        strategy_token = self._consume(
            TokenType.IDENTIFIER, "expected strategy name after 'backtest'"
        )
        self._consume(TokenType.ON, "expected 'on' after strategy name")
        symbol_token = self._consume(TokenType.IDENTIFIER, "expected symbol name after 'on'")
        self._consume(TokenType.FROM, "expected 'from' after symbol")
        start_token = self._consume(TokenType.NUMBER, "expected start value after 'from'")
        self._consume(TokenType.TO, "expected 'to' after start value")
        end_token = self._consume(TokenType.NUMBER, "expected end value after 'to'")
        self._consume(TokenType.SEMICOLON, "expected ';' after backtest statement")
        return BacktestStmt(
            span=start_span,
            strategy_name=strategy_token.lexeme,
            symbol=symbol_token.lexeme,
            start=start_token.literal,
            end=end_token.literal,
        )

    def _expr_statement(self) -> ExprStmt:
        start_span = self._span(self._peek())
        expr = self._expression()
        self._consume(TokenType.SEMICOLON, "expected ';' after expression")
        return ExprStmt(span=start_span, expression=expr)

    # ---- types ----

    def _parse_type(self) -> TypeAnnotation:
        name_token = self._consume(TokenType.IDENTIFIER, "expected a type name")
        type_arg = None
        if self._match(TokenType.LBRACKET):
            type_arg = self._parse_type()
            self._consume(TokenType.RBRACKET, "expected ']' after type argument")
        return TypeAnnotation(
            name=name_token.lexeme, span=self._span(name_token), type_arg=type_arg
        )

    # ---- expressions (precedence climbing, matching docs/grammar.md) ----

    def _expression(self) -> Expr:
        return self._assignment()

    def _assignment(self) -> Expr:
        expr = self._logic_or()
        if self._match(TokenType.EQUALS):
            equals_span = self._span(self._previous())
            value = self._assignment()
            if isinstance(expr, Identifier):
                return Assignment(span=expr.span, name=expr.name, value=value)
            self._error_at_span(equals_span, "invalid assignment target")
            raise _ParseError
        return expr

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
        expr = self._comparison()
        while self._match(TokenType.EQUALS_EQUALS, TokenType.BANG_EQUALS):
            operator = self._previous()
            right = self._comparison()
            expr = Binary(
                span=self._span(operator), operator=operator.lexeme, left=expr, right=right
            )
        return expr

    def _comparison(self) -> Expr:
        expr = self._term()
        while self._match(
            TokenType.LESS, TokenType.GREATER, TokenType.LESS_EQUALS, TokenType.GREATER_EQUALS
        ):
            operator = self._previous()
            right = self._term()
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
            operand = self._unary()
            return Unary(span=self._span(operator), operator=operator.lexeme, operand=operand)
        return self._call()

    def _call(self) -> Expr:
        expr = self._primary()
        while True:
            if self._match(TokenType.LPAREN):
                expr = self._finish_call(expr)
            elif self._match(TokenType.DOT):
                name = self._consume(TokenType.IDENTIFIER, "expected property name after '.'")
                expr = FieldAccess(span=self._span(name), obj=expr, name=name.lexeme)
            else:
                break
        return expr

    def _finish_call(self, callee: Expr) -> Call:
        start_span = self._span(self._previous())
        arguments: list[Expr] = []
        if not self._check(TokenType.RPAREN):
            while True:
                arguments.append(self._expression())
                if not self._match(TokenType.COMMA):
                    break
        self._consume(TokenType.RPAREN, "expected ')' after arguments")
        return Call(span=start_span, callee=callee, arguments=arguments)

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
            expr = self._expression()
            self._consume(TokenType.RPAREN, "expected ')' after expression")
            return expr
        if self._match(TokenType.LBRACKET):
            return self._array_literal()

        self._error(self._peek(), "expected an expression")
        raise _ParseError

    def _array_literal(self) -> ArrayLiteral:
        start_span = self._span(self._previous())
        elements: list[Expr] = []
        if not self._check(TokenType.RBRACKET):
            while True:
                elements.append(self._expression())
                if not self._match(TokenType.COMMA):
                    break
        self._consume(TokenType.RBRACKET, "expected ']' after array elements")
        return ArrayLiteral(span=start_span, elements=elements)

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
        self._error(self._peek(), message)
        raise _ParseError

    def _span(self, token: Token) -> SourceSpan:
        return SourceSpan(
            line=token.line, column=token.column, offset=0, length=len(token.lexeme) or 1
        )

    def _error(self, token: Token, message: str) -> None:
        self.diagnostics.error(message, self._span(token))

    def _error_at_span(self, span: SourceSpan, message: str) -> None:
        self.diagnostics.error(message, span)

    def _synchronize(self) -> None:
        self._advance()
        while not self._is_at_end():
            if self._previous().type == TokenType.SEMICOLON:
                return
            if self._peek().type in _SYNCHRONIZE_STARTS:
                return
            self._advance()
