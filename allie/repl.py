"""`allie repl`: an interactive session.

    $ allie repl
    allie 0.1.0 -- statements end with ';', :help for commands
    >>> let c = load("tests/fixtures/AAPL_sample.csv").close;
    >>> mean(SMA(c, 20))
    158.98179268292682
    >>> :type SMA(c, 5) > c
    Series<Bool>

Every submission goes through the same static passes as a file -- syntax,
scopes, types, look-ahead -- *incrementally*: the resolver, type checker
and look-ahead analysis keep their state between submissions, so a
submission is checked against everything accepted before it, and its
errors point at the line you just typed. A submission that fails, at
compile time or at run time, is not kept, so a later one can't depend on
a name it never defined. It runs on the tree-walking interpreter, in one
environment that lasts the session.

An expression typed on its own has its value printed; a missing final
`;` is supplied. A line that leaves a `{`, `(` or `[` open continues on
the next.
"""

from __future__ import annotations

import contextlib
import sys
from typing import TextIO

import allie
import allie.stdlib as stdlib
from allie.ast.nodes import ExprStmt, Program
from allie.errors import DiagnosticBag
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.runtime.errors import AllieRuntimeError
from allie.runtime.interpreter import evaluate, run_program
from allie.semantic.lookahead import LookaheadAnalysis
from allie.semantic.resolver import Resolver
from allie.stdlib.io import print_value
from allie.typechecker.checker import TypeChecker
from allie.typechecker.errors import QSTypeError
from allie.typechecker.types import VOID

_HELP = """\
  a statement      let x = 1 + 2;     runs it
  an expression    x * 2              prints its value
  :type EXPR       the type of an expression, without running it
  :reset           forget everything defined so far
  :help            this
  :quit            leave (or Ctrl-D)"""


class Session:
    """The state of one REPL session. `submit` takes one complete input
    and returns whether it was accepted."""

    def __init__(self, out: TextIO | None = None, err: TextIO | None = None) -> None:
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self.reset()

    def reset(self) -> None:
        self.env = stdlib.create_global_environment()
        self.resolver = Resolver()
        self.checker = TypeChecker()
        for name, signature in stdlib.builtin_function_types().items():
            self.checker.env.define(name, signature)
        self.lookahead = LookaheadAnalysis()

    # ---- one submission ----

    def submit(self, text: str) -> bool:
        source = _terminated(text)
        before = dict(self.checker.env.types)
        checkpoint = self.lookahead.checkpoint()
        program = self._check(source)
        if program is None:
            self.lookahead.rollback(checkpoint, ran=False)
            return False
        statements = program.statements
        try:
            with contextlib.redirect_stdout(self.out):
                if len(statements) == 1 and isinstance(statements[0], ExprStmt):
                    expression = statements[0].expression
                    value = evaluate(expression, self.env)
                    if expression.inferred_type != VOID:
                        print_value(value)
                else:
                    run_program(program, self.env)
        except (AllieRuntimeError, RecursionError) as error:
            # It type-checked, so its names went into the checker; it
            # didn't finish, so they mustn't stay there.
            self.checker.env.types = before
            self.lookahead.rollback(checkpoint, ran=True)
            message = (
                error.render(source)
                if isinstance(error, AllieRuntimeError)
                else "AllieRuntimeError: recursion too deep"
            )
            print(message, file=self.err)
            return False
        return True

    def type_of(self, text: str) -> str | None:
        """The type of an expression, checked but not run or kept."""
        source = _terminated(text)
        before = dict(self.checker.env.types)
        program = self._check(source, keep=False)
        self.checker.env.types = before
        if program is None:
            return None
        statement = program.statements[0] if len(program.statements) == 1 else None
        if not isinstance(statement, ExprStmt):
            print("error: :type takes one expression", file=self.err)
            return None
        return str(statement.expression.inferred_type)

    def _check(self, source: str, *, keep: bool = True) -> Program | None:
        lexer = Lexer(source)
        diagnostics = DiagnosticBag()
        program = Parser(lexer.scan_tokens(), diagnostics).parse()
        if lexer.errors:
            for error in lexer.errors:
                print(error.render(source), file=self.err)
            return None
        if not diagnostics.has_errors():
            self.resolver.diagnostics = diagnostics
            self.resolver.resolve(program)
        if diagnostics.has_errors():
            diagnostics.print_all(source, file=self.err)
            return None

        # Restored on failure, so a rejected submission leaves no trace.
        before = dict(self.checker.env.types)
        self.checker.errors = []
        try:
            self.checker.check(program)
        except QSTypeError:
            self.checker.env.types = before
            for type_error in self.checker.errors:
                print(type_error.render(source), file=self.err)
            return None

        if keep:
            self.lookahead.diagnostics = DiagnosticBag()
            self.lookahead.analyze(program)
            if self.lookahead.diagnostics.has_errors():
                self.checker.env.types = before
                self.lookahead.diagnostics.print_all(source, file=self.err)
                return None
        return program


def _terminated(text: str) -> str:
    """`text`, with the final `;` an expression typed on its own leaves off."""
    stripped = text.rstrip()
    return stripped if stripped.endswith((";", "}")) else stripped + ";"


def _open_brackets(text: str) -> int:
    """How many `{ ( [` are still open, ignoring brackets inside strings
    and comments -- a rough count, enough to know to keep reading."""
    depth, in_string, i = 0, False, 0
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\":
                i += 1
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif text.startswith("//", i):
            newline = text.find("\n", i)
            i = len(text) if newline < 0 else newline
            continue
        elif c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
        i += 1
    return depth


def run_repl(stdin: TextIO | None = None) -> int:
    stdin = stdin or sys.stdin
    interactive = stdin.isatty()
    if interactive:
        try:
            import readline  # noqa: F401 -- line editing and history, where available
        except ImportError:
            pass
        print(f"allie {allie.__version__} -- statements end with ';', :help for commands")
    session = Session()
    buffer: list[str] = []
    while True:
        prompt = "... " if buffer else ">>> "
        try:
            line = input(prompt) if interactive else stdin.readline()
        except EOFError:
            line = ""
        except KeyboardInterrupt:
            print("\n(cancelled)")
            buffer = []
            continue
        if not interactive and not line:
            return 0
        if interactive and line == "" and not buffer:
            continue
        if not buffer and line.strip().startswith(":"):
            command, _, rest = line.strip().partition(" ")
            if command in (":quit", ":q", ":exit"):
                return 0
            if command == ":help":
                print(_HELP)
            elif command == ":reset":
                session.reset()
                print("(reset)")
            elif command == ":type" and rest.strip():
                shown = session.type_of(rest)
                if shown is not None:
                    print(shown)
            else:
                print(f"unknown command {command} -- :help lists them", file=sys.stderr)
            continue
        buffer.append(line.rstrip("\n"))
        text = "\n".join(buffer)
        if _open_brackets(text) > 0:
            continue
        buffer = []
        if text.strip():
            session.submit(text)
