"""Command-line entry point for Allie.

Phase 0: wires up the argument surface and debug flags the lexer/parser
phases will need (--dump-tokens, --dump-ast, --trace). The full pipeline
is now wired: lex -> parse -> resolve -> type-check -> run, halting with
every collected diagnostic printed if any static pass fails, before
execution is even attempted. Importing `allie.stdlib` here triggers its
builtin registration as a side effect, before a TypeChecker/Environment
that need those builtins visible are constructed below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import allie.stdlib as stdlib
from allie.errors import DiagnosticBag
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.runtime.errors import AllieRuntimeError
from allie.runtime.interpreter import run_program
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker
from allie.typechecker.errors import QSTypeError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="allie", description="Allie compiler/interpreter CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a .ql source file")
    run.add_argument("file", type=Path, help="Path to a .ql source file")
    run.add_argument("--dump-tokens", action="store_true", help="Print the token stream and exit")
    run.add_argument("--dump-ast", action="store_true", help="Print the parsed AST and exit")
    run.add_argument("--trace", action="store_true", help="Print interpreter execution trace")

    return parser


def run_command(args: argparse.Namespace) -> int:
    source_path: Path = args.file
    if not source_path.exists():
        print(f"error: no such file: {source_path}", file=sys.stderr)
        return 1

    source = source_path.read_text(encoding="utf-8")

    lexer = Lexer(source)
    tokens = lexer.scan_tokens()
    if args.dump_tokens:
        for token in tokens:
            print(f"{token.line}:{token.column} {token.type.name} {token.lexeme!r}")

    diagnostics = DiagnosticBag()
    parser = Parser(tokens, diagnostics)
    program = parser.parse()
    if args.dump_ast:
        for stmt in program.statements:
            print(stmt)

    if lexer.errors:
        for lex_error in lexer.errors:
            print(lex_error, file=sys.stderr)
        return 1
    if diagnostics.has_errors():
        diagnostics.print_all()
        return 1

    if args.dump_tokens or args.dump_ast:
        return 0

    resolver = Resolver(diagnostics)
    resolver.resolve(program)
    if diagnostics.has_errors():
        diagnostics.print_all()
        return 1

    checker = TypeChecker()
    for name, signature in stdlib.builtin_function_types().items():
        checker.env.define(name, signature)
    try:
        checker.check(program)
    except QSTypeError as error:
        print(error, file=sys.stderr)
        return 1

    env = stdlib.create_global_environment()
    try:
        run_program(program, env)
    except AllieRuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
