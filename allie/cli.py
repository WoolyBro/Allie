"""Command-line entry point for Allie.

Phase 0: wires up the argument surface and debug flags the lexer/parser
phases will need (--dump-tokens, --dump-ast, --trace). The actual
lex -> parse -> run pipeline is stubbed until Phase 2 lands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from allie.errors import DiagnosticBag


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
    diagnostics = DiagnosticBag()

    # Phase 0 stub: lexer/parser land in Phase 2. The evaluator itself
    # exists (allie.runtime.interpreter, Phase 1) but has nothing to run
    # until the parser can produce trees for it.
    if args.dump_tokens:
        print("(lexer not implemented yet - Phase 2)")
        return 0
    if args.dump_ast:
        print("(parser not implemented yet - Phase 2)")
        return 0

    print(
        f"(lexer/parser not implemented yet - Phase 2) "
        f"read {len(source)} bytes from {source_path}"
    )
    diagnostics.print_all()
    return 1 if diagnostics.has_errors() else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
