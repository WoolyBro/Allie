"""Command-line entry point for Allie.

Runs the full pipeline: lex -> parse -> resolve -> type-check -> lower to
IR -> optimize -> execute, halting with every collected diagnostic
printed if any static pass fails, before execution is attempted.
Importing `allie.stdlib` here triggers its builtin registration as a
side effect, before a TypeChecker/Environment that need those builtins
visible are constructed.

## Two entry points, deliberately

Phase 6.4 specified `python main.py <file.qs> [--debug-tokens]
[--debug-ast] [--debug-ir] [--no-optimize]`, defaulting to the LLVM/JIT
path. This repo already had `allie run <file>` (an installed console
script, referenced by README/docs/tests) defaulting to the tree-walking
interpreter. Rather than break one to satisfy the other, both exist and
share `run_pipeline` below:

- `main.py` at the repo root implements the Phase 6.4 interface exactly,
  including the LLVM/JIT default.
- `allie run` keeps its existing shape and tree-walking default, and
  gained the same debug flags plus `--backend {tree,ir,llvm}`.

`--dump-tokens`/`--dump-ast` remain as aliases of `--debug-tokens`/
`--debug-ast` so nothing already written against the old names breaks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import allie.stdlib as stdlib
from allie.codegen.jit import compile_and_run
from allie.codegen.llvm_lowering import LLVMLowering, LLVMLoweringError
from allie.errors import DiagnosticBag
from allie.ir.interpreter import run_ir_program
from allie.ir.lowering import lower_program
from allie.lexer.lexer import Lexer
from allie.optimizer import constant_fold, cse, dce, run_passes
from allie.parser.parser import Parser
from allie.runtime.errors import AllieRuntimeError
from allie.runtime.interpreter import run_program
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker
from allie.typechecker.errors import QSTypeError

BACKENDS = ("tree", "ir", "llvm")

_DEFAULT_PASSES = [constant_fold.run, dce.run, cse.run]


def run_pipeline(
    source_path: Path,
    *,
    backend: str = "tree",
    debug_tokens: bool = False,
    debug_ast: bool = False,
    debug_ir: bool = False,
    optimize: bool = True,
) -> int:
    """Runs `source_path` end to end. Returns a process exit code."""
    if not source_path.exists():
        print(f"error: no such file: {source_path}", file=sys.stderr)
        return 1

    source = source_path.read_text(encoding="utf-8")

    lexer = Lexer(source)
    tokens = lexer.scan_tokens()
    if debug_tokens:
        print("=== TOKENS ===")
        for token in tokens:
            print(f"{token.line}:{token.column} {token.type.name} {token.lexeme!r}")

    diagnostics = DiagnosticBag()
    parser = Parser(tokens, diagnostics)
    program = parser.parse()
    if debug_ast:
        print("=== AST ===")
        for stmt in program.statements:
            print(stmt)

    if lexer.errors:
        for lex_error in lexer.errors:
            print(lex_error, file=sys.stderr)
        return 1
    if diagnostics.has_errors():
        diagnostics.print_all()
        return 1

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

    if backend == "tree" and not debug_ir:
        # The tree-walking interpreter runs straight off the AST -- no
        # IR lowering needed unless --debug-ir asked to see it.
        return _execute_tree(program)

    main_program, functions, register_types, function_info = lower_program(program)
    passes = _DEFAULT_PASSES if optimize else []
    main_program = run_passes(main_program, passes, enabled=optimize)
    functions = {
        name: (run_passes(prog, passes, enabled=optimize), params)
        for name, (prog, params) in functions.items()
    }

    if debug_ir:
        print("=== ALLIE IR ===")
        for instr in main_program.instructions:
            print(instr)
        for name, (prog, params) in functions.items():
            print(f"--- function {name}({', '.join(params)}) ---")
            for instr in prog.instructions:
                print(instr)

    if backend == "tree":
        return _execute_tree(program)
    if backend == "ir":
        return _execute_ir(main_program, functions)
    return _execute_llvm(main_program, functions, register_types, function_info, debug_ir)


def _report_runtime_error(error: BaseException) -> int:
    """One place every backend funnels runtime failures through.

    `RecursionError` is translated rather than propagated: an Allie
    function call consumes several Python frames, so deeply-recursive
    Allie code trips CPython's limit and, without this, dumps a raw
    Python traceback full of interpreter internals -- bypassing the
    whole AllieRuntimeError hierarchy the rest of the language reports
    through. The limit is deliberately *not* raised instead: a higher
    ceiling trades a catchable exception for a C-stack overflow that
    takes the process down with no message at all.
    """
    if isinstance(error, RecursionError):
        print(
            "AllieRuntimeError: recursion too deep -- the call nesting exceeded "
            "the host interpreter's stack limit",
            file=sys.stderr,
        )
    else:
        print(error, file=sys.stderr)
    return 1


def _execute_tree(program: object) -> int:
    env = stdlib.create_global_environment()
    try:
        run_program(program, env)  # type: ignore[arg-type]
    except (AllieRuntimeError, RecursionError) as error:
        return _report_runtime_error(error)
    return 0


def _execute_ir(main_program: object, functions: object) -> int:
    env = stdlib.create_global_environment()
    try:
        run_ir_program(main_program, env, functions, stdlib.BUILTINS)  # type: ignore[arg-type]
    except (AllieRuntimeError, RecursionError) as error:
        return _report_runtime_error(error)
    return 0


def _execute_llvm(
    main_program: object,
    functions: object,
    register_types: object,
    function_info: object,
    debug_ir: bool,
) -> int:
    try:
        lowering = LLVMLowering(functions, function_info)  # type: ignore[arg-type]
        lowering.declare_user_functions()
        for name in function_info:  # type: ignore[attr-defined]
            lowering.lower_function(name)
        main_func = lowering.lower_main(main_program, register_types)  # type: ignore[arg-type]
    except LLVMLoweringError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if debug_ir:
        print("=== LLVM IR ===")
        print(lowering.module)

    try:
        compile_and_run(lowering.module, main_func.name)
    except (AllieRuntimeError, RecursionError) as error:
        return _report_runtime_error(error)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="allie", description="Allie compiler/interpreter CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a .ql source file")
    run.add_argument("file", type=Path, help="Path to a .ql source file")
    run.add_argument(
        "--backend",
        choices=BACKENDS,
        default="tree",
        help="Execution backend (default: tree-walking interpreter)",
    )
    run.add_argument(
        "--debug-tokens",
        "--dump-tokens",
        dest="debug_tokens",
        action="store_true",
        help="Print the token stream",
    )
    run.add_argument(
        "--debug-ast",
        "--dump-ast",
        dest="debug_ast",
        action="store_true",
        help="Print the parsed AST",
    )
    run.add_argument(
        "--debug-ir",
        action="store_true",
        help="Print both the Allie IR and the final LLVM IR",
    )
    run.add_argument(
        "--no-optimize",
        dest="optimize",
        action="store_false",
        help="Skip the IR optimizer passes",
    )
    run.add_argument("--trace", action="store_true", help="Print interpreter execution trace")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_pipeline(
            args.file,
            backend=args.backend,
            debug_tokens=args.debug_tokens,
            debug_ast=args.debug_ast,
            debug_ir=args.debug_ir,
            optimize=args.optimize,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
