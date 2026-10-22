"""Command-line entry point for Allie.

    allie run FILE [--backend tree|ir|llvm] [--verify] [--trace] [--time] ...
    allie check FILE...
    allie repl
    allie --version

`run` takes the full pipeline: lex -> parse -> resolve -> type-check ->
look-ahead analysis -> (lower to IR -> optimize) -> execute, stopping
with every collected diagnostic printed if a static pass fails. `check`
runs only the static passes, over any number of files, for an editor or
a CI step. `repl` is an interactive session on the tree-walking
interpreter (allie/repl.py). FILE may be `-` for standard input.

## Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | the program has an error: a compile error, or a runtime error |
| 2 | the command was used wrong: an unknown flag, a file that can't be read |
| 3 | `--verify`: the backends disagreed, so no result is reported |
| 70 | an internal compiler error -- a bug in Allie, not in the program |

## Two entry points, deliberately

`python main.py FILE` is `allie run --backend llvm FILE`;
both share `run_pipeline` below.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import allie
import allie.stdlib as stdlib
from allie.ast.nodes import Program, Stmt
from allie.ast.printer import format_ast
from allie.errors import DiagnosticBag, source_line
from allie.ir.interpreter import run_ir_program
from allie.ir.lowering import LoweringError, lower_program
from allie.lexer.errors import LexError
from allie.lexer.lexer import Lexer
from allie.optimizer import DEFAULT_PASSES
from allie.optimizer import optimize as optimize_ir
from allie.parser.parser import Parser
from allie.quant.backtest_engine import reset_strategy_registry
from allie.runtime.errors import AllieRuntimeError
from allie.runtime.interpreter import run_program
from allie.semantic.lookahead import check_lookahead
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker
from allie.typechecker.errors import QSTypeError

if TYPE_CHECKING:
    from allie.ir.instructions import IRProgram
    from allie.ir.lowering import FunctionInfo
    from allie.typechecker.types import Type

BACKENDS = ("tree", "ir", "llvm")

EXIT_OK = 0
EXIT_PROGRAM_ERROR = 1
EXIT_USAGE = 2
EXIT_DISAGREEMENT = 3
EXIT_INTERNAL = 70

# Past this, more errors rarely help (a pasted binary or a wrong file
# produces thousands) and only push the first, most useful ones off screen.
_MAX_REPORTED_ERRORS = 20


@dataclass
class _Phases:
    """Wall-clock seconds per phase, for `--time`."""

    seconds: dict[str, float] = field(default_factory=dict)
    _last: float = field(default_factory=time.perf_counter)

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.seconds[name] = self.seconds.get(name, 0.0) + now - self._last
        self._last = now

    def render(self) -> str:
        parts = [f"{name} {seconds * 1000:.2f} ms" for name, seconds in self.seconds.items()]
        total = sum(self.seconds.values()) * 1000
        return f"[time] {' · '.join(parts)} · total {total:.2f} ms"


# ---- reading and checking source ----


def _read_source(source_path: Path) -> str | None:
    """The program text, or None after reporting why it can't be read."""
    if str(source_path) == "-":
        return sys.stdin.read()
    try:
        return source_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"error: no such file: {source_path}", file=sys.stderr)
    except IsADirectoryError:
        print(f"error: {source_path} is a directory, not an Allie source file", file=sys.stderr)
    except PermissionError:
        print(f"error: no permission to read {source_path}", file=sys.stderr)
    except UnicodeDecodeError as error:
        print(
            f"error: {source_path} is not valid UTF-8 "
            f"(byte 0x{error.object[error.start]:02x} at offset {error.start})",
            file=sys.stderr,
        )
    return None


def check_source(
    source: str, *, debug_tokens: bool = False, debug_ast: bool = False
) -> Program | None:
    """Every static pass: lex, parse, resolve, type-check, look-ahead.
    Returns the checked program, or None after printing every error."""
    program = type_check_source(source, debug_tokens=debug_tokens)
    if program is None or not lookahead_ok(program, source):
        return None
    if debug_ast:
        # After checking, so every expression shows its type.
        print("=== AST ===")
        print(format_ast(program))
    return program


def type_check_source(source: str, *, debug_tokens: bool = False) -> Program | None:
    """The passes up to and including type checking. Returns the typed
    program, or None after printing every error."""
    lexer = Lexer(source)
    tokens = lexer.scan_tokens()
    if debug_tokens:
        print("=== TOKENS ===")
        for token in tokens:
            print(f"{token.line}:{token.column} {token.type.name} {token.lexeme!r}")

    diagnostics = DiagnosticBag()
    program = Parser(tokens, diagnostics).parse()
    if lexer.errors:
        # Lex errors win over parse errors: a bad token usually causes the
        # parse errors, so showing both would bury the real problem.
        _print_errors(lexer.errors, source, "lexical")
        return None
    if diagnostics.has_errors():
        diagnostics.print_all(source)
        return None

    Resolver(diagnostics).resolve(program)
    if diagnostics.has_errors():
        diagnostics.print_all(source)
        return None

    checker = TypeChecker()
    for name, signature in stdlib.builtin_function_types().items():
        checker.env.define(name, signature)
    try:
        checker.check(program)
    except QSTypeError:
        # check() raises only after checking everything; report it all.
        _print_errors(checker.errors, source, "type")
        return None

    return program


def lookahead_ok(program: Program, source: str) -> bool:
    """The look-ahead analysis, which runs last: it reads the
    `inferred_type` annotations the checker wrote, and a data-provenance
    problem isn't worth reporting in a program that doesn't type-check."""
    diagnostics = check_lookahead(program)
    if diagnostics.has_errors():
        diagnostics.print_all(source)
        return False
    return True


def _print_errors(errors: Sequence[LexError | QSTypeError], source: str, kind: str) -> None:
    for error in errors[:_MAX_REPORTED_ERRORS]:
        print(error.render(source), file=sys.stderr)
    hidden = len(errors) - _MAX_REPORTED_ERRORS
    if hidden > 0:
        print(f"... and {hidden} more {kind} error(s)", file=sys.stderr)


# ---- running ----


def run_pipeline(
    source_path: Path,
    *,
    backend: str = "tree",
    debug_tokens: bool = False,
    debug_ast: bool = False,
    debug_ir: bool = False,
    optimize: bool = True,
    verify: bool = False,
    trace: bool = False,
    timing: bool = False,
) -> int:
    """Runs `source_path` end to end. Returns a process exit code."""
    source = _read_source(source_path)
    if source is None:
        return EXIT_USAGE
    try:
        return _run_source(
            source,
            backend=backend,
            debug_tokens=debug_tokens,
            debug_ast=debug_ast,
            debug_ir=debug_ir,
            optimize=optimize,
            verify=verify,
            trace=trace,
            timing=timing,
        )
    except _InternalError as error:
        return _report_internal_error(error.__cause__ or error, known=True)
    except Exception as error:  # noqa: BLE001 -- a bug in Allie, reported as one
        return _report_internal_error(error, known=False)


class _InternalError(Exception):
    """A failure that is Allie's fault, not the program's."""


def _run_source(
    source: str,
    *,
    backend: str,
    debug_tokens: bool,
    debug_ast: bool,
    debug_ir: bool,
    optimize: bool,
    verify: bool,
    trace: bool,
    timing: bool,
) -> int:
    phases = _Phases()
    program = check_source(source, debug_tokens=debug_tokens, debug_ast=debug_ast)
    if program is None:
        return EXIT_PROGRAM_ERROR
    phases.lap("front end")

    if verify:
        return _execute_verified(program, source, optimize=optimize)

    def execute() -> None:
        if backend == "tree" and not debug_ir:
            # Straight off the AST -- no lowering unless --debug-ir asked to
            # see it.
            _run_tree(program, trace=_tracer(source) if trace else None)
            return
        lowered = _lower(program, optimize, debug_ir)
        phases.lap("lowering")
        if backend == "tree":
            _run_tree(program, trace=_tracer(source) if trace else None)
        elif backend == "ir":
            _run_ir(*lowered[:2])
        else:
            _run_llvm(*lowered, debug_ir=debug_ir, optimize=optimize, phases=phases)

    try:
        execute()
        code = EXIT_OK
    except (AllieRuntimeError, RecursionError) as error:
        print(_runtime_error_text(error, source), file=sys.stderr)
        code = EXIT_PROGRAM_ERROR
    phases.lap("run")
    if timing:
        print(phases.render(), file=sys.stderr)
    return code


_Lowered = tuple[
    "IRProgram",
    dict[str, tuple["IRProgram", list[str]]],
    dict[str, "Type"],
    dict[str, "FunctionInfo"],
]


def _lower(program: Program, optimize: bool, debug_ir: bool = False) -> _Lowered:
    try:
        main_program, functions, register_types, function_info = lower_program(program)
    except LoweringError as error:
        raise _InternalError("lowering failed") from error
    # --debug-ir also verifies the IR after every pass and reports what
    # each pass removed.
    pass_report: list[str] = []
    main_program, functions = optimize_ir(
        main_program,
        functions,
        DEFAULT_PASSES,
        optimize,
        verify=debug_ir,
        report=pass_report if debug_ir else None,
    )
    if debug_ir:
        if pass_report:
            print("=== OPTIMIZER ===")
            for line in pass_report:
                print(line)
        print("=== ALLIE IR ===")
        for instr in main_program.instructions:
            print(instr)
        for name, (prog, params) in functions.items():
            print(f"--- function {name}({', '.join(params)}) ---")
            for instr in prog.instructions:
                print(instr)
    return main_program, functions, register_types, function_info


def _runtime_error_text(error: BaseException, source: str) -> str:
    """How every backend's runtime failure is shown: rendered with its
    source line and caret.

    `RecursionError` is translated rather than shown: an Allie call
    consumes several Python frames, so deeply-recursive Allie code trips
    CPython's limit. The limit is deliberately *not* raised: a higher
    ceiling trades a catchable exception for a C-stack overflow that takes
    the process down with no message at all."""
    if isinstance(error, RecursionError):
        return (
            "AllieRuntimeError: recursion too deep -- the call nesting exceeded "
            "the host interpreter's stack limit"
        )
    if isinstance(error, AllieRuntimeError):
        return error.render(source)
    return str(error)


def _report_internal_error(error: BaseException, *, known: bool) -> int:
    """`known`: a compiler stage reported it (LLVM rejected the module);
    otherwise an exception nothing anticipated."""
    what = str(error) if known else f"{type(error).__name__}: {error}"
    print(f"internal compiler error: {what}", file=sys.stderr)
    if os.environ.get("ALLIE_TRACEBACK"):
        traceback.print_exception(error)
    else:
        print(
            "  This is a bug in Allie, not in your program. "
            "Set ALLIE_TRACEBACK=1 to see the Python traceback.",
            file=sys.stderr,
        )
    return EXIT_INTERNAL


def _run_tree(program: Program, *, trace: Callable[[Stmt], None] | None = None) -> None:
    run_program(program, stdlib.create_global_environment(), trace=trace)


def _tracer(source: str) -> Callable[[Stmt], None]:
    """`--trace`: one line on stderr per statement executed."""

    def trace(statement: Stmt) -> None:
        line = statement.span.line
        text = (source_line(source, line) or "").strip()
        print(f"[trace] {line:>4} | {text}", file=sys.stderr)

    return trace


def _run_ir(main_program: IRProgram, functions: object) -> None:
    env = stdlib.create_global_environment()
    run_ir_program(main_program, env, functions, stdlib.BUILTINS)  # type: ignore[arg-type]


def _run_llvm(
    main_program: IRProgram,
    functions: dict[str, tuple[IRProgram, list[str]]],
    register_types: dict[str, Type],
    function_info: dict[str, FunctionInfo],
    *,
    debug_ir: bool = False,
    optimize: bool = True,
    phases: _Phases | None = None,
) -> None:
    """Compiles and runs on the JIT. `optimize` also turns LLVM's own
    `-O2` pipeline on or off, matching the IR passes. LLVM is imported
    here, not at the top, so the other backends start without loading it."""
    from allie.codegen.jit import JITError, compile_and_run
    from allie.codegen.llvm_lowering import LLVMLowering, LLVMLoweringError

    try:
        lowering = LLVMLowering(functions, function_info)
        lowering.declare_user_functions()
        for name in function_info:
            lowering.lower_function(name)
        main_func = lowering.lower_main(main_program, register_types)
    except LLVMLoweringError as error:
        raise _InternalError("LLVM lowering failed") from error

    if debug_ir:
        print("=== LLVM IR ===")
        print(lowering.module)

    def show_optimized(text: str) -> None:
        print("=== LLVM IR (after LLVM -O2) ===")
        print(text)

    try:
        compile_and_run(
            lowering.module,
            main_func.name,
            opt_level=2 if optimize else 0,
            show_optimized_ir=show_optimized if debug_ir else None,
            on_ready=(lambda: phases.lap("codegen")) if phases is not None else None,
        )
    except JITError as error:
        raise _InternalError(str(error)) from error


# ---- --verify ----


@dataclass(frozen=True)
class _Outcome:
    code: int
    output: str
    # What ended the run, as compared across backends; the error itself is
    # kept to render it, with its source line, once they've agreed.
    error_text: str = ""
    error: BaseException | None = field(default=None, compare=False)


def _run_capturing(program: Program, backend: str, *, optimize: bool) -> _Outcome:
    """Runs `program` on one backend with its output and any runtime
    error captured, so `--verify` compares both. A backend that can't
    run the program at all reports that as its error rather than
    raising, so a crash reads as a disagreement, not as Allie dying."""
    reset_strategy_registry()
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            if backend == "tree":
                _run_tree(program)
            else:
                lowered = _lower(program, optimize)
                if backend == "ir":
                    _run_ir(*lowered[:2])
                else:
                    _run_llvm(*lowered, optimize=optimize)
    except (AllieRuntimeError, RecursionError) as error:
        text = _runtime_error_text(error, "")
        return _Outcome(EXIT_PROGRAM_ERROR, out.getvalue(), text, error)
    except Exception as error:  # noqa: BLE001 -- surfaced in the report below
        cause = error.__cause__ or error
        return _Outcome(EXIT_INTERNAL, out.getvalue(), f"<{type(cause).__name__}: {cause}>")
    return _Outcome(EXIT_OK, out.getvalue())


def _execute_verified(program: Program, source: str, *, optimize: bool) -> int:
    """Runs the program on all three backends and reports its result only
    if they agree -- on the output, on the exit code, and on the error
    that ended the run, if one did.

    The three backends are independent implementations that share a front
    end but no execution logic. The test suite cross-validates them;
    `--verify` makes that a property of a user's own run. Disagreement is
    a hard failure, not a warning: if they don't agree, at least one is
    wrong, and there's no principled way to pick a winner.
    """
    results = {backend: _run_capturing(program, backend, optimize=optimize) for backend in BACKENDS}
    outcomes = set(results.values())

    if len(outcomes) == 1:
        outcome = results[BACKENDS[0]]
        print(outcome.output, end="")
        if outcome.error is not None:
            print(_runtime_error_text(outcome.error, source), file=sys.stderr)
        status = (
            "verified" if outcome.code == EXIT_OK else "verified (all backends failed identically)"
        )
        print(f"[{status}: {', '.join(BACKENDS)} agree]", file=sys.stderr)
        return outcome.code

    print("error: VERIFICATION FAILED -- backends disagree", file=sys.stderr)
    print(
        "  Three independent implementations produced different results, so at "
        "least one is wrong.\n"
        "  Refusing to report a result that cannot be trusted.",
        file=sys.stderr,
    )
    for backend, outcome in results.items():
        shown = outcome.output + (f"{outcome.error_text}\n" if outcome.error_text else "")
        rendered = shown if shown.strip() else "<no output>"
        indented = "\n".join(f"      {line}" for line in rendered.rstrip("\n").split("\n"))
        print(f"\n  --- {backend} (exit {outcome.code}) ---\n{indented}", file=sys.stderr)
    return EXIT_DISAGREEMENT


# ---- check ----


def check_files(paths: Sequence[Path], *, audit: bool = False) -> int:
    """`allie check`: the static passes over every file, every error
    reported, nothing run -- unless `audit`, which then also runs each
    program that type-checks through the look-ahead audit, even one the
    analysis rejected: it shows whether the leak is real on this data."""
    failed = unreadable = 0
    for path in paths:
        source = _read_source(path)
        if source is None:
            unreadable += 1
            continue
        program = type_check_source(source)
        if program is None:
            failed += 1
            continue
        passed = lookahead_ok(program, source)
        if audit:
            passed = _audit(program, source) and passed
        failed += not passed
    checked = len(paths) - unreadable
    if not failed and not unreadable:
        print(f"ok: {checked} file{'s' if checked != 1 else ''} checked", file=sys.stderr)
        return EXIT_OK
    return EXIT_USAGE if unreadable and not failed else EXIT_PROGRAM_ERROR


def _audit(program: Program, source: str) -> bool:
    from allie.semantic.audit import audit

    try:
        report = audit(program)
    except (AllieRuntimeError, RecursionError) as error:
        print(_runtime_error_text(error, source), file=sys.stderr)
        print("look-ahead audit: not run -- the program fails on its data", file=sys.stderr)
        return False
    print(report.render(), file=sys.stderr)
    return report.clean


# ---- arguments ----


def add_run_arguments(parser: argparse.ArgumentParser, *, default_backend: str) -> None:
    """The `run` options -- shared with main.py so the two can't drift."""
    parser.add_argument("file", type=Path, help="an Allie source file (.qs), or - for stdin")
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=default_backend,
        help=f"execution backend (default: {default_backend})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run on all three backends and report a result only if they agree",
    )
    parser.add_argument(
        "--no-optimize",
        dest="optimize",
        action="store_false",
        help="turn off the IR optimizer passes and LLVM's -O2",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="print each statement as it runs (tree backend)",
    )
    parser.add_argument(
        "--time",
        dest="timing",
        action="store_true",
        help="print how long each phase took",
    )
    parser.add_argument(
        "--debug-tokens",
        "--dump-tokens",
        dest="debug_tokens",
        action="store_true",
        help="print the token stream",
    )
    parser.add_argument(
        "--debug-ast",
        "--dump-ast",
        dest="debug_ast",
        action="store_true",
        help="print the type-checked AST",
    )
    parser.add_argument(
        "--debug-ir",
        action="store_true",
        help="print the Allie IR, and the LLVM IR on the llvm backend",
    )


_EPILOG = """\
exit codes:
  0   success
  1   the program has an error (compile or runtime)
  2   usage error: bad arguments, or a file that can't be read
  3   --verify: the backends disagreed
  70  internal compiler error (a bug in Allie)
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="allie",
        description="Allie: a statically typed language for quantitative research.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"allie {allie.__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = subparsers.add_parser(
        "run",
        help="run a program",
        description="Check and run an Allie program.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_run_arguments(run, default_backend="tree")

    check = subparsers.add_parser(
        "check",
        help="check programs without running them",
        description="Run every static check -- syntax, scopes, types, look-ahead bias -- "
        "and report every error, without running anything.",
    )
    check.add_argument("files", type=Path, nargs="+", help="Allie source files")
    check.add_argument(
        "--audit",
        action="store_true",
        help="also run each program once per bar, its data cut after that bar, and report "
        "any trading decision that changes (a look-ahead leak, shown by experiment)",
    )

    subparsers.add_parser(
        "repl",
        help="an interactive session",
        description="Type statements and expressions; each is checked and run as you go.",
    )
    return parser


def validate_run_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.trace and (args.backend != "tree" or args.verify):
        parser.error("--trace needs the tree backend (and not --verify)")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        validate_run_arguments(parser, args)
        return run_pipeline(
            args.file,
            backend=args.backend,
            debug_tokens=args.debug_tokens,
            debug_ast=args.debug_ast,
            debug_ir=args.debug_ir,
            optimize=args.optimize,
            verify=args.verify,
            trace=args.trace,
            timing=args.timing,
        )
    if args.command == "check":
        return check_files(args.files, audit=args.audit)
    if args.command == "repl":
        from allie.repl import run_repl

        return int(run_repl())
    parser.print_help()
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
