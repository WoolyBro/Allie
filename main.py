"""`allie run`, with the LLVM/JIT backend as the default:

    python main.py <file.qs> [--debug-tokens] [--debug-ast] [--debug-ir] [--no-optimize]

The same options, from the same definition (`allie.cli.add_run_arguments`),
so the two entry points can't drift apart. The pipeline: lex -> parse ->
resolve -> type-check -> look-ahead -> lower to IR -> optimize -> LLVM
lowering -> MCJIT execute.
"""

from __future__ import annotations

import argparse
import sys

from allie.cli import _EPILOG, add_run_arguments, run_pipeline, validate_run_arguments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Compile and run an Allie source file (LLVM/JIT by default).",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_run_arguments(parser, default_backend="llvm")
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
