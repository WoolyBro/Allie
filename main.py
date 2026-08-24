"""Phase 6.4 CLI entry point:

    python main.py <file.qs> [--debug-tokens] [--debug-ast] [--debug-ir] [--no-optimize]

All four flags are independent and combinable. Defaults to the LLVM/JIT
backend, per Phase 6.4's specified pipeline (lex -> parse -> resolve ->
type-check -> lower to IR -> optimize -> LLVM lowering -> MCJIT execute).

`allie run <file>` (the installed console script) is the other entry
point -- same pipeline, but defaulting to the tree-walking interpreter
and offering `--backend {tree,ir,llvm}`. See `allie/cli.py`'s module
docstring for why both exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from allie.cli import BACKENDS, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py", description="Compile and run an Allie source file"
    )
    parser.add_argument("file", type=Path, help="Path to an Allie source file")
    parser.add_argument("--debug-tokens", action="store_true", help="Print the token stream")
    parser.add_argument("--debug-ast", action="store_true", help="Print the parsed AST")
    parser.add_argument(
        "--debug-ir",
        action="store_true",
        help="Print both the Allie IR and the final LLVM IR",
    )
    parser.add_argument(
        "--no-optimize",
        dest="optimize",
        action="store_false",
        help="Skip the IR optimizer passes",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="llvm",
        help="Execution backend (default: llvm)",
    )
    args = parser.parse_args(argv)

    return run_pipeline(
        args.file,
        backend=args.backend,
        debug_tokens=args.debug_tokens,
        debug_ast=args.debug_ast,
        debug_ir=args.debug_ir,
        optimize=args.optimize,
    )


if __name__ == "__main__":
    sys.exit(main())
