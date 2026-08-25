"""Phase 6: a third, real execution backend -- LLVM codegen + MCJIT,
built on top of the custom IR (Phase 5). `allie run` still executes via
the tree-walking interpreter by default (docs/spec.md "Considered
alternatives" -- that decision hasn't changed); this backend is
selectable via `--backend llvm` for the reasons in `allie/codegen/
llvm_lowering.py`'s module docstring (native LLVM for scalar arithmetic
and user functions, bridged calls back into the existing Python/numpy
runtime for everything else).
"""

from allie.codegen.jit import compile_and_run
from allie.codegen.llvm_lowering import LLVMLowering, LLVMLoweringError, llvm_type_for

__all__ = ["LLVMLowering", "LLVMLoweringError", "compile_and_run", "llvm_type_for"]
