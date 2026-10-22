"""The third execution backend: LLVM codegen and MCJIT, compiling the
custom IR (allie.ir). Native LLVM for scalar arithmetic and user
functions, bridged calls back into the Python/numpy runtime for
everything else -- see llvm_lowering.py's module docstring.
`--backend llvm` selects it (`python main.py` defaults to it); `allie
run` defaults to the tree-walking interpreter, which starts faster.
"""

from allie.codegen.jit import compile_and_run
from allie.codegen.llvm_lowering import LLVMLowering, LLVMLoweringError, llvm_type_for

__all__ = ["LLVMLowering", "LLVMLoweringError", "compile_and_run", "llvm_type_for"]
