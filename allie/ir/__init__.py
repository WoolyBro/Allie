"""The intermediate representation: a flat list of instructions over
virtual registers (instructions.py), lowered from the typed AST
(lowering.py), checked by a verifier (verify.py), analyzed as a
control-flow graph (cfg.py), and run by its own interpreter
(interpreter.py).

It is the middle of two of the three execution paths: the IR
interpreter runs it, and the LLVM backend (allie.codegen) compiles it.
The tree-walking interpreter skips it. The optimizer (allie.optimizer)
works on this representation, so both IR-based paths run optimized
code, and `--verify` runs all three and reports a result only if they
agree.

See allie.ir.lowering's module docstring for the two significant
lowering decisions: a separate IRProgram per function, and functions
that are static and called by name rather than first-class values.
"""

from allie.ir.instructions import (
    BinOp,
    Call,
    IRInstr,
    IRProgram,
    Jump,
    JumpIfFalse,
    Label,
    LoadConst,
    LoadVar,
    MemberGet,
    Move,
    RegisterAllocator,
    Return,
    StoreVar,
    UnaryOp,
)
from allie.ir.interpreter import run_ir_program
from allie.ir.lowering import IRLowering, LoweringError, lower_program

__all__ = [
    "BinOp",
    "Call",
    "IRInstr",
    "IRLowering",
    "IRProgram",
    "Jump",
    "JumpIfFalse",
    "Label",
    "LoadConst",
    "LoadVar",
    "LoweringError",
    "MemberGet",
    "Move",
    "RegisterAllocator",
    "Return",
    "StoreVar",
    "UnaryOp",
    "lower_program",
    "run_ir_program",
]
