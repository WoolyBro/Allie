"""Phase 5: intermediate representation, lowering, and a second IR-based
interpreter -- a real, working second execution path, not the
aspirational skeleton this package started as.

**Note on scope, since this goes beyond what was earlier agreed**: when
this package was first scaffolded (Phase 0), the plan was "tree-walking
interpreter stays what actually runs for the foreseeable phases; ir/
optimizer/codegen are aspirational placeholders" (docs/spec.md
"Considered alternatives"). Phase 5 explicitly asked for a real,
independent execution path with a cross-validation test suite against
the tree-walking interpreter, so that's what got built here -- the
tree-walking interpreter (allie.runtime.interpreter) is still what
`allie run` actually executes; this IR path exists for the optimizer
passes (allie.optimizer) to operate on and be demonstrated against, per
Phase 5.3/5.4.

See allie.ir.lowering's module docstring for the two significant
lowering decisions (separate IRProgram per function; functions are
static/name-registered, not first-class closures) and their consequence
for what the cross-validation suite can and can't cover.
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
    LoadSeries,
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
    "LoadSeries",
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
