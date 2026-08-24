"""JIT execution -- Phase 6.4.

Verified against currently-installed llvmlite (0.49, bundling LLVM
22.1.0) rather than assumed from memory, per the spec's own warning that
this API drifts across versions:

- `llvm.initialize()` is deprecated *and removed* in this version -- a
  bare call to it raises `RuntimeError` outright, not a warning.
  `initialize_native_target()` + `initialize_native_asmprinter()` are
  still present and still required before any code generation.
- `ExecutionEngine.add_global_mapping()` needs the function's `ValueRef`
  from the *parsed/verified backing module*
  (`llvm.parse_assembly(...).get_function(name)`), not the
  `llvmlite.ir.Function` builder object used to construct the module --
  passing the latter raises a `ctypes.ArgumentError`
  ("expected LP_LLVMValue instance instead of Function"), not something
  obvious from either object's docstring.
- `backing_module.get_function(name)` raises `NameError` (not returning
  `None`) when the module never referenced that symbol -- verified
  directly rather than guessed, since guessing wrong here would silently
  skip binding a symbol the module actually needs.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable

import llvmlite.binding as llvm
import llvmlite.ir as ir

from allie.codegen import runtime_bridge

_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    _initialized = True


_I64 = ctypes.c_int64
_F64 = ctypes.c_double

# name -> (ctypes signature, Python implementation). Must stay in sync
# with the external declarations allie.codegen.llvm_lowering emits.
_BRIDGES: dict[str, tuple[type[ctypes._FuncPointer], Callable[..., object]]] = {
    "box_int": (ctypes.CFUNCTYPE(_I64, _I64), runtime_bridge.box_int),
    "box_float": (ctypes.CFUNCTYPE(_I64, _F64), runtime_bridge.box_float),
    "box_bool": (ctypes.CFUNCTYPE(_I64, _I64), runtime_bridge.box_bool),
    "unbox_int": (ctypes.CFUNCTYPE(_I64, _I64), runtime_bridge.unbox_int),
    "unbox_float": (ctypes.CFUNCTYPE(_F64, _I64), runtime_bridge.unbox_float),
    "unbox_bool": (ctypes.CFUNCTYPE(_I64, _I64), runtime_bridge.unbox_bool),
    "binop_bridge": (ctypes.CFUNCTYPE(_I64, _I64, _I64, _I64), runtime_bridge.binop_bridge),
    "unaryop_bridge": (ctypes.CFUNCTYPE(_I64, _I64, _I64), runtime_bridge.unaryop_bridge),
    "member_get_bridge": (ctypes.CFUNCTYPE(_I64, _I64, _I64), runtime_bridge.member_get_bridge),
    "make_series_bridge": (
        ctypes.CFUNCTYPE(_I64, _I64, _I64),
        runtime_bridge.make_series_bridge,
    ),
    "raise_runtime_error_bridge": (
        ctypes.CFUNCTYPE(_I64, _I64),
        runtime_bridge.raise_runtime_error_bridge,
    ),
    "call_builtin_bridge": (
        ctypes.CFUNCTYPE(_I64, _I64, _I64, _I64, _I64, _I64, _I64),
        runtime_bridge.call_builtin_bridge,
    ),
}


def compile_and_run(module: ir.Module, entry_fn_name: str) -> None:
    """Parses/verifies `module`, JIT-compiles it, binds every bridge
    symbol it actually references to its Python implementation, and
    calls `entry_fn_name` (must take no arguments and return nothing --
    the shape `allie.codegen.llvm_lowering.LLVMLowering.lower_main`
    produces).

    Re-raises any error the runtime bridge recorded during the run
    (`allie.codegen.runtime_bridge`'s module docstring explains why a
    bridge can't just raise directly: an exception raised inside a
    ctypes callback invoked from JIT code is swallowed, never reaching
    the Python caller). Clears the box registry either way.
    """
    _ensure_initialized()
    runtime_bridge.take_pending_exception()  # discard any stale flag

    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly(str(module))
    backing_mod.verify()

    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)

    # Keep every ctypes callback object alive for the engine's lifetime:
    # ctypes doesn't hold its own reference to a CFUNCTYPE instance, and
    # a garbage-collected callback becomes a dangling function pointer
    # that JIT-compiled code would crash calling.
    keepalive: list[object] = []
    for name, (signature, impl) in _BRIDGES.items():
        try:
            backing_fn = backing_mod.get_function(name)
        except NameError:
            continue  # this module never referenced this bridge symbol
        callback = signature(impl)
        keepalive.append(callback)
        addr = ctypes.cast(callback, ctypes.c_void_p).value
        engine.add_global_mapping(backing_fn, addr)

    engine.finalize_object()
    engine.run_static_constructors()

    fn_ptr = engine.get_function_address(entry_fn_name)
    if fn_ptr == 0:
        raise RuntimeError(f"JIT engine produced a null address for '{entry_fn_name}'")
    entry_cfunc = ctypes.CFUNCTYPE(None)(fn_ptr)
    try:
        entry_cfunc()  # keepalive/engine must still be alive at this point
    finally:
        del keepalive

    pending = runtime_bridge.take_pending_exception()
    runtime_bridge.clear_registry()
    if pending is not None:
        raise pending
