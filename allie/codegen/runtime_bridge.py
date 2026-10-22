"""Python-side runtime bridge. JIT-compiled machine code
cannot touch a `SeriesValue`/`PriceSeriesValue`/`str`/stdlib call
directly; every one of those crosses the LLVM/Python boundary as an
*opaque handle* (a plain 64-bit integer -- `id()` of a live Python
object), never a real struct or C string. `box()`/`unbox()` are the two
halves of that: `box()` registers an object and hands back a stable
handle a JIT function can pass around as an `i64`; `unbox()` reverses it.
A module-level registry (`_BOXED`) keeps every boxed object alive for the
JIT run's lifetime -- a bare `id()` doesn't hold a reference on its own,
so without this the object could be garbage-collected out from under
JIT-compiled code still holding its "pointer". `clear_registry()` resets
it between runs so a long-lived process (tests!) doesn't grow unbounded.

Strings are handles too, not pointers to global constants: `+`
concatenates them at run time, so a string's *value* isn't always known
at compile time the way a fixed global constant would require. Every
non-native-scalar type (String, Series, Matrix, PriceSeries, ...) uses
this same boxed-handle scheme -- one mechanism, not a special case per
type.

`binop_bridge`/`unaryop_bridge` reuse `allie.ir.interpreter._apply_binop`/
`_apply_unary` verbatim rather than reimplementing operator semantics a
third time (tree-walker, IR interpreter, now LLVM's bridged path all
agree by construction, not by re-derivation).

## Errors: a sticky pending-exception flag, not a raised exception

Verified directly, not assumed: a Python exception raised *inside* a
ctypes callback that's invoked from native code called by Python does
NOT propagate to that Python caller. ctypes prints "Exception ignored on
calling ctypes callback function" to stderr and the callback returns
whatever garbage happens to be in its return register -- the caller
never sees a `try`/`except`-catchable exception. Every bridge function
below is wrapped in `@_guarded`, which catches any exception, stores it
in `_pending_exception`, and returns a type-appropriate sentinel instead
of letting the exception try (and fail) to cross that boundary directly.
Once `_pending_exception` is set, `_guarded` short-circuits every
*subsequent* bridge call in the same JIT run to the sentinel too, without
doing any real work -- this stops a first failure from cascading into
confusing secondary errors (e.g. `unbox()` raising `KeyError` on a
sentinel handle that was never actually boxed) while the now-meaningless
JIT-compiled code runs to completion on garbage values. `jit.
compile_and_run` checks `_pending_exception` immediately after the JIT
call returns control to Python and re-raises it there, which is the
first point where a normal `raise` actually reaches the caller.
"""

from __future__ import annotations

import ctypes
import functools
from collections.abc import Callable
from typing import Any, TypeVar

from allie.ir.interpreter import _apply_binop, _apply_unary
from allie.quant.backtest_engine import (
    BacktestResult,
    add_strategy_clause,
    declare_strategy,
    run_registered_backtest,
    set_backtest_option,
)
from allie.runtime.elementwise import series_literal
from allie.runtime.errors import AllieNameError, AllieRuntimeError, AllieValueError
from allie.runtime.values import PriceSeriesValue
from allie.stdlib.registry import BUILTINS

_PRICE_SERIES_FIELDS = ("open", "high", "low", "close", "volume", "dates")

_BOXED: dict[int, Any] = {}

_pending_exception: BaseException | None = None
# Address of the JIT module's `__allie_error` byte, while one is running.
# Set to 1 whenever an error is recorded, so native code -- which checks
# it after every call -- stops at the next one instead of carrying on.
_error_flag_address: int | None = None

_F = TypeVar("_F", bound=Callable[..., Any])


def take_pending_exception() -> BaseException | None:
    """Returns and clears the pending exception, if any -- called once
    by `jit.compile_and_run` right after a JIT call returns."""
    global _pending_exception
    error, _pending_exception = _pending_exception, None
    return error


def set_error_flag_address(address: int | None) -> None:
    global _error_flag_address
    _error_flag_address = address


def _record(error: BaseException) -> None:
    global _pending_exception
    _pending_exception = error
    if _error_flag_address is not None:
        ctypes.c_uint8.from_address(_error_flag_address).value = 1


def _guarded(sentinel: Any) -> Callable[[_F], _F]:
    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any) -> Any:
            global _pending_exception
            if _pending_exception is not None:
                return sentinel
            try:
                return fn(*args)
            except Exception as error:  # noqa: BLE001 -- see module docstring
                _record(error)
                return sentinel

        return wrapper  # type: ignore[return-value]

    return decorator


def box(value: Any) -> int:
    """Registers `value` and returns a stable handle for JIT code to
    carry around as an opaque `i64`."""
    handle = id(value)
    _BOXED[handle] = value
    return handle


def unbox(handle: int) -> Any:
    return _BOXED[handle]


def clear_registry() -> None:
    """Releases every boxed object from this run. Call once a JIT
    invocation has returned and its result has been retrieved."""
    _BOXED.clear()


# ---- scalar box/unbox: crossing between a native LLVM i64/double/i1 and
# a boxed handle at a call boundary that expects everything boxed. ----


@_guarded(0)
def box_int(value: int) -> int:
    return box(int(value))


@_guarded(0)
def box_float(value: float) -> int:
    return box(float(value))


@_guarded(0)
def box_bool(value: int) -> int:
    return box(bool(value))


@_guarded(0)
def unbox_int(handle: int) -> int:
    return int(unbox(handle))


@_guarded(0.0)
def unbox_float(handle: int) -> float:
    return float(unbox(handle))


@_guarded(0)
def unbox_bool(handle: int) -> int:
    return int(bool(unbox(handle)))


# ---- generic bridges ----

_MAX_BRIDGED_ARITY = 4


@_guarded(0)
def call_builtin_bridge(
    name_handle: int, arg0: int, arg1: int, arg2: int, arg3: int, argc: int
) -> int:
    """One generic entry point for every fixed-arity bridged `Call` --
    stdlib builtins plus the three backtest-DSL calls
    (`__strategy_declare`/`__strategy_clause`/`__backtest_run`), which
    route into the same `allie.quant.backtest_engine` the tree-walking
    and IR paths use. `__make_series` is NOT dispatched here -- array
    literals can have arbitrarily many elements, more than this fixed
    4-argument signature allows, so it gets its own variable-arity
    `make_series_bridge` below."""
    name = unbox(name_handle)
    handles = [arg0, arg1, arg2, arg3][:argc]
    args = [unbox(h) for h in handles]

    if name == "__strategy_declare":
        declare_strategy(args[0])
        return 0
    if name == "__strategy_clause":
        add_strategy_clause(args[0], args[1])
        return 0
    if name == "__backtest_option":
        set_backtest_option(args[0], args[1])
        return 0
    if name == "__backtest_run":
        return box(run_registered_backtest(args[0], args[1], args[2], args[3]))
    if name not in BUILTINS:
        raise AllieValueError(f"cannot call undefined function '{name}'")
    _, fn = BUILTINS[name]
    result = fn(*args)
    return box(result)


@_guarded(0)
def make_series_bridge(elems_ptr: int, count: int) -> int:
    """Bridge target for `__make_series` (array-literal construction).
    Variable-arity, unlike `call_builtin_bridge`: `elems_ptr` points to
    `count` consecutive boxed `i64` handles in memory (an `alloca`'d
    array on the JIT side), read back here via ctypes rather than passed
    as fixed positional arguments."""
    if count == 0:
        return box(series_literal([]))
    array_type = ctypes.c_int64 * count
    handles = ctypes.cast(elems_ptr, ctypes.POINTER(array_type)).contents
    return box(series_literal([unbox(h) for h in handles]))


@_guarded(0)
def binop_bridge(op_handle: int, left_handle: int, right_handle: int) -> int:
    op = unbox(op_handle)
    left = unbox(left_handle)
    right = unbox(right_handle)
    return box(_apply_binop(op, left, right))


@_guarded(0)
def unaryop_bridge(op_handle: int, operand_handle: int) -> int:
    op = unbox(op_handle)
    operand = unbox(operand_handle)
    return box(_apply_unary(op, operand))


@_guarded(0)
def member_get_bridge(obj_handle: int, field_handle: int) -> int:
    """Bridge target for `MemberGet` (`prices.close`) -- same field
    logic as the tree-walker's/IR interpreter's field access, reused
    rather than reimplemented."""
    obj = unbox(obj_handle)
    field = unbox(field_handle)
    if isinstance(obj, PriceSeriesValue) and field in _PRICE_SERIES_FIELDS:
        return box(getattr(obj, field))
    if isinstance(obj, BacktestResult):
        return box(obj.field(field))
    raise AllieValueError(f"cannot access field '{field}' on {type(obj).__name__}")


@_guarded(0)
def raise_runtime_error_bridge(message_handle: int) -> int:
    """Bridge target for errors the native code itself detects: a zero
    divisor (checked before the division, since native division by zero
    is undefined behavior, not a controlled trap), Int overflow, a
    module-level variable used before it was assigned, recursion past
    the depth limit. The handle is a message (an AllieValueError) or a
    `(class name, message)` pair, so each error is raised as the same
    class the interpreters raise."""
    payload = unbox(message_handle)
    if isinstance(payload, tuple):
        class_name, message, *span = payload
        raise _ERROR_CLASSES[class_name](message, span[0] if span else None)
    raise AllieValueError(payload)


def locate_error_bridge(span_handle: int) -> int:
    """Called by native code on its way out after a failed call: gives
    the pending error the failing instruction's source span, unless an
    inner instruction already gave it one. Not `_guarded` -- it must run
    precisely when an error is pending."""
    error = _pending_exception
    if isinstance(error, AllieRuntimeError) and error.span is None:
        error.span = unbox(span_handle)
    return 0


_ERROR_CLASSES: dict[str, type[AllieRuntimeError]] = {
    "AllieNameError": AllieNameError,
    "AllieRuntimeError": AllieRuntimeError,
    "AllieValueError": AllieValueError,
}
