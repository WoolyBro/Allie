"""Runtime error hierarchy — distinct from qubit.errors.

Design decision (Phase 0.6): runtime errors raise Python exceptions and
halt the running strategy, rather than propagating as Result-style
values. Compile-time problems (syntax, gradual-typing violations) are
DiagnosticBag entries; this module is for failures that only manifest
while a strategy is executing (division by zero, missing bar data,
out-of-bounds series access).

Result-style propagation for strategy code is deferred — see
docs/spec.md "Deferred" list. Revisit if halting on first runtime
error proves too coarse for long backtests.
"""

from __future__ import annotations

from qubit.errors import SourceSpan


class QubitRuntimeError(Exception):
    """Base class for all errors raised while executing Qubit code."""

    def __init__(self, message: str, span: SourceSpan | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.span = span


class QubitTypeError(QubitRuntimeError):
    """Raised when a gradual-typing annotation is violated at runtime."""


class QubitNameError(QubitRuntimeError):
    """Raised when an identifier is not bound in the current scope."""


class QubitValueError(QubitRuntimeError):
    """Raised for invalid operand values (e.g. division by zero)."""
