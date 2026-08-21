"""Runtime error hierarchy — distinct from allie.errors.

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

from allie.errors import SourceSpan


class AllieRuntimeError(Exception):
    """Base class for all errors raised while executing Allie code."""

    def __init__(self, message: str, span: SourceSpan | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.span = span


class AllieTypeError(AllieRuntimeError):
    """Raised when a gradual-typing annotation is violated at runtime."""


class AllieNameError(AllieRuntimeError):
    """Raised when an identifier is not bound in the current scope."""


class AllieValueError(AllieRuntimeError):
    """Raised for invalid operand values (e.g. division by zero)."""
