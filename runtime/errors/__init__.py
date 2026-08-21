"""Runtime error hierarchy — distinct from compiler.errors.

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

from compiler.errors import SourceSpan


class QuantLangRuntimeError(Exception):
    """Base class for all errors raised while executing QuantLang code."""

    def __init__(self, message: str, span: SourceSpan | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.span = span


class QuantLangTypeError(QuantLangRuntimeError):
    """Raised when a gradual-typing annotation is violated at runtime."""


class QuantLangNameError(QuantLangRuntimeError):
    """Raised when an identifier is not bound in the current scope."""


class QuantLangValueError(QuantLangRuntimeError):
    """Raised for invalid operand values (e.g. division by zero)."""
