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
    """Base class for all errors raised while executing Allie code.

    `__str__` matches the shared format used by `LexError` and
    `QSTypeError` -- `{Kind}: {message}. Line {L}, Column {C}` -- but
    drops the location clause when no span is available.

    **Known limitation, stated rather than hidden:** almost no runtime
    error carries a span today. The overwhelming majority are raised
    inside stdlib functions operating on *data* ("mean: series must not
    be empty"), where the meaningful context is the value, not a source
    position -- and the runtime doesn't thread the currently-executing
    node's span down into those calls. Doing so is a v2 change touching
    every stdlib entry point; see docs/how-it-works.md "What's not
    done". Until then, printing a bare message is more honest than
    inventing a location.
    """

    def __init__(self, message: str, span: SourceSpan | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.span = span

    def __str__(self) -> str:
        kind = type(self).__name__
        if self.span is None:
            return f"{kind}: {self.message}"
        return f"{kind}: {self.message}. Line {self.span.line}, Column {self.span.column}"


class AllieTypeError(AllieRuntimeError):
    """Raised when a gradual-typing annotation is violated at runtime."""


class AllieNameError(AllieRuntimeError):
    """Raised when an identifier is not bound in the current scope."""


class AllieValueError(AllieRuntimeError):
    """Raised for invalid operand values (e.g. division by zero)."""
