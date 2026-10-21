"""Runtime error hierarchy — distinct from allie.errors.

Design decision (Phase 0.6): runtime errors raise Python exceptions and
halt the running strategy, rather than propagating as Result-style
values. Compile-time problems (syntax, type errors) are
DiagnosticBag entries; this module is for failures that only manifest
while a strategy is executing (division by zero, missing bar data,
out-of-bounds series access).

Result-style propagation for strategy code is deferred — see
docs/spec.md "Deferred" list. Revisit if halting on first runtime
error proves too coarse for long backtests.
"""

from __future__ import annotations

from allie.errors import SourceSpan, format_excerpt, source_line


class AllieRuntimeError(Exception):
    """Base class for all errors raised while executing Allie code.

    `__str__` matches the shared format used by `LexError` and
    `QSTypeError` -- `{Kind}: {message}. Line {L}, Column {C}` -- and
    `render(source)` adds the source line with a caret, as compile errors
    do. The span is the innermost expression that failed, attached by
    whichever backend ran it: the tree-walking interpreter from the AST
    node, the IR interpreter and the JIT from the IR instruction lowered
    from that node (allie.ir.instructions.IRInstr.span). So all three
    report a failure at the same line and column, which their
    cross-validation checks. An error raised with no span (inside a
    stdlib function, say) gets the span of the call that reached it.
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

    def render(self, source: str) -> str:
        """`str(self)`, then the failing line with a caret under the
        expression that failed, when there is one."""
        rendered = str(self)
        if self.span is not None:
            text = source_line(source, self.span.line)
            if text is not None:
                rendered += "\n" + format_excerpt(text, self.span.column, self.span.length)
        return rendered


class AllieTypeError(AllieRuntimeError):
    """Raised when a value has the wrong type at runtime -- which a
    type-checked program never does; it guards the interpreters against
    an unchecked AST."""


class AllieNameError(AllieRuntimeError):
    """Raised when an identifier is not bound in the current scope."""


class AllieValueError(AllieRuntimeError):
    """Raised for invalid operand values (e.g. division by zero)."""
