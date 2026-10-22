"""Cross-cutting checklist: every user-facing error type carries
location information where it has any, and formats it identically.

Before this file existed the three types had drifted: `LexError` used
`"... (line 1, col 7)"`, `QSTypeError` used `"... . Line 1, Column 7"`,
and `AllieRuntimeError` had no `__str__` at all -- so runtime errors
printed with no location and no type prefix. `QSTypeError`'s wording was
pinned by an explicit spec requirement and a test asserting the exact
string, so it became canonical and the other two were brought into line.
"""

from __future__ import annotations

import re

import pytest

from allie.errors import SourceSpan
from allie.lexer.errors import LexError
from allie.runtime.errors import (
    AllieNameError,
    AllieRuntimeError,
    AllieTypeError,
    AllieValueError,
)
from allie.typechecker.errors import QSTypeError

# "{Kind}: {message}. Line {L}, Column {C}"
_LOCATED = re.compile(r"^[A-Za-z]+: .+\. Line \d+, Column \d+$")


@pytest.mark.parametrize(
    "error",
    [
        LexError("unexpected character '@'", 3, 7),
        QSTypeError("cannot assign String to Int", 3, 7),
        AllieValueError("division by zero", SourceSpan(line=3, column=7, offset=0)),
    ],
    ids=["LexError", "QSTypeError", "AllieRuntimeError"],
)
def test_located_errors_share_one_format(error: Exception) -> None:
    assert _LOCATED.match(str(error)), f"{type(error).__name__} formats as {str(error)!r}"


@pytest.mark.parametrize(
    ("error", "line", "column"),
    [
        (LexError("boom", 3, 7), 3, 7),
        (QSTypeError("boom", 3, 7), 3, 7),
    ],
    ids=["LexError", "QSTypeError"],
)
def test_line_and_column_are_exposed_as_attributes(
    error: LexError | QSTypeError, line: int, column: int
) -> None:
    """Not just embedded in the string -- callers (the CLI, tests, a
    future LSP) need them structured."""
    assert error.line == line
    assert error.column == column


def test_runtime_error_exposes_span_when_it_has_one() -> None:
    span = SourceSpan(line=4, column=9, offset=0)
    error = AllieValueError("boom", span)
    assert error.span is span
    assert "Line 4, Column 9" in str(error)


def test_runtime_error_without_a_span_omits_the_location_clause() -> None:
    """Most runtime errors originate inside stdlib functions operating
    on data, where no source position is available. Printing a bare
    message is correct there -- inventing a location would be worse.
    See AllieRuntimeError's docstring."""
    error = AllieValueError("mean: series must not be empty")
    assert str(error) == "AllieValueError: mean: series must not be empty"
    assert error.span is None


@pytest.mark.parametrize(
    "error_class", [AllieRuntimeError, AllieTypeError, AllieNameError, AllieValueError]
)
def test_every_runtime_error_subclass_names_itself(error_class: type[AllieRuntimeError]) -> None:
    """The prefix comes from the concrete class, so a new subclass gets
    correct formatting for free rather than silently reporting as its
    base class."""
    assert str(error_class("boom")).startswith(f"{error_class.__name__}: ")


def test_exception_message_still_matches_str_message_attribute() -> None:
    """`.message` is the raw text; `str()` adds the prefix and location.
    Keeping both available means callers can format their own output
    without string-parsing ours."""
    error = QSTypeError("cannot assign String to Int", 1, 14)
    assert error.message == "cannot assign String to Int"
    assert str(error) == "TypeError: cannot assign String to Int. Line 1, Column 14"
